"""
Tests for MCP (Message Control Protocol)
"""

import pytest
import asyncio

from grace.mcp import MCPClient, MCPMessage, MCPMessageType, MCPSchema, MCPValidator


def test_mcp_schema_validation():
    """Assert MCP schema validates correctly"""
    schema = MCPSchema(
        name="test_schema",
        version="1.0",
        fields={
            "name": {"type": "string"},
            "count": {"type": "number"}
        },
        required_fields=["name"]
    )
    
    # Valid data
    valid_data = {"name": "test", "count": 5}
    is_valid, errors = schema.validate(valid_data)
    assert is_valid is True
    assert len(errors) == 0
    
    # Missing required field
    invalid_data = {"count": 5}
    is_valid, errors = schema.validate(invalid_data)
    assert is_valid is False
    assert "Missing required field: name" in errors
    
    # Wrong type
    wrong_type = {"name": 123}
    is_valid, errors = schema.validate(wrong_type)
    assert is_valid is False
    assert "must be string" in errors[0]


def test_mcp_message_creation():
    """Assert MCP messages are created correctly"""
    message = MCPMessage(
        message_id="msg_123",
        message_type=MCPMessageType.REQUEST,
        source="kernel_a",
        destination="kernel_b",
        payload={"data": "test"},
        trust_score=0.8
    )
    
    assert message.message_id == "msg_123"
    assert message.source == "kernel_a"
    assert message.trust_score == 0.8
    
    # Convert to dict and back
    data = message.to_dict()
    recreated = MCPMessage.from_dict(data)
    
    assert recreated.message_id == message.message_id
    assert recreated.source == message.source
    assert recreated.trust_score == message.trust_score


def test_mcp_validator():
    """Assert MCP validator works correctly"""
    validator = MCPValidator()
    
    # Valid message
    valid_msg = MCPMessage(
        message_id="msg_456",
        message_type=MCPMessageType.HEARTBEAT,
        source="test_kernel",
        destination="monitor",
        payload={"kernel": "test", "uptime_seconds": 100},
        trust_score=0.9
    )
    valid_msg.headers["schema"] = "heartbeat"
    
    is_valid, errors = validator.validate_message(valid_msg)
    assert is_valid is True
    
    # Invalid trust score
    invalid_trust = MCPMessage(
        message_id="msg_789",
        message_type=MCPMessageType.REQUEST,
        source="test",
        destination="dest",
        payload={},
        trust_score=1.5  # Invalid
    )
    
    is_valid, errors = validator.validate_message(invalid_trust)
    assert is_valid is False
    assert any("trust_score" in e for e in errors)


@pytest.mark.asyncio
async def test_mcp_client_send():
    """Assert MCP client sends messages correctly"""
    from grace.integration.event_bus import EventBus
    
    event_bus = EventBus()
    mcp_client = MCPClient(
        kernel_name="test_kernel",
        event_bus=event_bus,
        minimum_trust=0.5
    )
    
    # Track sent events
    sent_events = []
    event_bus.subscribe("mcp.request", lambda e: sent_events.append(e))
    
    # Send message
    message = await mcp_client.send_message(
        destination="target_kernel",
        payload={"test": "data"},
        message_type=MCPMessageType.REQUEST,
        trust_score=0.8
    )
    
    assert message.source == "test_kernel"
    assert message.destination == "target_kernel"
    assert mcp_client.messages_sent == 1
    
    # Wait for async delivery
    await asyncio.sleep(0.1)
    assert len(sent_events) > 0


@pytest.mark.asyncio
async def test_mcp_client_validation_failure():
    """Assert MCP client rejects invalid messages"""
    from grace.integration.event_bus import EventBus
    
    event_bus = EventBus()
    mcp_client = MCPClient(
        kernel_name="test_kernel",
        event_bus=event_bus,
        minimum_trust=0.8
    )
    
    # Try to send message with low trust
    with pytest.raises(ValueError) as exc_info:
        await mcp_client.send_message(
            destination="target",
            payload={},
            trust_score=0.5  # Below minimum
        )
    
    assert "Trust score" in str(exc_info.value)
    assert mcp_client.validation_failures == 1


@pytest.mark.asyncio
async def test_mcp_schema_enforcement():
    """Assert MCP enforces schema on messages"""
    from grace.integration.event_bus import EventBus
    
    event_bus = EventBus()
    mcp_client = MCPClient(
        kernel_name="test_kernel",
        event_bus=event_bus
    )
    
    # Send with invalid schema
    with pytest.raises(ValueError):
        await mcp_client.send_message(
            destination="target",
            payload={"count": 5},  # Missing required 'kernel' field
            schema_name="heartbeat",  # Uses heartbeat schema
            trust_score=0.9
        )
