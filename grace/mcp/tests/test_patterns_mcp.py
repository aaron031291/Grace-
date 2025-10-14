"""
Unit tests for PatternsMCP handler using the compatibility shims.

These tests verify:
- Basic handler instantiation and method invocation
- Semantic search with vector store integration
- Audit log creation via FusionDB
- Pushback scenarios (governance rejection, retry, escalation)
"""
import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone
from dataclasses import dataclass

from grace.mcp.handlers.patterns_mcp import (
    PatternsMCP,
    PatternCreateRequest,
    SemanticSearchRequest,
)
from grace.mcp.base_mcp import MCPContext
from grace.mcp.pushback import PushbackHandler


def make_test_context(request_id: str = "test_req") -> MCPContext:
    """Helper to create a test MCPContext with proper structure."""
    @dataclass
    class MockCaller:
        id: str = "test_user"
        user_id: str = "test_user"
        session_id: str = "test_session"
        roles: list = None
        def __post_init__(self):
            if self.roles is None:
                self.roles = []
    
    @dataclass 
    class MockRequest:
        method: str = "POST"
        path: str = "/patterns"
        body: dict = None
        def __post_init__(self):
            if self.body is None:
                self.body = {}
    
    return MCPContext(
        caller=MockCaller(),
        request=MockRequest(),
        manifest={},
        endpoint_config={},
        start_time=time.time(),
        request_id=request_id
    )


@pytest.fixture
def mcp_handler():
    """Create a PatternsMCP instance with mocked dependencies."""
    handler = PatternsMCP()
    # Mock external services
    handler.event_bus = Mock()
    handler.event_bus.emit = AsyncMock()
    handler.governance = Mock()
    handler.governance.check_policy = AsyncMock(return_value={"allowed": True})
    return handler


@pytest.fixture
def mcp_context():
    """Create a minimal MCPContext for testing."""
    return make_test_context("test_req_001")


@pytest.mark.asyncio
async def test_create_pattern_basic(mcp_handler, mcp_context):
    """Test creating a pattern with basic metadata."""
    import time
    request = PatternCreateRequest(
        who="test_user",
        what="Test workflow pattern",
        where="test_environment",
        when=time.time(),
        why="Testing pattern creation",
        how="Manual test execution",
        raw_text="Test user created a workflow pattern in test environment for testing pattern creation via manual test execution",
        metadata={"version": "1.0", "author": "test"},
        tags=["test", "workflow"]
    )
    
    result = await mcp_handler.create_pattern(request, mcp_context)
    
    assert result["status"] == "success"
    assert "pattern_id" in result
    # Verify event emission
    assert mcp_handler.event_bus.emit.called


@pytest.mark.asyncio
async def test_semantic_search(mcp_handler, mcp_context):
    """Test semantic search across patterns."""
    import time
    # First create a pattern to search for
    create_req = PatternCreateRequest(
        who="resilience_kernel",
        what="Error handling pattern with retries",
        where="resilience_domain",
        when=time.time(),
        why="Improve system resilience",
        how="Automatic pattern detection",
        raw_text="Resilience kernel detected error handling pattern with retry logic in resilience domain to improve system resilience through automatic pattern detection",
        metadata={"domain": "resilience"},
        tags=["resilience", "error_handling", "retry"]
    )
    await mcp_handler.create_pattern(create_req, mcp_context)
    
    # Now search for it
    search_req = SemanticSearchRequest(
        query="error handling retry logic",
        top_k=5,
        filters={"tags": "resilience"},
    )
    
    results = await mcp_handler.semantic_search(search_req, mcp_context)
    
    assert results["status"] == "success"
    assert "results" in results
    # Should find at least the pattern we just created
    assert len(results["results"]) > 0


@pytest.mark.asyncio
async def test_audit_trail():
    """Test that FusionDB audit logs are created correctly."""
    from grace.ingress_kernel.db.fusion_db import FusionDB
    
    db = FusionDB.get_instance()
    
    # Insert test audit log using actual schema
    audit_payload = {
        "category": "test_action",
        "data": {"test_key": "test_value"},
        "hash": "test_hash_001",
        "prev_hash": "test_hash_000",
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }
    
    entry_id = await db.insert("audit_logs", audit_payload)
    assert entry_id is not None
    
    # Verify retrieval
    row = await db.query_one(
        "SELECT * FROM audit_logs WHERE entry_id = ?", (entry_id,)
    )
    assert row is not None
    assert row["category"] == "test_action"


@pytest.mark.asyncio
async def test_pushback_governance_rejection():
    """Test pushback flow when governance rejects a request."""
    import time
    from grace.mcp.pushback import PushbackPayload, PushbackCategory, PushbackSeverity
    
    pushback = PushbackHandler()
    
    payload = PushbackPayload(
        error_code="GOV_QUOTA_EXCEEDED",
        category=PushbackCategory.GOVERNANCE,
        severity=PushbackSeverity.MEDIUM,
        message="Quota exceeded for pattern creation",
        domain="patterns",
        timestamp=time.time(),
        caller_id="test_user",
        request_id="test_req_gov_reject",
        retry_after_seconds=60,
        remediation_steps=["Wait 60 seconds", "Retry request"],
        metadata={"reason": "quota_exceeded", "retry_after": 60}
    )
    
    with patch.object(pushback, '_emit_events', new_callable=AsyncMock):
        result = await pushback.handle_pushback(payload)
    
    assert result["audit_id"] is not None
    assert result["observation_id"] is not None


@pytest.mark.asyncio
async def test_pushback_retry_logic():
    """Test that pushback correctly schedules retries."""
    import time
    from grace.mcp.pushback import PushbackPayload, PushbackCategory, PushbackSeverity
    
    pushback = PushbackHandler()
    
    payload = PushbackPayload(
        error_code="TRANSIENT_TIMEOUT",
        category=PushbackCategory.SERVICE_DEGRADATION,
        severity=PushbackSeverity.LOW,
        message="Vector search operation timed out",
        domain="patterns",
        timestamp=time.time(),
        caller_id="test_user",
        request_id="test_req_retry",
        retry_after_seconds=5,
        remediation_steps=["Retry with exponential backoff"],
        metadata={"error": "timeout", "retryable": True, "operation": "vector_search"}
    )
    
    # Simulate transient error
    with patch.object(pushback, '_emit_events', new_callable=AsyncMock):
        result = await pushback.handle_pushback(payload)
    
    assert result["audit_id"] is not None
    assert result["observation_id"] is not None


@pytest.mark.asyncio
async def test_memory_orchestrator_healing():
    """Test MemoryOrchestrator healing request."""
    from grace.mlt_kernel_ml.memory_orchestrator import MemoryOrchestrator
    
    orchestrator = MemoryOrchestrator.get_instance()
    
    heal_context = {
        "error_type": "vector_store_corruption",
        "affected_domain": "patterns",
        "severity": "high",
    }
    
    ticket = await orchestrator.request_healing(heal_context)
    
    assert ticket["ticket_id"] is not None
    assert ticket["status"] == "scheduled"
    assert ticket["context"] == heal_context


@pytest.mark.asyncio
async def test_observation_recording():
    """Test that observations are recorded in FusionDB."""
    from grace.ingress_kernel.db.fusion_db import FusionDB
    
    db = FusionDB.get_instance()
    
    obs_payload = {
        "observation_type": "pattern_created",
        "source_module": "patterns_mcp",
        "observation_data": {"pattern_id": "p123", "name": "test"},
        "context": {"session_id": "sess_001"},
        "credibility_score": 0.95,
        "novelty_score": 0.3,
        "observed_at": datetime.now(timezone.utc).timestamp(),
    }
    
    obs_id = await db.insert("observations", obs_payload)
    assert obs_id is not None
    
    # Retrieve and verify
    row = await db.query_one(
        "SELECT * FROM observations WHERE observation_id = ?", (obs_id,)
    )
    assert row["observation_type"] == "pattern_created"
    assert row["credibility_score"] == 0.95


@pytest.mark.asyncio
async def test_full_mcp_lifecycle(mcp_handler, mcp_context):
    """End-to-end test: create pattern, search, verify audit, check observation."""
    import time
    from grace.ingress_kernel.db.fusion_db import FusionDB
    
    db = FusionDB.get_instance()
    
    # 1. Create a pattern
    create_req = PatternCreateRequest(
        who="test_system",
        what="Full lifecycle test pattern",
        where="integration_tests",
        when=time.time(),
        why="Testing complete MCP lifecycle",
        how="Automated integration test",
        raw_text="Test system created full lifecycle test pattern in integration tests for testing complete MCP lifecycle via automated integration test",
        metadata={"test": True},
        tags=["test", "integration", "lifecycle"]
    )
    
    create_result = await mcp_handler.create_pattern(create_req, mcp_context)
    assert create_result["status"] == "success"
    pattern_id = create_result["pattern_id"]
    
    # 2. Search for it
    search_req = SemanticSearchRequest(
        query="lifecycle test integration",
        top_k=3,
    )
    search_result = await mcp_handler.semantic_search(search_req, mcp_context)
    assert search_result["status"] == "success"
    
    # 3. Verify audit logs exist
    audit_logs = await db.query_many(
        "SELECT * FROM audit_logs WHERE action LIKE '%pattern%' ORDER BY created_at DESC LIMIT 5"
    )
    assert len(audit_logs) > 0
    
    # 4. Verify observations
    observations = await db.query_many(
        "SELECT * FROM observations WHERE source_module = 'patterns_mcp' ORDER BY observed_at DESC LIMIT 5"
    )
    assert len(observations) > 0


if __name__ == "__main__":
    # Run tests directly (for debugging)
    pytest.main([__file__, "-v", "-s"])
