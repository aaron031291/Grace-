"""
Tests for TriggerMesh event routing
"""

import pytest
import asyncio
from pathlib import Path


@pytest.mark.asyncio
async def test_trigger_mesh_load_config():
    """Assert TriggerMesh loads configuration from YAML"""
    from grace.trigger_mesh import TriggerMesh
    from grace.integration.event_bus import EventBus
    
    event_bus = EventBus()
    mesh = TriggerMesh(event_bus, config_path="config/trigger_mesh.yaml")
    
    mesh.load_config()
    
    assert mesh._loaded is True
    assert len(mesh.routes) > 0
    assert "memory_writes" in [r.name for r in mesh.routes]


@pytest.mark.asyncio
async def test_trigger_mesh_route_matching():
    """Assert routes match event patterns correctly"""
    from grace.trigger_mesh import TriggerMesh, TriggerMeshRoute
    from grace.schemas.events import GraceEvent
    from grace.integration.event_bus import EventBus
    
    event_bus = EventBus()
    mesh = TriggerMesh(event_bus)
    mesh.load_config()
    
    # Create test event
    event = GraceEvent(
        event_type="memory.write",
        source="test",
        payload={"key": "test"}
    )
    
    # Find matching routes
    matching = [r for r in mesh.routes if r.matches(event)]
    
    assert len(matching) > 0
    assert any(r.name == "memory_writes" for r in matching)


@pytest.mark.asyncio
async def test_trigger_mesh_filters():
    """Assert route filters work correctly"""
    from grace.trigger_mesh import TriggerMeshRoute
    from grace.schemas.events import GraceEvent
    
    route_config = {
        "name": "test_route",
        "pattern": "test.*",
        "targets": ["target1"],
        "filters": [
            {"type": "trust_threshold", "threshold": 0.7}
        ]
    }
    
    route = TriggerMeshRoute(route_config)
    
    # Event with high trust
    event_high = GraceEvent(
        event_type="test.event",
        source="test",
        trust_score=0.8
    )
    assert route.apply_filters(event_high) is True
    
    # Event with low trust
    event_low = GraceEvent(
        event_type="test.event",
        source="test",
        trust_score=0.5
    )
    assert route.apply_filters(event_low) is False


@pytest.mark.asyncio
async def test_trigger_mesh_emit_with_routing():
    """Assert emit applies routing and targets"""
    from grace.trigger_mesh import TriggerMesh
    from grace.integration.event_bus import EventBus
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    mesh = TriggerMesh(event_bus)
    mesh.load_config()
    
    event = GraceEvent(
        event_type="memory.write",
        source="test",
        payload={"test": True},
        trust_score=0.9
    )
    
    # Emit with routing
    success = await mesh.emit(event, apply_routes=True)
    
    assert success is True
    # Targets should be added by routes
    assert len(event.targets) > 0


@pytest.mark.asyncio
async def test_trigger_mesh_subscribe():
    """Assert subscribe wrapper works"""
    from grace.trigger_mesh import TriggerMesh
    from grace.integration.event_bus import EventBus
    
    event_bus = EventBus()
    mesh = TriggerMesh(event_bus)
    
    received = []
    
    def handler(event):
        received.append(event)
    
    mesh.subscribe("test.event", handler, subscriber_name="test_subscriber")
    
    # Emit event
    from grace.schemas.events import GraceEvent
    event = GraceEvent(event_type="test.event", source="test")
    await event_bus.emit(event)
    
    await asyncio.sleep(0.1)
    assert len(received) > 0


@pytest.mark.asyncio
async def test_trigger_mesh_wait_for():
    """Assert wait_for wrapper works"""
    from grace.trigger_mesh import TriggerMesh
    from grace.integration.event_bus import EventBus
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    mesh = TriggerMesh(event_bus)
    
    # Start waiting
    async def wait_task():
        return await mesh.wait_for(
            "test.wait",
            lambda e: e.payload.get("value") == "expected",
            timeout=1.0
        )
    
    task = asyncio.create_task(wait_task())
    
    # Emit matching event
    await asyncio.sleep(0.1)
    event = GraceEvent(
        event_type="test.wait",
        source="test",
        payload={"value": "expected"}
    )
    await event_bus.emit(event)
    
    result = await task
    assert result is not None
    assert result.payload["value"] == "expected"
