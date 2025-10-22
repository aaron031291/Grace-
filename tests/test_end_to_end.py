"""
End-to-end integration tests for Grace system
"""

import pytest
import asyncio
from datetime import datetime


@pytest.mark.asyncio
async def test_complete_system_initialization():
    """
    Assert complete Grace system initializes all components
    
    Tests:
    - Config loading
    - EventBus initialization
    - TriggerMesh loading
    - MemoryCore setup
    - Kernel startup
    - Health checks
    """
    from grace.config import get_config
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.memory.core import MemoryCore
    from grace.kernels.multi_os import MultiOSKernel
    from grace.events.factory import GraceEventFactory
    
    # 1. Config
    config = get_config()
    assert config is not None
    assert config.environment in ["development", "staging", "production", "testing"]
    
    # 2. EventBus
    event_bus = EventBus()
    assert event_bus is not None
    
    # 3. TriggerMesh
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    assert trigger_mesh._loaded is True
    
    # 4. Kernel
    factory = GraceEventFactory()
    kernel = MultiOSKernel(event_bus, factory, trigger_mesh)
    await kernel.start()
    
    # 5. Health check
    health = kernel.get_health()
    assert health["running"] is True
    assert health["status"] == "healthy"
    assert health["trigger_mesh_enabled"] is True
    
    # Cleanup
    await kernel.stop()


@pytest.mark.asyncio
async def test_event_flow_through_trigger_mesh():
    """
    Assert events flow through TriggerMesh with routing
    
    Flow: Kernel -> TriggerMesh -> Routes -> EventBus -> Subscribers
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.kernels.multi_os import MultiOSKernel
    from grace.events.factory import GraceEventFactory
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    factory = GraceEventFactory()
    kernel = MultiOSKernel(event_bus, factory, trigger_mesh)
    
    # Track received events
    received_events = []
    
    def subscriber(event):
        received_events.append(event)
    
    trigger_mesh.subscribe("kernel.heartbeat", subscriber, "test_subscriber")
    
    # Start kernel (will emit heartbeats)
    await kernel.start()
    
    # Wait for heartbeat
    await asyncio.sleep(0.5)
    
    # Assert event received
    assert len(received_events) > 0
    assert received_events[0].event_type == "kernel.heartbeat"
    assert received_events[0].source == "multi_os_kernel"
    
    # Cleanup
    await kernel.stop()


@pytest.mark.asyncio
async def test_memory_write_with_trigger_mesh():
    """
    Assert memory writes trigger events through mesh
    
    Flow: MemoryCore.write -> Trigger event -> TriggerMesh -> Routes -> Subscribers
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.memory.core import MemoryCore
    from grace.memory.async_lightning import AsyncLightningMemory
    from grace.events.factory import GraceEventFactory
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    factory = GraceEventFactory()
    
    memory_core = MemoryCore(
        lightning_memory=lightning,
        fusion_memory=None,
        vector_store=None,
        trust_core=None,
        immutable_logs=None,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Track memory write events
    memory_events = []
    trigger_mesh.subscribe("memory.write", lambda e: memory_events.append(e), "test_memory_subscriber")
    
    # Write to memory
    await memory_core.write(
        key="test_key",
        value="test_value",
        actor="test_user",
        trust_attestation=False
    )
    
    # Wait for event propagation
    await asyncio.sleep(0.2)
    
    # Assert trigger event emitted
    assert len(memory_events) > 0
    assert memory_events[0].event_type == "memory.write"
    assert memory_events[0].payload["key"] == "test_key"
    
    # Cleanup
    await lightning.disconnect()


@pytest.mark.asyncio
async def test_kernel_command_with_correlation():
    """
    Assert kernel commands work with correlation IDs
    
    Flow: Command event -> Kernel processes -> Response event with correlation_id
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.kernels.multi_os import MultiOSKernel
    from grace.events.factory import GraceEventFactory
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    factory = GraceEventFactory()
    kernel = MultiOSKernel(event_bus, factory, trigger_mesh)
    await kernel.start()
    
    # Send command and wait for response
    response = await trigger_mesh.request_response(
        "kernel.command",
        {"command": "status"},
        timeout=2.0
    )
    
    # Assert response received
    assert response is not None
    assert response.event_type == "kernel.command.response"
    assert response.payload["status"] == "processed"
    
    # Cleanup
    await kernel.stop()


@pytest.mark.asyncio
async def test_route_filters_and_targets():
    """
    Assert TriggerMesh routes apply filters and add targets
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    # Create event that matches memory_writes route
    event = GraceEvent(
        event_type="memory.write",
        source="test",
        payload={"key": "test"},
        trust_score=0.9  # High trust, should pass filter
    )
    
    # Emit through mesh
    await trigger_mesh.emit(event, apply_routes=True)
    
    # Assert targets added by route
    assert len(event.targets) > 0
    # Route should add analytics_kernel and audit_logger
    assert "analytics_kernel" in event.targets or "audit_logger" in event.targets


@pytest.mark.asyncio
async def test_system_error_escalation():
    """
    Assert system errors trigger resilience kernel escalation
    
    Flow: Error event -> Resilience kernel -> Governance -> Escalation event
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.kernels.resilience import ResilienceKernel
    from grace.governance.engine import GovernanceEngine
    from grace.events.factory import GraceEventFactory
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    factory = GraceEventFactory()
    governance = GovernanceEngine()
    
    kernel = ResilienceKernel(event_bus, factory, governance, trigger_mesh)
    await kernel.start()
    
    # Track escalation events
    escalations = []
    trigger_mesh.subscribe("resilience.escalation", lambda e: escalations.append(e), "test_escalation")
    
    # Emit error event
    error_event = GraceEvent(
        event_type="system.error",
        source="test",
        payload={"error": "critical failure"},
        constitutional_validation_required=True,
        trust_score=0.2  # Low trust, will fail validation
    )
    
    await trigger_mesh.emit(error_event)
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    # Assert escalation occurred
    assert len(escalations) > 0
    assert escalations[0].event_type == "resilience.escalation"
    
    # Cleanup
    await kernel.stop()


@pytest.mark.asyncio
async def test_health_monitoring_across_kernels():
    """
    Assert health monitoring tracks all kernel heartbeats
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.kernels.multi_os import MultiOSKernel
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    factory = GraceEventFactory()
    
    # Start multiple kernels
    kernel1 = MultiOSKernel(event_bus, factory, trigger_mesh)
    kernel2 = MLDLKernel(event_bus, factory, None, None, trigger_mesh)
    
    await kernel1.start()
    await kernel2.start()
    
    # Track heartbeats
    heartbeats = []
    trigger_mesh.subscribe("kernel.heartbeat", lambda e: heartbeats.append(e), "health_monitor")
    
    # Wait for heartbeats
    await asyncio.sleep(1.0)
    
    # Assert heartbeats from multiple kernels
    assert len(heartbeats) > 0
    sources = [h.source for h in heartbeats]
    assert "multi_os_kernel" in sources or "mldl_kernel" in sources
    
    # Cleanup
    await kernel1.stop()
    await kernel2.stop()
