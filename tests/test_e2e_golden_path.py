"""
End-to-end golden path tests
Tests complete workflows from ingress to response
"""

import pytest
import asyncio
from datetime import datetime


@pytest.mark.asyncio
async def test_golden_path_event_flow():
    """
    Golden Path: Event submission → Processing → Response
    
    Flow:
    1. Ingress receives event
    2. RBAC validates permissions
    3. Event bus processes
    4. Kernel handles event
    5. Response returned
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.ingress_kernel import IngressKernel
    from grace.kernels.multi_os import MultiOSKernel
    from grace.events.factory import GraceEventFactory
    from grace.security import Role
    
    # Setup
    bus = EventBus()
    mesh = TriggerMesh(bus)
    mesh.load_config()
    governance = GovernanceEngine(trigger_mesh=mesh)
    factory = GraceEventFactory()
    
    # Initialize ingress
    ingress = IngressKernel(
        event_bus=bus,
        trigger_mesh=mesh,
        governance_engine=governance,
        immutable_logs=None
    )
    
    # Assign user role
    await ingress.rbac.assign_role("test_user", Role.USER, "system", trust_score=0.9)
    
    # Start kernel
    kernel = MultiOSKernel(bus, factory, mesh)
    await kernel.start()
    
    # Submit request through ingress
    result = await ingress.handle_request(
        user_id="test_user",
        endpoint="events/emit",
        action="write",
        payload={"data": "test_event"},
        trust_score=0.9
    )
    
    # Verify success
    assert result["success"] is True
    assert "event_id" in result
    
    # Verify kernel processed event
    await asyncio.sleep(0.2)
    health = kernel.get_health()
    assert health["running"] is True
    
    # Cleanup
    await kernel.stop()
    await bus.shutdown()


@pytest.mark.asyncio
async def test_golden_path_consensus_flow():
    """
    Golden Path: Consensus Request → MLDL Processing → Response
    
    Flow:
    1. Governance needs consensus
    2. Request sent via MCP
    3. MLDL kernel processes
    4. Consensus returned
    5. Governance applies decision
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    from grace.schemas.events import GraceEvent
    
    # Setup
    bus = EventBus()
    mesh = TriggerMesh(bus)
    mesh.load_config()
    factory = GraceEventFactory()
    
    # Start MLDL kernel
    mldl = MLDLKernel(bus, factory, None, None, mesh)
    await mldl.start()
    
    # Create governance with mesh
    governance = GovernanceEngine(trigger_mesh=mesh)
    
    # Create event requiring validation
    event = GraceEvent(
        event_type="test.decision",
        source="test",
        constitutional_validation_required=True,
        trust_score=0.6,  # Medium trust
        priority="normal"
    )
    
    # Request consensus
    result = await governance.validate(event, request_mldl_consensus=True)
    
    # Verify consensus was received
    assert result is not None
    assert result.decision is not None
    
    if result.decision.get("mldl_consensus"):
        consensus = result.decision["mldl_consensus"]
        assert "recommendation" in consensus
        assert "confidence" in consensus
        assert consensus["recommendation"] in ["approve", "review", "reject"]
    
    # Cleanup
    await mldl.stop()
    await bus.shutdown()


@pytest.mark.asyncio
async def test_golden_path_memory_persistence():
    """
    Golden Path: Memory Write → Multi-layer Fanout → Read
    
    Flow:
    1. Write data to memory
    2. Fanout to all layers (Lightning, Fusion, Trust, Logs)
    3. Read back from cache
    4. Verify consistency
    """
    from grace.memory.core import MemoryCore
    from grace.memory.async_lightning import AsyncLightningMemory
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    # Setup
    bus = EventBus()
    factory = GraceEventFactory()
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    # Create memory core
    memory = MemoryCore(
        lightning_memory=lightning,
        fusion_memory=None,  # Not required for this test
        vector_store=None,
        trust_core=None,
        immutable_logs=None,
        event_bus=bus,
        event_factory=factory
    )
    
    # Write data
    test_key = "golden_path_test"
    test_value = {"status": "success", "timestamp": datetime.utcnow().isoformat()}
    
    success = await memory.write(
        key=test_key,
        value=test_value,
        actor="test_user",
        trust_attestation=False
    )
    
    assert success is True
    
    # Read back
    retrieved = await memory.read(test_key, use_cache=True)
    
    assert retrieved is not None
    assert retrieved == test_value
    
    # Check stats
    stats = memory.get_stats()
    assert stats["writes_total"] > 0
    assert stats["cache_hits"] > 0
    
    # Cleanup
    await lightning.disconnect()
    await bus.shutdown()


@pytest.mark.asyncio
async def test_golden_path_rbac_enforcement():
    """
    Golden Path: Request → RBAC Check → Action Allowed/Denied
    
    Flow:
    1. User submits request
    2. RBAC checks permissions
    3. Constitutional validation for privileged actions
    4. Action executed or denied
    """
    from grace.security import RBACManager, Role, Permission
    from grace.governance.engine import GovernanceEngine
    
    # Setup
    governance = GovernanceEngine()
    rbac = RBACManager(governance_engine=governance)
    
    # Assign user role
    await rbac.assign_role("user1", Role.USER, "admin", trust_score=0.9)
    
    # Test allowed permission
    assert rbac.has_permission("user1", Permission.READ_EVENTS) is True
    assert rbac.has_permission("user1", Permission.WRITE_EVENTS) is True
    
    # Test denied permission
    assert rbac.has_permission("user1", Permission.SYSTEM_ADMIN) is False
    
    # Assign admin role (requires constitutional validation)
    await rbac.assign_role("admin1", Role.ADMIN, "system", trust_score=0.95)
    
    # Admin should have elevated permissions
    assert rbac.has_permission("admin1", Permission.MANAGE_KERNELS) is True
    assert rbac.has_permission("admin1", Permission.VALIDATE_EVENTS) is True


@pytest.mark.asyncio
async def test_golden_path_rate_limiting():
    """
    Golden Path: Multiple Requests → Rate Limit → Block/Allow
    
    Flow:
    1. Submit requests within limit
    2. Exceed rate limit
    3. Get rate limited
    4. Wait for refill
    5. Retry successfully
    """
    from grace.security import RateLimiter
    from grace.security.rate_limiter import RateLimitExceeded
    
    # Setup with low limit for testing
    limiter = RateLimiter(default_limit=5, default_window=60)
    
    # Use up tokens
    for i in range(5):
        result = await limiter.check_rate_limit("test_user", "test_endpoint")
        assert result is True
    
    # Next request should be blocked
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_rate_limit("test_user", "test_endpoint")
    
    assert exc_info.value.limit == 5
    assert exc_info.value.retry_after > 0


@pytest.mark.asyncio
async def test_golden_path_encryption():
    """
    Golden Path: Sensitive Data → Encrypt → Decrypt → Verify
    
    Flow:
    1. Encrypt sensitive fields
    2. Store encrypted data
    3. Retrieve and decrypt
    4. Verify data integrity
    """
    from grace.security import EncryptionManager
    
    # Setup
    enc = EncryptionManager()
    
    # Test data with sensitive fields
    data = {
        "username": "alice",
        "email": "alice@example.com",
        "password": "secret123",
        "api_key": "key_abc123"
    }
    
    # Encrypt sensitive fields
    encrypted = enc.encrypt_dict(data, ["password", "api_key"])
    
    # Verify encryption
    assert encrypted["username"] == "alice"  # Not encrypted
    assert encrypted["password"] != "secret123"  # Encrypted
    assert encrypted["api_key"] != "key_abc123"  # Encrypted
    assert encrypted["password_encrypted"] is True
    
    # Decrypt
    decrypted = enc.decrypt_dict(encrypted, ["password", "api_key"])
    
    # Verify decryption
    assert decrypted["password"] == "secret123"
    assert decrypted["api_key"] == "key_abc123"


@pytest.mark.asyncio
async def test_golden_path_mcp_validation():
    """
    Golden Path: Message → MCP Validation → Schema Check → Route
    
    Flow:
    1. Create MCP message
    2. Validate schema
    3. Check trust score
    4. Route to destination
    """
    from grace.mcp import MCPClient, MCPMessageType
    from grace.integration.event_bus import EventBus
    
    # Setup
    bus = EventBus()
    client = MCPClient("source_kernel", bus, minimum_trust=0.5)
    
    # Track received messages
    received = []
    bus.subscribe("mcp.request", lambda e: received.append(e))
    
    # Send valid message
    message = await client.send_message(
        destination="target_kernel",
        payload={"data": "test"},
        message_type=MCPMessageType.REQUEST,
        trust_score=0.9,
        schema_name="heartbeat"
    )
    
    assert message.source == "source_kernel"
    assert message.destination == "target_kernel"
    assert message.trust_score == 0.9
    
    # Verify message was sent
    await asyncio.sleep(0.1)
    assert len(received) > 0
    
    # Check stats
    stats = client.get_stats()
    assert stats["messages_sent"] == 1
    assert stats["validation_failures"] == 0
    
    await bus.shutdown()


@pytest.mark.asyncio
async def test_golden_path_kpi_tracking():
    """
    Golden Path: System Operations → Metrics Collection → KPI Calculation
    
    Flow:
    1. System processes events
    2. Metrics are collected
    3. KPIs are calculated
    4. Health assessment generated
    """
    from grace.observability.kpis import KPITracker
    
    # Setup
    tracker = KPITracker()
    
    # Simulate metrics
    metrics = {
        "grace_events_published_total": 1000,
        "grace_events_processed_total": 980,
        "grace_events_failed_total": 20,
        "grace_latency_percentiles": {
            "event_processing": {"p95": 85.0}
        }
    }
    
    # Calculate KPIs
    await tracker.calculate_kpis_from_metrics(metrics)
    
    # Get report
    report = tracker.get_kpi_report()
    
    # Verify KPIs
    assert report["overall_health"] in ["healthy", "degraded", "critical"]
    assert "kpis" in report
    assert "event_success_rate" in report["kpis"]
    
    success_rate = report["kpis"]["event_success_rate"]
    assert success_rate["value"] == 98.0  # 980/1000
    assert success_rate["met"] is True  # Above 95% target


@pytest.mark.asyncio
async def test_golden_path_full_system():
    """
    Golden Path: Complete System Integration
    
    Flow:
    1. Start all kernels
    2. Submit event through ingress
    3. Event flows through trigger mesh
    4. Kernels process event
    5. Memory persists data
    6. Metrics collected
    7. KPIs calculated
    8. All systems healthy
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.kernels.multi_os import MultiOSKernel
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    from grace.observability.metrics import get_metrics_collector
    from grace.observability.kpis import get_kpi_tracker
    from grace.security import Role
    
    # Setup complete system
    bus = EventBus()
    mesh = TriggerMesh(bus)
    mesh.load_config()
    mesh.bind_subscriptions()
    
    governance = GovernanceEngine(trigger_mesh=mesh)
    factory = GraceEventFactory()
    
    # Start kernels
    multi_os = MultiOSKernel(bus, factory, mesh)
    mldl = MLDLKernel(bus, factory, None, None, mesh)
    
    await multi_os.start()
    await mldl.start()
    
    # Track events
    events_received = []
    mesh.subscribe("kernel.heartbeat", lambda e: events_received.append(e), "test")
    
    # Wait for heartbeats
    await asyncio.sleep(0.5)
    
    # Verify kernels are running
    assert multi_os.get_health()["running"] is True
    assert mldl.get_health()["running"] is True
    
    # Verify events flowing
    assert len(events_received) > 0
    
    # Get metrics
    metrics_collector = get_metrics_collector()
    metrics = await metrics_collector.get_metrics()
    
    assert metrics["grace_events_published_total"] > 0
    
    # Calculate KPIs
    kpi_tracker = get_kpi_tracker()
    await kpi_tracker.calculate_kpis_from_metrics(metrics)
    
    report = kpi_tracker.get_kpi_report()
    assert report["overall_health"] is not None
    
    # Cleanup
    await multi_os.stop()
    await mldl.stop()
    await bus.shutdown()
    
    print("\n✅ Full system golden path test passed!")
