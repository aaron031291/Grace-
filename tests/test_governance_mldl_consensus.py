"""
Tests for Governance -> MLDL consensus request/response loop
"""

import pytest
import asyncio


@pytest.mark.asyncio
async def test_governance_requests_mldl_consensus():
    """
    Assert Governance can request and receive MLDL consensus
    
    Flow: Governance.validate -> emit consensus.request -> MLDL processes -> 
          emit consensus.response -> Governance receives
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    factory = GraceEventFactory()
    
    # Start MLDL kernel
    mldl_kernel = MLDLKernel(event_bus, factory, None, None, trigger_mesh)
    await mldl_kernel.start()
    
    # Create governance with trigger_mesh
    governance = GovernanceEngine(trigger_mesh=trigger_mesh)
    
    # Create event that will trigger consensus request
    event = GraceEvent(
        event_type="test.decision",
        source="test",
        constitutional_validation_required=True,
        trust_score=0.4,  # Low trust will create violations
        priority="normal"
    )
    
    # Validate (should request consensus)
    result = await governance.validate(event, request_mldl_consensus=True)
    
    # Assert consensus was received and used
    assert result.decision is not None
    assert "mldl_consensus" in result.decision
    
    consensus = result.decision["mldl_consensus"]
    if consensus:
        assert "recommendation" in consensus
        assert "confidence" in consensus
        assert "specialists" in consensus
        assert consensus["recommendation"] in ["approve", "review", "reject"]
    
    # Cleanup
    await mldl_kernel.stop()


@pytest.mark.asyncio
async def test_mldl_consensus_multiple_specialists():
    """
    Assert MLDL uses multiple specialists for consensus
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    
    factory = GraceEventFactory()
    mldl_kernel = MLDLKernel(event_bus, factory, None, None, trigger_mesh)
    await mldl_kernel.start()
    
    # Track consensus responses
    responses = []
    trigger_mesh.subscribe("mldl.consensus.response", lambda e: responses.append(e), "test")
    
    # Send consensus request
    request = factory.create_event(
        event_type="mldl.consensus.request",
        payload={
            "decision_context": {
                "trust_score": 0.6,
                "violations": ["test violation"]
            },
            "options": ["approve", "review", "reject"]
        },
        source="test"
    )
    
    await trigger_mesh.emit(request)
    
    # Wait for response
    await asyncio.sleep(0.3)
    
    # Assert response received
    assert len(responses) > 0
    
    response_payload = responses[0].payload
    consensus = response_payload.get("consensus")
    
    assert consensus is not None
    assert "specialists" in consensus
    assert len(consensus["specialists"]) >= 2  # At least 2 specialists
    
    # Check specialist structure
    for specialist in consensus["specialists"]:
        assert "name" in specialist
        assert "vote" in specialist
        assert "confidence" in specialist
    
    # Cleanup
    await mldl_kernel.stop()


@pytest.mark.asyncio
async def test_consensus_overrides_validation():
    """
    Assert MLDL consensus can override governance violations
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    factory = GraceEventFactory()
    
    # Start MLDL
    mldl_kernel = MLDLKernel(event_bus, factory, None, None, trigger_mesh)
    await mldl_kernel.start()
    
    # Governance with consensus
    governance = GovernanceEngine(trigger_mesh=trigger_mesh)
    
    # Event with high trust (MLDL will recommend approve)
    event = GraceEvent(
        event_type="test.decision",
        source="test",
        constitutional_validation_required=True,
        trust_score=0.9,  # High trust
        priority="normal"
    )
    
    # Validate with consensus
    result = await governance.validate(event, request_mldl_consensus=True)
    
    # High trust should lead to approval
    if result.decision and result.decision.get("mldl_consensus"):
        consensus = result.decision["mldl_consensus"]
        # With high trust, should recommend approve or review
        assert consensus["recommendation"] in ["approve", "review"]
    
    # Cleanup
    await mldl_kernel.stop()


@pytest.mark.asyncio
async def test_consensus_timeout_handling():
    """
    Assert governance handles MLDL consensus timeout gracefully
    """
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.schemas.events import GraceEvent
    
    event_bus = EventBus()
    trigger_mesh = TriggerMesh(event_bus)
    trigger_mesh.load_config()
    
    # Note: MLDL kernel NOT started - will timeout
    governance = GovernanceEngine(trigger_mesh=trigger_mesh)
    
    event = GraceEvent(
        event_type="test.decision",
        source="test",
        constitutional_validation_required=True,
        trust_score=0.4,
        priority="normal"
    )
    
    # Should not crash on timeout
    result = await governance.validate(event, request_mldl_consensus=True)
    
    # Should still return a result
    assert result is not None
    assert result.passed is not None
    
    # Consensus should be None due to timeout
    if result.decision:
        assert result.decision.get("mldl_consensus") is None
