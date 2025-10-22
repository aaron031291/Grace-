"""
Test kernel management and health checks
"""

import pytest
import asyncio


@pytest.mark.asyncio
async def test_kernel_health_metrics():
    """Test that kernels expose health metrics"""
    from grace.kernels.multi_os import start, stop, get_health
    
    # Start kernel
    await start()
    
    # Wait for some events
    await asyncio.sleep(0.2)
    
    # Check health
    health = get_health()
    assert "events_processed" in health
    assert "errors" in health
    assert health["heartbeat_interval"] == 5
    
    # Stop kernel
    await stop()


@pytest.mark.asyncio
async def test_mldl_inference_routing():
    """Test MLDL kernel routes to LLM when available"""
    from grace.kernels.mldl import start, stop, get_health
    from grace.integration.event_bus import get_event_bus
    
    bus = get_event_bus()
    await start()
    
    # Check health before inference
    health = get_health()
    initial_count = health["inference_count"]
    
    # Request inference
    response = await bus.request_response(
        "mldl.infer",
        {"input": "test prompt", "task_type": "general"},
        timeout=1.0
    )
    
    # Check health after inference
    health = get_health()
    assert health["inference_count"] > initial_count
    assert "avg_inference_time_ms" in health
    
    if response:
        result = response.payload.get("result", {})
        assert "prediction" in result
        assert "model" in result
    
    await stop()


@pytest.mark.asyncio
async def test_resilience_escalation_metrics():
    """Test resilience kernel tracks escalations"""
    from grace.kernels.resilience import start, stop, get_health
    from grace.integration.event_bus import get_event_bus
    from grace.events.factory import GraceEventFactory
    
    bus = get_event_bus()
    factory = GraceEventFactory()
    
    await start()
    
    # Check initial metrics
    health = get_health()
    initial_escalations = health["escalations"]
    
    # Trigger an error that should escalate
    event = factory.create_event(
        event_type="system.error",
        payload={"error": "critical failure"},
        constitutional_validation_required=True,
        trust_score=0.3  # Low trust to trigger escalation
    )
    bus.publish(event)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check metrics updated
    health = get_health()
    assert health["validations_failed"] > 0
    assert "escalations" in health
    
    await stop()
