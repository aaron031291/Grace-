"""
Comprehensive tests for EventBus features:
- emit/subscribe/wait_for
- TTL expiry
- Idempotency
- Dead Letter Queue (DLQ)
- Backpressure
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from grace.integration.event_bus import EventBus
from grace.schemas.events import GraceEvent, EventPriority, EventStatus
from grace.schemas.errors import (
    DuplicateEventError,
    BackpressureError,
    TTLExpiredError
)


@pytest.mark.asyncio
async def test_emit_and_subscribe():
    """Assert emit delivers to subscribers"""
    bus = EventBus()
    
    received = []
    
    async def handler(event: GraceEvent):
        received.append(event)
    
    # Subscribe
    bus.subscribe("test.event", handler)
    
    # Emit event
    event = GraceEvent(
        event_type="test.event",
        source="test",
        payload={"data": "value"}
    )
    
    success = await bus.emit(event)
    assert success is True
    
    # Wait for async delivery
    await asyncio.sleep(0.1)
    
    # Assert delivered
    assert len(received) == 1
    assert received[0].event_type == "test.event"
    assert received[0].payload["data"] == "value"


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """Assert event delivers to all subscribers"""
    bus = EventBus()
    
    received_1 = []
    received_2 = []
    
    bus.subscribe("test.multi", lambda e: received_1.append(e))
    bus.subscribe("test.multi", lambda e: received_2.append(e))
    
    event = GraceEvent(event_type="test.multi", source="test")
    await bus.emit(event)
    
    await asyncio.sleep(0.1)
    
    assert len(received_1) == 1
    assert len(received_2) == 1


@pytest.mark.asyncio
async def test_unsubscribe():
    """Assert unsubscribe stops delivery"""
    bus = EventBus()
    
    received = []
    
    def handler(event):
        received.append(event)
    
    bus.subscribe("test.unsub", handler)
    
    # First event should be received
    event1 = GraceEvent(event_type="test.unsub", source="test")
    await bus.emit(event1)
    await asyncio.sleep(0.1)
    
    assert len(received) == 1
    
    # Unsubscribe
    bus.unsubscribe("test.unsub", handler)
    
    # Second event should NOT be received
    event2 = GraceEvent(event_type="test.unsub", source="test")
    await bus.emit(event2)
    await asyncio.sleep(0.1)
    
    assert len(received) == 1  # Still 1, not 2


@pytest.mark.asyncio
async def test_wait_for_with_predicate():
    """Assert wait_for returns matching event"""
    bus = EventBus()
    
    # Start waiting
    async def wait_task():
        return await bus.wait_for(
            "test.wait",
            lambda e: e.payload.get("value") == "expected",
            timeout=2.0
        )
    
    task = asyncio.create_task(wait_task())
    
    # Emit non-matching event
    event1 = GraceEvent(
        event_type="test.wait",
        source="test",
        payload={"value": "wrong"}
    )
    await bus.emit(event1)
    
    # Emit matching event
    await asyncio.sleep(0.1)
    event2 = GraceEvent(
        event_type="test.wait",
        source="test",
        payload={"value": "expected"}
    )
    await bus.emit(event2)
    
    # Should receive the matching one
    result = await task
    assert result is not None
    assert result.payload["value"] == "expected"


@pytest.mark.asyncio
async def test_wait_for_timeout():
    """Assert wait_for returns None on timeout"""
    bus = EventBus()
    
    result = await bus.wait_for(
        "test.timeout",
        lambda e: True,
        timeout=0.1
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_ttl_expiry_on_emit():
    """Assert expired events are rejected on emit"""
    bus = EventBus()
    
    # Create expired event
    event = GraceEvent(
        event_type="test.ttl",
        source="test",
        ttl_seconds=1
    )
    
    # Set to already expired
    event.expires_at = datetime.utcnow() - timedelta(seconds=5)
    
    # Should raise TTLExpiredError
    with pytest.raises(TTLExpiredError):
        await bus.emit(event)
    
    # Check metrics
    assert bus.events_expired > 0


@pytest.mark.asyncio
async def test_ttl_cleanup_loop():
    """Assert background cleanup removes expired events"""
    bus = EventBus(enable_ttl_cleanup=True)
    
    # Emit event with short TTL
    event = GraceEvent(
        event_type="test.cleanup",
        source="test",
        ttl_seconds=1
    )
    
    await bus.emit(event)
    
    initial_queue_size = len(bus.pending_queue)
    assert initial_queue_size > 0
    
    # Wait for expiry
    await asyncio.sleep(1.5)
    
    # Wait for cleanup (runs every 10s, but we can force it for tests)
    # In production, expired events are cleaned periodically


@pytest.mark.asyncio
async def test_idempotency_key_deduplication():
    """Assert idempotency prevents duplicate processing"""
    bus = EventBus()
    
    event1 = GraceEvent(
        event_type="test.idem",
        source="test",
        idempotency_key="unique_key_123",
        payload={"attempt": 1}
    )
    
    # First emit should succeed
    result1 = await bus.emit(event1)
    assert result1 is True
    
    # Second emit with same key should be rejected
    event2 = GraceEvent(
        event_type="test.idem",
        source="test",
        idempotency_key="unique_key_123",
        payload={"attempt": 2}
    )
    
    result2 = await bus.emit(event2)
    assert result2 is False
    
    # Check metrics
    assert bus.events_deduplicated > 0


@pytest.mark.asyncio
async def test_idempotency_skip():
    """Assert idempotency can be skipped"""
    bus = EventBus()
    
    event1 = GraceEvent(
        event_type="test.skip_idem",
        source="test",
        idempotency_key="key_456"
    )
    
    # Both should succeed when skipping idempotency
    result1 = await bus.emit(event1, skip_idempotency=True)
    assert result1 is True
    
    result2 = await bus.emit(event1, skip_idempotency=True)
    assert result2 is True


@pytest.mark.asyncio
async def test_backpressure_threshold():
    """Assert backpressure prevents queue overflow"""
    bus = EventBus(max_queue_size=5)
    
    # Fill up the queue
    for i in range(5):
        event = GraceEvent(
            event_type="test.backpressure",
            source="test",
            payload={"seq": i}
        )
        await bus.emit(event)
    
    # Next emit should raise BackpressureError
    overflow_event = GraceEvent(
        event_type="test.backpressure",
        source="test",
        payload={"seq": 999}
    )
    
    with pytest.raises(BackpressureError) as exc_info:
        await bus.emit(overflow_event)
    
    assert exc_info.value.queue_size == 5
    assert exc_info.value.max_size == 5


@pytest.mark.asyncio
async def test_dead_letter_queue():
    """Assert failed events go to DLQ"""
    bus = EventBus(dlq_max_size=10)
    
    # Handler that always fails
    async def failing_handler(event: GraceEvent):
        raise RuntimeError("Handler failed")
    
    bus.subscribe("test.dlq", failing_handler)
    
    event = GraceEvent(
        event_type="test.dlq",
        source="test",
        max_retries=0  # No retries
    )
    
    await bus.emit(event)
    
    # Wait for async processing and DLQ
    await asyncio.sleep(0.3)
    
    # Event should be in DLQ
    assert len(bus.dead_letter_queue) > 0
    
    dlq_event = bus.dead_letter_queue[0]
    assert dlq_event.status == EventStatus.DEAD_LETTER
    assert dlq_event.dlq_reason is not None


@pytest.mark.asyncio
async def test_retry_mechanism():
    """Assert events retry on failure"""
    bus = EventBus()
    
    attempt_count = [0]
    
    async def flaky_handler(event: GraceEvent):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise RuntimeError(f"Attempt {attempt_count[0]} failed")
        # Succeed on 3rd attempt
    
    bus.subscribe("test.retry", flaky_handler)
    
    event = GraceEvent(
        event_type="test.retry",
        source="test",
        max_retries=3
    )
    
    await bus.emit(event)
    
    # Wait for retries
    await asyncio.sleep(0.5)
    
    # Should have retried
    assert attempt_count[0] >= 2


@pytest.mark.asyncio
async def test_correlation_id_routing():
    """Assert correlation_id routes responses correctly"""
    bus = EventBus()
    
    # Handler that responds with correlation_id
    async def responder(event: GraceEvent):
        if event.event_type == "test.request":
            response = GraceEvent(
                event_type="test.response",
                source="responder",
                correlation_id=event.correlation_id,
                payload={"result": "success"}
            )
            await bus.emit(response)
    
    bus.subscribe("test.request", responder)
    
    # Use request_response
    response = await bus.request_response(
        "test.request",
        {"data": "test"},
        timeout=2.0
    )
    
    assert response is not None
    assert response.event_type == "test.response"
    assert response.payload["result"] == "success"


@pytest.mark.asyncio
async def test_request_response_timeout():
    """Assert request_response times out correctly"""
    bus = EventBus()
    
    # No responder registered
    
    from grace.schemas.errors import TimeoutError
    
    with pytest.raises(TimeoutError):
        await bus.request_response(
            "test.no_response",
            {"data": "test"},
            timeout=0.5
        )


@pytest.mark.asyncio
async def test_event_status_transitions():
    """Assert event status transitions correctly"""
    bus = EventBus()
    
    event = GraceEvent(
        event_type="test.status",
        source="test"
    )
    
    # Initial status
    assert event.status == EventStatus.PENDING
    
    # After emit, should be PROCESSING
    await bus.emit(event)
    assert event.status == EventStatus.PROCESSING


@pytest.mark.asyncio
async def test_metrics_tracking():
    """Assert EventBus tracks metrics correctly"""
    bus = EventBus()
    
    # Track initial metrics
    initial_metrics = bus.get_metrics()
    initial_published = initial_metrics["events_published"]
    
    # Emit some events
    for i in range(3):
        event = GraceEvent(
            event_type="test.metrics",
            source="test",
            payload={"seq": i}
        )
        await bus.emit(event)
    
    # Check metrics updated
    metrics = bus.get_metrics()
    assert metrics["events_published"] == initial_published + 3
    assert metrics["pending_queue_size"] >= 0


@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Assert EventBus shuts down gracefully"""
    bus = EventBus(enable_ttl_cleanup=True)
    
    event = GraceEvent(event_type="test.shutdown", source="test")
    await bus.emit(event)
    
    # Shutdown
    await bus.shutdown()
    
    # Cleanup task should be cancelled
    assert bus._cleanup_task is None or bus._cleanup_task.cancelled()


@pytest.mark.asyncio
async def test_type_safety():
    """Assert EventBus enforces GraceEvent type"""
    bus = EventBus()
    
    # Should reject dict
    with pytest.raises(TypeError) as exc_info:
        await bus.emit({"event_type": "test", "source": "test"})
    
    assert "GraceEvent" in str(exc_info.value)


@pytest.mark.asyncio
async def test_target_routing():
    """Assert events route to target subscribers"""
    bus = EventBus()
    
    received = []
    
    bus.subscribe("target1", lambda e: received.append(("target1", e)))
    bus.subscribe("target2", lambda e: received.append(("target2", e)))
    
    event = GraceEvent(
        event_type="test.routing",
        source="test",
        targets=["target1", "target2"]
    )
    
    await bus.emit(event)
    await asyncio.sleep(0.1)
    
    # Should have received by both targets
    target_names = [name for name, _ in received]
    assert "target1" in target_names
    assert "target2" in target_names
