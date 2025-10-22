"""
Grace Integration Event Bus - Canonical implementation
"""

from typing import Dict, Any, List, Callable, Optional, Coroutine
import asyncio
import logging
from datetime import datetime, timezone
from collections import deque

from grace.schemas.events import GraceEvent, EventStatus, EventPriority
from grace.schemas.errors import (
    DuplicateEventError,
    BackpressureError,
    TimeoutError,
    TTLExpiredError,
    DeadLetterError
)
from grace.observability.metrics import get_metrics_collector
from grace.observability.structured_logging import StructuredLogger
import time

logger = logging.getLogger(__name__)


class EventBus:
    """
    Canonical event bus implementation with metrics
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        dlq_max_size: int = 1000,
        enable_ttl_cleanup: bool = True
    ) -> None:
        self.subscribers: Dict[str, List[Callable[[GraceEvent], Any]]] = {}
        self.processed_events: Dict[str, GraceEvent] = {}
        self.pending_queue: deque = deque(maxlen=max_queue_size)
        self.dead_letter_queue: deque = deque(maxlen=dlq_max_size)
        
        self._waiters: Dict[str, asyncio.Future] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        
        self.max_queue_size = max_queue_size
        self.dlq_max_size = dlq_max_size
        self.enable_ttl_cleanup = enable_ttl_cleanup
        
        # Metrics
        self.metrics = get_metrics_collector()
        
        # Structured logging
        self.logger = StructuredLogger("event_bus")
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        if enable_ttl_cleanup:
            self._cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())
    
    async def emit(
        self,
        event: GraceEvent,
        skip_idempotency: bool = False
    ) -> bool:
        """
        Emit an event with idempotency checks
        
        Args:
            event: GraceEvent to emit (NOT a dict!)
            skip_idempotency: Whether to skip idempotency check
        
        Returns:
            True if event was emitted, False if duplicate
        
        Raises:
            TypeError: If event is not a GraceEvent
            BackpressureError: If queue is full
            TTLExpiredError: If event already expired
        """
        start_time = time.time()
        
        # Type check
        if not isinstance(event, GraceEvent):
            raise TypeError(f"EventBus.emit expects GraceEvent, got {type(event)}")
        
        # Log with correlation ID
        logger = self.logger.with_correlation_id(event.correlation_id or event.event_id)
        logger.debug("Event emit started", event_type=event.event_type, source=event.source)
        
        # Check TTL expiry
        if event.is_expired():
            self.events_expired += 1
            await self.metrics.record_event_expired()
            logger.warning("Event expired", event_id=event.event_id, ttl_seconds=event.ttl_seconds)
            raise TTLExpiredError(event.event_id, event.ttl_seconds or 0)
        
        # Backpressure check
        if len(self.pending_queue) >= self.max_queue_size:
            logger.error("Backpressure triggered", queue_size=len(self.pending_queue))
            raise BackpressureError(len(self.pending_queue), self.max_queue_size)
        
        # Idempotency check
        if not skip_idempotency and event.idempotency_key:
            async with self._lock:
                if event.idempotency_key in self.processed_events:
                    self.events_deduplicated += 1
                    await self.metrics.record_event_deduplicated()
                    logger.debug("Duplicate event blocked", idempotency_key=event.idempotency_key)
                    return False
                
                self.processed_events[event.idempotency_key] = event
        
        # Mark as processing
        event.mark_as_processing()
        
        # Add to queue
        self.pending_queue.append(event)
        self.events_published += 1
        
        # Record metrics
        await self.metrics.record_event_published()
        await self.metrics.increment_event_count(event.event_type)
        await self.metrics.set_gauge("pending_queue_size", len(self.pending_queue))
        
        # Dispatch to subscribers
        await self._dispatch(event)
        
        # Record latency
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_latency("event_emit", latency_ms)
        
        logger.info("Event emitted", 
                   event_id=event.event_id,
                   event_type=event.event_type,
                   latency_ms=latency_ms)
        
        return True
    
    async def _dispatch(self, event: GraceEvent):
        """Dispatch event to subscribers"""
        # Get handlers for event type
        handlers = list(self.subscribers.get(event.event_type, []))
        
        # Get handlers for targets
        for target in event.targets:
            handlers.extend(self.subscribers.get(target, []))
        
        # Dispatch to handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(self._safe_async_handler(handler, event))
                else:
                    # Run sync handlers in executor
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, self._safe_sync_handler, handler, event)
            
            except Exception as e:
                logger.exception(f"Error dispatching to handler: {e}")
                self.events_failed += 1
        
        # Handle correlation_id for request/response
        if event.correlation_id:
            async with self._lock:
                fut = self._waiters.get(event.correlation_id)
                if fut and not fut.done():
                    fut.set_result(event)
    
    async def _safe_async_handler(self, handler: Callable, event: GraceEvent):
        """Safe async handler execution with metrics"""
        start_time = time.time()
        logger = self.logger.with_correlation_id(event.correlation_id or event.event_id)
        
        try:
            await handler(event)
            self.events_processed += 1
            event.mark_as_completed()
            
            await self.metrics.record_event_processed()
            
            latency_ms = (time.time() - start_time) * 1000
            await self.metrics.record_latency("event_processing", latency_ms)
            
            logger.debug("Event processed", event_id=event.event_id, latency_ms=latency_ms)
        
        except Exception as e:
            logger.error("Handler error", 
                        event_id=event.event_id,
                        error=str(e),
                        error_type=type(e).__name__)
            
            self.events_failed += 1
            event.mark_as_failed(str(e))
            
            await self.metrics.record_event_failed()
            
            # Retry or send to DLQ
            if event.can_retry():
                event.increment_retry()
                logger.info("Retrying event", event_id=event.event_id, retry_count=event.retry_count)
                await self.emit(event, skip_idempotency=True)
            else:
                await self._send_to_dlq(event, f"Handler failed: {e}")
    
    def _safe_sync_handler(self, handler: Callable, event: GraceEvent):
        """Safe sync handler execution"""
        try:
            handler(event)
            self.events_processed += 1
            event.mark_as_completed()
        
        except Exception as e:
            logger.exception(f"Sync handler error: {e}")
            self.events_failed += 1
            event.mark_as_failed(str(e))
            
            # Queue for retry or DLQ
            if event.can_retry():
                event.increment_retry()
                asyncio.create_task(self.emit(event, skip_idempotency=True))
            else:
                asyncio.create_task(self._send_to_dlq(event, f"Handler failed: {e}"))
    
    async def _send_to_dlq(self, event: GraceEvent, reason: str):
        """Send event to dead letter queue with metrics"""
        event.mark_as_dead_letter(reason)
        
        if len(self.dead_letter_queue) >= self.dlq_max_size:
            self.logger.error("DLQ full, dropping event", event_id=event.event_id)
            return
        
        self.dead_letter_queue.append(event)
        
        await self.metrics.record_dlq_event()
        await self.metrics.set_gauge("dlq_size", len(self.dead_letter_queue))
        
        self.logger.warning("Event sent to DLQ", 
                           event_id=event.event_id,
                           reason=reason,
                           dlq_size=len(self.dead_letter_queue))
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[GraceEvent], Any]
    ) -> None:
        """
        Subscribe to event type
        
        Args:
            event_type: Event type to subscribe to
            handler: Callback function (sync or async)
        """
        self.subscribers.setdefault(event_type, []).append(handler)
        logger.debug(f"Subscribed handler to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type"""
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
    
    async def request_response(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timeout: float = 5.0,
        **kwargs: Any
    ) -> Optional[GraceEvent]:
        """
        Request/response pattern
        
        Args:
            event_type: Event type
            payload: Event payload
            timeout: Timeout in seconds
            **kwargs: Additional event parameters
        
        Returns:
            Response event or None if timeout
        
        Raises:
            TimeoutError: If request times out
        """
        correlation_id = f"req_{datetime.now(timezone.utc).timestamp()}"
        
        event = GraceEvent(
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
            **kwargs
        )
        
        fut = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._waiters[correlation_id] = fut
        
        try:
            await self.emit(event)
            return await asyncio.wait_for(fut, timeout=timeout)
        
        except asyncio.TimeoutError:
            logger.warning(f"Request timed out: {correlation_id}")
            raise TimeoutError(f"Request timeout after {timeout}s", timeout)
        
        finally:
            async with self._lock:
                self._waiters.pop(correlation_id, None)
    
    async def wait_for(
        self,
        event_type: str,
        predicate: Callable[[GraceEvent], bool],
        timeout: float = 5.0
    ) -> Optional[GraceEvent]:
        """
        Wait for event matching predicate
        
        Args:
            event_type: Event type to wait for
            predicate: Function to test if event matches
            timeout: Timeout in seconds
        
        Returns:
            Matching event or None if timeout
        """
        fut = asyncio.get_event_loop().create_future()
        
        async def _handler(e: GraceEvent):
            try:
                if predicate(e) and not fut.done():
                    fut.set_result(e)
            except Exception as ex:
                logger.exception(f"wait_for predicate error: {ex}")
        
        self.subscribe(event_type, _handler)
        
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(event_type, _handler)
    
    async def _ttl_cleanup_loop(self):
        """Background cleanup of expired events"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Clean expired events from queue
                expired = []
                for event in list(self.pending_queue):
                    if event.is_expired():
                        expired.append(event)
                        event.status = EventStatus.EXPIRED
                
                for event in expired:
                    try:
                        self.pending_queue.remove(event)
                        self.events_expired += 1
                        logger.debug(f"Expired event removed: {event.event_id}")
                    except ValueError:
                        pass
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"TTL cleanup error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return {
            "events_published": self.events_published,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "events_expired": self.events_expired,
            "events_deduplicated": self.events_deduplicated,
            "pending_queue_size": len(self.pending_queue),
            "dlq_size": len(self.dead_letter_queue),
            "active_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "processed_events_cache_size": len(self.processed_events)
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("EventBus shutdown complete", extra=self.get_metrics())


# Global instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
