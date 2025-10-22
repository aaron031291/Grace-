"""
Event Bus - Specification-compliant implementation
"""

from typing import Dict, Any, List, Callable, Optional, Coroutine
import asyncio
import logging
import uuid

from grace.events.schema import GraceEvent
from grace.events.factory import GraceEventFactory

logger = logging.getLogger(__name__)


class EventBus:
    """
    Event bus supporting:
    - fire-and-forget publish(event)
    - subscribe(event_type, callback) (sync or async callbacks)
    - request_response(event_type, payload, timeout) -> response event
    - wait_for(event_type, predicate, timeout)
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.factory = GraceEventFactory()
        self._waiters: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: str, callback: Callable):
        self.subscribers.setdefault(event_type, []).append(callback)
        logger.debug(f"Subscribed callback to {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable):
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)

    def publish(self, event: GraceEvent) -> bool:
        # accept GraceEvent only
        if not isinstance(event, GraceEvent):
            raise TypeError("EventBus.publish expects GraceEvent")

        # Deliver to type subscribers
        handlers = list(self.subscribers.get(event.event_type, []))
        # Deliver to target subscribers as well
        for target in event.targets:
            handlers += list(self.subscribers.get(target, []))

        for cb in handlers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(event))
                else:
                    # run sync callbacks in threadpool to avoid blocking
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, cb, event)
            except Exception as e:
                logger.exception(f"Error invoking subscriber: {e}")

        # If there is a waiter expecting a response by correlation_id, set it
        corr = event.correlation_id
        if corr:
            fut = self._waiters.get(corr)
            if fut and not fut.done():
                fut.set_result(event)

        return True

    async def request_response(self, event_type: str, payload: Dict[str, Any], timeout: float = 5.0, **kwargs) -> Optional[GraceEvent]:
        """
        Create a request event and wait for a response event with correlation_id.
        The responder should publish a GraceEvent whose event_type is 'response' or any agreed type,
        and must set correlation_id to match.
        """
        corr_id = str(uuid.uuid4())
        event = self.factory.create_event(event_type=event_type, payload=payload, correlation_id=corr_id, **kwargs)

        fut = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._waiters[corr_id] = fut

        try:
            self.publish(event)
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Request timed out for correlation_id={corr_id}")
            return None
        finally:
            async with self._lock:
                self._waiters.pop(corr_id, None)

    async def wait_for(self, event_type: str, predicate: Callable[[GraceEvent], bool], timeout: float = 5.0) -> Optional[GraceEvent]:
        """
        Wait for an event of event_type that satisfies predicate. Registers a temporary subscriber.
        """
        fut = asyncio.get_event_loop().create_future()

        async def _cb(e: GraceEvent):
            try:
                if predicate(e) and not fut.done():
                    fut.set_result(e)
            except Exception as ex:
                logger.exception(f"wait_for predicate error: {ex}")

        # subscribe
        self.subscribe(event_type, _cb)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            # cleanup: remove callback
            try:
                self.unsubscribe(event_type, _cb)
            except Exception:
                pass


# Provide a global bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
