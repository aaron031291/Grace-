"""
Grace AI Event Bus - Central communication hub for the entire system
"""

import asyncio
from typing import Callable, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for publishing and subscribing to system events."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = asyncio.Lock()

    async def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        async with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
            logger.info(f"Subscribed to event: {event_type}")

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to all subscribers."""
        logger.info(f"Publishing event: {event_type} with data: {data}")
        async with self.lock:
            handlers = self.subscribers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}")

    async def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type."""
        async with self.lock:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(handler)
