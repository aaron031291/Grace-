"""
Event Bus - Specification-compliant implementation
"""

from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timezone
import asyncio
import logging

from grace.events.schema import GraceEvent
from grace.events.factory import GraceEventFactory

logger = logging.getLogger(__name__)


class EventBus:
    """
    Specification-compliant event bus
    
    Only accepts GraceEvent objects, not dicts
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.events: List[GraceEvent] = []
        self.factory = GraceEventFactory()
        
        # Idempotency tracking
        self.processed_events: set = set()
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type}")
        
    def publish(self, event: GraceEvent) -> bool:
        """
        Publish GraceEvent (not dict!)
        
        Args:
            event: Must be a GraceEvent object
            
        Returns:
            True if published, False if duplicate
        """
        # Type check
        if not isinstance(event, GraceEvent):
            raise TypeError(f"Expected GraceEvent, got {type(event)}")
        
        # Idempotency check
        if event.idempotency_key and event.idempotency_key in self.processed_events:
            logger.debug(f"Skipping duplicate event: {event.idempotency_key}")
            return False
        
        # Store event
        self.events.append(event)
        
        # Mark as processed
        if event.idempotency_key:
            self.processed_events.add(event.idempotency_key)
        
        # Route to type subscribers
        event_type = event.event_type
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
        
        # Route by targets
        for target in event.targets:
            if target in self.subscribers:
                for callback in self.subscribers[target]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Target callback error: {e}")
        
        return True
    
    def create_and_publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        **kwargs
    ) -> GraceEvent:
        """Create GraceEvent and publish in one step"""
        event = self.factory.create_event(event_type, payload, **kwargs)
        self.publish(event)
        return event
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "total_events": len(self.events),
            "active_channels": len(self.subscribers),
            "subscribers": {
                channel: len(subs) 
                for channel, subs in self.subscribers.items()
            },
            "processed_events": len(self.processed_events)
        }


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
