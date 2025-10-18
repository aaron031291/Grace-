"""
EventBus - Multi-agent signal routing for distributed Grace instances
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import uuid
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents an event in the system"""
    event_id: str
    event_type: str
    source: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Central event bus for multi-agent signal routing
    Enables pub/sub communication between distributed Grace components
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.async_queue: asyncio.Queue = asyncio.Queue()
        self.max_history = 10000
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        self.subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                h for h in self.subscribers[event_type] if h != handler
            ]
            logger.debug(f"Unsubscribed handler from event type: {event_type}")
    
    def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "system",
        priority: int = 0,
        metadata: Optional[Dict] = None
    ):
        """Publish an event to all subscribers"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            payload=payload,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store in history
        self._add_to_history(event)
        
        # Notify subscribers
        handlers = self.subscribers.get(event_type, [])
        wildcard_handlers = self.subscribers.get("*", [])
        
        for handler in handlers + wildcard_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
        
        logger.debug(f"Published event: {event_type} from {source}")
    
    async def publish_async(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "system",
        priority: int = 0
    ):
        """Publish event asynchronously"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            payload=payload,
            priority=priority
        )
        
        await self.async_queue.put(event)
        logger.debug(f"Queued async event: {event_type}")
    
    def _add_to_history(self, event: Event):
        """Add event to history with size limit"""
        self.event_history.append(event)
        
        # Trim history if needed
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history with optional filters"""
        filtered = self.event_history
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if source:
            filtered = [e for e in filtered if e.source == source]
        
        return filtered[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            'total_subscribers': sum(len(handlers) for handlers in self.subscribers.values()),
            'event_types': len(self.subscribers),
            'total_events': len(self.event_history),
            'event_types_seen': len(set(e.event_type for e in self.event_history))
        }

class SimpleEventBus:
    """
    Simple in-memory event bus for pub/sub messaging
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.events: List[Dict[str, Any]] = []
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event: Dict[str, Any]):
        """Publish event to subscribers"""
        event_type = event.get("type", "unknown")
        self.events.append({
            **event,
            "published_at": datetime.now(timezone.utc).isoformat()
        })
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "total_events": len(self.events),
            "active_channels": len(self.subscribers),
            "subscribers": {
                channel: len(subs) 
                for channel, subs in self.subscribers.items()
            }
        }
