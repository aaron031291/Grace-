"""
EventBus - Central event routing system for Grace governance kernel.
"""
import asyncio
from typing import Dict, List, Callable, Any, Optional
import json
from dataclasses import asdict
from datetime import datetime
import logging

from .contracts import EventType, generate_correlation_id


logger = logging.getLogger(__name__)


class EventBus:
    """
    Event-driven messaging system for governance components.
    Handles event routing, subscription management, and message delivery.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.correlation_tracking: Dict[str, List[str]] = {}
    
    async def subscribe(self, event_type: str, handler: Callable):
        """Subscribe a handler to a specific event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type}")
    
    async def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe a handler from an event type."""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed handler from {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type}")
    
    async def publish(self, event_type: str, payload: Dict[str, Any], 
                     correlation_id: Optional[str] = None) -> str:
        """
        Publish an event to all subscribers.
        Returns the correlation ID for tracking.
        """
        if correlation_id is None:
            correlation_id = generate_correlation_id()
        
        event = {
            "type": event_type,
            "payload": payload,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "id": f"evt_{len(self.message_history):06d}"
        }
        
        # Store in history
        self.message_history.append(event)
        
        # Track correlation
        if correlation_id not in self.correlation_tracking:
            self.correlation_tracking[correlation_id] = []
        self.correlation_tracking[correlation_id].append(event["id"])
        
        # Deliver to subscribers
        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                tasks.append(self._safe_deliver(handler, event))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Published {event_type} event with correlation_id {correlation_id}")
        return correlation_id
    
    async def _safe_deliver(self, handler: Callable, event: Dict[str, Any]):
        """Safely deliver event to handler with error handling."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Error delivering event to handler: {e}")
    
    def get_events_by_correlation(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all events associated with a correlation ID."""
        if correlation_id not in self.correlation_tracking:
            return []
        
        event_ids = self.correlation_tracking[correlation_id]
        return [event for event in self.message_history 
                if event["id"] in event_ids]
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent events."""
        return self.message_history[-limit:]
    
    async def clear_history(self, keep_recent: int = 1000):
        """Clear old events, keeping only the most recent ones."""
        if len(self.message_history) > keep_recent:
            self.message_history = self.message_history[-keep_recent:]
            logger.info(f"Cleared event history, kept {keep_recent} recent events")