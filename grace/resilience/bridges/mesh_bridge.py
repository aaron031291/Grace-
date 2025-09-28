"""Bridge to Event Mesh for resilience events."""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class MeshBridge:
    """
    Bridge to Event Mesh for publishing and consuming resilience events.
    
    Handles communication with the Grace Event Mesh for resilience-related
    events like incidents, degradation modes, circuit breaker state changes, etc.
    """
    
    def __init__(self, mesh_client=None):
        """Initialize mesh bridge."""
        self.mesh_client = mesh_client
        self.event_handlers: Dict[str, Callable] = {}
        self.published_events = []  # For testing/debugging
        
        logger.debug("Mesh bridge initialized")
    
    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Publish event to the mesh.
        
        Args:
            event_type: Type of event (e.g., RES_INCIDENT_OPENED)
            payload: Event payload data
        """
        try:
            event = {
                "event_type": event_type,
                "source": "resilience",
                "timestamp": datetime.now().isoformat(),
                "correlation_id": self._generate_correlation_id(),
                "payload": payload
            }
            
            # Store for debugging
            self.published_events.append(event)
            
            # Limit stored events
            if len(self.published_events) > 1000:
                self.published_events = self.published_events[-500:]
            
            if self.mesh_client:
                await self.mesh_client.publish(event)
            else:
                # Fallback: log the event
                logger.info(f"Published event {event_type}: {payload}")
            
            logger.debug(f"Published resilience event: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
    
    async def subscribe_to_events(self, event_patterns: list):
        """
        Subscribe to events from other kernels.
        
        Args:
            event_patterns: List of event type patterns to subscribe to
        """
        try:
            if self.mesh_client:
                for pattern in event_patterns:
                    await self.mesh_client.subscribe(pattern, self._handle_incoming_event)
            else:
                logger.info(f"Subscribed to event patterns: {event_patterns}")
            
            logger.info(f"Subscribed to {len(event_patterns)} event patterns")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register handler for incoming events."""
        self.event_handlers[event_type] = handler
        logger.debug(f"Registered handler for event type: {event_type}")
    
    async def _handle_incoming_event(self, event: Dict[str, Any]):
        """Handle incoming events from the mesh."""
        try:
            event_type = event.get("event_type")
            
            if event_type in self.event_handlers:
                handler = self.event_handlers[event_type]
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                
                logger.debug(f"Handled incoming event: {event_type}")
            else:
                logger.debug(f"No handler for event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error handling incoming event: {e}")
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for event tracking."""
        import uuid
        return str(uuid.uuid4())
    
    def get_published_events(self, event_type: Optional[str] = None, limit: int = 100) -> list:
        """Get published events for debugging."""
        events = self.published_events
        
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        return events[-limit:] if limit else events