"""
Event Factory - uses canonical GraceEvent
"""

from typing import Dict, Any, List, Optional
from grace.schemas.events import GraceEvent, EventPriority


class GraceEventFactory:
    """Factory for creating GraceEvent instances"""
    
    def __init__(self, default_source: str = "grace"):
        self.default_source = default_source
        self.last_event_hash: Optional[str] = None
    
    def create_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: Optional[str] = None,
        targets: Optional[List[str]] = None,
        constitutional_validation_required: bool = False,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
        **kwargs
    ) -> GraceEvent:
        """Create a GraceEvent with all fields"""
        event = GraceEvent(
            event_type=event_type,
            source=source or self.default_source,
            targets=targets or [],
            payload=payload,
            constitutional_validation_required=constitutional_validation_required,
            priority=priority,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            headers=headers or {},
            ttl_seconds=ttl_seconds,
            previous_event_id=self.last_event_hash,
            **kwargs
        )
        
        # Calculate chain hash
        event.chain_hash = event.calculate_chain_hash(self.last_event_hash)
        self.last_event_hash = event.event_id
        
        return event
