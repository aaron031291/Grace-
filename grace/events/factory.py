"""
Event Factory - Create specification-compliant events
"""

from typing import Dict, Any, List, Optional
import hashlib

from .schema import GraceEvent, EventPriority


class GraceEventFactory:
    """
    Factory for creating GraceEvent instances
    
    Ensures all required fields are populated
    """
    
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
        priority: str = "normal",
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> GraceEvent:
        """
        Create a complete GraceEvent
        
        All specification fields are populated
        """
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
            previous_event_id=self.last_event_hash
        )
        
        # Calculate chain hash
        event.chain_hash = self._calculate_hash(event)
        self.last_event_hash = event.event_id
        
        return event
    
    def _calculate_hash(self, event: GraceEvent) -> str:
        """Calculate cryptographic hash for event chaining"""
        hash_input = f"{event.event_id}:{event.event_type}:{event.timestamp}:{event.previous_event_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
