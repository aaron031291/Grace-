"""Trigger ledger - event recording and notification system."""
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional

from ..contracts.dto_common import TriggerEvent
from .schemas import MemoryStore


class TriggerLedger:
    """Records and manages trigger events across the system."""
    
    def __init__(self, memory_store: MemoryStore):
        self.store = memory_store
    
    def record(self, event_type: str, source: str, target_id: str, payload: Optional[Dict] = None) -> str:
        """Record a trigger event."""
        event = TriggerEvent(
            event_type=event_type,
            source=source,
            target_id=target_id,
            payload=payload or {}
        )
        
        self.store.trigger_events.append(event)
        return event.id
    
    def get_events_for_target(self, target_id: str) -> List[TriggerEvent]:
        """Get all events for a specific target."""
        return [event for event in self.store.trigger_events if event.target_id == target_id]
    
    def get_events_by_type(self, event_type: str) -> List[TriggerEvent]:
        """Get all events of a specific type."""
        return [event for event in self.store.trigger_events if event.event_type == event_type]
    
    def get_recent_events(self, limit: int = 10) -> List[TriggerEvent]:
        """Get most recent events."""
        return sorted(self.store.trigger_events, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def clear_old_events(self, max_age_days: int = 30) -> int:
        """Clear old events (optional cleanup)."""
        cutoff = utc_now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        old_events = [
            event for event in self.store.trigger_events 
            if event.created_at.timestamp() < cutoff
        ]
        
        for event in old_events:
            self.store.trigger_events.remove(event)
        
        return len(old_events)