"""
GraceEvent Schema - Complete specification-compliant implementation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


class EventPriority(Enum):
    """Event priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GraceEvent:
    """
    Complete GraceEvent specification
    
    All required fields from specification
    """
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Source and routing
    source: str = ""
    targets: List[str] = field(default_factory=list)
    
    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Governance
    constitutional_validation_required: bool = False
    governance_approved: bool = False
    trust_score: float = 1.0
    
    # Metadata
    priority: str = "normal"
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Idempotency
    idempotency_key: Optional[str] = None
    
    # Headers (for routing, auth, etc)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Audit trail
    chain_hash: Optional[str] = None
    previous_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "source": self.source,
            "targets": self.targets,
            "payload": self.payload,
            "constitutional_validation_required": self.constitutional_validation_required,
            "governance_approved": self.governance_approved,
            "trust_score": self.trust_score,
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "idempotency_key": self.idempotency_key,
            "headers": self.headers,
            "chain_hash": self.chain_hash,
            "previous_event_id": self.previous_event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraceEvent":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
