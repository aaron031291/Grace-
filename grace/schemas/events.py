"""
Canonical GraceEvent structure - single source of truth
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import uuid
import hashlib


class EventPriority(Enum):
    """Event priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(Enum):
    """Event lifecycle status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    EXPIRED = "expired"


@dataclass
class GraceEvent:
    """
    Canonical event structure for Grace system
    
    All event buses must use this structure
    """
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Source and routing
    source: str = ""
    targets: List[str] = field(default_factory=list)
    
    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Governance
    constitutional_validation_required: bool = False
    governance_approved: bool = False
    trust_score: float = 1.0
    
    # Priority and status
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    
    # Correlation and tracing
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Idempotency and retry
    idempotency_key: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # TTL and expiry
    ttl_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    
    # Headers and metadata
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Audit trail
    chain_hash: Optional[str] = None
    previous_event_id: Optional[str] = None
    
    # Dead letter queue
    dlq_reason: Optional[str] = None
    original_queue: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Convert priority to enum if string
        if isinstance(self.priority, str):
            self.priority = EventPriority(self.priority)
        
        # Convert status to enum if string
        if isinstance(self.status, str):
            self.status = EventStatus(self.status)
        
        # Calculate expiry if TTL set
        if self.ttl_seconds and not self.expires_at:
            self.expires_at = self.timestamp + timedelta(seconds=self.ttl_seconds)
        
        # Generate idempotency key if not provided
        if not self.idempotency_key and self.event_type and self.source:
            self.idempotency_key = self.generate_idempotency_key()
    
    def generate_idempotency_key(self) -> str:
        """Generate deterministic idempotency key"""
        data = f"{self.event_type}:{self.source}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def calculate_chain_hash(self, previous_hash: Optional[str] = None) -> str:
        """Calculate cryptographic chain hash"""
        hash_input = (
            f"{self.event_id}:"
            f"{self.event_type}:"
            f"{self.timestamp.isoformat()}:"
            f"{previous_hash or ''}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if event has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if event can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1
    
    def mark_as_processing(self):
        """Mark event as being processed"""
        self.status = EventStatus.PROCESSING
    
    def mark_as_completed(self):
        """Mark event as completed"""
        self.status = EventStatus.COMPLETED
    
    def mark_as_failed(self, reason: Optional[str] = None):
        """Mark event as failed"""
        self.status = EventStatus.FAILED
        if reason:
            self.metadata["failure_reason"] = reason
    
    def mark_as_dead_letter(self, reason: str):
        """Mark event for dead letter queue"""
        self.status = EventStatus.DEAD_LETTER
        self.dlq_reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "targets": self.targets,
            "payload": self.payload,
            "constitutional_validation_required": self.constitutional_validation_required,
            "governance_approved": self.governance_approved,
            "trust_score": self.trust_score,
            "priority": self.priority.value,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "trace_id": self.trace_id,
            "idempotency_key": self.idempotency_key,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "headers": self.headers,
            "metadata": self.metadata,
            "chain_hash": self.chain_hash,
            "previous_event_id": self.previous_event_id,
            "dlq_reason": self.dlq_reason,
            "original_queue": self.original_queue
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraceEvent":
        """Create from dictionary"""
        # Convert timestamp strings to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        
        if "expires_at" in data and isinstance(data["expires_at"], str):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
        
        # Filter to only known fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
