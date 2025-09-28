"""Grace Message Envelope (GME) - Standard message format for all Grace events."""

from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
import hashlib
import json


class RBACContext(BaseModel):
    """Role-based access control context."""
    user_id: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)


class PIIFlags(BaseModel):
    """PII handling flags."""
    contains_pii: bool = False
    pii_types: List[str] = Field(default_factory=list)
    redaction_level: str = Field(default="none", pattern="^(none|mask|hash|remove)$")


class GMEHeaders(BaseModel):
    """GME message headers."""
    traceparent: Optional[str] = Field(None, pattern=r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
    tracestate: Optional[str] = None
    source: str = Field(..., description="Source component that created this message")
    event_type: str = Field(..., description="Type of event being transmitted")
    priority: str = Field(default="normal", pattern="^(critical|high|normal|low)$")
    rbac: Optional[RBACContext] = None
    pii_flags: Optional[PIIFlags] = None
    consent_scope: List[str] = Field(default_factory=list)


class GraceMessageEnvelope(BaseModel):
    """
    Grace Message Envelope (GME) - Standard message format for all Grace events.
    
    Provides:
    - Unique message identification
    - W3C trace context support
    - Idempotency guarantees
    - RBAC and consent tracking
    - PII handling flags
    - Message TTL and retry handling
    """
    
    msg_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    headers: GMEHeaders
    payload: Dict[str, Any]
    idempotency_key: str = Field(default_factory=lambda: f"idem_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = Field(default=0, ge=0)
    ttl_seconds: int = Field(default=3600, ge=1)
    schema_version: str = Field(default="1.0.0", pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of message content for integrity checking."""
        content = {
            "headers": self.headers.dict(),
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat()
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        age_seconds = (utc_now() - self.timestamp).total_seconds()
        return age_seconds > self.ttl_seconds
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "msg_id": self.msg_id,
            "headers": self.headers.dict(),
            "payload": self.payload,
            "idempotency_key": self.idempotency_key,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "ttl_seconds": self.ttl_seconds,
            "schema_version": self.schema_version,
            "hash": self.compute_hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraceMessageEnvelope':
        """Create GME from dictionary."""
        headers_data = data["headers"]
        headers = GMEHeaders(**headers_data)
        
        return cls(
            msg_id=data["msg_id"],
            headers=headers,
            payload=data["payload"],
            idempotency_key=data["idempotency_key"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            retry_count=data.get("retry_count", 0),
            ttl_seconds=data.get("ttl_seconds", 3600),
            schema_version=data.get("schema_version", "1.0.0")
        )
    
    @classmethod
    def create_event(cls, 
                    event_type: str,
                    payload: Dict[str, Any],
                    source: str,
                    priority: str = "normal",
                    rbac: Optional[RBACContext] = None,
                    pii_flags: Optional[PIIFlags] = None,
                    consent_scope: Optional[List[str]] = None,
                    traceparent: Optional[str] = None) -> 'GraceMessageEnvelope':
        """Create a new GME for an event."""
        headers = GMEHeaders(
            source=source,
            event_type=event_type,
            priority=priority,
            rbac=rbac,
            pii_flags=pii_flags,
            consent_scope=consent_scope or [],
            traceparent=traceparent
        )
        
        return cls(headers=headers, payload=payload)


# Event type constants
class EventTypes:
    """Grace event type constants."""
    
    # Governance events
    GOVERNANCE_VALIDATION = "GOVERNANCE_VALIDATION"
    GOVERNANCE_APPROVED = "GOVERNANCE_APPROVED"
    GOVERNANCE_REJECTED = "GOVERNANCE_REJECTED"
    GOVERNANCE_NEEDS_REVIEW = "GOVERNANCE_NEEDS_REVIEW"
    GOVERNANCE_ROLLBACK = "GOVERNANCE_ROLLBACK"
    
    # Orchestration events
    ORCHESTRATION_TASK_CREATED = "ORCHESTRATION_TASK_CREATED"
    ORCHESTRATION_TASK_COMPLETED = "ORCHESTRATION_TASK_COMPLETED"
    ORCHESTRATION_TASK_FAILED = "ORCHESTRATION_TASK_FAILED"
    
    # Memory events
    MEMORY_WRITE_REQUESTED = "MEMORY_WRITE_REQUESTED"
    MEMORY_WRITE_COMPLETED = "MEMORY_WRITE_COMPLETED"
    MEMORY_READ_REQUESTED = "MEMORY_READ_REQUESTED"
    MEMORY_READ_COMPLETED = "MEMORY_READ_COMPLETED"
    
    # Resilience events
    RESILIENCE_INCIDENT_OPENED = "RESILIENCE_INCIDENT_OPENED"
    RESILIENCE_INCIDENT_RESOLVED = "RESILIENCE_INCIDENT_RESOLVED"
    RESILIENCE_CIRCUIT_OPENED = "RESILIENCE_CIRCUIT_OPENED"
    RESILIENCE_CIRCUIT_CLOSED = "RESILIENCE_CIRCUIT_CLOSED"
    
    # MLDL events
    MLDL_TRAINING_STARTED = "MLDL_TRAINING_STARTED"
    MLDL_CANDIDATE_READY = "MLDL_CANDIDATE_READY"
    MLDL_EVALUATED = "MLDL_EVALUATED"
    MLDL_DEPLOYMENT_REQUESTED = "MLDL_DEPLOYMENT_REQUESTED"
    MLDL_DEPLOYED = "MLDL_DEPLOYED"
    
    # Immune/AVN events
    ADV_TEST_FAILED = "ADV_TEST_FAILED"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    IMMUNE_SANDBOXED = "IMMUNE_SANDBOXED"
    IMMUNE_HARDENED = "IMMUNE_HARDENED"
    IMMUNE_ROLLED_BACK = "IMMUNE_ROLLED_BACK"
    
    # System events
    TRUST_UPDATED = "TRUST_UPDATED"
    SNAPSHOT_EXPORTED = "SNAPSHOT_EXPORTED"
    ROLLBACK_REQUESTED = "ROLLBACK_REQUESTED"
    ROLLBACK_COMPLETED = "ROLLBACK_COMPLETED"