"""Grace Message Envelope (GME) - Standard message format for all Grace events (simplified version)."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
import hashlib
import json


class RBACContext:
    """Role-based access control context."""
    def __init__(self, user_id: str = None, roles: List[str] = None, permissions: List[str] = None):
        self.user_id = user_id
        self.roles = roles or []
        self.permissions = permissions or []
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "roles": self.roles,
            "permissions": self.permissions
        }


class PIIFlags:
    """PII handling flags."""
    def __init__(self, contains_pii: bool = False, pii_types: List[str] = None, redaction_level: str = "none"):
        self.contains_pii = contains_pii
        self.pii_types = pii_types or []
        self.redaction_level = redaction_level
    
    def to_dict(self):
        return {
            "contains_pii": self.contains_pii,
            "pii_types": self.pii_types,
            "redaction_level": self.redaction_level
        }


class GMEHeaders:
    """GME message headers."""
    def __init__(self, source: str, event_type: str, traceparent: str = None, tracestate: str = None,
                 priority: str = "normal", rbac: RBACContext = None, pii_flags: PIIFlags = None,
                 consent_scope: List[str] = None):
        self.traceparent = traceparent
        self.tracestate = tracestate
        self.source = source
        self.event_type = event_type
        self.priority = priority
        self.rbac = rbac
        self.pii_flags = pii_flags
        self.consent_scope = consent_scope or []
    
    def to_dict(self):
        return {
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
            "source": self.source,
            "event_type": self.event_type,
            "priority": self.priority,
            "rbac": self.rbac.to_dict() if self.rbac else None,
            "pii_flags": self.pii_flags.to_dict() if self.pii_flags else None,
            "consent_scope": self.consent_scope
        }


class GraceMessageEnvelope:
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
    
    def __init__(self, headers: GMEHeaders, payload: Dict[str, Any],
                 msg_id: str = None, idempotency_key: str = None,
                 timestamp: datetime = None, retry_count: int = 0,
                 ttl_seconds: int = 3600, schema_version: str = "1.0.0"):
        self.msg_id = msg_id or f"msg_{uuid.uuid4().hex[:12]}"
        self.headers = headers
        self.payload = payload
        self.idempotency_key = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
        self.timestamp = timestamp or datetime.utcnow()
        self.retry_count = retry_count
        self.ttl_seconds = ttl_seconds
        self.schema_version = schema_version
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of message content for integrity checking."""
        content = {
            "headers": self.headers.to_dict(),
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat()
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        age_seconds = (datetime.utcnow() - self.timestamp).total_seconds()
        return age_seconds > self.ttl_seconds
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "msg_id": self.msg_id,
            "headers": self.headers.to_dict(),
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
        
        # Reconstruct RBAC context
        rbac = None
        if headers_data.get("rbac"):
            rbac_data = headers_data["rbac"]
            rbac = RBACContext(
                user_id=rbac_data.get("user_id"),
                roles=rbac_data.get("roles", []),
                permissions=rbac_data.get("permissions", [])
            )
        
        # Reconstruct PII flags
        pii_flags = None
        if headers_data.get("pii_flags"):
            pii_data = headers_data["pii_flags"]
            pii_flags = PIIFlags(
                contains_pii=pii_data.get("contains_pii", False),
                pii_types=pii_data.get("pii_types", []),
                redaction_level=pii_data.get("redaction_level", "none")
            )
        
        headers = GMEHeaders(
            source=headers_data["source"],
            event_type=headers_data["event_type"],
            traceparent=headers_data.get("traceparent"),
            tracestate=headers_data.get("tracestate"),
            priority=headers_data.get("priority", "normal"),
            rbac=rbac,
            pii_flags=pii_flags,
            consent_scope=headers_data.get("consent_scope", [])
        )
        
        return cls(
            headers=headers,
            payload=data["payload"],
            msg_id=data["msg_id"],
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
                    rbac: RBACContext = None,
                    pii_flags: PIIFlags = None,
                    consent_scope: List[str] = None,
                    traceparent: str = None) -> 'GraceMessageEnvelope':
        """Create a new GME for an event."""
        headers = GMEHeaders(
            source=source,
            event_type=event_type,
            priority=priority,
            rbac=rbac,
            pii_flags=pii_flags,
            consent_scope=consent_scope,
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