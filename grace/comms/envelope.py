"""
Grace Message Envelope (GME) - Core envelope implementation.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class MessageKind(str, Enum):
    """Message types in Grace communications."""

    EVENT = "event"
    COMMAND = "command"
    QUERY = "query"
    REPLY = "reply"


class Priority(str, Enum):
    """Message priority levels."""

    P0 = "P0"  # Critical - governance, security
    P1 = "P1"  # High - deployments, alerts
    P2 = "P2"  # Normal - standard operations
    P3 = "P3"  # Low - background, bulk


class QoSClass(str, Enum):
    """Quality of Service classes."""

    REALTIME = "realtime"
    STANDARD = "standard"
    BULK = "bulk"


class GovernanceLabel(str, Enum):
    """Data governance classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class RetryPolicy(BaseModel):
    """Retry configuration for message delivery."""

    strategy: str = Field(default="exp", pattern="^(exp|lin|none)$")
    max_attempts: int = Field(default=5, ge=0)
    base_ms: int = Field(default=50, ge=1)
    jitter_ms: int = Field(default=40, ge=0)


class MessageHeaders(BaseModel):
    """Headers section of Grace Message Envelope."""

    schema_ref: str
    correlation_id: str = Field(pattern="^cor_[a-z0-9_-]{6,}$")
    traceparent: str
    priority: Priority
    qos: QoSClass
    partition_key: str

    # Optional headers
    causation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    deadline_ms: Optional[int] = Field(None, ge=1)
    idempotency_key: Optional[str] = None
    redelivery_count: int = Field(default=0, ge=0)
    retry_policy: Optional[RetryPolicy] = None
    compression: str = Field(default="none", pattern="^(none|gzip|zstd)$")
    checksum: Optional[str] = Field(None, pattern="^sha256:[a-f0-9]{64}$")
    signature: Optional[str] = None
    governance_label: GovernanceLabel = GovernanceLabel.INTERNAL
    pii_flags: List[str] = Field(default_factory=list)
    consent_scope: List[str] = Field(default_factory=list)
    rbac: List[str] = Field(default_factory=list)
    tracestate: Optional[str] = None
    hop_count: int = Field(default=0, ge=0, le=64)


class GraceMessageEnvelope(BaseModel):
    """Grace Message Envelope (GME) - Standard message wrapper."""

    msg_id: str = Field(pattern="^msg_[a-z0-9]{10,}$")
    kind: MessageKind
    domain: str
    name: str
    ts: datetime
    headers: MessageHeaders
    version: Optional[str] = Field(None, pattern="^[0-9]+\\.[0-9]+\\.[0-9]+$")
    payload: Optional[Dict[str, Any]] = None
    payload_ref: Optional[str] = None

    def model_post_init(self, __context):
        """Validate envelope after initialization."""
        if not self.payload and not self.payload_ref:
            raise ValueError("Either payload or payload_ref must be provided")
        if self.payload and self.payload_ref:
            raise ValueError("Only one of payload or payload_ref should be provided")


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_{uuid.uuid4().hex[:12]}"


def generate_correlation_id() -> str:
    """Generate a unique correlation ID."""
    return f"cor_{uuid.uuid4().hex[:8]}"


def create_envelope(
    kind: MessageKind,
    domain: str,
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    priority: Priority = Priority.P2,
    qos: QoSClass = QoSClass.STANDARD,
    schema_ref: Optional[str] = None,
    correlation_id: Optional[str] = None,
    partition_key: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    governance_label: GovernanceLabel = GovernanceLabel.INTERNAL,
    rbac: Optional[List[str]] = None,
    **kwargs,
) -> GraceMessageEnvelope:
    """Create a Grace Message Envelope with reasonable defaults."""

    # Generate required IDs
    msg_id = generate_message_id()
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    if partition_key is None:
        partition_key = correlation_id
    if schema_ref is None:
        schema_ref = f"grace://contracts/{domain}.{name.lower()}.schema.json"

    # Create headers
    headers = MessageHeaders(
        schema_ref=schema_ref,
        correlation_id=correlation_id,
        traceparent=f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01",
        priority=priority,
        qos=qos,
        partition_key=partition_key,
        idempotency_key=idempotency_key,
        governance_label=governance_label,
        rbac=rbac or [],
        **{k: v for k, v in kwargs.items() if hasattr(MessageHeaders, k)},
    )

    return GraceMessageEnvelope(
        msg_id=msg_id,
        kind=kind,
        domain=domain,
        name=name,
        ts=datetime.now(timezone.utc),
        headers=headers,
        version="1.0.0",
        payload=payload,
    )
