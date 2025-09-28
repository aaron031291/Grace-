"""
Ingress Event Contracts - Event schemas for Ingress⇄Mesh communication.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class IngressEventType(str, Enum):
    """Ingress event types."""
    SOURCE_REGISTERED = "ING_SOURCE_REGISTERED"
    CAPTURED_RAW = "ING_CAPTURED_RAW"
    PARSED = "ING_PARSED"
    NORMALIZED = "ING_NORMALIZED"
    ENRICHED = "ING_ENRICHED"
    VALIDATION_FAILED = "ING_VALIDATION_FAILED"
    PERSISTED = "ING_PERSISTED"
    PUBLISHED = "ING_PUBLISHED"
    SOURCE_HEALTH = "ING_SOURCE_HEALTH"
    EXPERIENCE = "ING_EXPERIENCE"
    ROLLBACK_REQUESTED = "ROLLBACK_REQUESTED"
    ROLLBACK_COMPLETED = "ROLLBACK_COMPLETED"


class SourceRegisteredPayload(BaseModel):
    """Payload for ING_SOURCE_REGISTERED event."""
    source: Dict[str, Any]  # SourceConfig


class CapturedRawPayload(BaseModel):
    """Payload for ING_CAPTURED_RAW event."""
    event: Dict[str, Any]  # RawEvent


class ParseReport(BaseModel):
    """Parse operation report."""
    ok: bool
    errors: List[str] = Field(default_factory=list)
    bytes_in: int
    bytes_out: int


class ParsedPayload(BaseModel):
    """Payload for ING_PARSED event."""
    raw_event_id: str
    parse_report: ParseReport


class NormalizedPayload(BaseModel):
    """Payload for ING_NORMALIZED event."""
    record: Dict[str, Any]  # NormRecord


class EnrichedPayload(BaseModel):
    """Payload for ING_ENRICHED event."""
    record_id: str
    enrichments: List[str]


class ValidationFailedPayload(BaseModel):
    """Payload for ING_VALIDATION_FAILED event."""
    record_id: Optional[str] = None
    raw_event_id: Optional[str] = None
    reasons: List[str]
    severity: str  # "warn" or "error"
    policy: str  # "pii", "schema", "format", "governance"


class PersistedPayload(BaseModel):
    """Payload for ING_PERSISTED event."""
    record_id: str
    tier: str  # "bronze", "silver", "gold"
    uri: str


class PublishedPayload(BaseModel):
    """Payload for ING_PUBLISHED event."""
    record_id: str
    topics: List[str]


class SourceHealthPayload(BaseModel):
    """Payload for ING_SOURCE_HEALTH event."""
    source_id: str
    status: str  # "ok", "degraded", "down"
    latency_ms: int
    backlog: int
    last_ok: datetime


class ExperiencePayload(BaseModel):
    """Payload for ING_EXPERIENCE event."""
    schema_version: str = "1.0.0"
    experience: Dict[str, Any]  # IngressExperience


class RollbackRequestedPayload(BaseModel):
    """Payload for ROLLBACK_REQUESTED event."""
    target: str = "ingress"
    to_snapshot: str


class RollbackCompletedPayload(BaseModel):
    """Payload for ROLLBACK_COMPLETED event."""
    target: str = "ingress"
    snapshot_id: str
    at: datetime = Field(default_factory=datetime.utcnow)


class IngressEvent(BaseModel):
    """Base ingress event structure."""
    event_type: IngressEventType
    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Any  # One of the payload types above
    source: str = "ingress_kernel"
    version: str = "1.0.0"