"""
ingress_events.py
Ingress Event Contracts — Production-grade event schemas for Ingress ⇄ Mesh communication.

- Strict Pydantic v2 models (extra=forbid, validate_assignment).
- Timezone-aware datetimes (UTC normalization).
- Enums for status, severity, storage tier.
- Payload type-checking against event_type.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------
# Base config
# ---------------------------

class GraceModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "populate_by_name": True,
        "use_enum_values": True,
        "ser_json_bytes": "utf8",
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------
# Event types & enums
# ---------------------------

class IngressEventType(StrEnum):
    SOURCE_REGISTERED   = "ING_SOURCE_REGISTERED"
    CAPTURED_RAW        = "ING_CAPTURED_RAW"
    PARSED              = "ING_PARSED"
    NORMALIZED          = "ING_NORMALIZED"
    ENRICHED            = "ING_ENRICHED"
    VALIDATION_FAILED   = "ING_VALIDATION_FAILED"
    PERSISTED           = "ING_PERSISTED"
    PUBLISHED           = "ING_PUBLISHED"
    SOURCE_HEALTH       = "ING_SOURCE_HEALTH"
    EXPERIENCE          = "ING_EXPERIENCE"
    ROLLBACK_REQUESTED  = "ROLLBACK_REQUESTED"
    ROLLBACK_COMPLETED  = "ROLLBACK_COMPLETED"


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class Severity(StrEnum):
    WARN = "warn"
    ERROR = "error"


class StorageTier(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


# ---------------------------
# Payloads
# ---------------------------

class SourceRegisteredPayload(GraceModel):
    """Payload for ING_SOURCE_REGISTERED."""
    source: Mapping[str, Any]  # SourceConfig dump


class CapturedRawPayload(GraceModel):
    """Payload for ING_CAPTURED_RAW."""
    event: Mapping[str, Any]  # RawEvent dump


class ParseReport(GraceModel):
    ok: bool
    errors: List[str] = Field(default_factory=list)
    bytes_in: int = Field(ge=0)
    bytes_out: int = Field(ge=0)


class ParsedPayload(GraceModel):
    """Payload for ING_PARSED."""
    raw_event_id: str = Field(pattern=r"^rev_[a-f0-9]{8,}$")
    parse_report: ParseReport


class NormalizedPayload(GraceModel):
    """Payload for ING_NORMALIZED."""
    record: Mapping[str, Any]  # NormRecord dump


class EnrichedPayload(GraceModel):
    """Payload for ING_ENRICHED."""
    record_id: str = Field(pattern=r"^rec_[a-f0-9]{8,}$")
    enrichments: List[str] = Field(default_factory=list)

    @field_validator("enrichments", mode="after")
    @classmethod
    def _norm_enrichments(cls, v: List[str]) -> List[str]:
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]


class ValidationFailedPayload(GraceModel):
    """Payload for ING_VALIDATION_FAILED."""
    record_id: Optional[str] = Field(default=None, pattern=r"^rec_[a-f0-9]{8,}$")
    raw_event_id: Optional[str] = Field(default=None, pattern=r"^rev_[a-f0-9]{8,}$")
    reasons: List[str]
    severity: Severity
    policy: str  # "pii", "schema", "format", "governance"

    @field_validator("reasons", mode="after")
    @classmethod
    def _norm_reasons(cls, v: List[str]) -> List[str]:
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]


class PersistedPayload(GraceModel):
    """Payload for ING_PERSISTED."""
    record_id: str = Field(pattern=r"^rec_[a-f0-9]{8,}$")
    tier: StorageTier
    uri: str = Field(min_length=7)


class PublishedPayload(GraceModel):
    """Payload for ING_PUBLISHED."""
    record_id: str = Field(pattern=r"^rec_[a-f0-9]{8,}$")
    topics: List[str] = Field(default_factory=list)

    @field_validator("topics", mode="after")
    @classmethod
    def _norm_topics(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for s in (x.strip().lower() for x in v if isinstance(x, str)):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out


class SourceHealthPayload(GraceModel):
    """Payload for ING_SOURCE_HEALTH."""
    source_id: str
    status: HealthStatus
    latency_ms: int = Field(ge=0)
    backlog: int = Field(ge=0)
    last_ok: datetime = Field(default_factory=_utcnow)

    @field_validator("last_ok", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


class ExperiencePayload(GraceModel):
    """Payload for ING_EXPERIENCE."""
    schema_version: str = "1.0.0"
    experience: Mapping[str, Any]  # IngressExperience dump


class RollbackRequestedPayload(GraceModel):
    """Payload for ROLLBACK_REQUESTED."""
    target: str = "ingress"
    to_snapshot: str = Field(min_length=8)


class RollbackCompletedPayload(GraceModel):
    """Payload for ROLLBACK_COMPLETED."""
    target: str = "ingress"
    snapshot_id: str = Field(min_length=8)
    at: datetime = Field(default_factory=_utcnow)

    @field_validator("at", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


# Map event types to payload schemas for validation
EVENT_PAYLOAD_MAP: Dict[IngressEventType, Type[GraceModel]] = {
    IngressEventType.SOURCE_REGISTERED:  SourceRegisteredPayload,
    IngressEventType.CAPTURED_RAW:       CapturedRawPayload,
    IngressEventType.PARSED:             ParsedPayload,
    IngressEventType.NORMALIZED:         NormalizedPayload,
    IngressEventType.ENRICHED:           EnrichedPayload,
    IngressEventType.VALIDATION_FAILED:  ValidationFailedPayload,
    IngressEventType.PERSISTED:          PersistedPayload,
    IngressEventType.PUBLISHED:          PublishedPayload,
    IngressEventType.SOURCE_HEALTH:      SourceHealthPayload,
    IngressEventType.EXPERIENCE:         ExperiencePayload,
    IngressEventType.ROLLBACK_REQUESTED: RollbackRequestedPayload,
    IngressEventType.ROLLBACK_COMPLETED: RollbackCompletedPayload,
}


# ---------------------------
# Event envelope
# ---------------------------

class IngressEvent(GraceModel):
    """Base ingress event structure."""
    event_type: IngressEventType
    correlation_id: str = Field(min_length=8)
    timestamp: datetime = Field(default_factory=_utcnow)
    payload: Union[
        SourceRegisteredPayload,
        CapturedRawPayload,
        ParsedPayload,
        NormalizedPayload,
        EnrichedPayload,
        ValidationFailedPayload,
        PersistedPayload,
        PublishedPayload,
        SourceHealthPayload,
        ExperiencePayload,
        RollbackRequestedPayload,
        RollbackCompletedPayload,
    ]
    source: str = "ingress_kernel"
    version: str = "1.0.0"

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @model_validator(mode="after")
    def _validate_payload_type(self) -> "IngressEvent":
        expected = EVENT_PAYLOAD_MAP.get(self.event_type)
        if expected and not isinstance(self.payload, expected):
            # Attempt a best-effort coercion if payload is a plain mapping
            if isinstance(self.payload, Mapping):
                coerced = expected.model_validate(self.payload)  # type: ignore[arg-type]
                object.__setattr__(self, "payload", coerced)
            else:
                raise TypeError(f"payload must be {expected.__name__} for event_type={self.event_type}")
        return self


# ---------------------------
# Factory helper
# ---------------------------

def make_event(
    event_type: IngressEventType,
    payload: Mapping[str, Any] | GraceModel,
    *,
    correlation_id: str,
    source: str = "ingress_kernel",
    version: str = "1.0.0",
    timestamp: Optional[datetime] = None,
) -> IngressEvent:
    """
    Convenience builder that accepts either a typed payload model or a plain mapping.
    Ensures payload type matches event_type and normalizes timestamp to UTC.
    """
    ts = timestamp or _utcnow()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    expected = EVENT_PAYLOAD_MAP[event_type]
    model_payload = payload if isinstance(payload, expected) else expected.model_validate(payload)  # type: ignore[arg-type]
    return IngressEvent(
        event_type=event_type,
        correlation_id=correlation_id,
        timestamp=ts,
        payload=model_payload,
        source=source,
        version=version,
    )


__all__ = [
    # enums
    "IngressEventType", "HealthStatus", "Severity", "StorageTier",
    # payloads
    "SourceRegisteredPayload", "CapturedRawPayload", "ParseReport", "ParsedPayload",
    "NormalizedPayload", "EnrichedPayload", "ValidationFailedPayload",
    "PersistedPayload", "PublishedPayload", "SourceHealthPayload",
    "ExperiencePayload", "RollbackRequestedPayload", "RollbackCompletedPayload",
    # envelope & helpers
    "IngressEvent", "make_event",
]
