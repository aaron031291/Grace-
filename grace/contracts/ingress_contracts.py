"""
ingress_contracts.py
Ingress Kernel Contracts — Production-grade data models for the ingestion pipeline.

Key properties:
- Strict Pydantic v2 models (extra=forbid, validate_assignment).
- Timezone-aware datetimes (UTC normalization).
- Stable ID/hash helpers and pattern-validated identifiers.
- Light normalization for tags/entities/flags and safety checks (sizes, ranges).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, StrEnum
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------
# Helpers
# ---------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def generate_event_id() -> str:
    """Generate a raw event ID."""
    return f"rev_{uuid4().hex[:12]}"

def generate_record_id() -> str:
    """Generate a normalized record ID."""
    return f"rec_{uuid4().hex[:12]}"

def generate_source_id(prefix: str = "src") -> str:
    """Generate a source ID."""
    return f"{prefix}_{uuid4().hex[:8]}"

def content_hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()

def content_hash_obj(obj: Any) -> str:
    return sha256(repr(obj).encode("utf-8")).hexdigest()


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


# ---------------------------
# Enums
# ---------------------------

class SourceKind(StrEnum):
    HTTP = "http"
    RSS = "rss"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    GITHUB = "github"
    YOUTUBE = "youtube"
    PODCAST = "podcast"
    SOCIAL = "social"
    KAFKA = "kafka"
    MQTT = "mqtt"
    SQL = "sql"
    CSV_LOCAL = "csv_local"


class AuthMode(StrEnum):
    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SIGNED_URL = "signed_url"
    AWS_IAM = "aws_iam"
    GCP_SA = "gcp_sa"


class ParserType(StrEnum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    XML = "xml"


class PIIPolicy(StrEnum):
    BLOCK = "block"
    MASK = "mask"
    HASH = "hash"
    ALLOW_WITH_CONSENT = "allow_with_consent"


class GovernanceLabel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class RawEventKind(StrEnum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    XML = "xml"
    BIN = "bin"


# ---------------------------
# Core models
# ---------------------------

class SourceConfig(GraceModel):
    """Source registration configuration."""
    source_id: str = Field(default_factory=generate_source_id, pattern=r"^src_[a-f0-9]{8}$")
    kind: SourceKind
    uri: str = Field(min_length=5)
    auth_mode: AuthMode
    secrets_ref: Optional[str] = None
    schedule: str = Field(min_length=1, description='cron expression or "stream"')
    parser: ParserType
    parser_opts: Optional[Mapping[str, Any]] = None
    target_contract: str = Field(min_length=5, description='e.g., "contract:article.v1"')
    retention_days: int = Field(..., ge=1)
    pii_policy: PIIPolicy
    governance_label: GovernanceLabel
    enabled: bool = True

    @field_validator("uri")
    @classmethod
    def _trim_uri(cls, v: str) -> str:
        return v.strip()

    @field_validator("schedule")
    @classmethod
    def _trim_schedule(cls, v: str) -> str:
        return v.strip()

    @field_validator("target_contract")
    @classmethod
    def _normalize_contract(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("contract:"):
            raise ValueError('target_contract must start with "contract:"')
        return v


class RawEvent(GraceModel):
    """Raw event before normalization (Bronze tier)."""
    event_id: str = Field(default_factory=generate_event_id, pattern=r"^rev_[a-f0-9]{12}$")
    source_id: str = Field(min_length=5)
    kind: RawEventKind
    payload: Union[str, bytes, Mapping[str, Any]]
    headers: Optional[Mapping[str, Any]] = None
    ingestion_ts: datetime = Field(default_factory=_utcnow)
    offset: str = Field(min_length=1)  # stream position
    watermark: datetime = Field(default_factory=_utcnow)
    hash: str = Field(min_length=16)  # content hash for dedup
    size_bytes: Optional[int] = Field(default=None, ge=0)

    @field_validator("ingestion_ts", "watermark", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @model_validator(mode="after")
    def _derive_size_and_hash(self) -> "RawEvent":
        # derive size if missing; hash must be provided or computed
        if self.size_bytes is None:
            if isinstance(self.payload, bytes):
                object.__setattr__(self, "size_bytes", len(self.payload))
            elif isinstance(self.payload, str):
                object.__setattr__(self, "size_bytes", len(self.payload.encode("utf-8")))
            else:
                object.__setattr__(self, "size_bytes", len(repr(self.payload).encode("utf-8")))
        # compute hash if placeholder provided as empty
        if not self.hash or len(self.hash) < 16:
            obj = self.payload if isinstance(self.payload, (bytes, bytearray)) else repr(self.payload).encode("utf-8")
            object.__setattr__(self, "hash", content_hash_bytes(bytes(obj)))
        return self


class SourceInfo(GraceModel):
    """Source information in normalized record."""
    source_id: str = Field(min_length=5)
    uri: str = Field(min_length=5)
    fetched_at: datetime = Field(default_factory=_utcnow)
    parser: ParserType
    content_hash: str = Field(min_length=16)

    @field_validator("fetched_at", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


class QualityMetrics(GraceModel):
    """Quality metrics for normalized record."""
    validity_score: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    freshness_minutes: float = Field(..., ge=0.0)
    pii_flags: List[str] = Field(default_factory=list)
    trust_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("pii_flags", mode="after")
    @classmethod
    def _normalize_flags(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for s in (x.strip().lower() for x in v if isinstance(x, str)):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out


class LineageInfo(GraceModel):
    """Lineage information for traceability."""
    raw_event_id: str = Field(pattern=r"^rev_[a-f0-9]{12}$")
    transforms: List[str] = Field(default_factory=list)

    @field_validator("transforms", mode="after")
    @classmethod
    def _normalize_transforms(cls, v: List[str]) -> List[str]:
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]


class NormRecord(GraceModel):
    """Normalized record (Silver tier)."""
    record_id: str = Field(default_factory=generate_record_id, pattern=r"^rec_[a-f0-9]{12}$")
    contract: str = Field(min_length=5, description='e.g., "contract:article.v1"')
    body: Mapping[str, Any]  # conforms to contract schema
    source: SourceInfo
    quality: QualityMetrics
    lineage: LineageInfo
    ts: datetime = Field(default_factory=_utcnow)

    @field_validator("contract")
    @classmethod
    def _check_contract_prefix(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("contract:"):
            raise ValueError('contract must start with "contract:"')
        return v

    @field_validator("ts", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


# ---------------------------
# Contract schemas
# ---------------------------

class ArticleContract(GraceModel):
    """Article content contract (contract:article.v1)."""
    title: str = Field(min_length=1)
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    url: str = Field(min_length=7)
    language: str = Field(min_length=2, max_length=10)  # ISO-like guard
    text: str = Field(min_length=1)
    entities: Dict[str, List[str]] = Field(
        default_factory=lambda: {"persons": [], "orgs": [], "locations": []}
    )
    topics: List[str] = Field(default_factory=list)
    embeddings_ref: Optional[str] = None

    @field_validator("published_at", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @field_validator("topics", mode="after")
    @classmethod
    def _normalize_topics(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for s in (x.strip().lower() for x in v if isinstance(x, str)):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @field_validator("entities", mode="after")
    @classmethod
    def _normalize_entities(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for k, items in (v or {}).items():
            seen, vals = set(), []
            for s in (x.strip() for x in items if isinstance(x, str)):
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    vals.append(s)
            out[k.strip().lower()] = vals
        return out


class TranscriptSegment(GraceModel):
    """Transcript segment."""
    t0: float = Field(ge=0.0)  # start time
    t1: float = Field(ge=0.0)  # end time
    text: str = Field(min_length=1)
    speaker: Optional[str] = None

    @model_validator(mode="after")
    def _check_bounds(self) -> "TranscriptSegment":
        if self.t1 < self.t0:
            raise ValueError("t1 must be >= t0")
        return self


class TranscriptContract(GraceModel):
    """Transcript content contract (contract:transcript.v1)."""
    media_id: str = Field(min_length=1)
    start_at: Optional[datetime] = None
    duration_s: Optional[float] = Field(default=None, ge=0.0)
    lang: str = Field(min_length=2, max_length=10)
    segments: List[TranscriptSegment] = Field(default_factory=list)
    summary: Optional[str] = None

    @field_validator("start_at", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


class TabularColumn(GraceModel):
    """Column definition for tabular data."""
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)  # keep as free string; validate against runtime schema if needed


class TabularContract(GraceModel):
    """Tabular data contract (contract:tabular.v1)."""
    dataset_id: str = Field(min_length=1)
    columns: List[TabularColumn]
    rows_uri: str = Field(min_length=7)  # pointer to parquet/csv
    row_count: int = Field(ge=0)


# ---------------------------
# Experience / Snapshot
# ---------------------------

class IngressExperience(GraceModel):
    """Experience data sent to MLT for learning."""
    exp_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str = Field(min_length=5)
    stage: str = Field(pattern=r"^(capture|parse|normalize|validate|enrich|persist|publish)$")
    metrics: Mapping[str, Any]
    samples: Optional[Mapping[str, Any]] = None
    timestamp: datetime = Field(default_factory=_utcnow)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


class IngressSnapshot(GraceModel):
    """Ingress system snapshot for rollback."""
    snapshot_id: str = Field(min_length=8)
    active_sources: List[str] = Field(default_factory=list)
    registry_hash: str = Field(min_length=16)
    parser_versions: Mapping[str, str] = Field(default_factory=dict)
    dedupe_threshold: float = Field(ge=0.0, le=1.0)
    pii_policy_defaults: PIIPolicy
    offsets: Mapping[str, str] = Field(default_factory=dict)          # source_id -> offset
    watermarks: Mapping[str, datetime] = Field(default_factory=dict)  # source_id -> watermark
    gold_views_version: str = Field(min_length=1)
    hash: str = Field(min_length=16)  # snapshot hash
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("watermarks", mode="after")
    @classmethod
    def _ensure_tz_map(cls, mp: Mapping[str, datetime]) -> Mapping[str, datetime]:
        # normalize any naive datetime to UTC
        norm: Dict[str, datetime] = {}
        for k, dt in mp.items():
            if isinstance(dt, datetime) and dt.tzinfo is None:
                norm[k] = dt.replace(tzinfo=timezone.utc)
            else:
                norm[k] = dt
        return norm


# ---------------------------
# Convenience factories
# ---------------------------

def make_article_record(
    *,
    source: SourceInfo,
    article: ArticleContract,
    lineage: LineageInfo,
    quality: QualityMetrics,
    record_id: Optional[str] = None,
) -> NormRecord:
    """Helper to build a normalized Article record (contract:article.v1)."""
    contract_name = "contract:article.v1"
    body = article.model_dump()
    return NormRecord(
        record_id=record_id or generate_record_id(),
        contract=contract_name,
        body=body,
        source=source,
        quality=quality,
        lineage=lineage,
    )


def make_transcript_record(
    *,
    source: SourceInfo,
    transcript: TranscriptContract,
    lineage: LineageInfo,
    quality: QualityMetrics,
    record_id: Optional[str] = None,
) -> NormRecord:
    """Helper to build a normalized Transcript record (contract:transcript.v1)."""
    contract_name = "contract:transcript.v1"
    body = transcript.model_dump()
    return NormRecord(
        record_id=record_id or generate_record_id(),
        contract=contract_name,
        body=body,
        source=source,
        quality=quality,
        lineage=lineage,
    )


__all__ = [
    # enums
    "SourceKind", "AuthMode", "ParserType", "PIIPolicy", "GovernanceLabel", "RawEventKind",
    # core
    "SourceConfig", "RawEvent", "SourceInfo", "QualityMetrics", "LineageInfo", "NormRecord",
    # contracts
    "ArticleContract", "TranscriptSegment", "TranscriptContract",
    "TabularColumn", "TabularContract",
    # experience/snapshot
    "IngressExperience", "IngressSnapshot",
    # helpers
    "generate_event_id", "generate_record_id", "generate_source_id",
    "content_hash_bytes", "content_hash_obj",
    "make_article_record", "make_transcript_record",
]
