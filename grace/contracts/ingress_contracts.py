"""
Ingress Kernel Contracts - Data models for ingestion pipeline.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid


def generate_event_id() -> str:
    """Generate a raw event ID."""
    return f"rev_{uuid.uuid4().hex[:12]}"


def generate_record_id() -> str:
    """Generate a normalized record ID."""
    return f"rec_{uuid.uuid4().hex[:12]}"


def generate_source_id(prefix: str = "src") -> str:
    """Generate a source ID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class SourceKind(str, Enum):
    """Source types for ingestion."""
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


class AuthMode(str, Enum):
    """Authentication modes."""
    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SIGNED_URL = "signed_url"
    AWS_IAM = "aws_iam"
    GCP_SA = "gcp_sa"


class ParserType(str, Enum):
    """Parser types for content processing."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    XML = "xml"


class PIIPolicy(str, Enum):
    """PII handling policies."""
    BLOCK = "block"
    MASK = "mask"
    HASH = "hash"
    ALLOW_WITH_CONSENT = "allow_with_consent"


class GovernanceLabel(str, Enum):
    """Governance classification labels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class RawEventKind(str, Enum):
    """Raw event content types."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    XML = "xml"
    BIN = "bin"


class SourceConfig(BaseModel):
    """Source registration configuration."""
    source_id: str = Field(..., pattern=r"src_[a-z0-9_-]{3,40}")
    kind: SourceKind
    uri: str
    auth_mode: AuthMode
    secrets_ref: Optional[str] = None
    schedule: str  # cron or "stream"
    parser: ParserType
    parser_opts: Optional[Dict[str, Any]] = None
    target_contract: str  # e.g., "contract:article.v1"
    retention_days: int = Field(..., ge=1)
    pii_policy: PIIPolicy
    governance_label: GovernanceLabel
    enabled: bool = True


class RawEvent(BaseModel):
    """Raw event before normalization (Bronze tier)."""
    event_id: str = Field(..., pattern=r"rev_[a-z0-9]{8,}")
    source_id: str
    kind: RawEventKind
    payload: Union[str, bytes, Dict[str, Any]]
    headers: Optional[Dict[str, Any]] = None
    ingestion_ts: datetime = Field(default_factory=datetime.utcnow)
    offset: str  # stream position
    watermark: datetime = Field(default_factory=datetime.utcnow)
    hash: str  # content hash for dedup


class SourceInfo(BaseModel):
    """Source information in normalized record."""
    source_id: str
    uri: str
    fetched_at: datetime
    parser: str
    content_hash: str


class QualityMetrics(BaseModel):
    """Quality metrics for normalized record."""
    validity_score: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    freshness_minutes: float = Field(..., ge=0.0)
    pii_flags: List[str] = Field(default_factory=list)
    trust_score: float = Field(..., ge=0.0, le=1.0)


class LineageInfo(BaseModel):
    """Lineage information for traceability."""
    raw_event_id: str
    transforms: List[str] = Field(default_factory=list)


class NormRecord(BaseModel):
    """Normalized record (Silver tier)."""
    record_id: str = Field(..., pattern=r"rec_[a-z0-9]{8,}")
    contract: str  # e.g., contract:article.v1
    body: Dict[str, Any]  # conforms to contract schema
    source: SourceInfo
    quality: QualityMetrics
    lineage: LineageInfo
    ts: datetime = Field(default_factory=datetime.utcnow)


# Contract schemas for specific content types
class ArticleContract(BaseModel):
    """Article content contract (contract:article.v1)."""
    title: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    url: str
    language: str
    text: str
    entities: Dict[str, List[str]] = Field(
        default_factory=lambda: {"persons": [], "orgs": [], "locations": []}
    )
    topics: List[str] = Field(default_factory=list)
    embeddings_ref: Optional[str] = None


class TranscriptSegment(BaseModel):
    """Transcript segment."""
    t0: float  # start time
    t1: float  # end time
    text: str
    speaker: Optional[str] = None


class TranscriptContract(BaseModel):
    """Transcript content contract (contract:transcript.v1)."""
    media_id: str
    start_at: Optional[datetime] = None
    duration_s: Optional[float] = None
    lang: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    summary: Optional[str] = None


class TabularColumn(BaseModel):
    """Column definition for tabular data."""
    name: str
    type: str  # data type


class TabularContract(BaseModel):
    """Tabular data contract (contract:tabular.v1)."""
    dataset_id: str
    columns: List[TabularColumn]
    rows_uri: str  # pointer to parquet/csv
    row_count: int


# Experience data for MLT feedback
class IngressExperience(BaseModel):
    """Experience data sent to MLT for learning."""
    exp_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    stage: str  # capture, parse, normalize, validate, enrich, persist, publish
    metrics: Dict[str, Any]
    samples: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Snapshot structure for rollback capability
class IngressSnapshot(BaseModel):
    """Ingress system snapshot for rollback."""
    snapshot_id: str
    active_sources: List[str]
    registry_hash: str
    parser_versions: Dict[str, str]
    dedupe_threshold: float
    pii_policy_defaults: PIIPolicy
    offsets: Dict[str, str]  # source_id -> offset
    watermarks: Dict[str, datetime]  # source_id -> watermark
    gold_views_version: str
    hash: str  # snapshot hash
    created_at: datetime = Field(default_factory=datetime.utcnow)