"""Interface data models based on JSON schemas."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import uuid
import re


class A11yPreferences(BaseModel):
    """Accessibility preferences."""
    high_contrast: Optional[bool] = False
    reduce_motion: Optional[bool] = False


class UserIdentity(BaseModel):
    """User identity and profile."""
    user_id: str = Field(pattern=r"usr_[a-z0-9_-]{4,}")
    display_name: Optional[str] = None
    roles: List[Literal["owner", "admin", "dev", "analyst", "viewer"]]
    labels: Optional[List[Literal["internal", "restricted", "external"]]] = None
    locale: str = "en-GB"
    a11y: Optional[A11yPreferences] = None


class ClientInfo(BaseModel):
    """Client connection information."""
    agent: Optional[str] = None
    ip: Optional[str] = None
    device: Optional[str] = None


class UISession(BaseModel):
    """UI session with user and client info."""
    session_id: str = Field(pattern=r"ses_[a-z0-9]{8,}")
    user: UserIdentity
    client: ClientInfo
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    permissions: Optional[List[str]] = None


class PolicyCondition(BaseModel):
    """Policy rule condition."""
    label_in: Optional[List[str]] = None
    time_after: Optional[str] = None
    time_before: Optional[str] = None


class PolicyRule(BaseModel):
    """RBAC policy rule."""
    rule_id: str
    effect: Literal["allow", "deny"]
    actions: List[str]
    resources: List[str]
    condition: PolicyCondition


class MessageContent(BaseModel):
    """Task thread message content."""
    text: Optional[str] = None
    code: Optional[str] = None
    rich_blocks: Optional[List[Dict[str, Any]]] = None


class ThreadMessage(BaseModel):
    """Task card thread message."""
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "grace", "specialist", "governance"]
    author: str
    at: datetime = Field(default_factory=datetime.utcnow)
    content: MessageContent


class TaskMetrics(BaseModel):
    """Task execution metrics."""
    latency_ms: Optional[float] = None
    steps: Optional[int] = None


class TaskCard(BaseModel):
    """Primary UX unit - task card with thread."""
    card_id: str = Field(pattern=r"card_[a-z0-9]{8,}")
    title: str
    kind: Literal["analysis", "build", "ingest", "govern", "intel", "memory", "mlt", "debug"]
    owner: str  # user_id
    state: Literal["open", "running", "paused", "waiting_approval", "done", "failed"]
    context: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    thread: Optional[List[ThreadMessage]] = None
    metrics: Optional[TaskMetrics] = None
    approvals: Optional[List[str]] = None


class ConsentRecord(BaseModel):
    """User consent record."""
    consent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    scope: Literal["autonomy", "pii_use", "external_share", "canary_participation"]
    status: Literal["granted", "denied", "revoked", "pending"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    evidence_uri: Optional[str] = None


class UIAction(BaseModel):
    """User interface action."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    type: Literal[
        "task.create", "task.update", "task.run", "task.pause", "task.cancel",
        "memory.search", "intel.request", "ingress.register_source",
        "governance.request_approval", "consent.grant", "consent.revoke",
        "settings.update", "snapshot.export", "snapshot.rollback"
    ]
    payload: Dict[str, Any]
    at: datetime = Field(default_factory=datetime.utcnow)


class NotificationAction(BaseModel):
    """Notification action button."""
    label: str
    action: str


class Notification(BaseModel):
    """User notification."""
    notif_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    level: Literal["info", "success", "warning", "error"]
    message: str
    actions: Optional[List[NotificationAction]] = None
    read: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UIExperienceMetrics(BaseModel):
    """UX telemetry metrics."""
    p95_interaction_ms: Optional[float] = None
    task_completion_rate: Optional[float] = None
    approval_cycle_time_s: Optional[float] = None
    drop_off_rate: Optional[float] = None
    a11y_violations: Optional[int] = None
    ws_disconnects_per_hr: Optional[float] = None
    error_rate: Optional[float] = None
    satisfaction_csat: Optional[float] = None


class UIExperience(BaseModel):
    """UX telemetry experience data."""
    exp_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: Literal["latency", "interaction", "approval_flow", "error", "a11y", "i18n"]
    metrics: UIExperienceMetrics
    segment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Event payloads
class UISessionStartedPayload(BaseModel):
    """UI_SESSION_STARTED event payload."""
    session: UISession


class UIActionPayload(BaseModel):
    """UI_ACTION event payload."""
    action: UIAction


class UITaskCardUpdatedPayload(BaseModel):
    """UI_TASKCARD_UPDATED event payload."""
    card: TaskCard


class UINotificationPayload(BaseModel):
    """UI_NOTIFICATION event payload."""
    notification: Notification
    user_id: str


class UIPolicyViolationPayload(BaseModel):
    """UI_POLICY_VIOLATION event payload."""
    action_id: str
    reasons: List[str]
    severity: Literal["warn", "error"]


class UIExperiencePayload(BaseModel):
    """UI_EXPERIENCE event payload."""
    schema_version: str = "1.0.0"
    experience: UIExperience


class RollbackRequestedPayload(BaseModel):
    """ROLLBACK_REQUESTED event payload."""
    target: str = "interface"
    to_snapshot: str


class RollbackCompletedPayload(BaseModel):
    """ROLLBACK_COMPLETED event payload."""
    target: str = "interface"
    snapshot_id: str
    at: datetime


# Helper functions
def generate_card_id() -> str:
    """Generate a new card ID."""
    return f"card_{uuid.uuid4().hex[:8]}"


def generate_session_id() -> str:
    """Generate a new session ID."""
    return f"ses_{uuid.uuid4().hex[:8]}"


def generate_user_id(username: str) -> str:
    """Generate a user ID from username."""
    clean_name = re.sub(r'[^a-z0-9_-]', '', username.lower())
    return f"usr_{clean_name}"