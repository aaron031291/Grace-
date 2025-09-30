"""
governed_request.py
Production-grade governed request contracts for Grace Governance Kernel.

- Strict Pydantic v2 models (extra=forbid, validate_assignment).
- Enums for request_type, risk_level, and priority.
- Tags/context normalization; requester/content validation.
- Optional expiry (UTC-normalized) and correlation_id for tracing.
- Stable content_hash for idempotency/caching.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, IntEnum
from hashlib import sha256
from typing import Any, List, Mapping, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .dto_common import BaseDTO  # inherits id, created_at, updated_at, metadata


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
# Enums & literals
# ---------------------------

class RequestType(str, Enum):
    AUTONOMOUS_DECISION = "AUTONOMOUS_DECISION"
    POLICY_CHANGE       = "POLICY_CHANGE"
    CONSTITUTION_CHANGE = "CONSTITUTION_CHANGE"
    OPERATIONAL_ACTION  = "OPERATIONAL_ACTION"
    MEMORY_WRITE        = "MEMORY_WRITE"
    MEMORY_DELETE       = "MEMORY_DELETE"
    DATA_ACCESS         = "DATA_ACCESS"
    EXTERNAL_ACTION     = "EXTERNAL_ACTION"
    OTHER               = "OTHER"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Priority(IntEnum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# ---------------------------
# Request contract
# ---------------------------

class GovernedRequest(BaseDTO):
    """
    A request that requires governance evaluation.

    Fields:
        request_type: Typed category of the request.
        content: Canonical payload (stringified for hashing/audit).
        requester: Entity/component id initiating the request.
        context: Minimal extra context (strict Mapping).
        priority: 1..10 (use Priority enum for clarity).
        tags: Normalized keyword tags (lowercased, deduped).
        policy_domains: Policy areas to evaluate (e.g., ["security","privacy"]).
        risk_level: low|medium|high (affects guardrails).
        requires_quorum: Whether MLDL/committee quorum is required.
        expiry: Optional hard deadline for processing (UTC).
        correlation_id: For cross-component tracing.
        content_hash: Stable SHA-256 over (request_type + "\n" + content).
    """

    # core
    request_type: RequestType
    content: str = Field(min_length=1)
    requester: str = Field(min_length=1)

    # aux
    context: Mapping[str, Any] = Field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    tags: List[str] = Field(default_factory=list)

    # policy
    policy_domains: List[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_quorum: bool = False

    # lifecycle
    expiry: Optional[datetime] = None
    correlation_id: UUID = Field(default_factory=uuid4)

    # computed
    content_hash: Optional[str] = None

    # -------- Validators / Normalizers --------

    @field_validator("requester")
    @classmethod
    def _trim_requester(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("requester cannot be empty")
        return v

    @field_validator("tags", mode="after")
    @classmethod
    def _normalize_tags(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for s in (x.strip().lower() for x in v if isinstance(x, str)):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @field_validator("policy_domains", mode="after")
    @classmethod
    def _normalize_domains(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for s in (x.strip().lower() for x in v if isinstance(x, str)):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @field_validator("expiry", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @model_validator(mode="after")
    def _derive_quorum_from_risk(self) -> "GovernedRequest":
        # Guardrail: high risk implies quorum unless caller explicitly disables (not recommended)
        if self.risk_level == RiskLevel.HIGH:
            object.__setattr__(self, "requires_quorum", True)
        return self

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "GovernedRequest":
        if not self.content_hash:
            payload = (self.request_type.value + "\n" + self.content).encode("utf-8")
            object.__setattr__(self, "content_hash", sha256(payload).hexdigest())
        return self

    # -------- Factories / Helpers --------

    @classmethod
    def from_payload(
        cls,
        *,
        request_type: RequestType,
        payload: str,
        requester: str,
        context: Mapping[str, Any] | None = None,
        priority: Priority = Priority.NORMAL,
        tags: List[str] | None = None,
        policy_domains: List[str] | None = None,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        requires_quorum: Optional[bool] = None,
        expiry: Optional[datetime] = None,
        correlation_id: Optional[UUID] = None,
    ) -> "GovernedRequest":
        """
        Convenience constructor when `payload` already represents the canonical content string.
        If requires_quorum is None, it will be inferred by risk level (HIGH -> True).
        """
        return cls(
            request_type=request_type,
            content=payload,
            requester=requester,
            context=dict(context or {}),
            priority=priority,
            tags=list(tags or []),
            policy_domains=list(policy_domains or []),
            risk_level=risk_level,
            requires_quorum=bool(requires_quorum) if requires_quorum is not None else False,
            expiry=expiry,
            correlation_id=correlation_id or uuid4(),
        )

    def escalate(self) -> None:
        """Escalate priority one step up, capped at CRITICAL."""
        new_val = min(int(self.priority) + 1, int(Priority.CRITICAL))
        object.__setattr__(self, "priority", Priority(new_val))

    def add_tags(self, *tags: str) -> None:
        """Add tags with normalization & dedupe."""
        merged = self.tags + list(tags)
        object.__setattr__(self, "tags", self.__class__._normalize_tags(merged))

    def add_policy_domains(self, *domains: str) -> None:
        """Add policy domains with normalization & dedupe."""
        merged = self.policy_domains + list(domains)
        object.__setattr__(self, "policy_domains", self.__class__._normalize_domains(merged))


__all__ = [
    "GovernedRequest",
    "RequestType",
    "RiskLevel",
    "Priority",
]
