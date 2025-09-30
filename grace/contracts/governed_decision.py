"""
governed_decision.py
Production-grade governed decision contracts for Grace Governance Kernel.

- Strict Pydantic v2 models (extra=forbid, validate_assignment).
- Timezone-aware datetimes (UTC normalization).
- Strongly-typed outcomes for policy, verification, and quorum.
- Audit snapshots, immutable-log references, and optional signatures.
- Convenience factory: GovernedDecision.from_request(...)
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

# Import your shared DTO base
from .dto_common import BaseDTO  # noqa: F401


# ---------------------------
# Base config for all models
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
# Result primitives
# ---------------------------

class PolicyOutcome(GraceModel):
    name: str = Field(min_length=1)
    passed: bool
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    details: Mapping[str, Any] = Field(default_factory=dict)


class VerificationOutcome(GraceModel):
    name: str = Field(min_length=1)
    passed: bool
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_hash: Optional[str] = None
    details: Mapping[str, Any] = Field(default_factory=dict)


class SpecialistVote(GraceModel):
    specialist_id: str = Field(min_length=1)
    vote: str = Field(pattern=r"^(approve|reject|abstain)$")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rationale: Optional[str] = None


class QuorumOutcome(GraceModel):
    participating: int = Field(ge=0)
    threshold: float = Field(ge=0.0, le=1.0)  # e.g., 0.65 consensus
    votes_for: int = Field(ge=0)
    votes_against: int = Field(ge=0)
    abstained: int = Field(ge=0)
    consensus: bool
    aggregate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    votes: List[SpecialistVote] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_counts(self) -> "QuorumOutcome":
        total = self.votes_for + self.votes_against + self.abstained
        if total != self.participating:
            raise ValueError("votes_for + votes_against + abstained must equal participating")
        return self


class DecisionCondition(GraceModel):
    """Runtime conditions that must hold prior to execution."""
    description: str = Field(min_length=1)
    must_hold: bool = True
    kpi_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trust_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    window_seconds: Optional[int] = Field(default=None, ge=0)


class DecisionSignature(GraceModel):
    """Optional crypto signature for audit/immutability."""
    signer: str = Field(min_length=1)            # component/entity id
    algorithm: str = Field(min_length=1)         # e.g., "ed25519"
    signature: str = Field(min_length=1)         # base64/hex per your convention


# ---------------------------
# Main contract
# ---------------------------

DecisionMaker = str  # constrain via config/policy ("grace-governance", "parliament", "auto-rollback", etc.)


class GovernedDecision(BaseDTO):  # inherits id/created_at/updated_at/metadata
    """
    The canonical result of a governance evaluation.

    Fields:
        request_id: UUID/str for the governed request.
        approved: Overall approval flag from governance.
        confidence: Confidence in the decision (0..1).
        reasoning: Human-readable explanation.
        policy_results: Detailed policy checks (pass/fail/score).
        verification_results: Verification checks (pass/fail/score/evidence).
        quorum_results: MLDL/committee quorum outcome, if applicable.
        decision_maker: Provenance of this decision (e.g., "grace-governance").
        execution_approved: If True, execution may proceed (subject to conditions).
        conditions: Conditions that must hold to execute (KPI/trust windows, etc.).
        expiry: Optional hard deadline for decision execution.
        thresholds_snapshot: Governance thresholds captured at decision time.
        metrics_snapshot: Core metrics captured at decision time (e.g., {"kpi":0.96,"trust":0.93}).
        audit_refs: Immutable log record ids.
        signatures: Optional crypto signatures.
        extras: Free-form, strictly a mapping, for future-proofing.
        decision_hash: Stable content hash of this decision (excludes signatures by default).
    """

    # identity of the governed request
    request_id: UUID | str

    # core outcome
    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    # structured outcomes
    policy_results: List[PolicyOutcome] = Field(default_factory=list)
    verification_results: List[VerificationOutcome] = Field(default_factory=list)
    quorum_results: Optional[QuorumOutcome] = None

    # decision metadata
    decision_maker: DecisionMaker = "grace-governance"
    execution_approved: bool = False
    conditions: List[DecisionCondition] = Field(default_factory=list)
    expiry: Optional[datetime] = None  # hard deadline for execution (UTC)

    # audit snapshots
    thresholds_snapshot: Mapping[str, Any] = Field(default_factory=dict)
    metrics_snapshot: Mapping[str, Any] = Field(default_factory=dict)

    # audit wiring
    audit_refs: List[str] = Field(default_factory=list)
    signatures: List[DecisionSignature] = Field(default_factory=list)

    # extension point
    extras: Mapping[str, Any] = Field(default_factory=dict)

    # computed
    decision_hash: Optional[str] = None

    # ------------- Validators -------------
    @field_validator("expiry", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @model_validator(mode="after")
    def _guard_execution_flag(self) -> "GovernedDecision":
        # execution_approved requires either approval or an auto-rollback authority
        if self.execution_approved and not (
            self.approved or self.decision_maker.lower() == "auto-rollback"
        ):
            raise ValueError(
                "execution_approved requires approved=True or decision_maker='auto-rollback'"
            )
        return self

    @model_validator(mode="after")
    def _compute_hash_if_missing(self) -> "GovernedDecision":
        if not self.decision_hash:
            object.__setattr__(self, "decision_hash", self.compute_hash())
        return self

    # ------------- API helpers -------------

    def compute_hash(self, *, include_signatures: bool = False) -> str:
        """
        Compute a stable SHA-256 over key, order-stable fields.
        By default excludes signatures to keep the hash stable before/after signing.
        """
        # Build a canonical dict (avoid non-deterministic Mapping order by conversion)
        base: Dict[str, Any] = {
            "request_id": str(self.request_id),
            "approved": self.approved,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "policy_results": [po.model_dump() for po in self.policy_results],
            "verification_results": [vo.model_dump() for vo in self.verification_results],
            "quorum_results": self.quorum_results.model_dump() if self.quorum_results else None,
            "decision_maker": self.decision_maker,
            "execution_approved": self.execution_approved,
            "conditions": [c.model_dump() for c in self.conditions],
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "thresholds_snapshot": dict(self.thresholds_snapshot),
            "metrics_snapshot": dict(self.metrics_snapshot),
            "audit_refs": list(self.audit_refs),
            "extras": dict(self.extras),
        }
        if include_signatures:
            base["signatures"] = [s.model_dump() for s in self.signatures]

        serialized = repr(base).encode("utf-8")
        return sha256(serialized).hexdigest()

    def to_audit_record(self) -> Dict[str, Any]:
        """
        Produce a compact, append-only audit payload.
        Store in immutable logs; reference id can be pushed into `audit_refs`.
        """
        return {
            "decision_id": str(self.id),
            "request_id": str(self.request_id),
            "timestamp": self.created_at.isoformat(),
            "maker": self.decision_maker,
            "approved": self.approved,
            "confidence": self.confidence,
            "hash": self.decision_hash,
            "audit_refs": list(self.audit_refs),
            "thresholds": dict(self.thresholds_snapshot),
            "metrics": dict(self.metrics_snapshot),
        }

    # --------- Factory: from request ---------

    @classmethod
    def from_request(
        cls,
        *,
        request_id: UUID | str,
        approved: bool,
        confidence: float,
        reasoning: str,
        policy_results: Iterable[PolicyOutcome] | None = None,
        verification_results: Iterable[VerificationOutcome] | None = None,
        quorum_results: QuorumOutcome | None = None,
        decision_maker: DecisionMaker = "grace-governance",
        execution_approved: bool = False,
        conditions: Iterable[DecisionCondition] | None = None,
        expiry: Optional[datetime] = None,
        thresholds_snapshot: Mapping[str, Any] | None = None,
        metrics_snapshot: Mapping[str, Any] | None = None,
        audit_refs: Iterable[str] | None = None,
        signatures: Iterable[DecisionSignature] | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> "GovernedDecision":
        """
        Build a decision from raw evaluation artifacts.
        You can pass config-derived thresholds via `thresholds_snapshot` and live KPIs via `metrics_snapshot`.
        """
        return cls(
            request_id=request_id,
            approved=approved,
            confidence=confidence,
            reasoning=reasoning,
            policy_results=list(policy_results or []),
            verification_results=list(verification_results or []),
            quorum_results=quorum_results,
            decision_maker=decision_maker,
            execution_approved=execution_approved,
            conditions=list(conditions or []),
            expiry=expiry,
            thresholds_snapshot=dict(thresholds_snapshot or {}),
            metrics_snapshot=dict(metrics_snapshot or {}),
            audit_refs=list(audit_refs or []),
            signatures=list(signatures or []),
            extras=dict(extras or {}),
        )

    # --------- Mutation helpers (safe) ---------

    def add_signature(self, signer: str, algorithm: str, signature: str) -> None:
        """Append a signature and refresh decision_hash only if configured to include signatures."""
        self.signatures.append(DecisionSignature(signer=signer, algorithm=algorithm, signature=signature))

    def add_audit_ref(self, ref_id: str) -> None:
        """Attach an immutable-log id (e.g., CID, tx id)."""
        self.audit_refs.append(ref_id)

    def require_conditions(self, *conds: DecisionCondition) -> None:
        """Append one or more runtime conditions."""
        self.conditions.extend(list(conds))


__all__ = [
    "GovernedDecision",
    "PolicyOutcome",
    "VerificationOutcome",
    "SpecialistVote",
    "QuorumOutcome",
    "DecisionCondition",
    "DecisionSignature",
    "DecisionMaker",
]
