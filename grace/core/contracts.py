"""
Core data structures and contracts for the Grace governance kernel.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid


class DecisionSubject(Enum):
    ACTION = "action"
    POLICY = "policy"
    CLAIM = "claim"
    DEPLOYMENT = "deployment"


class EventType(Enum):
    GOVERNANCE_VALIDATION = "GOVERNANCE_VALIDATION"
    GOVERNANCE_APPROVED = "GOVERNANCE_APPROVED"
    GOVERNANCE_REJECTED = "GOVERNANCE_REJECTED"
    GOVERNANCE_NEEDS_REVIEW = "GOVERNANCE_NEEDS_REVIEW"
    GOVERNANCE_SNAPSHOT_CREATED = "GOVERNANCE_SNAPSHOT_CREATED"
    GOVERNANCE_ROLLBACK = "GOVERNANCE_ROLLBACK"


@dataclass
class Source:
    uri: str
    credibility: float  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    type: str  # "doc", "db", "api"
    pointer: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogicStep:
    step: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    id: str
    statement: str
    sources: List[Source]
    evidence: List[Evidence]
    confidence: float  # 0.0 to 1.0
    logical_chain: List[LogicStep]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "sources": [source.to_dict() for source in self.sources],
            "evidence": [evidence.to_dict() for evidence in self.evidence],
            "confidence": self.confidence,
            "logical_chain": [step.to_dict() for step in self.logical_chain],
        }


@dataclass
class ComponentSignal:
    component: str
    signal: str
    weight: float  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnifiedDecision:
    decision_id: str
    topic: str
    inputs: Dict[str, Any]
    recommendation: str  # "approve", "reject", "review"
    rationale: str
    confidence: float  # 0.0 to 1.0
    trust_score: float  # 0.0 to 1.0
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "topic": self.topic,
            "inputs": self.inputs,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "trust_score": self.trust_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VerifiedClaims:
    claims: List[Claim]
    overall_confidence: float
    verification_status: str  # "verified", "refuted", "inconclusive"
    contradictions: List[str]

    def to_dict(self) -> dict:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "overall_confidence": self.overall_confidence,
            "verification_status": self.verification_status,
            "contradictions": self.contradictions,
        }


@dataclass
class LogicReport:
    argument: Dict[str, Any]
    validity: bool
    confidence: float
    logical_errors: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GovernanceSnapshot:
    snapshot_id: str
    instance_id: str
    version: str
    policies: Dict[str, Any]
    thresholds: Dict[str, float]
    model_weights: Dict[str, float]
    state_hash: str
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "instance_id": self.instance_id,
            "version": self.version,
            "policies": self.policies,
            "thresholds": self.thresholds,
            "model_weights": self.model_weights,
            "state_hash": self.state_hash,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Experience:
    type: str
    component_id: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    success_score: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "component_id": self.component_id,
            "context": self.context,
            "outcome": self.outcome,
            "success_score": self.success_score,
            "timestamp": self.timestamp.isoformat(),
        }


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracing requests."""
    return f"corr_{uuid.uuid4().hex[:12]}"


def generate_decision_id() -> str:
    """Generate a unique decision ID."""
    return f"dec_{uuid.uuid4().hex[:12]}"


def generate_snapshot_id() -> str:
    """Generate a unique snapshot ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"govsnap_{timestamp}_{uuid.uuid4().hex[:8]}"
