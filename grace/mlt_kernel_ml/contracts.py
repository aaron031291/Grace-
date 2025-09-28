"""
Core data contracts for MLT Kernel ML components.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from enum import Enum
import uuid


class ExperienceSource(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    GOVERNANCE = "governance"
    OPS = "ops"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMRED = "dimred"
    RL = "rl"


class InsightType(Enum):
    PERFORMANCE = "performance"
    DRIFT = "drift"
    FAIRNESS = "fairness"
    CALIBRATION = "calibration"
    STABILITY = "stability"
    GOVERNANCE_ALIGNMENT = "governance_alignment"


class RecommendationType(Enum):
    RETRAIN = "retrain"
    REWEIGHT = "reweight"
    RECALIBRATE = "recalibrate"
    HPO = "hpo"
    POLICY_TUNE = "policy_tune"
    SEGMENT_ROUTE = "segment_route"


class ActionType(Enum):
    HPO = "hpo"
    REWEIGHT_SPECIALISTS = "reweight_specialists"
    POLICY_DELTA = "policy_delta"
    CANARY = "canary"


@dataclass
class Experience:
    """Input experience from various sources."""
    experience_id: str
    source: ExperienceSource
    task: TaskType
    context: Dict[str, Any]
    signals: Dict[str, Any]
    ground_truth_lag_s: int
    timestamp: datetime
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        return cls(
            experience_id=data["experience_id"],
            source=ExperienceSource(data["source"]),
            task=TaskType(data["task"]),
            context=data["context"],
            signals=data["signals"],
            ground_truth_lag_s=data["ground_truth_lag_s"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def to_dict(self) -> Dict:
        return {
            "experience_id": self.experience_id,
            "source": self.source.value,
            "task": self.task.value,
            "context": self.context,
            "signals": self.signals,
            "ground_truth_lag_s": self.ground_truth_lag_s,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Insight:
    """Generated insight from experience analysis."""
    insight_id: str
    type: InsightType
    scope: str
    evidence: Dict[str, Any]
    confidence: float
    recommendation: RecommendationType
    timestamp: datetime
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Insight':
        return cls(
            insight_id=data["insight_id"],
            type=InsightType(data["type"]),
            scope=data["scope"],
            evidence=data["evidence"],
            confidence=data["confidence"],
            recommendation=RecommendationType(data["recommendation"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "type": self.type.value,
            "scope": self.scope,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "recommendation": self.recommendation.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Action:
    """Single action in an adaptation plan."""
    type: ActionType
    target: Optional[str] = None
    budget: Optional[Dict[str, Any]] = None
    success_metric: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    path: Optional[str] = None
    from_value: Optional[Any] = None
    to_value: Optional[Any] = None
    target_model: Optional[str] = None
    steps: Optional[List[int]] = None
    
    def to_dict(self) -> Dict:
        result = {"type": self.type.value}
        if self.target:
            result["target"] = self.target
        if self.budget:
            result["budget"] = self.budget
        if self.success_metric:
            result["success_metric"] = self.success_metric
        if self.weights:
            result["weights"] = self.weights
        if self.path:
            result["path"] = self.path
        if self.from_value is not None:
            result["from"] = self.from_value
        if self.to_value is not None:
            result["to"] = self.to_value
        if self.target_model:
            result["target_model"] = self.target_model
        if self.steps:
            result["steps"] = self.steps
        return result


@dataclass
class AdaptationPlan:
    """Concrete adaptation plan for governance approval."""
    plan_id: str
    actions: List[Action]
    expected_effect: Dict[str, str]
    risk_controls: Dict[str, Union[int, float]]
    timestamp: datetime
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AdaptationPlan':
        actions = [Action(**action) if isinstance(action, dict) else action for action in data["actions"]]
        return cls(
            plan_id=data["plan_id"],
            actions=actions,
            expected_effect=data["expected_effect"],
            risk_controls=data["risk_controls"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "actions": [action.to_dict() for action in self.actions],
            "expected_effect": self.expected_effect,
            "risk_controls": self.risk_controls,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MLTSnapshot:
    """Versioned snapshot of MLT state."""
    snapshot_id: str
    planner_version: str
    search_spaces: Dict[str, str]
    weights: Dict[str, float]
    policies: Dict[str, Any]
    active_jobs: List[Dict[str, Any]]
    hash: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "snapshot_id": self.snapshot_id,
            "planner_version": self.planner_version,
            "search_spaces": self.search_spaces,
            "weights": self.weights,
            "policies": self.policies,
            "active_jobs": self.active_jobs,
            "hash": self.hash,
            "timestamp": self.timestamp.isoformat()
        }


def generate_experience_id() -> str:
    """Generate unique experience ID."""
    return f"exp_{uuid.uuid4().hex[:12]}"


def generate_insight_id() -> str:
    """Generate unique insight ID."""
    return f"ins_{uuid.uuid4().hex[:12]}"


def generate_plan_id() -> str:
    """Generate unique plan ID."""
    return f"plan_{uuid.uuid4().hex[:12]}"


def generate_snapshot_id() -> str:
    """Generate unique snapshot ID."""
    timestamp = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"mlt_{timestamp}"