"""Learning kernel - comprehensive data-centric learning system."""

import os
import asyncio
from typing import Dict, List, Optional, Any
from grace.utils.time import now_utc
from datetime import datetime
from pydantic import BaseModel

# Import new learning kernel components
from .learning_service import LearningService
from .registry.datasets import DatasetRegistry
from .labeling.hitl.queue import HITLQueue
from .labeling.weak.rules import WeakSupervision
from .labeling.policies.service import PolicyService
from .active.strategies import ActiveLearningStrategies
from .semi_supervised.self_train import SemiSupervisedLearning
from .augmentation.pipelines import AugmentationPipelines
from .evaluation.quality import QualityEvaluator
from .feature_store.views import FeatureStore
from .snapshots.manager import SnapshotManager
from .bridges.mesh_bridge import MeshBridge
from .bridges.mlt_bridge import MLTBridge
from .bridges.gov_bridge import GovernanceBridge
from .bridges.ingress_bridge import IngressBridge
from .bridges.memory_bridge import MemoryBridge


class LearningOutcome(BaseModel):
    """Legacy learning outcome record - maintained for backward compatibility."""

    outcome_type: str
    context: Dict
    result: Dict
    success: bool
    confidence: float
    timestamp: datetime


class LearningKernel:
    """
    Comprehensive data-centric learning kernel.

    Manages the complete data→label→dataset→curriculum lifecycle including:
    - Dataset registration and versioning with lineage
    - Human-in-the-loop and weak supervision labeling
    - Active learning query strategies
    - Data augmentation and curriculum learning
    - Quality evaluation and bias/fairness monitoring
    - Feature store and train/serve parity
    - Snapshot management and rollback capabilities
    """

    def __init__(
        self, db_path: Optional[str] = None, event_bus=None, governance_engine=None
    ):
        # Database setup
        if db_path is None:
            db_path = os.environ.get("LEARNING_DB_PATH", "/tmp/learning.db")
        self.db_path = db_path

        # External integrations
        self.event_bus = event_bus
        self.governance_engine = governance_engine

        # Initialize core components
        self.dataset_registry = DatasetRegistry(self.db_path)
        self.hitl_queue = HITLQueue(self.db_path)
        self.weak_supervision = WeakSupervision(self.db_path)
        self.policy_service = PolicyService(self.db_path)
        self.active_learning = ActiveLearningStrategies(self.db_path)
        self.semi_supervised = SemiSupervisedLearning(self.db_path)
        self.augmentation = AugmentationPipelines(self.db_path)
        self.quality_evaluator = QualityEvaluator(self.db_path)
        self.feature_store = FeatureStore(self.db_path)
        self.snapshot_manager = SnapshotManager(self.db_path)

        # Initialize bridges
        self.mesh_bridge = MeshBridge()
        self.mlt_bridge = MLTBridge()
        self.gov_bridge = GovernanceBridge()
        self.ingress_bridge = IngressBridge()
        self.memory_bridge = MemoryBridge()

        # Initialize service layer
        self.learning_service = LearningService(self.db_path)

        # Legacy outcome tracking for backward compatibility
        self.outcomes = []
        self.adaptation_rules = {}

        # Running state
        self.running = False

    # === Core Learning Kernel Methods ===

    def register_dataset(self, manifest: Dict[str, Any]) -> str:
        """Register a new dataset."""
        return self.dataset_registry.register(manifest)

    def publish_version(
        self, dataset_id: str, version: str, refs: List[str]
    ) -> Dict[str, Any]:
        """Publish a new dataset version."""
        return self.dataset_registry.publish_version(dataset_id, version, refs)

    def create_label_task(self, task: Dict[str, Any]) -> str:
        """Create a new labeling task."""
        return self.hitl_queue.create_task(task)

    def submit_label(self, label: Dict[str, Any]) -> str:
        """Submit a label."""
        return self.hitl_queue.submit_label(label)

    def query_active_batch(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Query active learning batch."""
        return self.active_learning.select_batch(cfg)

    def apply_augmentation(
        self, dataset_id: str, version: str, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply data augmentation."""
        return self.augmentation.apply_spec(dataset_id, version, spec)

    def build_feature_view(self, dataset_id: str, version: str) -> Dict[str, Any]:
        """Build feature view for train/serve parity."""
        return self.feature_store.build_view(dataset_id, version)

    def generate_quality_report(
        self, dataset_id: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        return self.quality_evaluator.generate_report(dataset_id, version)

    def create_snapshot(self, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a learning kernel snapshot."""
        return self.snapshot_manager.create_snapshot(description)

    def rollback_to_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Rollback to a specific snapshot."""
        return self.snapshot_manager.rollback(snapshot_id)

    # === Experience Generation for MLT Integration ===

    async def send_labeling_experience(
        self,
        label_throughput: float,
        agreement: float,
        gold_accuracy: Optional[float] = None,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Send labeling experience to MLT kernel."""
        return await self.mlt_bridge.send_labeling_experience(
            label_throughput, agreement, gold_accuracy, dataset_id
        )

    async def send_active_learning_experience(
        self,
        query_gain_f1: float,
        strategy: str,
        batch_size: int,
        coverage: float,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Send active learning experience to MLT kernel."""
        return await self.mlt_bridge.send_active_query_experience(
            query_gain_f1, strategy, batch_size, coverage, dataset_id
        )

    async def send_quality_experience(
        self,
        bias_delta: float,
        leakage_flags: int,
        ds_drift_psi: float,
        coverage: float,
        dataset_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Send quality evaluation experience to MLT kernel."""
        return await self.mlt_bridge.send_evaluation_experience(
            bias_delta, leakage_flags, ds_drift_psi, coverage, dataset_id, version
        )

    # === Event Handling ===

    async def handle_rollback_request(self, event_data: Dict[str, Any]):
        """Handle rollback requests from governance."""
        await self.mesh_bridge.handle_rollback_request(event_data)

    async def handle_mlt_adaptation_plan(self, event_data: Dict[str, Any]):
        """Handle adaptation plans from MLT kernel."""
        await self.mesh_bridge.handle_mlt_adaptation_plan(event_data)

    # === Governance Integration ===

    async def submit_policy_proposal(self, policy_change: Dict[str, Any]) -> str:
        """Submit policy change proposal to governance."""
        return await self.gov_bridge.submit_policy_proposal(policy_change)

    async def submit_dataset_publication(
        self, dataset_id: str, version: str, governance_label: str = "internal"
    ) -> str:
        """Submit dataset publication request to governance."""
        return await self.gov_bridge.submit_dataset_publication_request(
            dataset_id, version, governance_label
        )

    # === Service Layer Access ===

    def get_service(self) -> LearningService:
        """Get the FastAPI service for external access."""
        return self.learning_service

    def get_app(self):
        """Get the FastAPI application for external access."""
        return self.learning_service.get_app()

    # === Statistics and Monitoring ===

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning kernel statistics."""
        # Get component stats
        dataset_stats = {
            "datasets": len(self.dataset_registry.list_datasets()),
            "total_versions": len(self.dataset_registry.list_datasets()),  # Simplified
        }

        # Get experience stats
        experience_stats = self.mlt_bridge.get_experience_stats()

        # Get governance stats
        governance_stats = self.gov_bridge.get_governance_stats()

        # Legacy compatibility
        legacy_stats = {
            "total_outcomes": len(self.outcomes),
            "success_rate": sum(1 for o in self.outcomes if o.success)
            / max(1, len(self.outcomes)),
            "adaptation_rules": len(self.adaptation_rules),
        }

        return {
            "kernel_version": "2.0.0",
            "kernel_type": "data_centric_learning",
            "running": self.running,
            "components": {
                "datasets": dataset_stats,
                "experiences": experience_stats,
                "governance": governance_stats,
            },
            "legacy": legacy_stats,
            "capabilities": [
                "dataset_management",
                "active_learning",
                "weak_supervision",
                "human_in_the_loop",
                "data_augmentation",
                "quality_evaluation",
                "feature_store",
                "snapshot_rollback",
                "mlt_integration",
                "governance_integration",
            ],
        }

    # === Legacy Compatibility Methods ===

    def record_outcome(
        self,
        outcome_type: str,
        context: Dict,
        result: Dict,
        success: bool,
        confidence: float = 1.0,
    ):
        """Record a learning outcome (legacy compatibility)."""
        outcome = LearningOutcome(
            outcome_type=outcome_type,
            context=context,
            result=result,
            success=success,
            confidence=confidence,
            timestamp=now_utc(),
        )

        self.outcomes.append(outcome)
        self._trigger_adaptation(outcome)

        # Convert to modern experience format and send to MLT
        asyncio.create_task(self._convert_legacy_outcome_to_experience(outcome))

    async def _convert_legacy_outcome_to_experience(self, outcome: LearningOutcome):
        """Convert legacy outcome to modern learning experience."""
        try:
            # Map legacy outcome types to modern stages
            stage_mapping = {
                "labeling": "labeling",
                "active_learning": "active_query",
                "data_quality": "eval",
                "augmentation": "augmentation",
            }

            stage = stage_mapping.get(outcome.outcome_type, "eval")

            # Create basic metrics from legacy outcome
            metrics = {
                "success": 1.0 if outcome.success else 0.0,
                "confidence": outcome.confidence,
            }

            await self.mlt_bridge.send_learning_experience(stage, metrics)
        except Exception as e:
            print(f"Failed to convert legacy outcome to experience: {e}")

    def adapt(self) -> Dict:
        """Analyze outcomes and adapt system behavior (legacy compatibility)."""
        if not self.outcomes:
            return {"adaptations": 0, "message": "No outcomes to analyze"}

        recent_outcomes = self.outcomes[-100:]
        adaptations = {}

        # Analyze success rates by outcome type
        success_rates = {}
        for outcome in recent_outcomes:
            outcome_type = outcome.outcome_type
            if outcome_type not in success_rates:
                success_rates[outcome_type] = {"total": 0, "success": 0}

            success_rates[outcome_type]["total"] += 1
            if outcome.success:
                success_rates[outcome_type]["success"] += 1

        # Generate adaptations for low success rates
        for outcome_type, stats in success_rates.items():
            success_rate = stats["success"] / stats["total"]
            if success_rate < 0.7 and stats["total"] >= 5:
                adaptations[outcome_type] = (
                    f"Consider adjustment - success rate: {success_rate:.2%}"
                )

        return {
            "adaptations": len(adaptations),
            "recommendations": adaptations,
            "analyzed_outcomes": len(recent_outcomes),
            "note": "Legacy adapt() method - consider using modern quality_evaluator and mlt_bridge",
        }

    def _trigger_adaptation(self, outcome: LearningOutcome):
        """Trigger adaptation based on outcome (legacy compatibility)."""
        if not outcome.success and outcome.confidence > 0.8:
            outcome_type = outcome.outcome_type
            if outcome_type not in self.adaptation_rules:
                self.adaptation_rules[outcome_type] = {"failure_count": 0}

            self.adaptation_rules[outcome_type]["failure_count"] += 1
