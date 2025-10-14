"""MLT bridge for Learning Kernel integration with MLT (Meta-Learning & Tuning) kernel."""

from datetime import datetime
from typing import Dict, List, Optional, Any


class MLTBridge:
    """Bridge for sending learning experiences to MLT and receiving adaptation plans."""

    def __init__(self):
        self.sent_experiences: List[Dict[str, Any]] = []
        self.received_plans: List[Dict[str, Any]] = []

    async def send_learning_experience(
        self,
        stage: str,
        metrics: Dict[str, Any],
        segment: Optional[str] = None,
        dataset_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Send a learning experience to MLT kernel."""
        exp_id = f"learn_exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        experience = {
            "exp_id": exp_id,
            "stage": stage,
            "metrics": metrics,
            "segment": segment,
            "dataset_id": dataset_id,
            "version": version,
            "timestamp": datetime.now().isoformat(),
        }

        # Validate stage
        valid_stages = [
            "labeling",
            "active_query",
            "weak_supervision",
            "augmentation",
            "curriculum",
            "version_publish",
            "eval",
        ]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of: {valid_stages}")

        # Store for tracking
        self.sent_experiences.append(experience)

        # In practice, would send to actual MLT kernel via event mesh or API
        print(f"[LEARNING->MLT] Sent experience: {stage} - {exp_id}")

        return exp_id

    async def send_labeling_experience(
        self,
        label_throughput: float,
        agreement: float,
        gold_accuracy: Optional[float] = None,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Send labeling stage experience to MLT."""
        metrics = {
            "label_throughput_items_per_hr": label_throughput,
            "label_agreement": agreement,
        }

        if gold_accuracy is not None:
            metrics["gold_accuracy"] = gold_accuracy

        return await self.send_learning_experience(
            stage="labeling", metrics=metrics, dataset_id=dataset_id
        )

    async def send_active_query_experience(
        self,
        query_gain_f1: float,
        strategy: str,
        batch_size: int,
        coverage: float,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Send active learning query experience to MLT."""
        metrics = {"query_gain_f1": query_gain_f1, "coverage": coverage}

        segment = f"strategy_{strategy}_batch_{batch_size}"

        return await self.send_learning_experience(
            stage="active_query",
            metrics=metrics,
            segment=segment,
            dataset_id=dataset_id,
        )

    async def send_weak_supervision_experience(
        self,
        weak_precision: float,
        weak_recall: Optional[float] = None,
        labeler_count: int = 1,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Send weak supervision experience to MLT."""
        metrics = {"weak_precision": weak_precision, "labeler_count": labeler_count}

        if weak_recall is not None:
            metrics["weak_recall"] = weak_recall

        return await self.send_learning_experience(
            stage="weak_supervision", metrics=metrics, dataset_id=dataset_id
        )

    async def send_augmentation_experience(
        self,
        delta_rows: int,
        augment_type: str,
        quality_impact: Optional[float] = None,
        dataset_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Send data augmentation experience to MLT."""
        metrics = {"delta_rows": delta_rows, "augmentation_type": augment_type}

        if quality_impact is not None:
            metrics["quality_impact"] = quality_impact

        return await self.send_learning_experience(
            stage="augmentation",
            metrics=metrics,
            dataset_id=dataset_id,
            version=version,
        )

    async def send_evaluation_experience(
        self,
        bias_delta: float,
        leakage_flags: int,
        ds_drift_psi: float,
        coverage: float,
        dataset_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Send evaluation/quality assessment experience to MLT."""
        metrics = {
            "bias_delta": bias_delta,
            "leakage_flags": leakage_flags,
            "ds_drift_psi": ds_drift_psi,
            "coverage": coverage,
        }

        return await self.send_learning_experience(
            stage="eval", metrics=metrics, dataset_id=dataset_id, version=version
        )

    async def receive_adaptation_plan(self, plan: Dict[str, Any]) -> bool:
        """Receive and process adaptation plan from MLT kernel."""
        # Store received plan
        self.received_plans.append(
            {"plan": plan, "received_at": datetime.now().isoformat()}
        )

        # Extract learning-specific actions
        learning_actions = []
        for action in plan.get("actions", []):
            if self._is_learning_action(action):
                learning_actions.append(action)

        if learning_actions:
            print(
                f"[MLT->LEARNING] Received plan with {len(learning_actions)} learning actions"
            )

            # Process each learning action
            for action in learning_actions:
                await self._process_learning_action(action)

            return True

        return False

    def _is_learning_action(self, action: Dict[str, Any]) -> bool:
        """Check if an action is relevant to the learning kernel."""
        action_type = action.get("type", "")
        target = action.get("target", "")
        path = action.get("path", "")

        # Check for learning-specific actions
        learning_indicators = [
            "learning." in path,
            "active." in target,
            "weak." in target,
            "labeling." in path,
            "augmentation." in path,
            "curriculum" in target,
        ]

        return any(learning_indicators)

    async def _process_learning_action(self, action: Dict[str, Any]):
        """Process a specific learning action from MLT."""
        action_type = action.get("type")

        if action_type == "policy_delta":
            await self._apply_policy_change(action)
        elif action_type == "hpo":
            await self._start_hyperparameter_optimization(action)
        elif action_type == "canary":
            await self._start_canary_deployment(action)
        else:
            print(f"[LEARNING] Unknown action type: {action_type}")

    async def _apply_policy_change(self, action: Dict[str, Any]):
        """Apply policy change from adaptation plan."""
        path = action.get("path", "")
        from_value = action.get("from")
        to_value = action.get("to")

        print(f"[LEARNING] Applying policy change: {path} {from_value} -> {to_value}")

        # In practice, would update actual configuration
        # For now, just log the change

    async def _start_hyperparameter_optimization(self, action: Dict[str, Any]):
        """Start hyperparameter optimization based on MLT recommendation."""
        target = action.get("target", "")
        budget = action.get("budget", {})
        success_metric = action.get("success_metric", "")

        print(f"[LEARNING] Starting HPO for {target} with metric {success_metric}")
        print(f"[LEARNING] HPO budget: {budget}")

        # In practice, would trigger actual HPO process

    async def _start_canary_deployment(self, action: Dict[str, Any]):
        """Start canary deployment for a learning component."""
        target_model = action.get("target_model", "")
        steps = action.get("steps", [])

        print(f"[LEARNING] Starting canary for {target_model} with steps: {steps}")

        # In practice, would manage canary rollout

    def get_experience_stats(self) -> Dict[str, Any]:
        """Get statistics about sent experiences."""
        if not self.sent_experiences:
            return {"total_experiences": 0}

        # Count by stage
        stage_counts = {}
        for exp in self.sent_experiences:
            stage = exp["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        # Recent experiences
        recent_count = len(
            [
                exp
                for exp in self.sent_experiences
                if (
                    datetime.now() - datetime.fromisoformat(exp["timestamp"])
                ).total_seconds()
                < 3600
            ]
        )

        return {
            "total_experiences": len(self.sent_experiences),
            "by_stage": stage_counts,
            "recent_hour_count": recent_count,
            "plans_received": len(self.received_plans),
        }

    def clear_history(self):
        """Clear experience and plan history (for testing)."""
        self.sent_experiences.clear()
        self.received_plans.clear()
