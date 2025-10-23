"""
Adaptation Planner - Converts insights into concrete adaptation plans with specific actions.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .contracts import (
    Insight,
    AdaptationPlan,
    Action,
    ActionType,
    InsightType,
    RecommendationType,
    generate_plan_id,
)


logger = logging.getLogger(__name__)


class AdaptationPlanner:
    """Converts insights into concrete, executable adaptation plans."""

    def __init__(self):
        self.plans: List[AdaptationPlan] = []
        self.recommendation_handlers = {
            RecommendationType.RETRAIN: self._plan_retrain,
            RecommendationType.REWEIGHT: self._plan_reweight,
            RecommendationType.RECALIBRATE: self._plan_recalibrate,
            RecommendationType.HPO: self._plan_hpo,
            RecommendationType.POLICY_TUNE: self._plan_policy_tune,
            RecommendationType.SEGMENT_ROUTE: self._plan_segment_route,
        }

    async def create_adaptation_plan(
        self, insights: List[Insight]
    ) -> Optional[AdaptationPlan]:
        """Create an adaptation plan from a collection of insights."""
        if not insights:
            return None

        try:
            # Sort insights by confidence
            sorted_insights = sorted(insights, key=lambda x: x.confidence, reverse=True)

            # Generate actions for each insight
            actions = []
            expected_effects = {}
            risk_controls = {"max_regret_pct": 2, "halt_on_drift_z": 3}

            for insight in sorted_insights:
                handler = self.recommendation_handlers.get(insight.recommendation)
                if handler:
                    plan_actions, effects, controls = await handler(insight)
                    actions.extend(plan_actions)
                    expected_effects.update(effects)
                    risk_controls.update(controls)

            if not actions:
                return None

            plan = AdaptationPlan(
                plan_id=generate_plan_id(),
                actions=actions,
                expected_effect=expected_effects,
                risk_controls=risk_controls,
                timestamp=datetime.now(),
            )

            self.plans.append(plan)
            logger.info(
                f"Created adaptation plan {plan.plan_id} with {len(actions)} actions"
            )
            return plan

        except Exception as e:
            logger.error(f"Failed to create adaptation plan: {e}")
            return None

    async def _plan_retrain(
        self, insight: Insight
    ) -> tuple[List[Action], Dict[str, str], Dict[str, Any]]:
        """Plan retraining actions."""
        actions = []

        if insight.type == InsightType.PERFORMANCE:
            # Full retrain for performance issues
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="tabular.classification",
                    budget={"trials": 50},
                    success_metric="f1",
                )
            )

            expected_effects = {"f1": "+2.5%", "auroc": "+1.8%"}

        elif insight.type == InsightType.DRIFT:
            # More focused retrain for drift
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="drift_adaptation",
                    budget={"trials": 30},
                    success_metric="psi_reduction",
                )
            )

            expected_effects = {"drift_psi": "-0.15", "calibration": "+0.03"}

        else:
            # Generic retrain
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="general",
                    budget={"trials": 25},
                    success_metric="f1",
                )
            )

            expected_effects = {"f1": "+1.5%"}

        risk_controls = {"max_regret_pct": 3, "halt_on_drift_z": 2.5}

        return actions, expected_effects, risk_controls

    async def _plan_reweight(
        self, insight: Insight
    ) -> tuple[List[Action], Dict[str, str], Dict[str, Any]]:
        """Plan specialist reweighting actions."""
        actions = []
        weights = {}

        if insight.type == InsightType.FAIRNESS:
            # Increase fairness specialist weight
            weights["Fairness"] = 1.2
            weights["Bias_Detector"] = 1.15
            expected_effects = {"fairness_delta": "-0.03"}

        elif insight.type == InsightType.STABILITY:
            # Increase anomaly detection weight
            weights["Anomaly"] = 1.1
            weights["Uncertainty"] = 1.05
            expected_effects = {"stability": "+0.1"}

        else:
            # Generic reweighting
            weights["Performance"] = 1.1
            expected_effects = {"overall_performance": "+1.0%"}

        if weights:
            actions.append(
                Action(type=ActionType.REWEIGHT_SPECIALISTS, weights=weights)
            )

        risk_controls = {"max_regret_pct": 1.5}

        return actions, expected_effects, risk_controls

    async def _plan_recalibrate(
        self, insight: Insight
    ) -> tuple[List[Action], Dict[str, str], Dict[str, Any]]:
        """Plan recalibration actions."""
        actions = []

        if insight.type == InsightType.CALIBRATION:
            # Direct calibration improvement
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="calibration_only",
                    budget={"trials": 20},
                    success_metric="calibration_error",
                )
            )

            expected_effects = {"calibration": "+0.05", "uncertainty": "+0.02"}

        elif insight.type == InsightType.DRIFT:
            # Calibration after drift
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="drift_calibration",
                    budget={"trials": 15},
                    success_metric="calibration",
                )
            )

            expected_effects = {"calibration": "+0.03", "drift_psi": "-0.05"}

        else:
            # Generic recalibration
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="general_calibration",
                    budget={"trials": 10},
                    success_metric="calibration",
                )
            )

            expected_effects = {"calibration": "+0.02"}

        risk_controls = {"max_regret_pct": 1}

        return actions, expected_effects, risk_controls

    async def _plan_hpo(
        self, insight: Insight
    ) -> tuple[List[Action], Dict[str, str], Dict[str, Any]]:
        """Plan hyperparameter optimization actions."""
        actions = []

        if insight.type == InsightType.PERFORMANCE:
            # Performance-focused HPO
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="tabular.classification",
                    budget={"trials": 40},
                    success_metric="f1",
                )
            )

            expected_effects = {"f1": "+2.0%", "precision": "+1.5%"}

        elif insight.type == InsightType.STABILITY:
            # Stability-focused HPO
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="stability_optimization",
                    budget={"trials": 25},
                    success_metric="variance_reduction",
                )
            )

            expected_effects = {"latency_variance": "-30%", "error_rate": "-0.02"}

        else:
            # General HPO
            actions.append(
                Action(
                    type=ActionType.HPO,
                    target="general",
                    budget={"trials": 30},
                    success_metric="f1",
                )
            )

            expected_effects = {"f1": "+1.2%"}

        risk_controls = {"max_regret_pct": 2.5}

        return actions, expected_effects, risk_controls

    async def _plan_policy_tune(
        self, insight: Insight
    ) -> tuple[List[Action], Dict[str, str], Dict[str, Any]]:
        """Plan governance policy tuning actions."""
        actions = []

        if insight.type == InsightType.GOVERNANCE_ALIGNMENT:
            # Adjust governance thresholds
            evidence = insight.evidence.get("after", {})
            current_compliance = evidence.get("avg_compliance", 0.9)

            if current_compliance < 0.85:
                # Lower confidence threshold to improve compliance
                actions.append(
                    Action(
                        type=ActionType.POLICY_DELTA,
                        path="governance.thresholds.min_confidence",
                        from_value=0.8,
                        to_value=0.75,
                    )
                )

            # Also adjust calibration threshold
            actions.append(
                Action(
                    type=ActionType.POLICY_DELTA,
                    path="governance.thresholds.min_calibration",
                    from_value=0.95,
                    to_value=0.97,
                )
            )

            expected_effects = {
                "governance_compliance": "+0.05",
                "approval_rate": "+0.1",
            }

        elif insight.type == InsightType.FAIRNESS:
            # Adjust fairness policies
            actions.append(
                Action(
                    type=ActionType.POLICY_DELTA,
                    path="governance.fairness.max_delta",
                    from_value=0.05,
                    to_value=0.03,
                )
            )

            expected_effects = {"fairness_compliance": "+0.08"}

        else:
            # Generic policy adjustment
            actions.append(
                Action(
                    type=ActionType.POLICY_DELTA,
                    path="governance.thresholds.min_confidence",
                    from_value=0.8,
                    to_value=0.85,
                )
            )

            expected_effects = {"policy_alignment": "+0.05"}

        risk_controls = {"max_regret_pct": 1}

        return actions, expected_effects, risk_controls

    async def _plan_segment_route(
        self, insight: Insight
    ) -> tuple[List[Action], Dict[str, str], Dict[str, Any]]:
        """Plan segment-based routing actions."""
        actions = []

        # Implement canary deployment for segment routing
        actions.append(
            Action(
                type=ActionType.CANARY,
                target_model="segment_router@1.1.0",
                steps=[5, 15, 35, 70, 100],
            )
        )

        expected_effects = {"segment_accuracy": "+1.5%", "routing_efficiency": "+0.1"}
        risk_controls = {"max_regret_pct": 2}

        return actions, expected_effects, risk_controls

    def get_recent_plans(self, limit: int = 20) -> List[AdaptationPlan]:
        """Get recent adaptation plans."""
        return sorted(self.plans, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get adaptation planner statistics."""
        action_types = {}
        for action_type in ActionType:
            count = sum(
                1
                for plan in self.plans
                for action in plan.actions
                if action.type == action_type
            )
            action_types[action_type.value] = count

        return {
            "total_plans": len(self.plans),
            "total_actions": sum(len(plan.actions) for plan in self.plans),
            "action_types": action_types,
            "avg_actions_per_plan": sum(len(plan.actions) for plan in self.plans)
            / len(self.plans)
            if self.plans
            else 0,
        }
