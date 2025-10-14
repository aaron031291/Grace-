"""Bridge to MLT kernel for adaptation plan transparency."""

from datetime import datetime
from grace.utils.time import now_utc, iso_now_utc
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MLTBridge:
    """Bridges Interface to MLT kernel for plan transparency and approvals."""

    def __init__(self, mlt_kernel=None):
        self.mlt_kernel = mlt_kernel
        self.plan_history: List[Dict] = []

    async def get_adaptation_plans(
        self, limit: int = 10, status_filter: Optional[str] = None
    ) -> List[Dict]:
        """Get recent adaptation plans with structured summaries."""
        plans = []

        try:
            if self.mlt_kernel and hasattr(self.mlt_kernel, "adaptation_planner"):
                raw_plans = self.mlt_kernel.adaptation_planner.get_recent_plans(limit)

                for plan in raw_plans:
                    structured_plan = await self._structure_plan_summary(plan)

                    if status_filter and structured_plan.get("status") != status_filter:
                        continue

                    plans.append(structured_plan)

        except Exception as e:
            logger.error(f"Failed to get adaptation plans: {e}")

        return plans

    async def _structure_plan_summary(self, plan: Any) -> Dict:
        """Create structured summary of adaptation plan."""
        summary = {
            "plan_id": getattr(plan, "plan_id", "unknown"),
            "created_at": getattr(plan, "created_at", iso_now_utc()).isoformat() if getattr(plan, "created_at", None) else iso_now_utc(),
            "status": getattr(plan, "status", "unknown"),
            "priority": getattr(plan, "priority", 5),
            "summary": {
                "title": self._generate_plan_title(plan),
                "description": self._generate_plan_description(plan),
                "impact_assessment": self._assess_plan_impact(plan),
                "risk_level": self._assess_plan_risk(plan),
            },
            "actions": [],
            "governance_status": "pending",
        }

        # Extract action summaries
        if hasattr(plan, "actions"):
            for action in plan.actions:
                action_summary = {
                    "type": getattr(action, "type", "unknown"),
                    "target": getattr(action, "target", "unknown"),
                    "description": self._describe_action(action),
                }
                summary["actions"].append(action_summary)

        return summary

    def _generate_plan_title(self, plan: Any) -> str:
        """Generate human-readable title for plan."""
        if hasattr(plan, "actions") and plan.actions:
            action_types = set(
                getattr(action, "type", "unknown") for action in plan.actions
            )
            action_list = ", ".join(action_types)
            return f"Adaptation Plan: {action_list}"

        return "Adaptation Plan"

    def _generate_plan_description(self, plan: Any) -> str:
        """Generate human-readable description."""
        description_parts = []

        if hasattr(plan, "trigger"):
            description_parts.append(f"Triggered by: {plan.trigger}")

        if hasattr(plan, "objectives"):
            description_parts.append(f"Objectives: {', '.join(plan.objectives)}")

        return (
            "; ".join(description_parts)
            if description_parts
            else "Automated system adaptation"
        )

    def _assess_plan_impact(self, plan: Any) -> str:
        """Assess the impact level of the plan."""
        if hasattr(plan, "actions"):
            action_count = len(plan.actions)

            if action_count > 5:
                return "high"
            elif action_count > 2:
                return "medium"
            else:
                return "low"

        return "unknown"

    def _assess_plan_risk(self, plan: Any) -> str:
        """Assess risk level of the plan."""
        if hasattr(plan, "risk_controls"):
            max_regret = plan.risk_controls.get("max_regret_pct", 2)

            if max_regret > 5:
                return "high"
            elif max_regret > 2:
                return "medium"
            else:
                return "low"

        return "medium"  # Default to medium risk

    def _describe_action(self, action: Any) -> str:
        """Generate human-readable action description."""
        action_type = getattr(action, "type", "unknown")
        target = getattr(action, "target", "system")

        descriptions = {
            "HPO": f"Optimize hyperparameters for {target}",
            "REWEIGHT_SPECIALISTS": f"Reweight specialist contributions in {target}",
            "POLICY_DELTA": f"Update policy configuration for {target}",
            "CANARY": f"Deploy canary version of {target}",
        }

        return descriptions.get(action_type, f"Execute {action_type} on {target}")

    async def send_to_governance(self, plan_id: str, user_id: str) -> str:
        """Send adaptation plan to governance for approval."""
        try:
            plan_summary = await self._get_plan_by_id(plan_id)

            if not plan_summary:
                raise ValueError(f"Plan {plan_id} not found")

            # Create governance request
            governance_request = {
                "type": "mlt_adaptation_plan",
                "user_id": user_id,
                "action": "approve_adaptation_plan",
                "resource": f"mlt:plan:{plan_id}",
                "context": {
                    "plan_summary": plan_summary,
                    "impact_level": plan_summary.get("summary", {}).get(
                        "impact_assessment"
                    ),
                    "risk_level": plan_summary.get("summary", {}).get("risk_level"),
                },
                "priority": plan_summary.get("priority", 5),
            }

            # Would integrate with governance bridge here
            logger.info(f"Sent MLT plan {plan_id} to governance for approval")

            return f"governance_request_{plan_id}"

        except Exception as e:
            logger.error(f"Failed to send plan to governance: {e}")
            raise

    async def _get_plan_by_id(self, plan_id: str) -> Optional[Dict]:
        """Get plan by ID from MLT kernel."""
        if not self.mlt_kernel:
            return None

        try:
            if hasattr(self.mlt_kernel, "adaptation_planner"):
                plans = self.mlt_kernel.adaptation_planner.get_recent_plans(100)

                for plan in plans:
                    if getattr(plan, "plan_id", None) == plan_id:
                        return await self._structure_plan_summary(plan)

        except Exception as e:
            logger.error(f"Error retrieving plan {plan_id}: {e}")

        return None

    async def get_experience_data(self, limit: int = 20) -> List[Dict]:
        """Get recent experience data for transparency."""
        experiences = []

        try:
            if self.mlt_kernel and hasattr(self.mlt_kernel, "experience_collector"):
                raw_experiences = (
                    self.mlt_kernel.experience_collector.get_recent_experiences(limit)
                )

                for exp in raw_experiences:
                    experience_summary = {
                        "exp_id": getattr(exp, "experience_id", "unknown"),
                        "timestamp": getattr(exp, "timestamp", now_utc()).isoformat(),
                        "category": getattr(exp, "category", "unknown"),
                        "metrics": getattr(exp, "metrics", {}),
                        "insights_generated": getattr(exp, "insights_count", 0),
                    }
                    experiences.append(experience_summary)

        except Exception as e:
            logger.error(f"Failed to get experience data: {e}")

        return experiences

    async def get_insights(self, limit: int = 10) -> List[Dict]:
        """Get recent insights for transparency."""
        insights = []

        try:
            if self.mlt_kernel and hasattr(self.mlt_kernel, "insight_generator"):
                raw_insights = self.mlt_kernel.insight_generator.get_recent_insights(
                    limit
                )

                for insight in raw_insights:
                    insight_summary = {
                        "insight_id": getattr(insight, "insight_id", "unknown"),
                        "timestamp": getattr(insight, "timestamp", now_utc()).isoformat(),
                        "type": getattr(insight, "insight_type", "unknown"),
                        "confidence": getattr(insight, "confidence", 0.0),
                        "summary": getattr(insight, "summary", "No summary available"),
                    }
                    insights.append(insight_summary)

        except Exception as e:
            logger.error(f"Failed to get insights: {e}")

        return insights

    def set_mlt_kernel(self, mlt_kernel):
        """Set MLT kernel reference."""
        self.mlt_kernel = mlt_kernel
        logger.info("MLT kernel connected to MLT bridge")

    def get_stats(self) -> Dict:
        """Get bridge statistics."""
        return {
            "mlt_connected": bool(self.mlt_kernel),
            "plan_requests": len(self.plan_history),
        }
