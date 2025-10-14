"""Error budget tracking and management."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ErrorBudgetTracker:
    """
    Track and manage error budgets for services.

    Monitors SLO breaches and calculates remaining error budget
    to make informed decisions about deployments and risk.
    """

    def __init__(self):
        """Initialize error budget tracker."""
        self.policies: Dict[str, Dict] = {}
        self.budget_consumption: Dict[str, List[Dict]] = {}
        self.alerts_sent: Dict[str, datetime] = {}

        logger.debug("Error budget tracker initialized")

    def set_policy(self, service_id: str, slo_policy: Dict):
        """Set SLO policy for budget tracking."""
        self.policies[service_id] = slo_policy
        if service_id not in self.budget_consumption:
            self.budget_consumption[service_id] = []

        logger.info(f"Set error budget policy for service {service_id}")

    def record_breach(
        self, service_id: str, sli_type: str, breach_amount: float, duration_s: int = 60
    ):
        """
        Record an SLO breach and consume error budget.

        Args:
            service_id: Service identifier
            sli_type: Type of SLI that was breached
            breach_amount: Amount by which SLO was breached
            duration_s: Duration of the breach in seconds
        """
        if service_id not in self.policies:
            logger.warning(f"No error budget policy for service {service_id}")
            return

        policy = self.policies[service_id]
        error_budget_days = policy.get(
            "error_budget_days", 0.1
        )  # Default 2.4 hours per month

        # Calculate budget consumption based on breach severity and duration
        # More severe breaches consume more budget
        consumption_rate = self._calculate_consumption_rate(sli_type, breach_amount)
        budget_consumed = (
            duration_s / 86400
        ) * consumption_rate  # Convert to fraction of daily budget

        # Record the consumption
        consumption_record = {
            "timestamp": datetime.now().isoformat(),
            "sli_type": sli_type,
            "breach_amount": breach_amount,
            "duration_s": duration_s,
            "budget_consumed": budget_consumed,
        }

        self.budget_consumption[service_id].append(consumption_record)

        # Trim old records (keep last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.budget_consumption[service_id] = [
            record
            for record in self.budget_consumption[service_id]
            if datetime.fromisoformat(record["timestamp"]) >= cutoff
        ]

        logger.info(
            f"Recorded error budget consumption for {service_id}: {budget_consumed:.4f} days"
        )

    def get_remaining_budget(self, service_id: str, window_days: int = 30) -> Dict:
        """
        Get remaining error budget for a service.

        Args:
            service_id: Service identifier
            window_days: Budget calculation window in days

        Returns:
            Budget status dictionary
        """
        if service_id not in self.policies:
            return {
                "service_id": service_id,
                "error": "No policy configured",
                "remaining_pct": 0.0,
            }

        policy = self.policies[service_id]
        budget_per_period = policy.get("error_budget_days", 0.1)

        # Calculate total budget for the window
        total_budget = budget_per_period * (window_days / 30)  # Normalize to window

        # Calculate consumed budget in the window
        cutoff = datetime.now() - timedelta(days=window_days)
        recent_consumption = [
            record
            for record in self.budget_consumption.get(service_id, [])
            if datetime.fromisoformat(record["timestamp"]) >= cutoff
        ]

        consumed_budget = sum(
            record["budget_consumed"] for record in recent_consumption
        )
        remaining_budget = max(0, total_budget - consumed_budget)
        remaining_pct = (
            (remaining_budget / total_budget * 100) if total_budget > 0 else 100.0
        )

        # Determine status
        if remaining_pct > 50:
            status = "healthy"
        elif remaining_pct > 20:
            status = "warning"
        elif remaining_pct > 0:
            status = "critical"
        else:
            status = "exhausted"

        return {
            "service_id": service_id,
            "status": status,
            "total_budget_days": total_budget,
            "consumed_budget_days": consumed_budget,
            "remaining_budget_days": remaining_budget,
            "remaining_pct": remaining_pct,
            "window_days": window_days,
            "breach_count": len(recent_consumption),
            "calculated_at": datetime.now().isoformat(),
        }

    def should_block_deployment(
        self, service_id: str, risk_threshold_pct: float = 10.0
    ) -> Dict:
        """
        Determine if deployment should be blocked based on error budget.

        Args:
            service_id: Service identifier
            risk_threshold_pct: Minimum remaining budget percentage to allow deployment

        Returns:
            Deployment decision
        """
        budget_status = self.get_remaining_budget(service_id)
        remaining_pct = budget_status.get("remaining_pct", 0.0)

        should_block = remaining_pct < risk_threshold_pct

        return {
            "service_id": service_id,
            "should_block": should_block,
            "reason": f"Error budget at {remaining_pct:.1f}%, threshold is {risk_threshold_pct}%",
            "remaining_budget_pct": remaining_pct,
            "risk_threshold_pct": risk_threshold_pct,
            "recommendation": self._get_deployment_recommendation(
                remaining_pct, risk_threshold_pct
            ),
        }

    def get_budget_burn_rate(self, service_id: str, hours: int = 24) -> Dict:
        """
        Calculate error budget burn rate.

        Args:
            service_id: Service identifier
            hours: Time window for burn rate calculation

        Returns:
            Burn rate analysis
        """
        if service_id not in self.budget_consumption:
            return {
                "service_id": service_id,
                "burn_rate_per_hour": 0.0,
                "projected_exhaustion_hours": float("inf"),
            }

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_consumption = [
            record
            for record in self.budget_consumption[service_id]
            if datetime.fromisoformat(record["timestamp"]) >= cutoff
        ]

        if not recent_consumption:
            return {
                "service_id": service_id,
                "burn_rate_per_hour": 0.0,
                "projected_exhaustion_hours": float("inf"),
            }

        total_consumed = sum(record["budget_consumed"] for record in recent_consumption)
        burn_rate_per_hour = total_consumed / hours * 24  # Convert to daily rate

        # Get current remaining budget
        budget_status = self.get_remaining_budget(service_id)
        remaining_days = budget_status.get("remaining_budget_days", 0.0)

        # Project when budget will be exhausted
        if burn_rate_per_hour > 0:
            projected_exhaustion_hours = (remaining_days / burn_rate_per_hour) * 24
        else:
            projected_exhaustion_hours = float("inf")

        return {
            "service_id": service_id,
            "burn_rate_per_hour": burn_rate_per_hour * 24,  # Daily rate
            "recent_consumption_days": total_consumed,
            "projected_exhaustion_hours": projected_exhaustion_hours,
            "window_hours": hours,
            "calculated_at": datetime.now().isoformat(),
        }

    def get_budget_summary(self, service_ids: Optional[List[str]] = None) -> Dict:
        """Get error budget summary for services."""
        if service_ids is None:
            service_ids = list(self.policies.keys())

        summary = {
            "total_services": len(service_ids),
            "services": {},
            "overall_status": "healthy",
            "generated_at": datetime.now().isoformat(),
        }

        status_counts = {"healthy": 0, "warning": 0, "critical": 0, "exhausted": 0}

        for service_id in service_ids:
            budget_status = self.get_remaining_budget(service_id)
            summary["services"][service_id] = budget_status

            status = budget_status.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1

        # Determine overall status
        if status_counts["exhausted"] > 0:
            summary["overall_status"] = "exhausted"
        elif status_counts["critical"] > 0:
            summary["overall_status"] = "critical"
        elif status_counts["warning"] > 0:
            summary["overall_status"] = "warning"

        summary["status_counts"] = status_counts
        return summary

    def _calculate_consumption_rate(self, sli_type: str, breach_amount: float) -> float:
        """Calculate budget consumption rate based on breach severity."""
        base_rate = 1.0  # Base consumption rate

        # Adjust rate based on SLI type
        if sli_type == "availability_pct":
            # Availability breaches are severe
            base_rate *= 2.0
        elif sli_type == "error_rate_pct":
            # Error rate breaches consume budget faster
            base_rate *= 1.5

        # Scale by breach amount (more severe = more consumption)
        severity_multiplier = min(10.0, max(1.0, breach_amount / 100))

        return base_rate * severity_multiplier

    def _get_deployment_recommendation(
        self, remaining_pct: float, threshold_pct: float
    ) -> str:
        """Get deployment recommendation based on budget status."""
        if remaining_pct > threshold_pct * 2:
            return "Safe to deploy - good error budget remaining"
        elif remaining_pct > threshold_pct:
            return "Proceed with caution - monitor closely after deployment"
        elif remaining_pct > 5.0:
            return "High risk deployment - consider emergency procedures"
        else:
            return "Do not deploy - error budget exhausted or critical"
