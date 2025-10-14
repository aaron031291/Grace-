"""
Grace Governance Integration Examples
Demonstrates Layer-1 (constitutional) and Layer-2 (organizational) governance in action.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GovernanceLayer(Enum):
    """Governance layers."""

    LAYER_1_CONSTITUTIONAL = "layer_1_constitutional"
    LAYER_2_ORGANIZATIONAL = "layer_2_organizational"


@dataclass
class GovernanceViolation:
    """Governance violation detected."""

    layer: GovernanceLayer
    rule_id: str
    rule_name: str
    description: str
    severity: str
    can_override: bool = False
    suggested_actions: List[str] = None

    def __post_init__(self):
        if self.suggested_actions is None:
            self.suggested_actions = []


@dataclass
class GovernanceDecision:
    """Governance decision result."""

    approved: bool
    violations: List[GovernanceViolation]
    reasoning: str
    required_approvals: List[str] = None
    conditions: List[str] = None

    def __post_init__(self):
        if self.required_approvals is None:
            self.required_approvals = []
        if self.conditions is None:
            self.conditions = []


class GraceGovernanceEngine:
    """
    Grace Governance Engine implementing both constitutional and organizational governance.

    Layer-1: Constitutional rules (immutable, cannot be overridden)
    Layer-2: Organizational policies (configurable, can be overridden with approval)
    """

    def __init__(self):
        self.version = "1.0.0"

        # Layer-1 Constitutional Rules (immutable)
        self.layer_1_rules = {
            "privacy_protection": {
                "name": "Privacy Protection",
                "description": "No exposure of personally identifiable information",
                "severity": "critical",
                "can_override": False,
                "checks": ["pii_detection", "data_anonymization"],
            },
            "harm_prevention": {
                "name": "Harm Prevention",
                "description": "Prevent actions that could cause physical, emotional, or financial harm",
                "severity": "critical",
                "can_override": False,
                "checks": ["harm_assessment", "risk_evaluation"],
            },
            "legal_compliance": {
                "name": "Legal Compliance",
                "description": "Must comply with applicable laws and regulations",
                "severity": "critical",
                "can_override": False,
                "checks": ["regulatory_compliance", "legal_review"],
            },
            "transparency": {
                "name": "Transparency",
                "description": "Actions and decisions must be auditable and explainable",
                "severity": "high",
                "can_override": False,
                "checks": ["audit_trail", "decision_explanation"],
            },
            "fairness": {
                "name": "Fairness",
                "description": "No discrimination or bias in decisions and actions",
                "severity": "high",
                "can_override": False,
                "checks": ["bias_detection", "fairness_evaluation"],
            },
        }

        # Layer-2 Organizational Policies (configurable)
        self.layer_2_policies = {
            "trading_limits": {
                "name": "Trading Risk Limits",
                "description": "Maximum trading amounts and risk exposure",
                "severity": "medium",
                "can_override": True,
                "override_requires": ["risk_manager", "cfo"],
                "parameters": {
                    "max_single_trade": 50000,
                    "max_daily_exposure": 500000,
                    "max_risk_percentage": 0.02,
                },
            },
            "data_access": {
                "name": "Data Access Control",
                "description": "Cross-department data access restrictions",
                "severity": "medium",
                "can_override": True,
                "override_requires": ["data_owner", "privacy_lead"],
                "parameters": {
                    "require_approval": True,
                    "access_duration_days": 30,
                    "audit_frequency": "weekly",
                },
            },
            "marketing_timing": {
                "name": "Marketing Campaign Timing",
                "description": "Restrictions on marketing send times",
                "severity": "low",
                "can_override": True,
                "override_requires": ["marketing_lead"],
                "parameters": {
                    "allowed_hours": {"start": 9, "end": 17},
                    "allowed_days": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                    ],
                    "respect_timezone": True,
                },
            },
            "model_deployment": {
                "name": "Model Deployment Review",
                "description": "All ML models require review before production",
                "severity": "medium",
                "can_override": True,
                "override_requires": ["ml_lead", "security_lead"],
                "parameters": {
                    "require_testing": True,
                    "min_accuracy": 0.85,
                    "bias_check_required": True,
                },
            },
        }

        logger.info("Grace Governance Engine initialized")

    async def evaluate_action(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> GovernanceDecision:
        """
        Evaluate an action against both governance layers.

        Args:
            action_type: Type of action (trading, data_access, marketing, etc.)
            action_details: Specific details of the action
            user_context: User and context information

        Returns:
            GovernanceDecision with approval status and any violations
        """
        logger.info(
            f"Evaluating {action_type} action for user {user_context.get('user_id')}"
        )

        violations = []

        # Layer-1 Constitutional Checks (always enforced)
        layer_1_violations = await self._check_layer_1_rules(
            action_type, action_details, user_context
        )
        violations.extend(layer_1_violations)

        # Layer-2 Organizational Policy Checks
        layer_2_violations = await self._check_layer_2_policies(
            action_type, action_details, user_context
        )
        violations.extend(layer_2_violations)

        # Determine approval status
        # Any Layer-1 violation blocks action completely
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            return GovernanceDecision(
                approved=False,
                violations=violations,
                reasoning=f"Action blocked by {len(critical_violations)} constitutional violation(s)",
            )

        # Layer-2 violations may be overrideable
        high_violations = [v for v in violations if v.severity == "high"]
        medium_violations = [
            v for v in violations if v.severity == "medium" and not v.can_override
        ]

        if high_violations or medium_violations:
            return GovernanceDecision(
                approved=False,
                violations=violations,
                reasoning="Action requires governance review due to policy violations",
            )

        # Overrideable violations
        overrideable_violations = [v for v in violations if v.can_override]
        if overrideable_violations:
            required_approvals = []
            for violation in overrideable_violations:
                # Find the policy and required approvers
                for policy_id, policy in self.layer_2_policies.items():
                    if violation.rule_id == policy_id:
                        required_approvals.extend(policy.get("override_requires", []))

            return GovernanceDecision(
                approved=False,
                violations=violations,
                reasoning="Action can proceed with required approvals",
                required_approvals=list(set(required_approvals)),  # Remove duplicates
            )

        # No violations or only low-severity violations
        return GovernanceDecision(
            approved=True,
            violations=violations,
            reasoning="Action approved - no significant governance violations",
        )

    async def _check_layer_1_rules(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> List[GovernanceViolation]:
        """Check Layer-1 constitutional rules."""
        violations = []

        # Privacy Protection
        if await self._check_pii_exposure(action_details):
            violations.append(
                GovernanceViolation(
                    layer=GovernanceLayer.LAYER_1_CONSTITUTIONAL,
                    rule_id="privacy_protection",
                    rule_name="Privacy Protection",
                    description="Action would expose personally identifiable information",
                    severity="critical",
                    can_override=False,
                    suggested_actions=[
                        "Remove PII from data",
                        "Apply data anonymization",
                    ],
                )
            )

        # Harm Prevention
        harm_risk = await self._assess_harm_risk(action_type, action_details)
        if harm_risk["level"] == "high":
            violations.append(
                GovernanceViolation(
                    layer=GovernanceLayer.LAYER_1_CONSTITUTIONAL,
                    rule_id="harm_prevention",
                    rule_name="Harm Prevention",
                    description=f"Action poses high risk of harm: {harm_risk['reason']}",
                    severity="critical",
                    can_override=False,
                    suggested_actions=harm_risk.get("mitigation", []),
                )
            )

        # Transparency (audit trail)
        if not action_details.get("audit_enabled", True):
            violations.append(
                GovernanceViolation(
                    layer=GovernanceLayer.LAYER_1_CONSTITUTIONAL,
                    rule_id="transparency",
                    rule_name="Transparency",
                    description="Action must maintain audit trail",
                    severity="high",
                    can_override=False,
                    suggested_actions=[
                        "Enable audit logging",
                        "Add decision reasoning",
                    ],
                )
            )

        return violations

    async def _check_layer_2_policies(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> List[GovernanceViolation]:
        """Check Layer-2 organizational policies."""
        violations = []

        # Trading Limits
        if action_type == "trading" and "trading_limits" in self.layer_2_policies:
            policy = self.layer_2_policies["trading_limits"]
            trade_amount = action_details.get("amount", 0)

            if trade_amount > policy["parameters"]["max_single_trade"]:
                violations.append(
                    GovernanceViolation(
                        layer=GovernanceLayer.LAYER_2_ORGANIZATIONAL,
                        rule_id="trading_limits",
                        rule_name="Trading Risk Limits",
                        description=f"Trade amount {trade_amount} exceeds limit {policy['parameters']['max_single_trade']}",
                        severity="medium",
                        can_override=True,
                        suggested_actions=[
                            "Reduce trade size",
                            "Request risk manager approval",
                        ],
                    )
                )

        # Data Access Control
        if action_type == "data_access" and "data_access" in self.layer_2_policies:
            policy = self.layer_2_policies["data_access"]
            requester_dept = user_context.get("department")
            data_owner_dept = action_details.get("data_owner_department")

            if (
                requester_dept != data_owner_dept
                and policy["parameters"]["require_approval"]
            ):
                violations.append(
                    GovernanceViolation(
                        layer=GovernanceLayer.LAYER_2_ORGANIZATIONAL,
                        rule_id="data_access",
                        rule_name="Data Access Control",
                        description=f"Cross-department data access requires approval",
                        severity="medium",
                        can_override=True,
                        suggested_actions=[
                            "Request data owner approval",
                            "Specify access duration",
                        ],
                    )
                )

        # Marketing Campaign Timing
        if (
            action_type == "marketing_campaign"
            and "marketing_timing" in self.layer_2_policies
        ):
            policy = self.layer_2_policies["marketing_timing"]
            scheduled_time = action_details.get("scheduled_time")

            if scheduled_time:
                # Parse scheduled time (simplified)
                hour = int(scheduled_time.split(":")[0]) if ":" in scheduled_time else 0
                allowed_hours = policy["parameters"]["allowed_hours"]

                if hour < allowed_hours["start"] or hour >= allowed_hours["end"]:
                    violations.append(
                        GovernanceViolation(
                            layer=GovernanceLayer.LAYER_2_ORGANIZATIONAL,
                            rule_id="marketing_timing",
                            rule_name="Marketing Campaign Timing",
                            description=f"Campaign scheduled outside allowed hours ({allowed_hours['start']}-{allowed_hours['end']})",
                            severity="low",
                            can_override=True,
                            suggested_actions=[
                                "Reschedule to business hours",
                                "Request marketing lead approval",
                            ],
                        )
                    )

        return violations

    async def _check_pii_exposure(self, action_details: Dict[str, Any]) -> bool:
        """Check if action would expose PII."""
        # Simplified PII detection
        data = str(action_details.get("data", ""))

        pii_patterns = [
            "email",
            "phone",
            "ssn",
            "credit_card",
            "address",
            "@",
            "phone_number",
            "social_security",
        ]

        return any(pattern in data.lower() for pattern in pii_patterns)

    async def _assess_harm_risk(
        self, action_type: str, action_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess potential harm risk."""
        risk_level = "low"
        reason = ""
        mitigation = []

        # Financial harm
        if action_type == "trading":
            amount = action_details.get("amount", 0)
            if amount > 100000:  # Large trades
                risk_level = "medium"
                reason = "Large financial exposure"
                mitigation = ["Add stop-loss orders", "Reduce position size"]

            leverage = action_details.get("leverage", 1)
            if leverage > 10:
                risk_level = "high"
                reason = "Excessive leverage could cause significant losses"
                mitigation = ["Reduce leverage", "Add risk controls"]

        # Data exposure harm
        if action_type == "data_export":
            if action_details.get("contains_sensitive_data"):
                risk_level = "medium"
                reason = "Potential data privacy violation"
                mitigation = ["Apply data anonymization", "Restrict access"]

        return {"level": risk_level, "reason": reason, "mitigation": mitigation}

    async def request_override(
        self, decision: GovernanceDecision, requester_id: str, justification: str
    ) -> Dict[str, Any]:
        """Request override for governance decision."""
        if not decision.required_approvals:
            return {
                "success": False,
                "reason": "No overrides available for this decision",
            }

        override_request_id = f"override_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # In real implementation, this would create approval requests
        return {
            "success": True,
            "override_request_id": override_request_id,
            "required_approvals": decision.required_approvals,
            "status": "pending_approval",
        }


# Example User Journey Functions


async def developer_journey_example():
    """Developer building a module with governance hooks."""
    print("🧑‍💻 Developer Journey - Building NLP Module")
    print("-" * 50)

    governance = GraceGovernanceEngine()

    # Developer creates sentiment analysis module
    action_details = {
        "module_type": "nlp",
        "data_sources": ["twitter_api", "customer_reviews"],
        "data": "Processing tweets containing email addresses and user handles",
        "output": "sentiment_scores",
        "audit_enabled": True,
    }

    user_context = {
        "user_id": "dev_alice",
        "role": "developer",
        "department": "engineering",
    }

    # Governance evaluation
    decision = await governance.evaluate_action(
        "module_deployment", action_details, user_context
    )

    print(f"✅ Initial evaluation: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"📝 Reasoning: {decision.reasoning}")

    if decision.violations:
        print("⚠️  Violations found:")
        for violation in decision.violations:
            print(
                f"   • [{violation.layer.value}] {violation.rule_name}: {violation.description}"
            )
            if violation.suggested_actions:
                print(f"     Suggestions: {', '.join(violation.suggested_actions)}")

    # Developer fixes PII issue
    if any(v.rule_id == "privacy_protection" for v in decision.violations):
        print("\n🔧 Developer fixes PII exposure...")
        action_details["data"] = (
            "Processing anonymized tweets with sentiment indicators"
        )

        # Re-evaluate
        decision = await governance.evaluate_action(
            "module_deployment", action_details, user_context
        )
        print(f"✅ After fix: {'APPROVED' if decision.approved else 'NEEDS_REVIEW'}")

        if decision.required_approvals:
            print(f"👥 Required approvals: {', '.join(decision.required_approvals)}")

    print("✨ Developer journey completed\n")


async def sales_journey_example():
    """Sales manager with automated outreach and ethical bounds."""
    print("📈 Sales Journey - Email Campaign with Governance")
    print("-" * 50)

    governance = GraceGovernanceEngine()

    # Sales manager wants to send campaign
    action_details = {
        "campaign_type": "follow_up_emails",
        "recipient_list": ["new_signups", "trial_users"],
        "scheduled_time": "06:00",  # 6 AM - outside business hours
        "content_type": "promotional",
        "contains_sensitive_data": False,
        "audit_enabled": True,
    }

    user_context = {
        "user_id": "sales_bob",
        "role": "sales_manager",
        "department": "sales",
    }

    # Governance evaluation
    decision = await governance.evaluate_action(
        "marketing_campaign", action_details, user_context
    )

    print(f"✅ Initial evaluation: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"📝 Reasoning: {decision.reasoning}")

    if decision.violations:
        print("⚠️  Policy violations:")
        for violation in decision.violations:
            print(f"   • {violation.rule_name}: {violation.description}")

    # Manager reschedules to comply
    if any(v.rule_id == "marketing_timing" for v in decision.violations):
        print("\n⏰ Manager reschedules to business hours...")
        action_details["scheduled_time"] = "10:00"  # 10 AM

        # Re-evaluate
        decision = await governance.evaluate_action(
            "marketing_campaign", action_details, user_context
        )
        print(
            f"✅ After reschedule: {'APPROVED' if decision.approved else 'NEEDS_REVIEW'}"
        )

    print("✨ Sales journey completed\n")


async def operations_journey_example():
    """Data analyst with access control and Layer-2 policies."""
    print("🔍 Operations Journey - Data Access with Controls")
    print("-" * 50)

    governance = GraceGovernanceEngine()

    # Data analyst wants cross-department access
    action_details = {
        "data_source": "customer_purchase_history",
        "data_owner_department": "sales",
        "access_purpose": "churn_analysis",
        "duration_days": 30,
        "contains_sensitive_data": True,
        "data": "Customer purchase data with payment details and addresses",
        "audit_enabled": True,
    }

    user_context = {
        "user_id": "analyst_carol",
        "role": "data_analyst",
        "department": "operations",
    }

    # Governance evaluation
    decision = await governance.evaluate_action(
        "data_access", action_details, user_context
    )

    print(f"✅ Initial evaluation: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"📝 Reasoning: {decision.reasoning}")

    if decision.violations:
        print("🚫 Governance violations:")
        for violation in decision.violations:
            layer_emoji = (
                "🏛️"
                if violation.layer == GovernanceLayer.LAYER_1_CONSTITUTIONAL
                else "🏢"
            )
            print(f"   {layer_emoji} [{violation.layer.value}] {violation.rule_name}")
            print(f"      {violation.description}")
            if violation.can_override:
                print(f"      ✅ Can be overridden with approval")
            else:
                print(f"      ❌ Cannot be overridden")

    # Request override for Layer-2 policy
    if decision.required_approvals:
        print(
            f"\n📋 Requesting override from: {', '.join(decision.required_approvals)}"
        )
        override_result = await governance.request_override(
            decision,
            user_context["user_id"],
            "Need cross-department data for critical churn analysis",
        )
        print(f"✅ Override request: {override_result['override_request_id']}")
        print(f"📊 Status: {override_result['status']}")

    print("✨ Operations journey completed\n")


async def main():
    """Run all governance examples."""
    print("🏛️  Grace Governance Integration Examples")
    print("=" * 60)
    print(
        "Demonstrating Layer-1 (Constitutional) and Layer-2 (Organizational) governance\n"
    )

    await developer_journey_example()
    await sales_journey_example()
    await operations_journey_example()

    print("=" * 60)
    print("🎉 All governance examples completed!")
    print("\nKey Takeaways:")
    print("• Layer-1 rules are immutable and cannot be overridden")
    print("• Layer-2 policies are configurable and may allow overrides")
    print("• Governance is integrated into every action and decision")
    print("• Users are guided through compliant workflows")


if __name__ == "__main__":
    asyncio.run(main())
