"""Policy engine - evaluates requests against governance policies."""
from typing import Dict, List

from ..contracts.governed_request import GovernedRequest
from .types import PolicyResult, PolicyType


class PolicyEngine:
    """Evaluates requests against configured policies."""
    
    def __init__(self):
        self.policies = self._load_default_policies()
    
    def _load_default_policies(self) -> Dict[str, Dict]:
        """Load default policy configurations."""
        return {
            "content_safety": {
                "type": PolicyType.CONTENT_FILTER,
                "rules": ["no_harmful_content", "no_pii"],
                "severity": "high"
            },
            "access_control": {
                "type": PolicyType.ACCESS_CONTROL,
                "rules": ["authenticated_user", "authorized_domain"],
                "severity": "medium"
            },
            "risk_threshold": {
                "type": PolicyType.RISK_ASSESSMENT,
                "rules": ["risk_level_check"],
                "max_risk": "high",
                "severity": "medium"
            },
            "high_impact_quorum": {
                "type": PolicyType.QUORUM_REQUIRED,
                "rules": ["high_priority_requires_quorum"],
                "trigger_priority": 8,
                "severity": "low"
            }
        }
    
    def check(self, request: GovernedRequest) -> List[PolicyResult]:
        """Evaluate request against all applicable policies."""
        results = []
        
        for policy_id, policy_config in self.policies.items():
            result = self._evaluate_policy(policy_id, policy_config, request)
            results.append(result)
        
        return results
    
    def _evaluate_policy(self, policy_id: str, policy_config: Dict, request: GovernedRequest) -> PolicyResult:
        """Evaluate a single policy."""
        policy_type = policy_config["type"]
        
        if policy_type == PolicyType.CONTENT_FILTER:
            return self._check_content_filter(policy_id, policy_config, request)
        elif policy_type == PolicyType.ACCESS_CONTROL:
            return self._check_access_control(policy_id, policy_config, request)
        elif policy_type == PolicyType.RISK_ASSESSMENT:
            return self._check_risk_assessment(policy_id, policy_config, request)
        elif policy_type == PolicyType.QUORUM_REQUIRED:
            return self._check_quorum_requirement(policy_id, policy_config, request)
        else:
            return PolicyResult(
                policy_id=policy_id,
                policy_type=policy_type,
                passed=True,
                confidence=0.5,
                reasoning="Unknown policy type, defaulting to pass"
            )
    
    def _check_content_filter(self, policy_id: str, config: Dict, request: GovernedRequest) -> PolicyResult:
        """Check content filtering policy."""
        content = request.content.lower()
        
        # Simple content checks
        harmful_patterns = ["malicious", "harmful", "dangerous", "illegal"]
        pii_patterns = ["ssn:", "credit card:", "password:", "api key:"]
        
        issues = []
        for pattern in harmful_patterns:
            if pattern in content:
                issues.append(f"Potentially harmful content: {pattern}")
        
        for pattern in pii_patterns:
            if pattern in content:
                issues.append(f"PII detected: {pattern}")
        
        passed = len(issues) == 0
        
        return PolicyResult(
            policy_id=policy_id,
            policy_type=PolicyType.CONTENT_FILTER,
            passed=passed,
            confidence=0.85,
            reasoning="Passed content safety checks" if passed else f"Content issues: {', '.join(issues)}"
        )
    
    def _check_access_control(self, policy_id: str, config: Dict, request: GovernedRequest) -> PolicyResult:
        """Check access control policy."""
        # Simple access control - check requester is not empty
        has_requester = bool(request.requester and request.requester != "anonymous")
        
        return PolicyResult(
            policy_id=policy_id,
            policy_type=PolicyType.ACCESS_CONTROL,
            passed=has_requester,
            confidence=0.9,
            reasoning="Valid authenticated user" if has_requester else "No authenticated user"
        )
    
    def _check_risk_assessment(self, policy_id: str, config: Dict, request: GovernedRequest) -> PolicyResult:
        """Check risk assessment policy."""
        max_risk = config.get("max_risk", "high")
        request_risk = request.risk_level
        
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        max_level = risk_levels.get(max_risk, 3)
        request_level = risk_levels.get(request_risk, 2)
        
        passed = request_level <= max_level
        
        return PolicyResult(
            policy_id=policy_id,
            policy_type=PolicyType.RISK_ASSESSMENT,
            passed=passed,
            confidence=0.8,
            reasoning=f"Risk level {request_risk} {'within' if passed else 'exceeds'} threshold {max_risk}"
        )
    
    def _check_quorum_requirement(self, policy_id: str, config: Dict, request: GovernedRequest) -> PolicyResult:
        """Check if quorum is required."""
        trigger_priority = config.get("trigger_priority", 8)
        requires_quorum = request.priority >= trigger_priority
        
        # Set flag on request for downstream processing
        if requires_quorum:
            request.requires_quorum = True
        
        return PolicyResult(
            policy_id=policy_id,
            policy_type=PolicyType.QUORUM_REQUIRED,
            passed=True,  # This policy doesn't block, just sets requirements
            confidence=1.0,
            reasoning=f"{'Requires' if requires_quorum else 'No'} quorum based on priority {request.priority}"
        )