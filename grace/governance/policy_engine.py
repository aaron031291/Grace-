"""Policy engine - evaluates requests against governance policies."""
from typing import Dict, List, Any, Optional
import logging

# Support both contract systems for compatibility
try:
    from ..contracts.governed_request import GovernedRequest
    from ..governance_kernel.types import PolicyResult, PolicyType
    PYDANTIC_CONTRACTS = True
except ImportError:
    # Fallback for core contracts
    from core.contracts import DecisionSubject
    PYDANTIC_CONTRACTS = False

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Evaluates requests against configured policies."""
    
    def __init__(self):
        self.policies = self._load_default_policies()
    
    def _load_default_policies(self) -> Dict[str, Dict]:
        """Load default policy configurations."""
        return {
            "content_safety": {
                "type": "content_filter",
                "rules": ["no_harmful_content", "no_pii"],
                "severity": "high"
            },
            "access_control": {
                "type": "access_control", 
                "rules": ["authenticated_user", "authorized_domain"],
                "severity": "medium"
            },
            "risk_threshold": {
                "type": "risk_assessment",
                "rules": ["risk_level_check"],
                "max_risk": "high",
                "severity": "medium"
            },
            "high_impact_quorum": {
                "type": "quorum_required",
                "rules": ["high_priority_requires_quorum"],
                "trigger_priority": 8,
                "severity": "low"
            }
        }
    
    def check(self, request) -> List[Dict[str, Any]]:
        """Evaluate request against all applicable policies.
        
        Args:
            request: Either GovernedRequest (pydantic) or dict with request data
            
        Returns:
            List of policy evaluation results
        """
        results = []
        
        for policy_id, policy_config in self.policies.items():
            result = self._evaluate_policy(policy_id, policy_config, request)
            results.append(result)
        
        return results
    
    def _evaluate_policy(self, policy_id: str, policy_config: Dict, request) -> Dict[str, Any]:
        """Evaluate a single policy."""
        policy_type = policy_config["type"]
        
        if policy_type == "content_filter":
            return self._check_content_filter(policy_id, policy_config, request)
        elif policy_type == "access_control":
            return self._check_access_control(policy_id, policy_config, request)
        elif policy_type == "risk_assessment":
            return self._check_risk_assessment(policy_id, policy_config, request)
        elif policy_type == "quorum_required":
            return self._check_quorum_requirement(policy_id, policy_config, request)
        else:
            return {
                "policy_id": policy_id,
                "policy_type": policy_type,
                "passed": True,
                "confidence": 0.5,
                "reasoning": "Unknown policy type, defaulting to pass"
            }
    
    def _extract_content(self, request) -> str:
        """Extract content from request regardless of format."""
        if hasattr(request, 'content'):
            return str(request.content)
        elif isinstance(request, dict):
            return str(request.get('content', ''))
        else:
            return str(request)
    
    def _extract_requester(self, request) -> str:
        """Extract requester from request regardless of format."""
        if hasattr(request, 'requester'):
            return str(request.requester)
        elif isinstance(request, dict):
            return str(request.get('requester', ''))
        else:
            return ''
    
    def _extract_priority(self, request) -> int:
        """Extract priority from request regardless of format."""
        if hasattr(request, 'priority'):
            return int(request.priority)
        elif isinstance(request, dict):
            return int(request.get('priority', 0))
        else:
            return 0
    
    def _extract_risk_level(self, request) -> str:
        """Extract risk level from request regardless of format."""
        if hasattr(request, 'risk_level'):
            return str(request.risk_level)
        elif isinstance(request, dict):
            return str(request.get('risk_level', 'medium'))
        else:
            return 'medium'
    
    def _check_content_filter(self, policy_id: str, config: Dict, request) -> Dict[str, Any]:
        """Check content filtering policy."""
        content = self._extract_content(request).lower()
        
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
        
        return {
            "policy_id": policy_id,
            "policy_type": "content_filter",
            "passed": passed,
            "confidence": 0.85,
            "reasoning": "Passed content safety checks" if passed else f"Content issues: {', '.join(issues)}"
        }
    
    def _check_access_control(self, policy_id: str, config: Dict, request) -> Dict[str, Any]:
        """Check access control policy.""" 
        requester = self._extract_requester(request)
        has_requester = bool(requester and requester != "anonymous")
        
        return {
            "policy_id": policy_id,
            "policy_type": "access_control",
            "passed": has_requester,
            "confidence": 0.9,
            "reasoning": "Valid authenticated user" if has_requester else "No authenticated user"
        }
    
    def _check_risk_assessment(self, policy_id: str, config: Dict, request) -> Dict[str, Any]:
        """Check risk assessment policy."""
        max_risk = config.get("max_risk", "high")
        request_risk = self._extract_risk_level(request)
        
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        max_level = risk_levels.get(max_risk, 3)
        request_level = risk_levels.get(request_risk, 2)
        
        passed = request_level <= max_level
        
        return {
            "policy_id": policy_id,
            "policy_type": "risk_assessment",
            "passed": passed,
            "confidence": 0.8,
            "reasoning": f"Risk level {request_risk} {'within' if passed else 'exceeds'} threshold {max_risk}"
        }
    
    def _check_quorum_requirement(self, policy_id: str, config: Dict, request) -> Dict[str, Any]:
        """Check if quorum is required."""
        trigger_priority = config.get("trigger_priority", 8)
        priority = self._extract_priority(request)
        requires_quorum = priority >= trigger_priority
        
        # Set flag on request for downstream processing if possible
        if requires_quorum and hasattr(request, 'requires_quorum'):
            request.requires_quorum = True
        
        return {
            "policy_id": policy_id,
            "policy_type": "quorum_required",
            "passed": True,  # This policy doesn't block, just sets requirements
            "confidence": 1.0,
            "reasoning": f"{'Requires' if requires_quorum else 'No'} quorum based on priority {priority}",
            "requires_quorum": requires_quorum
        }