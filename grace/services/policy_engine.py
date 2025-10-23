"""
Grace AI Policy Engine - Governance and policy enforcement
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    SECURITY = "security"
    GOVERNANCE = "governance"
    RESOURCE = "resource"
    ACCESS_CONTROL = "access_control"

class Policy:
    """Represents a system policy."""
    
    def __init__(self, policy_id: str, name: str, policy_type: PolicyType, rules: List[Dict[str, Any]]):
        self.policy_id = policy_id
        self.name = name
        self.policy_type = policy_type
        self.rules = rules
        self.created_at = datetime.now().isoformat()
        self.enabled = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "policy_type": self.policy_type.value,
            "rules": self.rules,
            "created_at": self.created_at,
            "enabled": self.enabled
        }

class PolicyEngine:
    """Enforces policies across the system."""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.policy_violations: List[Dict[str, Any]] = []
    
    def register_policy(self, policy: Policy) -> bool:
        """Register a new policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered policy: {policy.name} ({policy.policy_type.value})")
        return True
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self.policies.get(policy_id)
    
    async def evaluate_policy(self, policy_id: str, context: Dict[str, Any]) -> bool:
        """Evaluate if a context complies with a policy."""
        policy = self.policies.get(policy_id)
        if not policy or not policy.enabled:
            return True
        
        # Simple rule evaluation
        for rule in policy.rules:
            field = rule.get("field")
            operator = rule.get("operator")
            value = rule.get("value")
            
            context_value = context.get(field)
            
            if operator == "equals" and context_value != value:
                self._record_violation(policy_id, context)
                return False
            elif operator == "greater_than" and not (context_value > value):
                self._record_violation(policy_id, context)
                return False
        
        return True
    
    def _record_violation(self, policy_id: str, context: Dict[str, Any]):
        """Record a policy violation."""
        violation = {
            "policy_id": policy_id,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.policy_violations.append(violation)
        logger.warning(f"Policy violation: {policy_id}")
    
    def get_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get policy violations."""
        return self.policy_violations[-limit:]
    
    def list_policies(self) -> List[Dict[str, Any]]:
        """List all policies."""
        return [p.to_dict() for p in self.policies.values()]
