"""RBAC policy evaluator for Interface Kernel."""
from datetime import datetime
from typing import Dict, List, Optional
import logging
import re

from ..models import PolicyRule, PolicyCondition

logger = logging.getLogger(__name__)


class RBACEvaluator:
    """Role-based access control evaluator with policy-checked actions."""
    
    def __init__(self):
        self.policies: Dict[str, PolicyRule] = {}
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load default RBAC policies."""
        default_policies = [
            # Owner can do everything
            PolicyRule(
                rule_id="owner_full_access",
                effect="allow",
                actions=["*"],
                resources=["*"],
                condition=PolicyCondition()
            ),
            
            # Admin can manage tasks, memory, and intel
            PolicyRule(
                rule_id="admin_task_management",
                effect="allow",
                actions=["task.create", "task.update", "task.run", "task.pause", "task.cancel"],
                resources=["task:*"],
                condition=PolicyCondition()
            ),
            PolicyRule(
                rule_id="admin_memory_access",
                effect="allow",
                actions=["memory.query", "memory.search"],
                resources=["memory:*"],
                condition=PolicyCondition(label_in=["internal"])
            ),
            
            # Developers can create and manage their own tasks
            PolicyRule(
                rule_id="dev_own_tasks",
                effect="allow",
                actions=["task.create", "task.update", "task.run", "task.pause"],
                resources=["task:{user_id}:*"],
                condition=PolicyCondition()
            ),
            
            # Analysts can query memory and create analysis tasks
            PolicyRule(
                rule_id="analyst_memory_query",
                effect="allow",
                actions=["memory.query", "memory.search"],
                resources=["memory:*"],
                condition=PolicyCondition(label_in=["internal", "restricted"])
            ),
            PolicyRule(
                rule_id="analyst_create_analysis",
                effect="allow",
                actions=["task.create"],
                resources=["task:analysis:*"],
                condition=PolicyCondition()
            ),
            
            # Viewers have read-only access
            PolicyRule(
                rule_id="viewer_readonly",
                effect="allow",
                actions=["task.view", "memory.search"],
                resources=["task:*", "memory:*"],
                condition=PolicyCondition(label_in=["internal"])
            ),
            
            # Deny dangerous operations for non-owners
            PolicyRule(
                rule_id="deny_snapshot_rollback",
                effect="deny",
                actions=["snapshot.rollback"],
                resources=["*"],
                condition=PolicyCondition()
            ),
            PolicyRule(
                rule_id="deny_external_share",
                effect="deny",
                actions=["consent.external_share"],
                resources=["*"],
                condition=PolicyCondition(label_in=["restricted"])
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.rule_id] = policy
    
    def evaluate(self, user_roles: List[str], action: str, resource: str, ctx: Dict) -> bool:
        """
        Evaluate if user roles can perform action on resource.
        
        Args:
            user_roles: List of user roles
            action: Action being attempted (e.g., "task.create")
            resource: Resource being accessed (e.g., "task:analysis:123")
            ctx: Context dictionary with user_id, labels, current_time, etc.
            
        Returns:
            True if action is allowed, False otherwise
        """
        logger.debug(f"Evaluating action {action} on {resource} for roles {user_roles}")
        
        # Default deny
        allow = False
        explicit_deny = False
        
        # Check each policy
        for policy_id, policy in self.policies.items():
            if self._policy_matches(policy, user_roles, action, resource, ctx):
                logger.debug(f"Policy {policy_id} matches")
                
                if policy.effect == "allow":
                    allow = True
                elif policy.effect == "deny":
                    explicit_deny = True
                    break  # Deny always wins
        
        # Apply decision logic: explicit deny overrides allow
        result = allow and not explicit_deny
        
        logger.info(f"RBAC decision: {result} for {action} on {resource} by {user_roles}")
        return result
    
    def _policy_matches(self, policy: PolicyRule, user_roles: List[str], action: str, resource: str, ctx: Dict) -> bool:
        """Check if a policy matches the current request."""
        
        # Check if user has required role (implicit in Grace - all policies apply to all users)
        # In more complex systems, policies might be role-specific
        
        # Check action match
        if not self._action_matches(policy.actions, action):
            return False
        
        # Check resource match
        if not self._resource_matches(policy.resources, resource, ctx):
            return False
        
        # Check conditions
        if not self._condition_matches(policy.condition, ctx):
            return False
        
        return True
    
    def _action_matches(self, policy_actions: List[str], requested_action: str) -> bool:
        """Check if requested action matches policy actions."""
        for policy_action in policy_actions:
            if policy_action == "*":
                return True
            elif policy_action == requested_action:
                return True
            elif policy_action.endswith("*"):
                prefix = policy_action[:-1]
                if requested_action.startswith(prefix):
                    return True
        return False
    
    def _resource_matches(self, policy_resources: List[str], requested_resource: str, ctx: Dict) -> bool:
        """Check if requested resource matches policy resources."""
        for policy_resource in policy_resources:
            if policy_resource == "*":
                return True
            elif policy_resource == requested_resource:
                return True
            elif "{user_id}" in policy_resource and "user_id" in ctx:
                # Replace {user_id} placeholder
                expanded_resource = policy_resource.replace("{user_id}", ctx["user_id"])
                if self._resource_pattern_matches(expanded_resource, requested_resource):
                    return True
            elif self._resource_pattern_matches(policy_resource, requested_resource):
                return True
        return False
    
    def _resource_pattern_matches(self, pattern: str, resource: str) -> bool:
        """Check if resource matches pattern with wildcards."""
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return resource.startswith(prefix)
        return pattern == resource
    
    def _condition_matches(self, condition: PolicyCondition, ctx: Dict) -> bool:
        """Check if context matches policy conditions."""
        
        # Check label conditions
        if condition.label_in:
            user_labels = ctx.get("labels", [])
            if not any(label in condition.label_in for label in user_labels):
                return False
        
        # Check time conditions
        current_time = ctx.get("current_time")
        if current_time:
            if condition.time_after:
                # Simple time format check (would need proper parsing in production)
                pass
            
            if condition.time_before:
                # Simple time format check (would need proper parsing in production)
                pass
        
        return True
    
    def add_policy(self, policy: PolicyRule):
        """Add or update a policy rule."""
        self.policies[policy.rule_id] = policy
        logger.info(f"Added policy {policy.rule_id}")
    
    def remove_policy(self, rule_id: str):
        """Remove a policy rule."""
        if rule_id in self.policies:
            del self.policies[rule_id]
            logger.info(f"Removed policy {rule_id}")
    
    def list_policies(self) -> List[PolicyRule]:
        """List all policy rules."""
        return list(self.policies.values())
    
    def get_user_permissions(self, user_roles: List[str], ctx: Dict) -> List[str]:
        """Get list of permissions for user based on their roles."""
        permissions = set()
        
        # Test common actions against common resources
        test_actions = [
            "task.create", "task.update", "task.run", "task.view",
            "memory.query", "memory.search", "intel.request",
            "governance.request_approval", "snapshot.export"
        ]
        
        for action in test_actions:
            if self.evaluate(user_roles, action, f"{action.split('.')[0]}:*", ctx):
                permissions.add(action)
        
        return list(permissions)


# Default evaluator instance
evaluator = RBACEvaluator()

def evaluate(user_roles: List[str], action: str, resource: str, ctx: Dict) -> bool:
    """Convenience function for policy evaluation."""
    return evaluator.evaluate(user_roles, action, resource, ctx)