"""
YAML-based policy rule definitions for Grace system.

Defines policies for dangerous operations and security constraints.
"""
import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class PolicyRule:
    """Individual policy rule definition."""
    
    def __init__(self, name: str, rule_data: Dict[str, Any]):
        self.name = name
        self.description = rule_data.get('description', '')
        self.severity = rule_data.get('severity', 'medium')
        self.enabled = rule_data.get('enabled', True)
        self.conditions = rule_data.get('conditions', [])
        self.actions = rule_data.get('actions', [])
        self.exceptions = rule_data.get('exceptions', [])
        self.metadata = rule_data.get('metadata', {})
    
    def matches_operation(self, operation: Dict[str, Any]) -> bool:
        """Check if this rule matches the given operation."""
        if not self.enabled:
            return False
        
        # Check exceptions first
        for exception in self.exceptions:
            if self._matches_condition(exception, operation):
                return False
        
        # Check conditions
        for condition in self.conditions:
            if self._matches_condition(condition, operation):
                return True
        
        return False
    
    def _matches_condition(self, condition: Dict[str, Any], operation: Dict[str, Any]) -> bool:
        """Check if a condition matches an operation."""
        condition_type = condition.get('type')
        
        if condition_type == 'operation_type':
            return operation.get('type') in condition.get('values', [])
        
        elif condition_type == 'content_pattern':
            content = operation.get('content', '')
            pattern = condition.get('pattern', '')
            if condition.get('regex', False):
                return bool(re.search(pattern, content, re.IGNORECASE))
            else:
                return pattern.lower() in content.lower()
        
        elif condition_type == 'file_path':
            file_path = operation.get('file_path', '')
            patterns = condition.get('patterns', [])
            for pattern in patterns:
                if re.search(pattern, file_path):
                    return True
            return False
        
        elif condition_type == 'user_role':
            user_roles = operation.get('user_roles', [])
            required_roles = condition.get('roles', [])
            return any(role in user_roles for role in required_roles)
        
        elif condition_type == 'scope_required':
            user_scopes = operation.get('user_scopes', [])
            required_scopes = condition.get('scopes', [])
            return any(scope in user_scopes for scope in required_scopes)
        
        return False


class PolicyEngine:
    """Main policy engine that loads and evaluates rules."""
    
    def __init__(self, policy_file: Optional[str] = None):
        self.rules: List[PolicyRule] = []
        self.policy_file = policy_file or self._get_default_policy_path()
        self.load_policies()
    
    def _get_default_policy_path(self) -> str:
        """Get default policy file path."""
        # Look for policy file in grace/policy directory
        policy_dir = Path(__file__).parent
        default_policy = policy_dir / "default_policies.yml"
        return str(default_policy)
    
    def load_policies(self):
        """Load policy rules from YAML file."""
        try:
            policy_path = Path(self.policy_file)
            if not policy_path.exists():
                logger.warning(f"Policy file not found: {self.policy_file}")
                self._create_default_policy_file(policy_path)
            
            with open(policy_path, 'r') as f:
                policy_data = yaml.safe_load(f)
            
            self.rules = []
            rules_data = policy_data.get('rules', {})
            
            for rule_name, rule_config in rules_data.items():
                rule = PolicyRule(rule_name, rule_config)
                self.rules.append(rule)
            
            logger.info(f"Loaded {len(self.rules)} policy rules from {self.policy_file}")
            
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            self.rules = []
    
    def _create_default_policy_file(self, policy_path: Path):
        """Create a default policy file."""
        default_policies = {
            'version': '1.0',
            'description': 'Grace System Default Security Policies',
            'rules': {
                'dangerous_file_operations': {
                    'description': 'Block dangerous file system operations',
                    'severity': 'high',
                    'enabled': True,
                    'conditions': [
                        {
                            'type': 'operation_type',
                            'values': ['file_write', 'file_delete', 'directory_create', 'directory_delete']
                        },
                        {
                            'type': 'file_path',
                            'patterns': [
                                r'/etc/.*',
                                r'/usr/bin/.*',
                                r'/var/log/.*',
                                r'.*\.sh$',
                                r'.*\.exe$',
                                r'.*\.bat$'
                            ]
                        }
                    ],
                    'actions': ['block', 'log', 'notify_admin'],
                    'exceptions': [
                        {
                            'type': 'user_role',
                            'roles': ['admin', 'system']
                        }
                    ]
                },
                'dangerous_code_execution': {
                    'description': 'Block dangerous code execution patterns',
                    'severity': 'critical',
                    'enabled': True,
                    'conditions': [
                        {
                            'type': 'content_pattern',
                            'pattern': r'(exec|eval|subprocess|os\.system|shell=True)',
                            'regex': True
                        },
                        {
                            'type': 'operation_type',
                            'values': ['code_execution', 'shell_command']
                        }
                    ],
                    'actions': ['block', 'log', 'alert'],
                    'exceptions': [
                        {
                            'type': 'scope_required',
                            'scopes': ['sandbox:build', 'govern:tasks']
                        }
                    ]
                },
                'network_access': {
                    'description': 'Control network access operations',
                    'severity': 'medium',
                    'enabled': True,
                    'conditions': [
                        {
                            'type': 'operation_type',
                            'values': ['network_request', 'socket_open']
                        },
                        {
                            'type': 'content_pattern',
                            'pattern': r'(requests\.|urllib\.|socket\.|http)',
                            'regex': True
                        }
                    ],
                    'actions': ['require_approval', 'log'],
                    'exceptions': [
                        {
                            'type': 'scope_required',
                            'scopes': ['network:access']
                        }
                    ]
                },
                'ide_apply_changes': {
                    'description': 'Gate IDE apply changes to sandbox branch with approval',
                    'severity': 'medium',
                    'enabled': True,
                    'conditions': [
                        {
                            'type': 'operation_type',
                            'values': ['ide_apply_changes', 'code_modification']
                        }
                    ],
                    'actions': ['require_sandbox_branch', 'require_policy_pass', 'require_human_approval'],
                    'metadata': {
                        'sandbox_branch_prefix': 'sandbox/',
                        'required_labels': ['policy:pass'],
                        'required_approvals': 1
                    }
                }
            }
        }
        
        # Ensure parent directory exists
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(policy_path, 'w') as f:
            yaml.dump(default_policies, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created default policy file: {policy_path}")
    
    def evaluate_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an operation against all policy rules.
        
        Args:
            operation: Operation details to evaluate
            
        Returns:
            Policy evaluation result
        """
        violations = []
        highest_severity = None
        actions_to_take = set()
        
        for rule in self.rules:
            if rule.matches_operation(operation):
                violation = {
                    'rule_name': rule.name,
                    'description': rule.description,
                    'severity': rule.severity,
                    'actions': rule.actions,
                    'metadata': rule.metadata
                }
                violations.append(violation)
                
                # Track highest severity
                if highest_severity is None or self._severity_level(rule.severity) > self._severity_level(highest_severity):
                    highest_severity = rule.severity
                
                # Collect actions
                actions_to_take.update(rule.actions)
        
        result = {
            'allowed': len(violations) == 0 or not any(
                'block' in v['actions'] for v in violations
            ),
            'violations': violations,
            'severity': highest_severity,
            'actions': list(actions_to_take),
            'operation': operation
        }
        
        # Log policy evaluation
        if violations:
            logger.warning(f"Policy violations detected for operation {operation.get('type', 'unknown')}: {len(violations)} violations")
        else:
            logger.debug(f"Policy evaluation passed for operation {operation.get('type', 'unknown')}")
        
        return result
    
    def _severity_level(self, severity: str) -> int:
        """Convert severity string to numeric level."""
        levels = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        return levels.get(severity.lower(), 0)
    
    def reload_policies(self):
        """Reload policies from file."""
        logger.info("Reloading policy rules")
        self.load_policies()
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of loaded policies."""
        summary = {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'policy_file': self.policy_file,
            'rules_by_severity': {}
        }
        
        for rule in self.rules:
            if rule.enabled:
                severity = rule.severity
                if severity not in summary['rules_by_severity']:
                    summary['rules_by_severity'][severity] = 0
                summary['rules_by_severity'][severity] += 1
        
        return summary


# Global policy engine instance
_policy_engine = None

def get_policy_engine() -> PolicyEngine:
    """Get global policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    return _policy_engine