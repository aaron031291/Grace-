"""
Class 6: Governance Validation - Constitution compliance checking
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of constitution violations"""
    ETHICAL = "ethical"
    SAFETY = "safety"
    SCOPE = "scope"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    PROCEDURAL = "procedural"


@dataclass
class ConstitutionRule:
    """Individual constitution rule"""
    rule_id: str
    rule_type: ViolationType
    description: str
    validator: str  # Python expression or function name
    severity: str = "medium"  # low, medium, high, critical
    mandatory: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of constitution validation"""
    passed: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 1.0
    recommendations: List[str] = field(default_factory=list)


class ConstitutionValidator:
    """
    Validates actions and decisions against Grace constitution
    Core implementation of Class 6
    """
    
    def __init__(self):
        self.rules: Dict[str, ConstitutionRule] = {}
        self.violation_history: List[Dict[str, Any]] = []
        self._initialize_default_rules()
        logger.info("ConstitutionValidator initialized")
    
    def _initialize_default_rules(self):
        """Initialize default constitution rules"""
        default_rules = [
            ConstitutionRule(
                rule_id="ethical_001",
                rule_type=ViolationType.ETHICAL,
                description="No harm to humans",
                validator="not action.get('harmful_to_humans', False)",
                severity="critical",
                mandatory=True
            ),
            ConstitutionRule(
                rule_id="ethical_002",
                rule_type=ViolationType.ETHICAL,
                description="Respect privacy",
                validator="not action.get('violates_privacy', False)",
                severity="high",
                mandatory=True
            ),
            ConstitutionRule(
                rule_id="safety_001",
                rule_type=ViolationType.SAFETY,
                description="Output safety check",
                validator="action.get('safety_score', 0) >= 0.7",
                severity="high",
                mandatory=True
            ),
            ConstitutionRule(
                rule_id="scope_001",
                rule_type=ViolationType.SCOPE,
                description="Within authorized scope",
                validator="action.get('scope') in ['reasoning', 'analysis', 'recommendation']",
                severity="medium",
                mandatory=False
            ),
            ConstitutionRule(
                rule_id="auth_001",
                rule_type=ViolationType.AUTHORIZATION,
                description="Proper authorization required",
                validator="action.get('authorized', False) == True",
                severity="high",
                mandatory=True
            ),
            ConstitutionRule(
                rule_id="resource_001",
                rule_type=ViolationType.RESOURCE,
                description="Resource usage within limits",
                validator="action.get('resource_usage', 0) <= 1.0",
                severity="medium",
                mandatory=False
            ),
            ConstitutionRule(
                rule_id="proc_001",
                rule_type=ViolationType.PROCEDURAL,
                description="Follow established procedures",
                validator="action.get('follows_procedure', True) == True",
                severity="low",
                mandatory=False
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def validate_against_constitution(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate an action against the constitution
        Core implementation of Class 6
        """
        violations = []
        warnings = []
        passed = True
        
        context = context or {}
        
        for rule in self.rules.values():
            try:
                # Evaluate rule
                is_valid = self._evaluate_rule(rule, action, context)
                
                if not is_valid:
                    violation = {
                        'rule_id': rule.rule_id,
                        'type': rule.rule_type.value,
                        'description': rule.description,
                        'severity': rule.severity,
                        'mandatory': rule.mandatory
                    }
                    
                    if rule.mandatory:
                        violations.append(violation)
                        if rule.severity in ['high', 'critical']:
                            passed = False
                    else:
                        warnings.append(violation)
                        
            except Exception as e:
                logger.error(f"Rule evaluation error for {rule.rule_id}: {e}")
                warnings.append({
                    'rule_id': rule.rule_id,
                    'error': str(e),
                    'description': 'Rule evaluation failed'
                })
        
        # Calculate compliance score
        total_rules = len(self.rules)
        violations_count = len(violations)
        warnings_count = len(warnings)
        
        score = 1.0 - (
            (violations_count * 0.3 + warnings_count * 0.1) / total_rules
        )
        score = max(0.0, min(1.0, score))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, warnings)
        
        result = ValidationResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            score=score,
            recommendations=recommendations
        )
        
        # Log violations
        if violations:
            self.violation_history.append({
                'action': action,
                'violations': violations,
                'timestamp': str(datetime.now())
            })
            logger.warning(f"Constitution violations detected: {len(violations)}")
        
        return result
    
    def _evaluate_rule(
        self,
        rule: ConstitutionRule,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single rule"""
        try:
            # Create safe evaluation environment
            eval_env = {
                'action': action,
                'context': context,
                **action  # Allow direct access to action fields
            }
            
            # Evaluate validator expression
            result = eval(rule.validator, {"__builtins__": {}}, eval_env)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {rule.rule_id} - {e}")
            return False
    
    def _generate_recommendations(
        self,
        violations: List[Dict],
        warnings: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        for violation in violations:
            if violation['type'] == 'ethical':
                recommendations.append(
                    f"Review ethical implications of {violation['description'].lower()}"
                )
            elif violation['type'] == 'safety':
                recommendations.append(
                    f"Enhance safety measures for {violation['description'].lower()}"
                )
            elif violation['type'] == 'authorization':
                recommendations.append(
                    "Obtain proper authorization before proceeding"
                )
        
        for warning in warnings:
            if warning['type'] == 'scope':
                recommendations.append(
                    "Consider if action is within authorized scope"
                )
            elif warning['type'] == 'resource':
                recommendations.append(
                    "Optimize resource usage to stay within limits"
                )
        
        return recommendations
    
    def add_rule(self, rule: ConstitutionRule):
        """Add a new constitution rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added constitution rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str):
        """Remove a constitution rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed constitution rule: {rule_id}")
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get violation statistics"""
        if not self.violation_history:
            return {'total_violations': 0}
        
        violation_types = {}
        for record in self.violation_history:
            for v in record['violations']:
                vtype = v['type']
                violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        return {
            'total_violations': len(self.violation_history),
            'by_type': violation_types,
            'total_rules': len(self.rules),
            'mandatory_rules': sum(1 for r in self.rules.values() if r.mandatory)
        }
