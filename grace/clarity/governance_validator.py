"""
Governance Validator - Constitutional constraint checking (Class 6)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of governance validation"""
    passed: bool
    violations: List[str] = field(default_factory=list)
    amendments: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    decision: Optional[Dict[str, Any]] = None


class ConstitutionalConstraint:
    """Represents a constitutional constraint"""
    
    def __init__(
        self,
        constraint_id: str,
        rule: str,
        validator: callable,
        severity: str = "error",  # error, warning
        auto_amend: bool = False
    ):
        self.constraint_id = constraint_id
        self.rule = rule
        self.validator = validator
        self.severity = severity
        self.auto_amend = auto_amend
        
        self.violation_count = 0
        self.amendment_count = 0
    
    def validate(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate decision against this constraint
        
        Returns:
            (passed, violation_message)
        """
        try:
            result = self.validator(decision, context)
            
            if not result:
                self.violation_count += 1
                return False, f"Violated: {self.rule}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Constraint validation error ({self.constraint_id}): {e}")
            self.violation_count += 1
            return False, f"Validation error: {str(e)}"


class GovernanceValidator:
    """
    Validates decisions against constitutional constraints and regulations
    
    Features:
    - Constitutional constraint checking
    - Regulation policy enforcement
    - Automatic amendment suggestions
    - Violation logging
    """
    
    def __init__(self, constitution_validator=None, immutable_logs=None):
        """
        Initialize governance validator
        
        Args:
            constitution_validator: Existing constitution validator
            immutable_logs: ImmutableLogs for violation logging
        """
        self.constitution_validator = constitution_validator
        self.immutable_logs = immutable_logs
        
        self.constraints: Dict[str, ConstitutionalConstraint] = {}
        self.regulation_policies: Dict[str, Dict[str, Any]] = {}
        
        self._register_default_constraints()
        
        logger.info("GovernanceValidator initialized")
    
    def _register_default_constraints(self):
        """Register default constitutional constraints"""
        # User privacy
        self.register_constraint(
            "privacy_protection",
            "Must not expose user PII without consent",
            lambda decision, ctx: not self._contains_pii(decision) or ctx.get("user_consent", False),
            severity="error"
        )
        
        # Data retention
        self.register_constraint(
            "data_retention",
            "Must not retain data beyond policy limits",
            lambda decision, ctx: self._check_retention_limits(decision, ctx),
            severity="warning"
        )
        
        # Fairness
        self.register_constraint(
            "fairness",
            "Must not discriminate based on protected attributes",
            lambda decision, ctx: not self._contains_discrimination(decision),
            severity="error"
        )
        
        # Transparency
        self.register_constraint(
            "transparency",
            "Must provide reasoning for decisions",
            lambda decision, ctx: "reasoning" in decision or "explanation" in decision,
            severity="warning",
            auto_amend=True
        )
    
    def register_constraint(
        self,
        constraint_id: str,
        rule: str,
        validator: callable,
        severity: str = "error",
        auto_amend: bool = False
    ):
        """Register a constitutional constraint"""
        constraint = ConstitutionalConstraint(
            constraint_id, rule, validator, severity, auto_amend
        )
        
        self.constraints[constraint_id] = constraint
        logger.info(f"Registered constraint: {constraint_id}")
    
    def register_regulation(
        self,
        regulation_id: str,
        policy: Dict[str, Any]
    ):
        """Register a regulation policy"""
        self.regulation_policies[regulation_id] = policy
        logger.info(f"Registered regulation: {regulation_id}")
    
    def validate_decision(
        self,
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a decision against all constraints and regulations
        
        Args:
            decision: Decision to validate
            context: Additional context
            
        Returns:
            ValidationResult with pass/fail and details
        """
        context = context or {}
        violations = []
        amendments = []
        passed = True
        
        # Check constitutional constraints
        for constraint in self.constraints.values():
            is_valid, violation_msg = constraint.validate(decision, context)
            
            if not is_valid:
                violations.append(f"{constraint.constraint_id}: {violation_msg}")
                
                if constraint.severity == "error":
                    passed = False
                
                # Try auto-amendment
                if constraint.auto_amend:
                    amendment = self._generate_amendment(constraint, decision, context)
                    if amendment:
                        amendments.append(amendment)
                        constraint.amendment_count += 1
        
        # Check regulation policies
        for reg_id, policy in self.regulation_policies.items():
            reg_valid, reg_violation = self._validate_regulation(decision, policy, context)
            
            if not reg_valid:
                violations.append(f"Regulation {reg_id}: {reg_violation}")
                passed = False
        
        # Use constitution validator if available
        if self.constitution_validator:
            try:
                const_result = self.constitution_validator.validate_against_constitution(
                    decision, context
                )
                
                if not const_result.passed:
                    violations.extend(const_result.violations)
                    passed = False
            except Exception as e:
                logger.error(f"Constitution validator error: {e}")
        
        # Calculate confidence
        confidence = 1.0 if passed else 0.0
        if violations and not passed:
            # Partial confidence based on severity
            error_count = sum(1 for v in violations if "error" in v.lower())
            warning_count = len(violations) - error_count
            confidence = max(0.0, 1.0 - (error_count * 0.4 + warning_count * 0.1))
        
        # Apply amendments if any
        amended_decision = decision.copy()
        for amendment in amendments:
            amended_decision = self._apply_amendment(amended_decision, amendment)
        
        result = ValidationResult(
            passed=passed,
            decision=amended_decision if amendments else decision,
            violations=violations,
            amendments=amendments,
            confidence=confidence,
            metadata={
                "total_constraints_checked": len(self.constraints),
                "total_regulations_checked": len(self.regulation_policies),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Log violations
        if violations and self.immutable_logs:
            self._log_violations(decision, violations, context)
        
        logger.info(
            f"Validation {'passed' if passed else 'failed'}: "
            f"{len(violations)} violations, {len(amendments)} amendments"
        )
        
        return result
    
    def _contains_pii(self, decision: Dict[str, Any]) -> bool:
        """Check if decision contains PII"""
        # Simple heuristic (in production, use NER)
        text = str(decision).lower()
        
        pii_indicators = [
            "ssn", "social security", "passport", "driver license",
            "credit card", "bank account", "phone number", "email@"
        ]
        
        return any(indicator in text for indicator in pii_indicators)
    
    def _check_retention_limits(
        self,
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check data retention limits"""
        # Check if decision involves data storage beyond limits
        retention_days = decision.get("retention_days", 0)
        max_retention = context.get("max_retention_days", 90)
        
        return retention_days <= max_retention
    
    def _contains_discrimination(self, decision: Dict[str, Any]) -> bool:
        """Check for discrimination based on protected attributes"""
        # Simple check for protected attribute usage
        protected_attrs = ["race", "gender", "age", "religion", "nationality"]
        
        text = str(decision).lower()
        
        # Check if decision explicitly mentions protected attributes
        for attr in protected_attrs:
            if f"based on {attr}" in text or f"because of {attr}" in text:
                return True
        
        return False
    
    def _validate_regulation(
        self,
        decision: Dict[str, Any],
        policy: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate against a regulation policy"""
        # Check policy requirements
        required_fields = policy.get("required_fields", [])
        
        for field in required_fields:
            if field not in decision:
                return False, f"Missing required field: {field}"
        
        # Check forbidden actions
        forbidden_actions = policy.get("forbidden_actions", [])
        decision_action = decision.get("action", "")
        
        if decision_action in forbidden_actions:
            return False, f"Forbidden action: {decision_action}"
        
        return True, None
    
    def _generate_amendment(
        self,
        constraint: ConstitutionalConstraint,
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate amendment suggestion"""
        if constraint.constraint_id == "transparency":
            return "add_reasoning: Provide explanation for this decision"
        
        return None
    
    def _apply_amendment(
        self,
        decision: Dict[str, Any],
        amendment: str
    ) -> Dict[str, Any]:
        """Apply an amendment to the decision"""
        amended = decision.copy()
        
        if amendment.startswith("add_reasoning"):
            amended["reasoning"] = "Automated amendment: Decision reasoning required by governance"
        
        return amended
    
    def _log_violations(
        self,
        decision: Dict[str, Any],
        violations: List[str],
        context: Dict[str, Any]
    ):
        """Log violations to immutable logs"""
        try:
            self.immutable_logs.log_constitutional_operation(
                operation_type="governance_violation",
                actor="governance_validator",
                action={"decision": decision, "context": context},
                result={"violations": violations},
                severity="error" if len(violations) > 0 else "warning",
                tags=["governance", "validation", "violation"]
            )
        except Exception as e:
            logger.error(f"Error logging violations: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "total_constraints": len(self.constraints),
            "total_regulations": len(self.regulation_policies),
            "constraint_violations": {
                cid: c.violation_count
                for cid, c in self.constraints.items()
            },
            "constraint_amendments": {
                cid: c.amendment_count
                for cid, c in self.constraints.items()
            }
        }
