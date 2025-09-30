"""
Constitutional Validator - Validates actions against constitutional principles.
Part of Phase 4: Governance Gate implementation.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from ..config.environment import get_grace_config
from ..core.immutable_logs import ImmutableLogs, TransparencyLevel

logger = logging.getLogger(__name__)


@dataclass
class ConstitutionalViolation:
    """Represents a constitutional principle violation."""
    principle: str
    violation_type: str
    severity: str  # 'minor', 'major', 'critical'
    description: str
    confidence: float
    recommendation: str


@dataclass
class ConstitutionalValidationResult:
    """Results from constitutional validation."""
    is_valid: bool
    compliance_score: float
    violations: List[ConstitutionalViolation]
    validation_timestamp: datetime
    validation_id: str
    audit_trail: List[str] = None
    
    @property
    def summary(self) -> str:
        """Generate a summary of validation results."""
        if self.is_valid:
            return f"Valid (score: {self.compliance_score:.3f})"
        
        violation_summary = ", ".join([
            f"{v.principle}:{v.severity}" for v in self.violations
        ])
        return f"Invalid (score: {self.compliance_score:.3f}) - {violation_summary}"


class ConstitutionalValidator:
    """
    Validates actions and decisions against constitutional principles.
    Integrates with the orchestrator's decision pathway.
    """
    
    def __init__(self, immutable_logs: Optional[ImmutableLogs] = None, 
                 event_publisher: Optional[Any] = None):
        self.config = get_grace_config()
        self.immutable_logs = immutable_logs
        self.event_publisher = event_publisher
        
        # Load constitutional principles from configuration
        self.constitutional_principles = self.config["constitutional_principles"]
        
        # Validation thresholds
        self.min_compliance_score = self.config["governance_thresholds"]["constitutional_compliance_min"]
        
        logger.info("ConstitutionalValidator initialized")
    
    async def validate_against_constitution(self, action: Dict[str, Any], 
                                          context: Optional[Dict[str, Any]] = None) -> ConstitutionalValidationResult:
        """
        Validate an action against constitutional principles.
        This is the main method called by the orchestrator.
        """
        validation_id = f"const_val_{int(datetime.now().timestamp() * 1000)}"
        validation_timestamp = datetime.now()
        
        # Log the validation request
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="constitutional_validation_started",
                component_id="constitutional_validator",
                event_data={
                    "validation_id": validation_id,
                    "action_type": action.get("type", "unknown"),
                    "action_id": action.get("id", "unknown")
                },
                transparency_level=TransparencyLevel.DEMOCRATIC_OVERSIGHT
            )
        
        violations = []
        
        # Validate against each constitutional principle
        for principle_name, principle_config in self.constitutional_principles.items():
            if principle_config.get("required", False):
                violation = await self._check_principle_compliance(
                    principle_name, principle_config, action, context
                )
                if violation:
                    violations.append(violation)
        
        # Calculate overall compliance score
        compliance_score = await self._calculate_compliance_score(violations)
        
        # Determine if action is constitutionally valid
        is_valid = compliance_score >= self.min_compliance_score
        
        # Create validation result
        result = ConstitutionalValidationResult(
            is_valid=is_valid,
            compliance_score=compliance_score,
            violations=violations,
            validation_timestamp=validation_timestamp,
            validation_id=validation_id,
            audit_trail=[f"Validated {len(violations)} principles", f"Score: {compliance_score:.3f}"]
        )
        
        # Log any violations
        if violations:
            await self._log_violations(result, action)
        
        # Log validation result
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="constitutional_validation_completed",
                component_id="constitutional_validator",
                event_data={
                    "validation_id": validation_id,
                    "is_valid": is_valid,
                    "compliance_score": compliance_score,
                    "violation_count": len(violations)
                },
                transparency_level=TransparencyLevel.DEMOCRATIC_OVERSIGHT
            )
        
        # Publish validation event
        if self.event_publisher:
            await self.event_publisher("constitutional_validation", asdict(result))
        
        logger.info(f"Constitutional validation {validation_id}: {'VALID' if is_valid else 'INVALID'} "
                   f"(score: {compliance_score:.3f})")
        
        return result
    
    async def validate_governance_decision(self, decision: Dict[str, Any]) -> ConstitutionalValidationResult:
        """
        Specialized validation for governance decisions.
        """
        # Enhanced context for governance decisions
        context = {
            "decision_type": "governance",
            "decision_id": decision.get("decision_id"),
            "confidence": decision.get("confidence", 0.0),
            "trust_score": decision.get("trust_score", 0.0),
            "has_rationale": bool(decision.get("rationale")),
            "transparency_level": decision.get("transparency_level", "internal")
        }
        
        return await self.validate_against_constitution(decision, context)
    
    async def validate_autonomous_action(self, action: Dict[str, Any]) -> ConstitutionalValidationResult:
        """
        Specialized validation for autonomous system actions.
        """
        # Enhanced context for autonomous actions
        context = {
            "action_type": "autonomous",
            "action_component": action.get("component_id"),
            "has_human_oversight": action.get("human_oversight", False),
            "risk_level": action.get("risk_level", "unknown"),
            "reversible": action.get("reversible", False),
            "impact_scope": action.get("impact_scope", "unknown")
        }
        
        return await self.validate_against_constitution(action, context)
    
    async def _check_principle_compliance(self, principle_name: str, principle_config: Dict[str, Any],
                                        action: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[ConstitutionalViolation]:
        """
        Check compliance with a specific constitutional principle.
        """
        if principle_name == "transparency":
            return await self._check_transparency_compliance(action, context)
        elif principle_name == "fairness":
            return await self._check_fairness_compliance(action, context)
        elif principle_name == "accountability":
            return await self._check_accountability_compliance(action, context)
        elif principle_name == "consistency":
            return await self._check_consistency_compliance(action, context)
        elif principle_name == "harm_prevention":
            return await self._check_harm_prevention_compliance(action, context)
        
        return None
    
    async def _check_transparency_compliance(self, action: Dict[str, Any], 
                                           context: Optional[Dict[str, Any]]) -> Optional[ConstitutionalViolation]:
        """Check transparency principle compliance."""
        issues = []
        
        # Check if action has adequate documentation
        if not action.get("rationale") and not action.get("description"):
            issues.append("No rationale or description provided")
        
        # Check if decision process is auditable
        if not action.get("audit_trail") and not context.get("has_rationale"):
            issues.append("No audit trail available")
        
        # Check transparency level
        transparency_level = context.get("transparency_level", "internal") if context else "internal"
        if transparency_level == "secret" or transparency_level == "hidden":
            issues.append("Action marked with inappropriate secrecy level")
        
        if issues:
            return ConstitutionalViolation(
                principle="transparency",
                violation_type="insufficient_transparency",
                severity="major" if len(issues) > 2 else "minor",
                description=f"Transparency violations: {'; '.join(issues)}",
                confidence=0.8,
                recommendation="Provide clear rationale and ensure audit trail"
            )
        
        return None
    
    async def _check_fairness_compliance(self, action: Dict[str, Any], 
                                       context: Optional[Dict[str, Any]]) -> Optional[ConstitutionalViolation]:
        """Check fairness principle compliance."""
        issues = []
        
        # Check for bias indicators
        if action.get("targets_specific_group"):
            issues.append("Action may unfairly target specific group")
        
        # Check for equal treatment
        if action.get("unequal_treatment") and not action.get("justified_differential"):
            issues.append("Unequal treatment without justification")
        
        # Check decision confidence and bias
        confidence = context.get("confidence", 1.0) if context else 1.0
        if confidence < 0.5:
            issues.append("Low confidence decision may indicate bias")
        
        if issues:
            return ConstitutionalViolation(
                principle="fairness",
                violation_type="unfair_treatment",
                severity="major" if "target" in str(issues) else "minor",
                description=f"Fairness violations: {'; '.join(issues)}",
                confidence=0.7,
                recommendation="Review for bias and ensure equal treatment"
            )
        
        return None
    
    async def _check_accountability_compliance(self, action: Dict[str, Any], 
                                             context: Optional[Dict[str, Any]]) -> Optional[ConstitutionalViolation]:
        """Check accountability principle compliance."""
        issues = []
        
        # Check if there's a clear decision maker
        if not action.get("decision_maker") and not action.get("component_id"):
            issues.append("No clear decision maker identified")
        
        # Check if action is traceable
        if not action.get("decision_id") and not action.get("correlation_id"):
            issues.append("Action is not traceable")
        
        # Check for autonomous actions without oversight
        if context and context.get("action_type") == "autonomous":
            if not context.get("has_human_oversight") and action.get("risk_level") == "high":
                issues.append("High-risk autonomous action without human oversight")
        
        if issues:
            return ConstitutionalViolation(
                principle="accountability",
                violation_type="lack_of_accountability",
                severity="critical" if "oversight" in str(issues) else "major",
                description=f"Accountability violations: {'; '.join(issues)}",
                confidence=0.9,
                recommendation="Ensure clear accountability chain and traceability"
            )
        
        return None
    
    async def _check_consistency_compliance(self, action: Dict[str, Any], 
                                          context: Optional[Dict[str, Any]]) -> Optional[ConstitutionalViolation]:
        """Check consistency principle compliance."""
        # This would involve checking against historical similar decisions
        # For now, implement basic consistency checks
        issues = []
        
        # Check for contradictory actions
        if action.get("contradicts_previous_decision"):
            if not action.get("contradiction_justification"):
                issues.append("Action contradicts previous decision without justification")
        
        if issues:
            return ConstitutionalViolation(
                principle="consistency",
                violation_type="inconsistent_decision",
                severity="major",
                description=f"Consistency violations: {'; '.join(issues)}",
                confidence=0.6,
                recommendation="Provide justification for deviation from precedent"
            )
        
        return None
    
    async def _check_harm_prevention_compliance(self, action: Dict[str, Any], 
                                              context: Optional[Dict[str, Any]]) -> Optional[ConstitutionalViolation]:
        """Check harm prevention principle compliance."""
        issues = []
        
        # Check for potential harm indicators
        risk_level = action.get("risk_level", "unknown")
        if risk_level == "high" or risk_level == "critical":
            if not action.get("risk_mitigation_plan"):
                issues.append("High-risk action without mitigation plan")
        
        # Check if action is reversible for high-risk situations
        if risk_level in ["high", "critical"]:
            if not action.get("reversible", False) and not action.get("harm_safeguards"):
                issues.append("High-risk irreversible action without safeguards")
        
        # Check for explicit harm indicators
        if action.get("potential_harm"):
            if not action.get("harm_justification"):
                issues.append("Action may cause harm without justification")
        
        if issues:
            return ConstitutionalViolation(
                principle="harm_prevention",
                violation_type="potential_harm",
                severity="critical" if "without safeguards" in str(issues) else "major",
                description=f"Harm prevention violations: {'; '.join(issues)}",
                confidence=0.85,
                recommendation="Implement safeguards and risk mitigation measures"
            )
        
        return None
    
    async def _calculate_compliance_score(self, violations: List[ConstitutionalViolation]) -> float:
        """
        Calculate overall compliance score based on violations.
        Score ranges from 0.0 (completely non-compliant) to 1.0 (fully compliant).
        """
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            "minor": 0.1,
            "major": 0.3,
            "critical": 0.6
        }
        
        total_penalty = 0.0
        max_possible_penalty = 0.0
        
        for violation in violations:
            penalty = severity_weights.get(violation.severity, 0.3)
            # Weight by confidence in the violation
            penalty *= violation.confidence
            total_penalty += penalty
            max_possible_penalty += 0.6  # Maximum critical penalty
        
        # Normalize to [0, 1] range
        if max_possible_penalty == 0:
            return 1.0
        
        compliance_score = max(0.0, 1.0 - (total_penalty / max_possible_penalty))
        return compliance_score
    
    async def _log_violations(self, result: ConstitutionalValidationResult, action: Dict[str, Any]):
        """Log constitutional violations to immutable logs."""
        if not self.immutable_logs:
            return
        
        for violation in result.violations:
            await self.immutable_logs.log_constitutional_violation(
                violation_data={
                    "validation_id": result.validation_id,
                    "principle": violation.principle,
                    "violation_type": violation.violation_type,
                    "severity": violation.severity,
                    "description": violation.description,
                    "confidence": violation.confidence,
                    "recommendation": violation.recommendation,
                    "action_summary": {
                        "type": action.get("type", "unknown"),
                        "id": action.get("id", "unknown"),
                        "component": action.get("component_id", "unknown")
                    }
                },
                component_id="constitutional_validator"
            )
    
    def get_principle_definitions(self) -> Dict[str, Any]:
        """Get all constitutional principle definitions."""
        return self.constitutional_principles.copy()
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics (would be enhanced with actual tracking)."""
        return {
            "principles_count": len(self.constitutional_principles),
            "min_compliance_score": self.min_compliance_score,
            "enforcement_enabled": self.config["infrastructure_config"]["constitutional_enforcement"]
        }