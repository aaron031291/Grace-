"""
Governance Liaison Specialist - Interface between ML/DL specialists and governance system.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from ..quorum import SpecialistModel, SpecialistType, SpecialistOutput


logger = logging.getLogger(__name__)


class GovernanceLiaisonSpecialist(SpecialistModel):
    """
    Specialist that ensures ML/DL model outputs comply with governance policies
    and constitutional principles before deployment decisions.
    """
    
    def __init__(self, specialist_id: str = "governance_liaison"):
        super().__init__(specialist_id, SpecialistType.GOVERNANCE_LIAISON, confidence_threshold=0.8)
        
        # Governance compliance rules
        self.compliance_rules = {
            "fairness_minimum": 0.8,
            "privacy_minimum": 0.9,
            "security_minimum": 0.9,
            "transparency_minimum": 0.7,
            "constitutional_minimum": 0.85
        }
        
        # Model deployment safety checks
        self.safety_checks = [
            "bias_assessment",
            "privacy_audit",
            "security_scan",
            "performance_validation",
            "constitutional_review"
        ]
    
    async def _generate_prediction(self, inputs: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Governance compliance assessment for ML/DL model outputs.
        
        Args:
            inputs: Contains specialist outputs and governance context
            
        Returns:
            Compliance decision and confidence
        """
        try:
            # Extract specialist outputs
            specialist_outputs = inputs.get("specialist_outputs", [])
            governance_context = inputs.get("governance_context", {})
            deployment_type = inputs.get("deployment_type", "standard")
            
            # Perform governance checks
            compliance_results = await self._assess_governance_compliance(
                specialist_outputs, governance_context, deployment_type
            )
            
            # Make deployment recommendation
            recommendation = await self._make_deployment_recommendation(compliance_results)
            
            # Calculate confidence based on compliance scores
            confidence = self._calculate_governance_confidence(compliance_results)
            
            return {
                "deployment_recommendation": recommendation,
                "compliance_results": compliance_results,
                "safety_score": compliance_results.get("overall_safety", 0.5)
            }, confidence
            
        except Exception as e:
            logger.error(f"Governance liaison prediction error: {e}")
            return {
                "deployment_recommendation": "reject",
                "compliance_results": {"error": str(e)},
                "safety_score": 0.0
            }, 0.0
    
    async def _assess_governance_compliance(self, specialist_outputs: List[Dict[str, Any]],
                                          governance_context: Dict[str, Any],
                                          deployment_type: str) -> Dict[str, Any]:
        """Assess governance compliance across all dimensions."""
        compliance_results = {
            "fairness_assessment": await self._assess_fairness(specialist_outputs),
            "privacy_assessment": await self._assess_privacy(specialist_outputs, governance_context),
            "security_assessment": await self._assess_security(specialist_outputs, governance_context),
            "transparency_assessment": await self._assess_transparency(specialist_outputs),
            "constitutional_assessment": await self._assess_constitutional_compliance(
                specialist_outputs, governance_context
            ),
            "deployment_type": deployment_type
        }
        
        # Calculate overall compliance score
        compliance_scores = [
            compliance_results["fairness_assessment"]["score"],
            compliance_results["privacy_assessment"]["score"],
            compliance_results["security_assessment"]["score"],
            compliance_results["transparency_assessment"]["score"],
            compliance_results["constitutional_assessment"]["score"]
        ]
        
        compliance_results["overall_compliance"] = sum(compliance_scores) / len(compliance_scores)
        compliance_results["passed_all_minimums"] = all(
            score >= self.compliance_rules.get(f"{category}_minimum", 0.5)
            for category, score in zip(
                ["fairness", "privacy", "security", "transparency", "constitutional"],
                compliance_scores
            )
        )
        
        return compliance_results
    
    async def _assess_fairness(self, specialist_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess fairness and bias concerns."""
        fairness_scores = []
        bias_indicators = []
        
        for output in specialist_outputs:
            specialist_type = output.get("specialist_type", "")
            prediction = output.get("prediction", {})
            
            # Look for fairness-specific outputs
            if "fairness" in specialist_type:
                if isinstance(prediction, dict) and "fairness_score" in prediction:
                    fairness_scores.append(prediction["fairness_score"])
                
                if isinstance(prediction, dict) and "bias_indicators" in prediction:
                    bias_indicators.extend(prediction["bias_indicators"])
            
            # Check for bias in other predictions
            elif isinstance(prediction, dict):
                confidence = output.get("confidence", 0.5)
                # High confidence with extreme predictions might indicate bias
                if confidence > 0.9 and isinstance(prediction.get("score"), (int, float)):
                    score = prediction["score"]
                    if score < 0.1 or score > 0.9:
                        bias_indicators.append(f"Extreme prediction from {specialist_type}")
        
        # Calculate fairness score
        if fairness_scores:
            fairness_score = sum(fairness_scores) / len(fairness_scores)
        else:
            # Default assessment based on bias indicators
            fairness_score = max(0.5, 1.0 - len(bias_indicators) * 0.1)
        
        return {
            "score": fairness_score,
            "bias_indicators": bias_indicators,
            "fairness_specialists_consulted": len(fairness_scores),
            "passes_minimum": fairness_score >= self.compliance_rules["fairness_minimum"]
        }
    
    async def _assess_privacy(self, specialist_outputs: List[Dict[str, Any]],
                            governance_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy protection measures."""
        privacy_scores = []
        privacy_violations = []
        
        # Check for privacy-specific assessments
        for output in specialist_outputs:
            specialist_type = output.get("specialist_type", "")
            prediction = output.get("prediction", {})
            
            if "privacy" in specialist_type or "security" in specialist_type:
                if isinstance(prediction, dict) and "privacy_score" in prediction:
                    privacy_scores.append(prediction["privacy_score"])
                
                if isinstance(prediction, dict) and "privacy_violations" in prediction:
                    privacy_violations.extend(prediction["privacy_violations"])
        
        # Check governance context for privacy requirements
        privacy_requirements = governance_context.get("privacy_requirements", [])
        data_sensitivity = governance_context.get("data_sensitivity", "standard")
        
        # Calculate privacy score
        base_privacy_score = sum(privacy_scores) / len(privacy_scores) if privacy_scores else 0.7
        
        # Adjust based on data sensitivity
        if data_sensitivity == "high":
            base_privacy_score *= 0.8  # Higher bar for sensitive data
        elif data_sensitivity == "low":
            base_privacy_score *= 1.1  # Slightly more lenient
        
        # Penalize for violations
        privacy_score = max(0.0, base_privacy_score - len(privacy_violations) * 0.1)
        
        return {
            "score": privacy_score,
            "privacy_violations": privacy_violations,
            "data_sensitivity": data_sensitivity,
            "privacy_requirements_met": len(privacy_requirements),
            "passes_minimum": privacy_score >= self.compliance_rules["privacy_minimum"]
        }
    
    async def _assess_security(self, specialist_outputs: List[Dict[str, Any]],
                             governance_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security measures and vulnerabilities."""
        security_scores = []
        security_risks = []
        
        for output in specialist_outputs:
            specialist_type = output.get("specialist_type", "")
            prediction = output.get("prediction", {})
            
            if "security" in specialist_type or "privacy" in specialist_type:
                if isinstance(prediction, dict) and "security_score" in prediction:
                    security_scores.append(prediction["security_score"])
                
                if isinstance(prediction, dict) and "security_risks" in prediction:
                    security_risks.extend(prediction["security_risks"])
        
        # Check for anomalies that might indicate security issues
        for output in specialist_outputs:
            specialist_type = output.get("specialist_type", "")
            if "anomaly" in specialist_type:
                prediction = output.get("prediction", {})
                if isinstance(prediction, dict) and prediction.get("is_anomaly", False):
                    security_risks.append("Anomaly detected during assessment")
        
        # Calculate security score
        base_security_score = sum(security_scores) / len(security_scores) if security_scores else 0.6
        
        # Penalize for security risks
        security_score = max(0.0, base_security_score - len(security_risks) * 0.15)
        
        return {
            "score": security_score,
            "security_risks": security_risks,
            "security_specialists_consulted": len(security_scores),
            "passes_minimum": security_score >= self.compliance_rules["security_minimum"]
        }
    
    async def _assess_transparency(self, specialist_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess model transparency and explainability."""
        explainability_scores = []
        transparency_issues = []
        
        for output in specialist_outputs:
            specialist_type = output.get("specialist_type", "")
            prediction = output.get("prediction", {})
            metadata = output.get("metadata", {})
            
            # Check for explainability information
            if "explainability" in specialist_type or "fairness" in specialist_type:
                if isinstance(prediction, dict) and "explainability_score" in prediction:
                    explainability_scores.append(prediction["explainability_score"])
            
            # Check if predictions have explanations
            has_explanation = bool(
                metadata.get("explanation") or 
                (isinstance(prediction, dict) and prediction.get("explanation"))
            )
            
            if not has_explanation and output.get("confidence", 0) > 0.8:
                transparency_issues.append(f"High confidence prediction from {specialist_type} lacks explanation")
        
        # Calculate transparency score
        if explainability_scores:
            transparency_score = sum(explainability_scores) / len(explainability_scores)
        else:
            # Base score reduced by transparency issues
            transparency_score = max(0.3, 0.8 - len(transparency_issues) * 0.1)
        
        return {
            "score": transparency_score,
            "transparency_issues": transparency_issues,
            "explainability_specialists_consulted": len(explainability_scores),
            "passes_minimum": transparency_score >= self.compliance_rules["transparency_minimum"]
        }
    
    async def _assess_constitutional_compliance(self, specialist_outputs: List[Dict[str, Any]],
                                              governance_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with constitutional principles."""
        constitutional_scores = []
        violations = []
        
        # Check for constitutional principle adherence
        constitutional_principles = governance_context.get("constitutional_principles", {})
        
        for principle, required in constitutional_principles.items():
            if not required:
                continue
            
            principle_satisfied = False
            
            # Check if any specialist addressed this principle
            for output in specialist_outputs:
                prediction = output.get("prediction", {})
                if isinstance(prediction, dict):
                    principle_score = prediction.get(f"{principle}_compliance")
                    if principle_score is not None:
                        constitutional_scores.append(principle_score)
                        principle_satisfied = True
                        break
            
            if not principle_satisfied:
                violations.append(f"Principle '{principle}' not addressed by any specialist")
        
        # Calculate constitutional compliance score
        if constitutional_scores:
            constitutional_score = sum(constitutional_scores) / len(constitutional_scores)
        else:
            # Default score reduced by violations
            constitutional_score = max(0.2, 0.7 - len(violations) * 0.1)
        
        return {
            "score": constitutional_score,
            "constitutional_violations": violations,
            "principles_assessed": len(constitutional_scores),
            "passes_minimum": constitutional_score >= self.compliance_rules["constitutional_minimum"]
        }
    
    async def _make_deployment_recommendation(self, compliance_results: Dict[str, Any]) -> str:
        """Make final deployment recommendation based on compliance assessment."""
        overall_compliance = compliance_results["overall_compliance"]
        passed_all_minimums = compliance_results["passed_all_minimums"]
        
        # Critical failures
        security_score = compliance_results["security_assessment"]["score"]
        privacy_score = compliance_results["privacy_assessment"]["score"]
        constitutional_score = compliance_results["constitutional_assessment"]["score"]
        
        # Automatic reject conditions
        if security_score < 0.5 or privacy_score < 0.5:
            return "reject"
        
        if constitutional_score < 0.6:
            return "reject"
        
        if not passed_all_minimums:
            return "conditional"  # Needs review
        
        # Recommendation based on overall compliance
        if overall_compliance >= 0.9:
            return "approve"
        elif overall_compliance >= 0.8:
            return "approve_with_monitoring"
        elif overall_compliance >= 0.7:
            return "conditional"
        else:
            return "reject"
    
    def _calculate_governance_confidence(self, compliance_results: Dict[str, Any]) -> float:
        """Calculate confidence in the governance assessment."""
        overall_compliance = compliance_results["overall_compliance"]
        
        # Confidence factors
        confidence_factors = []
        
        # Base confidence from compliance score
        confidence_factors.append(overall_compliance)
        
        # Confidence from specialist coverage
        specialists_consulted = sum([
            compliance_results["fairness_assessment"]["fairness_specialists_consulted"],
            compliance_results["security_assessment"]["security_specialists_consulted"],
            len(compliance_results["privacy_assessment"].get("privacy_requirements_met", [])),
            compliance_results["transparency_assessment"]["explainability_specialists_consulted"],
            compliance_results["constitutional_assessment"]["principles_assessed"]
        ])
        
        coverage_confidence = min(1.0, specialists_consulted / 10.0)  # Ideal: 10+ assessments
        confidence_factors.append(coverage_confidence)
        
        # Penalty for violations or issues
        total_issues = len(
            compliance_results["fairness_assessment"]["bias_indicators"] +
            compliance_results["privacy_assessment"]["privacy_violations"] +
            compliance_results["security_assessment"]["security_risks"] +
            compliance_results["transparency_assessment"]["transparency_issues"] +
            compliance_results["constitutional_assessment"]["constitutional_violations"]
        )
        
        issue_penalty = min(0.5, total_issues * 0.05)  # Up to 0.5 penalty
        
        # Calculate final confidence
        base_confidence = sum(confidence_factors) / len(confidence_factors)
        final_confidence = max(0.1, base_confidence - issue_penalty)
        
        return final_confidence
    
    async def update_compliance_rules(self, new_rules: Dict[str, float]):
        """Update compliance rule thresholds."""
        for rule_name, threshold in new_rules.items():
            if rule_name in self.compliance_rules:
                old_threshold = self.compliance_rules[rule_name]
                self.compliance_rules[rule_name] = max(0.0, min(1.0, threshold))
                logger.info(f"Updated {rule_name} threshold: {old_threshold:.3f} -> {threshold:.3f}")
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get current compliance rules and statistics."""
        return {
            "compliance_rules": self.compliance_rules.copy(),
            "safety_checks": self.safety_checks.copy(),
            "confidence_threshold": self.confidence_threshold,
            "specialist_id": self.specialist_id,
            "specialist_type": self.specialist_type.value
        }