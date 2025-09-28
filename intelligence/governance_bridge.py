"""
Governance Bridge - Policy validation and approval integration.

Handles:
1. Integration with Grace Governance Kernel
2. Policy validation for plans and results
3. Approval workflows for high-risk operations
4. Compliance checking and audit trail
"""
import logging
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GovernanceBridge:
    """Bridge to Grace Governance Kernel for policy enforcement."""
    
    def __init__(self, governance_kernel=None):
        self.governance_kernel = governance_kernel
        self.policy_cache: Dict[str, Dict] = {}
        self.approval_history: list[Dict] = []
        
        logger.info("Governance Bridge initialized")
    
    def request_approval(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request governance approval for execution plan.
        
        Args:
            plan: Execution plan requiring approval
            
        Returns:
            Approval decision with rationale
        """
        try:
            # Check if governance kernel is available
            if not self.governance_kernel:
                # Fallback: basic policy check without governance kernel
                return self._basic_policy_check(plan)
            
            # Prepare governance request
            governance_request = self._prepare_governance_request(plan)
            
            # Submit to governance kernel
            decision = self.governance_kernel.evaluate(governance_request)
            
            # Process governance decision
            approval_result = self._process_governance_decision(decision, plan)
            
            # Log approval history
            self._log_approval(plan, approval_result)
            
            return approval_result
            
        except Exception as e:
            logger.error(f"Governance approval request failed: {e}")
            return {
                "approved": False,
                "reason": f"Governance system error: {str(e)}",
                "fallback": True
            }
    
    def validate_policy(self, result: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inference result against policy requirements.
        
        Args:
            result: Inference result to validate
            policy: Policy requirements
            
        Returns:
            Validation result with compliance status
        """
        try:
            validation_result = {
                "compliant": True,
                "violations": [],
                "warnings": [],
                "policy_version": "1.0.0"
            }
            
            # Check confidence threshold
            min_confidence = policy.get("min_confidence", 0.5)
            actual_confidence = result.get("outputs", {}).get("confidence", 0.0)
            
            if actual_confidence < min_confidence:
                validation_result["compliant"] = False
                validation_result["violations"].append({
                    "type": "confidence_threshold",
                    "required": min_confidence,
                    "actual": actual_confidence,
                    "severity": "error"
                })
            
            # Check calibration requirement
            min_calibration = policy.get("min_calibration", 0.9)
            # Mock calibration check (in real system, would compute actual calibration)
            estimated_calibration = actual_confidence * 0.95
            
            if estimated_calibration < min_calibration:
                validation_result["compliant"] = False
                validation_result["violations"].append({
                    "type": "calibration_threshold", 
                    "required": min_calibration,
                    "actual": estimated_calibration,
                    "severity": "error"
                })
            
            # Check fairness constraints
            max_fairness_delta = policy.get("fairness_delta_max", 0.1)
            # Mock fairness check
            estimated_fairness_delta = 0.01  # Mock low delta
            
            if estimated_fairness_delta > max_fairness_delta:
                validation_result["compliant"] = False
                validation_result["violations"].append({
                    "type": "fairness_delta",
                    "required_max": max_fairness_delta,
                    "actual": estimated_fairness_delta,
                    "severity": "error"
                })
            
            # Check explanation requirements
            if policy.get("require_explanations", False):
                if not result.get("explanations"):
                    validation_result["warnings"].append({
                        "type": "missing_explanations",
                        "message": "Explanations required but not provided",
                        "severity": "warning"
                    })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Policy validation failed: {e}")
            return {
                "compliant": False,
                "violations": [{"type": "validation_error", "message": str(e)}],
                "error": True
            }
    
    def _prepare_governance_request(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request for governance kernel evaluation."""
        route = plan.get("route", {})
        policy = plan.get("policy", {})
        
        # Create governance request structure
        governance_request = {
            "claims": [
                {
                    "id": f"intel_plan_{plan.get('plan_id', 'unknown')}",
                    "statement": f"Intelligence plan execution with {len(route.get('models', []))} models",
                    "sources": [
                        {
                            "uri": f"intelligence://plan/{plan.get('plan_id')}",
                            "credibility": 0.9
                        }
                    ],
                    "evidence": [
                        {
                            "type": "plan",
                            "pointer": plan.get("plan_id"),
                            "content": {
                                "models": route.get("models", []),
                                "ensemble": route.get("ensemble", "none"),
                                "canary_pct": route.get("canary_pct", 0),
                                "shadow": route.get("shadow", False)
                            }
                        }
                    ],
                    "confidence": 0.85,
                    "logical_chain": [
                        {
                            "step": "Task routing completed successfully"
                        },
                        {
                            "step": "Model selection validated"
                        },
                        {
                            "step": f"Policy requirements: confidence >= {policy.get('min_confidence', 0.75)}"
                        }
                    ]
                }
            ],
            "context": {
                "decision_type": "intelligence_plan_approval",
                "urgency": "normal",
                "risk_level": self._assess_risk_level(plan),
                "environment": route.get("env", "unknown")
            }
        }
        
        return governance_request
    
    def _assess_risk_level(self, plan: Dict[str, Any]) -> str:
        """Assess risk level of execution plan."""
        route = plan.get("route", {})
        policy = plan.get("policy", {})
        
        # High risk conditions
        if (
            route.get("canary_pct", 0) > 50 or  # High canary percentage
            len(route.get("models", [])) > 10 or  # Many models
            policy.get("min_confidence", 1.0) < 0.5 or  # Very low confidence threshold
            route.get("env") == "prod"  # Production environment
        ):
            return "high"
        
        # Medium risk conditions
        elif (
            route.get("canary_pct", 0) > 10 or
            len(route.get("models", [])) > 3 or
            route.get("shadow", False)
        ):
            return "medium"
        
        return "low"
    
    def _process_governance_decision(self, decision: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Process governance kernel decision."""
        try:
            # Extract decision status
            # Governance kernel returns evaluation results
            decision_status = decision.get("decision_status", "unknown")
            approved = decision_status == "approved"
            
            # Extract rationale
            rationale = decision.get("rationale", "No rationale provided")
            
            # Create approval result
            approval_result = {
                "approved": approved,
                "reason": rationale,
                "governance_decision_id": decision.get("decision_id"),
                "policy_violations": decision.get("violations", []),
                "confidence": decision.get("confidence", 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional context if approval failed
            if not approved:
                failure_modes = decision.get("failure_modes", [])
                approval_result["failure_modes"] = failure_modes
                approval_result["remediation_suggestions"] = self._get_remediation_suggestions(failure_modes)
            
            return approval_result
            
        except Exception as e:
            logger.error(f"Error processing governance decision: {e}")
            return {
                "approved": False,
                "reason": f"Decision processing error: {str(e)}",
                "error": True
            }
    
    def _basic_policy_check(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Basic policy check when governance kernel is unavailable."""
        route = plan.get("route", {})
        policy = plan.get("policy", {})
        
        # Basic safety checks
        violations = []
        
        # Check for excessive canary deployment
        canary_pct = route.get("canary_pct", 0)
        if canary_pct > 75:
            violations.append("Canary percentage too high (>75%)")
        
        # Check for too many models
        num_models = len(route.get("models", []))
        if num_models > 15:
            violations.append(f"Too many models ({num_models} > 15)")
        
        # Check confidence threshold
        min_confidence = policy.get("min_confidence", 0.75)
        if min_confidence < 0.3:
            violations.append(f"Confidence threshold too low ({min_confidence} < 0.3)")
        
        # Approval decision
        approved = len(violations) == 0
        
        return {
            "approved": approved,
            "reason": "; ".join(violations) if violations else "Basic policy checks passed",
            "violations": violations,
            "fallback_mode": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_remediation_suggestions(self, failure_modes: list[str]) -> list[str]:
        """Get remediation suggestions for policy failures."""
        suggestions = []
        
        for failure_mode in failure_modes:
            if "confidence" in failure_mode.lower():
                suggestions.append("Consider using more confident models or ensemble methods")
            elif "calibration" in failure_mode.lower():
                suggestions.append("Recalibrate models or apply calibration techniques")
            elif "fairness" in failure_mode.lower():
                suggestions.append("Review model bias and apply fairness constraints")
            elif "canary" in failure_mode.lower():
                suggestions.append("Reduce canary deployment percentage")
            else:
                suggestions.append("Review and adjust plan parameters")
        
        return suggestions
    
    def _log_approval(self, plan: Dict[str, Any], approval_result: Dict[str, Any]):
        """Log approval decision for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "plan_id": plan.get("plan_id"),
            "req_id": plan.get("req_id"),
            "approved": approval_result["approved"],
            "reason": approval_result["reason"],
            "risk_level": self._assess_risk_level(plan),
            "governance_mode": "fallback" if approval_result.get("fallback_mode") else "full"
        }
        
        self.approval_history.append(log_entry)
        
        # Keep only recent history (last 1000 approvals)
        if len(self.approval_history) > 1000:
            self.approval_history = self.approval_history[-1000:]
        
        logger.info(f"Approval logged: Plan {plan.get('plan_id')} - {'APPROVED' if approval_result['approved'] else 'REJECTED'}")
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval statistics for monitoring."""
        if not self.approval_history:
            return {"message": "No approval history available"}
        
        recent_approvals = self.approval_history[-100:]  # Last 100 approvals
        
        total_requests = len(recent_approvals)
        approved_count = sum(1 for entry in recent_approvals if entry["approved"])
        approval_rate = approved_count / total_requests if total_requests > 0 else 0
        
        # Risk level distribution
        risk_distribution = {}
        for entry in recent_approvals:
            risk_level = entry.get("risk_level", "unknown")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            "total_requests": total_requests,
            "approval_rate": approval_rate,
            "approved_count": approved_count,
            "rejected_count": total_requests - approved_count,
            "risk_distribution": risk_distribution,
            "governance_connection": self.governance_kernel is not None
        }
    
    def set_governance_kernel(self, governance_kernel):
        """Set or update governance kernel instance."""
        self.governance_kernel = governance_kernel
        logger.info("Governance kernel connection updated")
    
    def clear_policy_cache(self):
        """Clear policy cache to force refresh."""
        self.policy_cache.clear()
        logger.info("Policy cache cleared")