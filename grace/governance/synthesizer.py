"""Synthesizer - merges governance results into final decisions."""
from typing import Dict, List, Optional, Any
import logging

# Try to import the contracts for compatibility
try:
    from ..contracts.governed_decision import GovernedDecision
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class Synthesizer:
    """Synthesizes policy, verification, and quorum results into governance decisions."""
    
    def merge(self, 
              request,
              policy_results: List[Dict[str, Any]],
              verification_result: Dict[str, Any],
              quorum_consensus: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge all governance results into a final decision."""
        
        # Analyze policy results
        policy_analysis = self._analyze_policies(policy_results)
        
        # Determine overall approval
        approved = self._determine_approval(
            policy_analysis, verification_result, quorum_consensus
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            policy_analysis, verification_result, quorum_consensus
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            approved, policy_analysis, verification_result, quorum_consensus
        )
        
        # Determine execution approval (more restrictive)
        execution_approved = approved and policy_analysis["all_passed"] and verification_result.get("verified", False)
        
        # Generate conditions if needed
        conditions = self._generate_conditions(policy_analysis, verification_result)
        
        # Extract request ID
        request_id = self._extract_request_id(request)
        
        decision_data = {
            "request_id": request_id,
            "approved": approved,
            "confidence": confidence,
            "reasoning": reasoning,
            "policy_results": {
                "total_policies": len(policy_results),
                "passed_policies": policy_analysis["passed_count"],
                "failed_policies": policy_analysis["failed_count"],
                "details": policy_results
            },
            "verification_results": verification_result,
            "quorum_results": quorum_consensus,
            "execution_approved": execution_approved,
            "conditions": conditions,
            "decision_maker": "grace-governance"
        }
        
        # Return Pydantic model if available for compatibility, otherwise dict
        if PYDANTIC_AVAILABLE:
            try:
                return GovernedDecision(**decision_data)
            except Exception as e:
                logger.warning(f"Failed to create GovernedDecision model: {e}, returning dict")
                return decision_data
        else:
            return decision_data
    
    def _extract_request_id(self, request) -> str:
        """Extract request ID from request regardless of format."""
        if hasattr(request, 'id'):
            return str(request.id)
        elif hasattr(request, 'request_id'):
            return str(request.request_id)
        elif isinstance(request, dict):
            return str(request.get('id', request.get('request_id', 'unknown')))
        else:
            return 'unknown'
    
    def _analyze_policies(self, policy_results: List[Dict[str, Any]]) -> Dict:
        """Analyze policy results."""
        total = len(policy_results)
        passed = sum(1 for pr in policy_results if pr.get("passed", False))
        failed = total - passed
        
        # Calculate weighted confidence
        if total > 0:
            total_confidence = sum(pr.get("confidence", 0.0) for pr in policy_results)
            avg_confidence = total_confidence / total
        else:
            avg_confidence = 0.0
        
        # Check for critical failures
        critical_failures = [
            pr for pr in policy_results 
            if not pr.get("passed", False) and pr.get("policy_type") in ["content_filter", "access_control"]
        ]
        
        return {
            "total_count": total,
            "passed_count": passed,
            "failed_count": failed,
            "all_passed": failed == 0,
            "avg_confidence": avg_confidence,
            "critical_failures": critical_failures,
            "has_critical_failures": len(critical_failures) > 0
        }
    
    def _determine_approval(self, 
                           policy_analysis: Dict,
                           verification_result: Dict[str, Any],
                           quorum_consensus: Optional[Dict[str, Any]]) -> bool:
        """Determine overall approval based on all factors."""
        
        # Must pass verification
        if not verification_result.get("verified", False):
            return False
        
        # Must not have critical policy failures
        if policy_analysis["has_critical_failures"]:
            return False
        
        # If quorum was required, must have consensus
        if quorum_consensus and not quorum_consensus.get("consensus_reached", False):
            return False
        
        # General policy compliance (allow some non-critical failures)
        policy_pass_rate = policy_analysis["passed_count"] / max(1, policy_analysis["total_count"])
        if policy_pass_rate < 0.7:  # Require 70% policy compliance
            return False
        
        return True
    
    def _calculate_confidence(self,
                            policy_analysis: Dict,
                            verification_result: Dict[str, Any],
                            quorum_consensus: Optional[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the decision."""
        components = []
        
        # Policy confidence
        components.append(policy_analysis["avg_confidence"] * 0.4)
        
        # Verification confidence
        components.append(verification_result.get("confidence", 0.0) * 0.3)
        
        # Quorum confidence (if available)
        if quorum_consensus:
            components.append(quorum_consensus.get("confidence", 0.0) * 0.3)
        else:
            # Boost other components if no quorum needed
            components = [c * 1.2 for c in components]
        
        return min(1.0, sum(components))
    
    def _generate_reasoning(self,
                           approved: bool,
                           policy_analysis: Dict,
                           verification_result: Dict[str, Any],
                           quorum_consensus: Optional[Dict[str, Any]]) -> str:
        """Generate human-readable reasoning for the decision."""
        parts = []
        
        if approved:
            parts.append("Request approved.")
        else:
            parts.append("Request denied.")
        
        # Policy reasoning
        if policy_analysis["has_critical_failures"]:
            parts.append("Critical policy violations detected.")
        elif policy_analysis["all_passed"]:
            parts.append("All policies passed.")
        else:
            pass_rate = policy_analysis["passed_count"] / max(1, policy_analysis["total_count"])
            parts.append(f"{int(pass_rate * 100)}% of policies passed.")
        
        # Verification reasoning
        if verification_result.get("verified", False):
            parts.append("Request verification successful.")
        else:
            parts.append("Request verification failed.")
        
        # Quorum reasoning
        if quorum_consensus:
            if quorum_consensus.get("consensus_reached", False):
                agreement_level = quorum_consensus.get("agreement_level", 0.0)
                parts.append(f"Quorum consensus reached ({agreement_level:.1%} agreement).")
            else:
                parts.append("Quorum consensus not reached.")
        
        return " ".join(parts)
    
    def _generate_conditions(self,
                           policy_analysis: Dict,
                           verification_result: Dict[str, Any]) -> List[str]:
        """Generate execution conditions if needed."""
        conditions = []
        
        # Add conditions based on non-critical policy failures
        if policy_analysis["failed_count"] > 0 and not policy_analysis["has_critical_failures"]:
            conditions.append("Monitor for compliance with failed non-critical policies")
        
        if verification_result.get("confidence", 0.0) < 0.8:
            conditions.append("Additional verification may be required")
        
        return conditions