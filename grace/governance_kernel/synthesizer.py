"""Synthesizer - merges governance results into final decisions."""
from typing import Dict, List, Optional

from ..contracts.governed_request import GovernedRequest
from ..contracts.governed_decision import GovernedDecision
from .types import PolicyResult, VerificationResult, QuorumConsensus


class Synthesizer:
    """Synthesizes policy, verification, and quorum results into governance decisions."""
    
    def merge(self, 
              request: GovernedRequest,
              policy_results: List[PolicyResult],
              verification_result: VerificationResult,
              quorum_consensus: Optional[QuorumConsensus] = None) -> GovernedDecision:
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
        execution_approved = approved and policy_analysis["all_passed"] and verification_result.verified
        
        # Generate conditions if needed
        conditions = self._generate_conditions(policy_analysis, verification_result)
        
        return GovernedDecision(
            request_id=request.id,
            approved=approved,
            confidence=confidence,
            reasoning=reasoning,
            policy_results={
                "total_policies": len(policy_results),
                "passed_policies": policy_analysis["passed_count"],
                "failed_policies": policy_analysis["failed_count"],
                "details": [
                    {
                        "policy_id": pr.policy_id,
                        "passed": pr.passed,
                        "confidence": pr.confidence,
                        "reasoning": pr.reasoning
                    } for pr in policy_results
                ]
            },
            verification_results={
                "verified": verification_result.verified,
                "confidence": verification_result.confidence,
                "method": verification_result.verification_method,
                "details": verification_result.details
            },
            quorum_results=self._format_quorum_results(quorum_consensus) if quorum_consensus else None,
            execution_approved=execution_approved,
            conditions=conditions
        )
    
    def _analyze_policies(self, policy_results: List[PolicyResult]) -> Dict:
        """Analyze policy results."""
        total = len(policy_results)
        passed = sum(1 for pr in policy_results if pr.passed)
        failed = total - passed
        
        # Calculate weighted confidence
        if total > 0:
            total_confidence = sum(pr.confidence for pr in policy_results)
            avg_confidence = total_confidence / total
        else:
            avg_confidence = 0.0
        
        # Check for critical failures
        critical_failures = [
            pr for pr in policy_results 
            if not pr.passed and pr.policy_type.value in ["content_filter", "access_control"]
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
                           verification_result: VerificationResult,
                           quorum_consensus: Optional[QuorumConsensus]) -> bool:
        """Determine overall approval based on all factors."""
        
        # Must pass verification
        if not verification_result.verified:
            return False
        
        # Must not have critical policy failures
        if policy_analysis["has_critical_failures"]:
            return False
        
        # If quorum was required, must have consensus
        if quorum_consensus and not quorum_consensus.consensus_reached:
            return False
        
        # General policy compliance (allow some non-critical failures)
        policy_pass_rate = policy_analysis["passed_count"] / max(1, policy_analysis["total_count"])
        if policy_pass_rate < 0.7:  # Require 70% policy compliance
            return False
        
        return True
    
    def _calculate_confidence(self,
                            policy_analysis: Dict,
                            verification_result: VerificationResult,
                            quorum_consensus: Optional[QuorumConsensus]) -> float:
        """Calculate overall confidence in the decision."""
        components = []
        
        # Policy confidence
        components.append(policy_analysis["avg_confidence"] * 0.4)
        
        # Verification confidence
        components.append(verification_result.confidence * 0.3)
        
        # Quorum confidence (if available)
        if quorum_consensus:
            components.append(quorum_consensus.confidence * 0.3)
        else:
            # Boost other components if no quorum needed
            components = [c * 1.2 for c in components]
        
        return min(1.0, sum(components))
    
    def _generate_reasoning(self,
                           approved: bool,
                           policy_analysis: Dict,
                           verification_result: VerificationResult,
                           quorum_consensus: Optional[QuorumConsensus]) -> str:
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
        if verification_result.verified:
            parts.append("Request verification successful.")
        else:
            parts.append("Request verification failed.")
        
        # Quorum reasoning
        if quorum_consensus:
            if quorum_consensus.consensus_reached:
                parts.append(f"Quorum consensus reached ({quorum_consensus.agreement_level:.1%} agreement).")
            else:
                parts.append("Quorum consensus not reached.")
        
        return " ".join(parts)
    
    def _generate_conditions(self,
                           policy_analysis: Dict,
                           verification_result: VerificationResult) -> List[str]:
        """Generate execution conditions if needed."""
        conditions = []
        
        # Add conditions based on non-critical policy failures
        non_critical_failures = [
            pr for pr in policy_analysis.get("failed_policies", [])
            if pr.policy_type.value not in ["content_filter", "access_control"]
        ]
        
        if non_critical_failures:
            conditions.append("Monitor for compliance with failed non-critical policies")
        
        if verification_result.confidence < 0.8:
            conditions.append("Additional verification may be required")
        
        return conditions
    
    def _format_quorum_results(self, quorum_consensus: QuorumConsensus) -> Dict:
        """Format quorum results for decision record."""
        return {
            "consensus_reached": quorum_consensus.consensus_reached,
            "agreement_level": quorum_consensus.agreement_level,
            "participant_count": len(quorum_consensus.participating_memories),
            "consensus_view": quorum_consensus.consensus_view,
            "confidence": quorum_consensus.confidence,
            "dissenting_views_count": len(quorum_consensus.dissenting_views)
        }