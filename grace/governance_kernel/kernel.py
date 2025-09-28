"""Governance kernel - main orchestration for policy evaluation and decision making."""
from typing import Optional

from ..contracts.governed_request import GovernedRequest
from ..contracts.governed_decision import GovernedDecision
from .policy_engine import PolicyEngine
from .verification_bridge import VerificationBridge
from .quorum_bridge import QuorumBridge
from .synthesizer import Synthesizer


class GovernanceKernel:
    """Main governance kernel orchestrating policy evaluation and decision making."""
    
    def __init__(self, mtl_kernel=None, intelligence_kernel=None):
        self.mtl_kernel = mtl_kernel
        self.policy_engine = PolicyEngine()
        self.verification_bridge = VerificationBridge()
        self.quorum_bridge = QuorumBridge(intelligence_kernel)
        self.synthesizer = Synthesizer()
    
    def evaluate(self, request: GovernedRequest) -> GovernedDecision:
        """
        Main governance evaluation pipeline:
        1) policy_engine.check(request)
        2) verification_bridge.verify(request)
        3) feed = mtl.feed_for_quorum(filters_from(request))
        4) result = quorum_bridge.consensus(feed)
        5) decision = synthesizer.merge(request, results)
        6) mtl.store_decision(decision)
        """
        
        # Step 1: Policy evaluation
        policy_results = self.policy_engine.check(request)
        
        # Step 2: Request verification
        verification_result = self.verification_bridge.verify(request)
        
        # Step 3 & 4: Quorum consensus (if required)
        quorum_consensus = None
        if request.requires_quorum and self.mtl_kernel:
            # Generate filters from request for MTL feed
            filters = self._filters_from_request(request)
            feed_ids = self.mtl_kernel.feed_for_quorum(filters)
            quorum_consensus = self.quorum_bridge.consensus(feed_ids, context={"request": request.dict()})
        
        # Step 5: Synthesize final decision
        decision = self.synthesizer.merge(
            request=request,
            policy_results=policy_results,
            verification_result=verification_result,
            quorum_consensus=quorum_consensus
        )
        
        # Step 6: Store decision in MTL
        if self.mtl_kernel:
            try:
                self.mtl_kernel.store_decision(decision)
            except Exception as e:
                # Log error but don't fail the decision
                pass
        
        return decision
    
    def _filters_from_request(self, request: GovernedRequest) -> dict:
        """Generate MTL query filters from governance request."""
        filters = {}
        
        # Filter by request type
        if request.request_type:
            filters["tags"] = [request.request_type]
        
        # Filter by policy domains
        if request.policy_domains:
            filters["tags"] = filters.get("tags", []) + request.policy_domains
        
        # Filter by risk level
        if request.risk_level:
            filters["risk_level"] = request.risk_level
        
        return filters
    
    def get_stats(self) -> dict:
        """Get governance kernel statistics."""
        policy_count = len(self.policy_engine.policies)
        
        return {
            "active_policies": policy_count,
            "verification_methods": len(self.verification_bridge.verification_methods),
            "quorum_enabled": bool(self.quorum_bridge.intelligence_kernel),
            "mtl_connected": bool(self.mtl_kernel),
            "components": {
                "policy_engine": True,
                "verification_bridge": True,
                "quorum_bridge": True,
                "synthesizer": True
            }
        }
    
    def set_mtl_kernel(self, mtl_kernel):
        """Set the MTL kernel for decision storage and feed generation."""
        self.mtl_kernel = mtl_kernel
    
    def set_intelligence_kernel(self, intelligence_kernel):
        """Set the intelligence kernel for quorum operations."""
        self.quorum_bridge.set_intelligence_kernel(intelligence_kernel)