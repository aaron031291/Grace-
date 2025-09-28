"""
MLT-Governance Bridge - Packs proposals into governance events and handles approvals.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .contracts import AdaptationPlan, Action, ActionType


logger = logging.getLogger(__name__)


class GovernanceProposal:
    """A proposal to governance for MLT actions."""
    
    def __init__(self, proposal_id: str, plan: AdaptationPlan, priority: str = "normal"):
        self.proposal_id = proposal_id
        self.plan = plan
        self.priority = priority
        self.status = "pending"
        self.submitted_at = datetime.now()
        self.approved_at: Optional[datetime] = None
        self.rejection_reason: Optional[str] = None
    
    def to_governance_event(self) -> Dict[str, Any]:
        """Convert to governance event format."""
        return {
            "event_type": "ADAPTATION_PLAN_PROPOSED",
            "payload": {
                "proposal_id": self.proposal_id,
                "plan": self.plan.to_dict(),
                "priority": self.priority,
                "risk_assessment": self._assess_risk(),
                "governance_requirements": self._determine_governance_requirements()
            },
            "timestamp": self.submitted_at.isoformat()
        }
    
    def _assess_risk(self) -> Dict[str, Any]:
        """Assess risk level of the plan."""
        risk_score = 0.0
        risk_factors = []
        
        for action in self.plan.actions:
            if action.type == ActionType.HPO:
                risk_score += 0.3
                risk_factors.append("hyperparameter_optimization")
                
            elif action.type == ActionType.REWEIGHT_SPECIALISTS:
                risk_score += 0.2
                risk_factors.append("specialist_reweighting")
                
            elif action.type == ActionType.POLICY_DELTA:
                risk_score += 0.4
                risk_factors.append("policy_change")
                
            elif action.type == ActionType.CANARY:
                risk_score += 0.1
                risk_factors.append("canary_deployment")
        
        # Consider risk controls
        max_regret = self.plan.risk_controls.get("max_regret_pct", 2)
        if max_regret > 3:
            risk_score += 0.2
            risk_factors.append("high_regret_tolerance")
        
        return {
            "risk_score": min(1.0, risk_score),
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "risk_factors": risk_factors
        }
    
    def _determine_governance_requirements(self) -> Dict[str, Any]:
        """Determine what governance approval is needed."""
        requirements = {
            "approval_level": "automatic",
            "review_required": False,
            "constitutional_check": False,
            "parliament_review": False
        }
        
        # Check for policy changes
        has_policy_changes = any(action.type == ActionType.POLICY_DELTA for action in self.plan.actions)
        if has_policy_changes:
            requirements["approval_level"] = "governance_review"
            requirements["review_required"] = True
            requirements["constitutional_check"] = True
        
        # Check for high-risk actions
        risk_assessment = self._assess_risk()
        if risk_assessment["risk_level"] == "high":
            requirements["approval_level"] = "full_review"
            requirements["review_required"] = True
            requirements["constitutional_check"] = True
            requirements["parliament_review"] = True
        
        return requirements


class MLTGovernanceBridge:
    """Bridges MLT kernel with governance system for proposal approval."""
    
    def __init__(self, governance_engine=None, event_bus=None):
        self.governance_engine = governance_engine
        self.event_bus = event_bus
        self.pending_proposals: Dict[str, GovernanceProposal] = {}
        self.approved_proposals: Dict[str, GovernanceProposal] = {}
        self.rejected_proposals: Dict[str, GovernanceProposal] = {}
    
    async def submit_plan_for_approval(self, plan: AdaptationPlan, priority: str = "normal") -> str:
        """Submit an adaptation plan to governance for approval."""
        try:
            proposal_id = f"mlt_proposal_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            proposal = GovernanceProposal(proposal_id, plan, priority)
            self.pending_proposals[proposal_id] = proposal
            
            # Send to governance through event bus
            if self.event_bus:
                governance_event = proposal.to_governance_event()
                await self.event_bus.publish("ADAPTATION_PLAN_PROPOSED", governance_event)
                
                logger.info(f"Submitted plan {plan.plan_id} to governance as proposal {proposal_id}")
            
            return proposal_id
            
        except Exception as e:
            logger.error(f"Failed to submit plan to governance: {e}")
            raise
    
    async def handle_governance_decision(self, decision_event: Dict[str, Any]) -> bool:
        """Handle governance approval or rejection decision."""
        try:
            event_type = decision_event.get("event_type")
            payload = decision_event.get("payload", {})
            proposal_id = payload.get("proposal_id")
            
            if not proposal_id or proposal_id not in self.pending_proposals:
                logger.warning(f"Unknown proposal ID in governance decision: {proposal_id}")
                return False
            
            proposal = self.pending_proposals.pop(proposal_id)
            
            if event_type == "GOVERNANCE_APPROVED":
                proposal.status = "approved"
                proposal.approved_at = datetime.now()
                self.approved_proposals[proposal_id] = proposal
                
                # Apply the plan atomically
                await self._apply_approved_plan(proposal)
                
                logger.info(f"Governance approved proposal {proposal_id}")
                return True
                
            elif event_type == "GOVERNANCE_REJECTED":
                proposal.status = "rejected"
                proposal.rejection_reason = payload.get("rationale", "No reason provided")
                self.rejected_proposals[proposal_id] = proposal
                
                logger.info(f"Governance rejected proposal {proposal_id}: {proposal.rejection_reason}")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to handle governance decision: {e}")
            return False
    
    async def _apply_approved_plan(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Apply an approved adaptation plan atomically."""
        try:
            plan = proposal.plan
            applied_actions = []
            
            for action in plan.actions:
                result = await self._apply_action(action)
                applied_actions.append({
                    "action": action.to_dict(),
                    "result": result,
                    "applied_at": datetime.now().isoformat()
                })
            
            # Emit plan applied event
            if self.event_bus:
                await self.event_bus.publish("MLT_PLAN_APPLIED", {
                    "proposal_id": proposal.proposal_id,
                    "plan_id": plan.plan_id,
                    "applied_actions": applied_actions,
                    "snapshot_id": None  # Would be populated by snapshot manager
                })
            
            logger.info(f"Applied plan {plan.plan_id} with {len(applied_actions)} actions")
            return {"status": "applied", "actions": applied_actions}
            
        except Exception as e:
            logger.error(f"Failed to apply approved plan {proposal.plan.plan_id}: {e}")
            
            # Emit rollback request
            if self.event_bus:
                await self.event_bus.publish("ROLLBACK_REQUESTED", {
                    "reason": f"Plan application failed: {str(e)}",
                    "proposal_id": proposal.proposal_id
                })
            
            raise
    
    async def _apply_action(self, action: Action) -> Dict[str, Any]:
        """Apply a single action."""
        if action.type == ActionType.HPO:
            return await self._apply_hpo_action(action)
            
        elif action.type == ActionType.REWEIGHT_SPECIALISTS:
            return await self._apply_reweight_action(action)
            
        elif action.type == ActionType.POLICY_DELTA:
            return await self._apply_policy_delta_action(action)
            
        elif action.type == ActionType.CANARY:
            return await self._apply_canary_action(action)
        
        else:
            raise ValueError(f"Unknown action type: {action.type}")
    
    async def _apply_hpo_action(self, action: Action) -> Dict[str, Any]:
        """Apply hyperparameter optimization action."""
        # In a real implementation, this would launch HPO jobs
        return {
            "action_type": "hpo",
            "target": action.target,
            "budget": action.budget,
            "status": "job_scheduled",
            "job_id": f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    async def _apply_reweight_action(self, action: Action) -> Dict[str, Any]:
        """Apply specialist reweighting action."""
        # In a real implementation, this would update specialist weights
        if self.governance_engine and hasattr(self.governance_engine, 'unifier'):
            await self.governance_engine.unifier.update_weights(action.weights)
        
        return {
            "action_type": "reweight",
            "weights": action.weights,
            "status": "applied"
        }
    
    async def _apply_policy_delta_action(self, action: Action) -> Dict[str, Any]:
        """Apply policy change action."""
        # In a real implementation, this would update governance policies
        if self.governance_engine:
            # Update the specific policy path
            path_parts = action.path.split('.')
            # This is simplified - real implementation would navigate nested dicts
            
            return {
                "action_type": "policy_delta",
                "path": action.path,
                "from": action.from_value,
                "to": action.to_value,
                "status": "applied"
            }
        
        return {"action_type": "policy_delta", "status": "simulated"}
    
    async def _apply_canary_action(self, action: Action) -> Dict[str, Any]:
        """Apply canary deployment action."""
        # In a real implementation, this would set up canary deployment
        return {
            "action_type": "canary",
            "target_model": action.target_model,
            "steps": action.steps,
            "status": "canary_initiated"
        }
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific proposal."""
        # Check pending
        if proposal_id in self.pending_proposals:
            proposal = self.pending_proposals[proposal_id]
            return {
                "proposal_id": proposal_id,
                "status": "pending",
                "submitted_at": proposal.submitted_at.isoformat(),
                "priority": proposal.priority
            }
        
        # Check approved
        if proposal_id in self.approved_proposals:
            proposal = self.approved_proposals[proposal_id]
            return {
                "proposal_id": proposal_id,
                "status": "approved",
                "submitted_at": proposal.submitted_at.isoformat(),
                "approved_at": proposal.approved_at.isoformat() if proposal.approved_at else None,
                "priority": proposal.priority
            }
        
        # Check rejected
        if proposal_id in self.rejected_proposals:
            proposal = self.rejected_proposals[proposal_id]
            return {
                "proposal_id": proposal_id,
                "status": "rejected",
                "submitted_at": proposal.submitted_at.isoformat(),
                "rejection_reason": proposal.rejection_reason,
                "priority": proposal.priority
            }
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "pending_proposals": len(self.pending_proposals),
            "approved_proposals": len(self.approved_proposals),
            "rejected_proposals": len(self.rejected_proposals),
            "total_proposals": len(self.pending_proposals) + len(self.approved_proposals) + len(self.rejected_proposals),
            "approval_rate": (
                len(self.approved_proposals) / 
                max(1, len(self.approved_proposals) + len(self.rejected_proposals))
            )
        }