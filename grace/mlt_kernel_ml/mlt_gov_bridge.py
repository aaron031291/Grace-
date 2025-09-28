"""
MLT-Governance Bridge - Packs proposals into governance events and handles approvals.
Integrated with MTL Kernel for memory, trust, and audit capabilities.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename

from .contracts import AdaptationPlan, Action, ActionType
from ..contracts.dto_common import MemoryEntry
from ..mtl_kernel.kernel import MTLKernel


logger = logging.getLogger(__name__)


class GovernanceProposal:
    """A proposal to governance for MLT actions."""
    
    def __init__(self, proposal_id: str, plan: AdaptationPlan, priority: str = "normal"):
        self.proposal_id = proposal_id
        self.plan = plan
        self.priority = priority
        self.status = "pending"
        self.submitted_at = utc_now()
        self.approved_at: Optional[datetime] = None
        self.rejection_reason: Optional[str] = None
        self.memory_id: Optional[str] = None  # MTL kernel memory reference
    
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
    """Bridges MLT kernel with governance system for proposal approval.
    Integrated with MTL Kernel for comprehensive memory, trust, and audit capabilities.
    """
    
    def __init__(self, governance_engine=None, event_bus=None, mtl_kernel=None):
        self.governance_engine = governance_engine
        self.event_bus = event_bus
        self.mtl_kernel = mtl_kernel or MTLKernel()  # Create MTL kernel if not provided
        
        # Local caches for quick access
        self.pending_proposals: Dict[str, GovernanceProposal] = {}
        self.approved_proposals: Dict[str, GovernanceProposal] = {}
        self.rejected_proposals: Dict[str, GovernanceProposal] = {}
    
    async def submit_plan_for_approval(self, plan: AdaptationPlan, priority: str = "normal") -> str:
        """Submit an adaptation plan to governance for approval."""
        try:
            proposal_id = f"mlt_proposal_{utc_now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            proposal = GovernanceProposal(proposal_id, plan, priority)
            self.pending_proposals[proposal_id] = proposal
            
            # Store proposal in MTL memory system
            memory_id = self._store_proposal_in_memory(proposal)
            proposal.memory_id = memory_id  # Add memory_id to proposal
            
            # Send to governance through event bus
            if self.event_bus:
                governance_event = proposal.to_governance_event()
                governance_event["payload"]["memory_id"] = memory_id
                await self.event_bus.publish("ADAPTATION_PLAN_PROPOSED", governance_event)
                
                logger.info(f"Submitted plan {plan.plan_id} to governance as proposal {proposal_id} (memory: {memory_id})")
            
            return proposal_id
            
        except Exception as e:
            logger.error(f"Failed to submit plan to governance: {e}")
            raise
    
    def _store_proposal_in_memory(self, proposal: GovernanceProposal) -> str:
        """Store governance proposal in MTL memory system."""
        try:
            memory_entry = MemoryEntry(
                content=f"Governance Proposal: {proposal.plan.plan_id}",
                content_type="application/json",
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "plan_id": proposal.plan.plan_id,
                    "priority": proposal.priority,
                    "status": proposal.status,
                    "submitted_at": proposal.submitted_at.isoformat(),
                    "type": "governance_proposal",
                    "risk_assessment": proposal._assess_risk(),
                    "governance_requirements": proposal._determine_governance_requirements(),
                    "actions_count": len(proposal.plan.actions),
                    "actions_summary": [{"type": action.type.value, "target": getattr(action, 'target', 'unknown')} for action in proposal.plan.actions]
                }
            )
            
            # Store in MTL kernel with full fan-out (memory, trust, immutable log, triggers)
            memory_id = self.mtl_kernel.write(memory_entry)
            
            # Add initial trust attestation for the proposal
            self.mtl_kernel.attest(memory_id, {
                "trust_score": 0.5,  # Neutral trust for new proposals
                "confidence": 0.8,
                "source": "mlt_governance_bridge",
                "hash": proposal.proposal_id
            })
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store proposal in MTL memory: {e}")
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
                proposal.approved_at = utc_now()
                self.approved_proposals[proposal_id] = proposal
                
                # Store decision in MTL memory system
                self._store_governance_decision(proposal, "approved", payload.get("rationale", ""))
                
                # Update trust score for approved proposal
                if hasattr(proposal, 'memory_id'):
                    self.mtl_kernel.attest(proposal.memory_id, {
                        "trust_score": 0.9,  # High trust for approved proposals
                        "confidence": 1.0,
                        "source": "governance_system",
                        "hash": f"{proposal_id}_approved",
                        "decision_timestamp": iso_format()
                    })
                
                # Apply the plan atomically
                await self._apply_approved_plan(proposal)
                
                logger.info(f"Governance approved proposal {proposal_id}")
                return True
                
            elif event_type == "GOVERNANCE_REJECTED":
                proposal.status = "rejected"
                proposal.rejection_reason = payload.get("rationale", "No reason provided")
                self.rejected_proposals[proposal_id] = proposal
                
                # Store decision in MTL memory system
                self._store_governance_decision(proposal, "rejected", proposal.rejection_reason)
                
                # Update trust score for rejected proposal
                if hasattr(proposal, 'memory_id'):
                    self.mtl_kernel.attest(proposal.memory_id, {
                        "trust_score": 0.1,  # Low trust for rejected proposals
                        "confidence": 1.0,
                        "source": "governance_system",
                        "hash": f"{proposal_id}_rejected",
                        "rejection_reason": proposal.rejection_reason,
                        "decision_timestamp": iso_format()
                    })
                
                logger.info(f"Governance rejected proposal {proposal_id}: {proposal.rejection_reason}")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to handle governance decision: {e}")
            return False
    
    def _store_governance_decision(self, proposal: GovernanceProposal, decision: str, rationale: str) -> str:
        """Store governance decision in MTL memory system."""
        try:
            memory_entry = MemoryEntry(
                content=f"Governance Decision: {decision.upper()} - {proposal.plan.plan_id}",
                content_type="application/json",
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "plan_id": proposal.plan.plan_id,
                    "decision": decision,
                    "rationale": rationale,
                    "decided_at": iso_format(),
                    "type": "governance_decision",
                    "original_proposal_memory_id": getattr(proposal, 'memory_id', None),
                    "risk_level": proposal._assess_risk().get("risk_level", "unknown"),
                    "actions_count": len(proposal.plan.actions)
                }
            )
            
            # Store decision with high trust
            memory_id = self.mtl_kernel.write(memory_entry)
            self.mtl_kernel.attest(memory_id, {
                "trust_score": 0.95,  # Very high trust for governance decisions
                "confidence": 1.0,
                "source": "governance_system",
                "hash": f"{proposal.proposal_id}_{decision}"
            })
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store governance decision in MTL memory: {e}")
            raise
    
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
                    "applied_at": iso_format()
                })
            
            # Store applied plan in MTL memory system
            applied_plan_memory_id = self._store_applied_plan(proposal, applied_actions)
            
            # Emit plan applied event
            if self.event_bus:
                await self.event_bus.publish("MLT_PLAN_APPLIED", {
                    "proposal_id": proposal.proposal_id,
                    "plan_id": plan.plan_id,
                    "applied_actions": applied_actions,
                    "memory_id": applied_plan_memory_id,
                    "snapshot_id": None  # Would be populated by snapshot manager
                })
            
            logger.info(f"Applied plan {plan.plan_id} with {len(applied_actions)} actions (memory: {applied_plan_memory_id})")
            return {"status": "applied", "actions": applied_actions, "memory_id": applied_plan_memory_id}
            
        except Exception as e:
            logger.error(f"Failed to apply approved plan {proposal.plan.plan_id}: {e}")
            
            # Emit rollback request
            if self.event_bus:
                await self.event_bus.publish("ROLLBACK_REQUESTED", {
                    "reason": f"Plan application failed: {str(e)}",
                    "proposal_id": proposal.proposal_id
                })
            
            raise
    
    def _store_applied_plan(self, proposal: GovernanceProposal, applied_actions: List[Dict]) -> str:
        """Store applied adaptation plan in MTL memory system."""
        try:
            memory_entry = MemoryEntry(
                content=f"Applied Adaptation Plan: {proposal.plan.plan_id}",
                content_type="application/json",
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "plan_id": proposal.plan.plan_id,
                    "applied_at": iso_format(),
                    "type": "applied_adaptation_plan",
                    "actions_count": len(applied_actions),
                    "actions_summary": applied_actions,
                    "original_proposal_memory_id": getattr(proposal, 'memory_id', None),
                    "success": True,
                    "priority": proposal.priority
                }
            )
            
            # Store with high trust since this was governance approved and successfully applied
            memory_id = self.mtl_kernel.write(memory_entry)
            self.mtl_kernel.attest(memory_id, {
                "trust_score": 0.95,  # Very high trust for successfully applied plans
                "confidence": 1.0,
                "source": "mlt_governance_bridge",
                "hash": f"{proposal.proposal_id}_applied",
                "application_timestamp": iso_format()
            })
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store applied plan in MTL memory: {e}")
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
            "job_id": f"hpo_{format_for_filename()}"
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
        """Get bridge statistics including MTL kernel metrics."""
        base_stats = {
            "pending_proposals": len(self.pending_proposals),
            "approved_proposals": len(self.approved_proposals),
            "rejected_proposals": len(self.rejected_proposals),
            "total_proposals": len(self.pending_proposals) + len(self.approved_proposals) + len(self.rejected_proposals),
            "approval_rate": (
                len(self.approved_proposals) / 
                max(1, len(self.approved_proposals) + len(self.rejected_proposals))
            )
        }
        
        # Add MTL kernel statistics
        try:
            mtl_stats = self.mtl_kernel.get_stats()
            base_stats.update({
                "mtl_integration": {
                    "memory_entries": mtl_stats.get("memory_entries", 0),
                    "trust_records": mtl_stats.get("trust_records", 0),
                    "audit_records": mtl_stats.get("audit_records", 0),
                    "trigger_events": mtl_stats.get("trigger_events", 0),
                    "merkle_root": mtl_stats.get("merkle_root", None),
                    "integrated": True
                }
            })
        except Exception as e:
            logger.warning(f"Failed to get MTL kernel stats: {e}")
            base_stats["mtl_integration"] = {"integrated": False, "error": str(e)}
        
        return base_stats
    
    def get_memory_audit_trail(self, proposal_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail from MTL memory system for a proposal."""
        try:
            # Search for all memory entries related to this proposal
            results = self.mtl_kernel.recall(
                query=proposal_id,
                filters={"metadata.proposal_id": proposal_id}
            )
            
            audit_trail = []
            for entry in results:
                # Get trust information
                trust_score = self.mtl_kernel.trust_service.get_trust_score(entry.id)
                
                audit_trail.append({
                    "memory_id": entry.id,
                    "content": entry.content,
                    "type": entry.metadata.get("type", "unknown"),
                    "timestamp": entry.metadata.get("submitted_at") or entry.metadata.get("decided_at") or entry.metadata.get("applied_at"),
                    "trust_score": trust_score,
                    "metadata": entry.metadata
                })
            
            # Sort by timestamp
            audit_trail.sort(key=lambda x: x["timestamp"] or "")
            return audit_trail
            
        except Exception as e:
            logger.error(f"Failed to get memory audit trail for proposal {proposal_id}: {e}")
            return []
    
    def set_mtl_kernel(self, mtl_kernel: MTLKernel):
        """Set or update the MTL kernel instance."""
        self.mtl_kernel = mtl_kernel