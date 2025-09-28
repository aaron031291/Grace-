"""Bridge to Governance kernel for approvals and policy prompts."""
import asyncio
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Any
import logging
import uuid

logger = logging.getLogger(__name__)


class GovernanceBridge:
    """Bridges Interface to Governance kernel for approvals and policy enforcement."""
    
    def __init__(self, governance_kernel=None):
        self.governance_kernel = governance_kernel
        self.pending_approvals: Dict[str, Dict] = {}
    
    async def request_approval(self, request_data: Dict) -> str:
        """Request governance approval for an action."""
        approval_id = str(uuid.uuid4())
        
        approval_request = {
            "approval_id": approval_id,
            "request_type": request_data.get("type", "general"),
            "requester": request_data.get("user_id"),
            "action": request_data.get("action"),
            "resource": request_data.get("resource"),
            "context": request_data.get("context", {}),
            "priority": request_data.get("priority", 5),
            "created_at": utc_now(),
            "status": "pending"
        }
        
        self.pending_approvals[approval_id] = approval_request
        
        # If governance kernel is available, submit for review
        if self.governance_kernel:
            try:
                # Submit to governance for evaluation
                result = await self._submit_to_governance(approval_request)
                approval_request["governance_result"] = result
                
                if result.get("approved", False):
                    approval_request["status"] = "approved"
                else:
                    approval_request["status"] = "denied"
                    approval_request["reason"] = result.get("reason", "Policy violation")
                
            except Exception as e:
                logger.error(f"Governance submission failed: {e}")
                approval_request["status"] = "error"
                approval_request["error"] = str(e)
        
        logger.info(f"Created approval request {approval_id} for {request_data.get('action')}")
        return approval_id
    
    async def _submit_to_governance(self, request: Dict) -> Dict:
        """Submit request to governance kernel."""
        if not self.governance_kernel:
            return {"approved": False, "reason": "Governance kernel unavailable"}
        
        try:
            # Create governed request
            governed_request = {
                "request_type": request["request_type"],
                "content": f"{request['action']} on {request['resource']}",
                "requester": request["requester"],
                "context": request["context"],
                "priority": request["priority"]
            }
            
            # Evaluate through governance
            decision = await self.governance_kernel.evaluate_request(governed_request)
            
            return {
                "approved": decision.approved if hasattr(decision, 'approved') else False,
                "confidence": decision.confidence if hasattr(decision, 'confidence') else 0.0,
                "reason": decision.reasoning if hasattr(decision, 'reasoning') else "No reason provided",
                "decision_id": decision.decision_id if hasattr(decision, 'decision_id') else None
            }
            
        except Exception as e:
            logger.error(f"Governance evaluation error: {e}")
            return {"approved": False, "reason": f"Evaluation error: {str(e)}"}
    
    def get_approval_status(self, approval_id: str) -> Optional[Dict]:
        """Get status of approval request."""
        return self.pending_approvals.get(approval_id)
    
    def list_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict]:
        """List pending approval requests."""
        approvals = list(self.pending_approvals.values())
        
        if user_id:
            approvals = [a for a in approvals if a["requester"] == user_id]
        
        return [a for a in approvals if a["status"] == "pending"]
    
    async def check_policy_compliance(self, action: Dict) -> Dict:
        """Check if action complies with policies."""
        if not self.governance_kernel:
            return {"compliant": True, "warnings": []}
        
        try:
            # Use policy engine if available
            if hasattr(self.governance_kernel, 'policy_engine'):
                violations = self.governance_kernel.policy_engine.check(action)
                
                compliant = len(violations) == 0
                warnings = [v.get("message", "Policy violation") for v in violations]
                
                return {
                    "compliant": compliant,
                    "warnings": warnings,
                    "violations": violations
                }
            
            return {"compliant": True, "warnings": []}
            
        except Exception as e:
            logger.error(f"Policy compliance check failed: {e}")
            return {"compliant": False, "warnings": [f"Check failed: {str(e)}"]}
    
    def generate_policy_prompt(self, violation: Dict) -> Dict:
        """Generate user-friendly policy prompt for violations."""
        return {
            "title": "Policy Review Required",
            "message": violation.get("message", "This action requires review"),
            "severity": violation.get("severity", "warning"),
            "actions": [
                {"label": "Request Approval", "action": "request_approval"},
                {"label": "Modify Request", "action": "modify_request"},
                {"label": "Cancel", "action": "cancel"}
            ]
        }
    
    def set_governance_kernel(self, governance_kernel):
        """Set the governance kernel reference."""
        self.governance_kernel = governance_kernel
        logger.info("Governance kernel connected to bridge")
    
    def get_stats(self) -> Dict:
        """Get bridge statistics."""
        approvals = list(self.pending_approvals.values())
        
        status_counts = {}
        for approval in approvals:
            status = approval["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_approvals": len(approvals),
            "status_distribution": status_counts,
            "governance_connected": bool(self.governance_kernel)
        }