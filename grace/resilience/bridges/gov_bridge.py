"""Bridge to Governance kernel for policy decisions."""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class GovernanceBridge:
    """
    Bridge to Governance kernel for resilience policy decisions.
    
    Handles communication with the Grace Governance kernel for
    approving chaos experiments, policy changes, and risk decisions.
    """
    
    def __init__(self, governance_client=None):
        """Initialize governance bridge."""
        self.governance_client = governance_client
        self.approval_history: List[Dict] = []
        
        logger.debug("Governance bridge initialized")
    
    async def request_approval(
        self, 
        action: str, 
        risk_level: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request approval from governance for a resilience action.
        
        Args:
            action: Action requiring approval (e.g., "chaos_experiment", "policy_change")
            risk_level: Risk level (low, medium, high, critical)
            context: Additional context for the decision
            
        Returns:
            Approval decision with reasoning
        """
        try:
            approval_request = {
                "requester": "resilience",
                "action": action,
                "risk_level": risk_level,
                "context": context,
                "timestamp": self._get_timestamp()
            }
            
            if self.governance_client:
                response = await self.governance_client.request_approval(approval_request)
            else:
                # Fallback approval logic
                response = self._fallback_approval(action, risk_level, context)
            
            # Store approval history
            approval_record = {
                **approval_request,
                "decision": response["approved"],
                "reasoning": response.get("reasoning", ""),
                "conditions": response.get("conditions", [])
            }
            self.approval_history.append(approval_record)
            
            # Trim history
            if len(self.approval_history) > 1000:
                self.approval_history = self.approval_history[-500:]
            
            logger.info(f"Governance approval for {action}: {response['approved']}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to request governance approval for {action}: {e}")
            return {
                "approved": False,
                "reasoning": f"Governance communication failed: {str(e)}",
                "conditions": []
            }
    
    async def request_chaos_approval(
        self, 
        target: str, 
        experiment_type: str, 
        blast_radius_pct: float,
        duration_s: int
    ) -> Dict[str, Any]:
        """Request approval for chaos experiment."""
        risk_level = self._assess_chaos_risk(blast_radius_pct, duration_s)
        
        context = {
            "target": target,
            "experiment_type": experiment_type,
            "blast_radius_pct": blast_radius_pct,
            "duration_s": duration_s,
            "estimated_impact": self._estimate_chaos_impact(blast_radius_pct)
        }
        
        return await self.request_approval("chaos_experiment", risk_level, context)
    
    async def request_policy_change_approval(
        self, 
        service_id: str, 
        policy_type: str, 
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request approval for policy changes."""
        risk_level = self._assess_policy_risk(policy_type, changes)
        
        context = {
            "service_id": service_id,
            "policy_type": policy_type,
            "changes": changes,
            "impact_assessment": self._assess_policy_impact(changes)
        }
        
        return await self.request_approval("policy_change", risk_level, context)
    
    async def request_incident_escalation(
        self, 
        incident_id: str, 
        severity: str, 
        proposed_actions: List[str]
    ) -> Dict[str, Any]:
        """Request approval for incident escalation actions."""
        risk_level = self._map_severity_to_risk(severity)
        
        context = {
            "incident_id": incident_id,
            "severity": severity,
            "proposed_actions": proposed_actions,
            "escalation_reason": "Automated resilience response"
        }
        
        return await self.request_approval("incident_escalation", risk_level, context)
    
    def get_approval_history(self, action_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get approval history."""
        history = self.approval_history
        
        if action_type:
            history = [h for h in history if h["action"] == action_type]
        
        return history[-limit:] if limit else history
    
    def _fallback_approval(self, action: str, risk_level: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback approval logic when governance is unavailable."""
        # Conservative approval logic
        if risk_level in ["critical", "high"]:
            approved = False
            reasoning = f"Auto-denied: {risk_level} risk actions require manual governance approval"
        elif action == "chaos_experiment":
            blast_radius = context.get("blast_radius_pct", 0)
            approved = blast_radius <= 5.0  # Only approve low blast radius
            reasoning = f"Auto-approved chaos experiment with {blast_radius}% blast radius"
        elif action == "policy_change":
            # Only approve minor policy changes
            approved = self._is_minor_policy_change(context.get("changes", {}))
            reasoning = "Auto-approved minor policy change" if approved else "Policy change requires manual approval"
        else:
            approved = risk_level == "low"
            reasoning = f"Auto-approved {risk_level} risk {action}"
        
        return {
            "approved": approved,
            "reasoning": reasoning,
            "conditions": ["Fallback approval - verify with governance when available"] if approved else [],
            "approval_method": "fallback"
        }
    
    def _assess_chaos_risk(self, blast_radius_pct: float, duration_s: int) -> str:
        """Assess risk level for chaos experiments."""
        if blast_radius_pct > 20 or duration_s > 3600:  # >20% or >1 hour
            return "high"
        elif blast_radius_pct > 10 or duration_s > 1800:  # >10% or >30 minutes
            return "medium"
        else:
            return "low"
    
    def _assess_policy_risk(self, policy_type: str, changes: Dict[str, Any]) -> str:
        """Assess risk level for policy changes."""
        if policy_type == "slo" and any(key in changes for key in ["error_budget_days"]):
            return "medium"
        elif policy_type == "resilience" and any(key in changes for key in ["circuit_breaker", "degradation_modes"]):
            return "medium"
        else:
            return "low"
    
    def _map_severity_to_risk(self, severity: str) -> str:
        """Map incident severity to risk level."""
        severity_risk_map = {
            "sev1": "critical",
            "sev2": "high", 
            "sev3": "medium"
        }
        return severity_risk_map.get(severity, "medium")
    
    def _estimate_chaos_impact(self, blast_radius_pct: float) -> str:
        """Estimate impact of chaos experiment."""
        if blast_radius_pct > 50:
            return "severe"
        elif blast_radius_pct > 20:
            return "moderate"
        elif blast_radius_pct > 5:
            return "minor"
        else:
            return "minimal"
    
    def _assess_policy_impact(self, changes: Dict[str, Any]) -> str:
        """Assess impact of policy changes."""
        # Simplified impact assessment
        if len(changes) > 5:
            return "major"
        elif any(isinstance(v, dict) and len(v) > 3 for v in changes.values()):
            return "moderate"
        else:
            return "minor"
    
    def _is_minor_policy_change(self, changes: Dict[str, Any]) -> bool:
        """Determine if policy change is minor."""
        # Consider changes minor if they only adjust thresholds within reasonable bounds
        minor_keys = {"max", "threshold", "timeout", "interval"}
        return all(key in minor_keys for key in changes.keys()) and len(changes) <= 2
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
        return iso_format()