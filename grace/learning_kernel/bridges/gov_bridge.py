"""Governance bridge for policy approval and compliance integration."""

import json
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Any


class GovernanceBridge:
    """Bridge for integrating Learning Kernel with Grace Governance system."""
    
    def __init__(self):
        self.submitted_proposals: List[Dict[str, Any]] = []
        self.approved_actions: List[Dict[str, Any]] = []
        self.rejected_actions: List[Dict[str, Any]] = []
    
    async def submit_policy_proposal(self, policy_change: Dict[str, Any]) -> str:
        """Submit policy change proposal to governance."""
        proposal_id = f"learn_policy_{format_for_filename()}"
        
        proposal = {
            "proposal_id": proposal_id,
            "type": "learning_policy_change",
            "change": policy_change,
            "submitted_at": iso_format(),
            "status": "pending",
            "impact_assessment": self._assess_policy_impact(policy_change)
        }
        
        self.submitted_proposals.append(proposal)
        
        # In practice, would submit to actual governance system
        print(f"[LEARNING->GOV] Submitted policy proposal: {proposal_id}")
        
        return proposal_id
    
    async def submit_dataset_publication_request(self, dataset_id: str, version: str,
                                                governance_label: str = "internal") -> str:
        """Submit dataset publication request to governance."""
        proposal_id = f"learn_publish_{dataset_id}_{version}_{utc_now().strftime('%H%M%S')}"
        
        proposal = {
            "proposal_id": proposal_id,
            "type": "dataset_publication",
            "dataset_id": dataset_id,
            "version": version,
            "governance_label": governance_label,
            "submitted_at": iso_format(),
            "status": "pending",
            "compliance_checks": self._run_compliance_checks(dataset_id, version, governance_label)
        }
        
        self.submitted_proposals.append(proposal)
        
        print(f"[LEARNING->GOV] Submitted dataset publication request: {proposal_id}")
        
        return proposal_id
    
    async def submit_rollback_request(self, snapshot_id: str, reason: str) -> str:
        """Submit rollback request to governance."""
        proposal_id = f"learn_rollback_{format_for_filename()}"
        
        proposal = {
            "proposal_id": proposal_id,
            "type": "learning_rollback",
            "target": "learning",
            "to_snapshot": snapshot_id,
            "reason": reason,
            "submitted_at": iso_format(),
            "status": "pending",
            "risk_assessment": self._assess_rollback_risk(snapshot_id)
        }
        
        self.submitted_proposals.append(proposal)
        
        print(f"[LEARNING->GOV] Submitted rollback request: {proposal_id}")
        
        return proposal_id
    
    def _assess_policy_impact(self, policy_change: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact of a policy change."""
        change_type = policy_change.get("type", "unknown")
        
        # Mock impact assessment
        impact = {
            "risk_level": "medium",
            "affected_components": ["labeling_service", "quality_assurance"],
            "estimated_downtime": "0 minutes",
            "rollback_complexity": "low"
        }
        
        if "qa.min_agreement" in str(policy_change):
            impact.update({
                "risk_level": "low",
                "affected_components": ["quality_assurance"],
                "description": "Changes minimum agreement threshold for label quality"
            })
        elif "pii" in str(policy_change):
            impact.update({
                "risk_level": "high",
                "affected_components": ["labeling_service", "data_privacy"],
                "description": "Changes PII handling policy - requires careful review"
            })
        
        return impact
    
    def _run_compliance_checks(self, dataset_id: str, version: str, governance_label: str) -> Dict[str, Any]:
        """Run compliance checks for dataset publication."""
        # Mock compliance checks
        checks = {
            "pii_scan": {"status": "passed", "flags": 0},
            "license_compliance": {"status": "passed", "license": "internal"},
            "data_classification": {"status": "passed", "classification": governance_label},
            "export_restrictions": {"status": "passed", "restricted": False},
            "retention_policy": {"status": "passed", "retention_days": 365}
        }
        
        # Add some risk for restricted data
        if governance_label == "restricted":
            checks["additional_review_required"] = True
            checks["reviewer_clearance"] = "security_cleared"
        
        overall_status = "passed" if all(
            check.get("status") == "passed" 
            for check in checks.values() 
            if isinstance(check, dict) and "status" in check
        ) else "failed"
        
        checks["overall_status"] = overall_status
        
        return checks
    
    def _assess_rollback_risk(self, snapshot_id: str) -> Dict[str, Any]:
        """Assess risk of rollback operation."""
        # Mock risk assessment
        return {
            "risk_level": "medium",
            "data_loss_risk": "low",
            "service_interruption": "minimal", 
            "affected_users": "labeling_team",
            "estimated_duration": "5-10 minutes",
            "requires_approval": True,
            "recommended_window": "maintenance_hours"
        }
    
    async def handle_approval(self, proposal_id: str, approved: bool, 
                            reason: Optional[str] = None) -> bool:
        """Handle approval/rejection from governance system."""
        # Find the proposal
        proposal = None
        for p in self.submitted_proposals:
            if p["proposal_id"] == proposal_id:
                proposal = p
                break
        
        if not proposal:
            print(f"[GOV->LEARNING] Proposal {proposal_id} not found")
            return False
        
        proposal["status"] = "approved" if approved else "rejected"
        proposal["decision_at"] = iso_format()
        proposal["reason"] = reason
        
        if approved:
            self.approved_actions.append(proposal)
            print(f"[GOV->LEARNING] Approved: {proposal_id}")
            
            # Execute the approved action
            await self._execute_approved_action(proposal)
        else:
            self.rejected_actions.append(proposal)
            print(f"[GOV->LEARNING] Rejected: {proposal_id} - {reason}")
        
        return True
    
    async def _execute_approved_action(self, proposal: Dict[str, Any]):
        """Execute an approved action."""
        proposal_type = proposal.get("type")
        
        if proposal_type == "learning_policy_change":
            await self._apply_policy_change(proposal["change"])
        elif proposal_type == "dataset_publication":
            await self._publish_dataset(proposal)
        elif proposal_type == "learning_rollback":
            await self._execute_rollback(proposal)
    
    async def _apply_policy_change(self, change: Dict[str, Any]):
        """Apply an approved policy change."""
        print(f"[LEARNING] Applying approved policy change: {change}")
        # In practice, would update actual policies
    
    async def _publish_dataset(self, proposal: Dict[str, Any]):
        """Publish an approved dataset."""
        dataset_id = proposal["dataset_id"]
        version = proposal["version"]
        governance_label = proposal["governance_label"]
        
        print(f"[LEARNING] Publishing approved dataset: {dataset_id}@{version} ({governance_label})")
        # In practice, would update dataset visibility/access
    
    async def _execute_rollback(self, proposal: Dict[str, Any]):
        """Execute an approved rollback."""
        snapshot_id = proposal["to_snapshot"]
        
        print(f"[LEARNING] Executing approved rollback to: {snapshot_id}")
        # In practice, would trigger actual rollback process
    
    def check_policy_compliance(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action complies with current policies."""
        # Mock compliance check
        compliance = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "approval_required": False
        }
        
        # Check for actions that require approval
        if action.get("type") in ["dataset_publication", "policy_change", "rollback"]:
            compliance["approval_required"] = True
        
        # Check for PII-related actions
        if "pii" in str(action).lower():
            compliance["warnings"].append("PII-related action requires extra scrutiny")
            compliance["approval_required"] = True
        
        # Check for high-impact changes  
        if action.get("impact", {}).get("risk_level") == "high":
            compliance["approval_required"] = True
        
        return compliance
    
    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance interaction statistics."""
        return {
            "total_proposals": len(self.submitted_proposals),
            "approved_count": len(self.approved_actions),
            "rejected_count": len(self.rejected_actions),
            "pending_count": len([
                p for p in self.submitted_proposals 
                if p["status"] == "pending"
            ]),
            "approval_rate": len(self.approved_actions) / max(1, len(self.submitted_proposals))
        }
    
    def clear_history(self):
        """Clear governance history (for testing)."""
        self.submitted_proposals.clear()
        self.approved_actions.clear()
        self.rejected_actions.clear()