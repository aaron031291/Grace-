"""
Parliament - Democratic review system for major governance decisions.
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..core.contracts import Experience, generate_correlation_id


logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class VoteType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    NEEDS_INFO = "needs_info"


class ParliamentMember:
    """Represents a member of the governance parliament."""
    
    def __init__(self, member_id: str, name: str, expertise: List[str], 
                 weight: float = 1.0):
        self.member_id = member_id
        self.name = name
        self.expertise = expertise  # Areas of expertise
        self.weight = weight  # Voting weight
        self.active = True
        self.vote_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "expertise": self.expertise,
            "weight": self.weight,
            "active": self.active,
            "vote_count": len(self.vote_history)
        }


class ReviewProposal:
    """Represents a proposal for parliamentary review."""
    
    def __init__(self, proposal_id: str, title: str, description: str,
                 proposal_type: str, urgency: str = "normal"):
        self.proposal_id = proposal_id
        self.title = title
        self.description = description
        self.proposal_type = proposal_type  # "policy", "constitutional", "operational"
        self.urgency = urgency  # "low", "normal", "high", "critical"
        self.submitted_at = datetime.now()
        self.status = ReviewStatus.PENDING
        self.votes = {}
        self.discussion = []
        self.deadline = self._calculate_deadline()
    
    def _calculate_deadline(self) -> datetime:
        """Calculate review deadline based on urgency."""
        days_map = {
            "critical": 1,
            "high": 3,
            "normal": 7,
            "low": 14
        }
        days = days_map.get(self.urgency, 7)
        return self.submitted_at + timedelta(days=days)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "proposal_type": self.proposal_type,
            "urgency": self.urgency,
            "submitted_at": self.submitted_at.isoformat(),
            "status": self.status.value,
            "deadline": self.deadline.isoformat(),
            "vote_count": len(self.votes),
            "discussion_count": len(self.discussion)
        }


class Parliament:
    """
    Democratic review system for major governance decisions and policy updates.
    Manages committee reviews, voting processes, and democratic validation.
    """
    
    def __init__(self, event_bus, memory_core):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.members = self._initialize_default_members()
        self.active_proposals = {}
        self.review_history = []
        
        # Voting thresholds
        self.voting_thresholds = {
            "policy": 0.6,      # 60% for policy changes
            "constitutional": 0.75,  # 75% for constitutional changes
            "operational": 0.5       # 50% for operational decisions
        }
        
        # Setup event subscriptions
        asyncio.create_task(self._setup_event_subscriptions())
    
    def _initialize_default_members(self) -> Dict[str, ParliamentMember]:
        """Initialize default parliament members."""
        default_members = [
            ParliamentMember("ethics_chair", "Ethics Committee Chair", 
                           ["ethics", "fairness", "harm_prevention"], 1.2),
            ParliamentMember("tech_lead", "Technical Lead", 
                           ["security", "privacy", "technical"], 1.1),
            ParliamentMember("transparency_officer", "Transparency Officer", 
                           ["transparency", "accountability", "audit"], 1.1),
            ParliamentMember("user_advocate", "User Advocate", 
                           ["user_rights", "accessibility", "usability"], 1.0),
            ParliamentMember("legal_counsel", "Legal Counsel", 
                           ["legal", "compliance", "constitutional"], 1.2),
            ParliamentMember("domain_expert_1", "Domain Expert (AI/ML)", 
                           ["machine_learning", "ai_safety"], 1.0),
            ParliamentMember("domain_expert_2", "Domain Expert (Governance)", 
                           ["governance", "policy", "process"], 1.0),
        ]
        
        return {member.member_id: member for member in default_members}
    
    async def _setup_event_subscriptions(self):
        """Setup event subscriptions for parliament operations."""
        await self.event_bus.subscribe(
            "GOVERNANCE_NEEDS_REVIEW",
            self._handle_review_request
        )
        
        await self.event_bus.subscribe(
            "PARLIAMENT_VOTE_CAST",
            self._handle_vote_cast
        )
    
    async def submit_for_review(self, title: str, description: str,
                              proposal_type: str, urgency: str = "normal",
                              context: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a proposal for parliamentary review.
        
        Args:
            title: Proposal title
            description: Detailed description
            proposal_type: Type of proposal ("policy", "constitutional", "operational")
            urgency: Urgency level
            context: Additional context data
            
        Returns:
            Proposal ID
        """
        proposal_id = f"prop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_proposals)}"
        
        proposal = ReviewProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposal_type=proposal_type,
            urgency=urgency
        )
        
        self.active_proposals[proposal_id] = proposal
        
        # Select relevant reviewers based on expertise
        assigned_reviewers = await self._assign_reviewers(proposal, context)
        
        # Notify reviewers
        await self._notify_reviewers(proposal, assigned_reviewers)
        
        logger.info(f"Submitted proposal {proposal_id} for {proposal_type} review")
        
        return proposal_id
    
    async def _assign_reviewers(self, proposal: ReviewProposal,
                              context: Optional[Dict[str, Any]]) -> List[str]:
        """Assign reviewers based on proposal type and member expertise."""
        assigned = []
        
        # Always include chair members for important decisions
        if proposal.proposal_type in ["constitutional", "policy"]:
            for member_id, member in self.members.items():
                if "chair" in member_id or member.weight > 1.1:
                    assigned.append(member_id)
        
        # Add domain experts based on context
        if context:
            required_expertise = context.get("required_expertise", [])
            for expertise in required_expertise:
                for member_id, member in self.members.items():
                    if expertise in member.expertise and member_id not in assigned:
                        assigned.append(member_id)
        
        # Ensure minimum reviewers
        min_reviewers = {
            "constitutional": 5,
            "policy": 4,
            "operational": 3
        }
        
        required_count = min_reviewers.get(proposal.proposal_type, 3)
        while len(assigned) < required_count:
            # Add active members not yet assigned
            for member_id, member in self.members.items():
                if member.active and member_id not in assigned:
                    assigned.append(member_id)
                    if len(assigned) >= required_count:
                        break
        
        return assigned
    
    async def _notify_reviewers(self, proposal: ReviewProposal,
                              assigned_reviewers: List[str]):
        """Notify assigned reviewers of new proposal."""
        notification = {
            "type": "PARLIAMENT_REVIEW_ASSIGNED",
            "proposal": proposal.to_dict(),
            "assigned_reviewers": assigned_reviewers,
            "deadline": proposal.deadline.isoformat()
        }
        
        await self.event_bus.publish("PARLIAMENT_NOTIFICATION", notification)
    
    async def cast_vote(self, proposal_id: str, member_id: str,
                       vote: VoteType, rationale: str = "") -> bool:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: ID of proposal to vote on
            member_id: ID of voting member
            vote: Vote type
            rationale: Optional rationale for vote
            
        Returns:
            True if vote was recorded successfully
        """
        if proposal_id not in self.active_proposals:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        if member_id not in self.members:
            logger.error(f"Member {member_id} not found")
            return False
        
        proposal = self.active_proposals[proposal_id]
        member = self.members[member_id]
        
        # Record vote
        proposal.votes[member_id] = {
            "vote": vote.value,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
            "weight": member.weight
        }
        
        # Update member vote history
        member.vote_history.append({
            "proposal_id": proposal_id,
            "vote": vote.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if voting is complete
        await self._check_voting_completion(proposal)
        
        logger.info(f"Member {member_id} voted {vote.value} on proposal {proposal_id}")
        
        return True
    
    async def _check_voting_completion(self, proposal: ReviewProposal):
        """Check if voting is complete and determine outcome."""
        if not proposal.votes:
            return
        
        # Calculate vote totals
        approve_weight = sum(
            vote_data["weight"] for vote_data in proposal.votes.values()
            if vote_data["vote"] == VoteType.APPROVE.value
        )
        
        total_weight = sum(vote_data["weight"] for vote_data in proposal.votes.values())
        reject_weight = sum(
            vote_data["weight"] for vote_data in proposal.votes.values()
            if vote_data["vote"] == VoteType.REJECT.value
        )
        
        if total_weight == 0:
            return
        
        approval_ratio = approve_weight / total_weight
        rejection_ratio = reject_weight / total_weight
        
        # Check thresholds
        threshold = self.voting_thresholds.get(proposal.proposal_type, 0.6)
        
        # Determine if we have enough votes to make a decision
        assigned_members = len([m for m in self.members.values() if m.active])
        voted_members = len(proposal.votes)
        participation_ratio = voted_members / assigned_members
        
        # Decision criteria
        if approval_ratio >= threshold and participation_ratio >= 0.6:
            proposal.status = ReviewStatus.APPROVED
            await self._finalize_proposal(proposal, "approved", approval_ratio)
        elif rejection_ratio >= 0.4 and participation_ratio >= 0.6:
            proposal.status = ReviewStatus.REJECTED
            await self._finalize_proposal(proposal, "rejected", rejection_ratio)
        elif datetime.now() >= proposal.deadline:
            # Deadline reached
            if approval_ratio > rejection_ratio:
                proposal.status = ReviewStatus.APPROVED
                await self._finalize_proposal(proposal, "approved_by_deadline", approval_ratio)
            else:
                proposal.status = ReviewStatus.NEEDS_REVISION
                await self._finalize_proposal(proposal, "needs_revision", rejection_ratio)
    
    async def _finalize_proposal(self, proposal: ReviewProposal,
                               outcome: str, final_ratio: float):
        """Finalize a proposal and notify stakeholders."""
        # Move to history
        self.review_history.append({
            "proposal": proposal.to_dict(),
            "outcome": outcome,
            "final_ratio": final_ratio,
            "finalized_at": datetime.now().isoformat(),
            "votes": proposal.votes
        })
        
        # Remove from active proposals
        if proposal.proposal_id in self.active_proposals:
            del self.active_proposals[proposal.proposal_id]
        
        # Record experience
        experience = Experience(
            type="DEMOCRATIC_REVIEW",
            component_id="parliament",
            context={
                "proposal_type": proposal.proposal_type,
                "urgency": proposal.urgency,
                "participation_ratio": len(proposal.votes) / len(self.members)
            },
            outcome={
                "decision": outcome,
                "approval_ratio": final_ratio,
                "vote_count": len(proposal.votes)
            },
            success_score=final_ratio if outcome.startswith("approved") else 1.0 - final_ratio,
            timestamp=datetime.now()
        )
        
        self.memory_core.store_experience(experience)
        
        # Publish result
        await self.event_bus.publish("PARLIAMENT_DECISION", {
            "proposal_id": proposal.proposal_id,
            "outcome": outcome,
            "final_ratio": final_ratio,
            "votes": proposal.votes
        })
        
        logger.info(f"Finalized proposal {proposal.proposal_id}: {outcome} (ratio: {final_ratio:.3f})")
    
    async def _handle_review_request(self, event: Dict[str, Any]):
        """Handle incoming review requests from governance engine."""
        payload = event.get("payload", {})
        
        # Extract review details
        decision_id = payload.get("decision_id", "")
        rationale = payload.get("rationale", "")
        
        # Create review proposal
        await self.submit_for_review(
            title=f"Review Required: {decision_id}",
            description=f"Governance decision requires parliamentary review. {rationale}",
            proposal_type="policy",  # Default to policy review
            urgency="normal"
        )
    
    async def _handle_vote_cast(self, event: Dict[str, Any]):
        """Handle vote casting events."""
        payload = event.get("payload", {})
        
        proposal_id = payload.get("proposal_id", "")
        member_id = payload.get("member_id", "")
        vote_str = payload.get("vote", "")
        rationale = payload.get("rationale", "")
        
        try:
            vote = VoteType(vote_str)
            await self.cast_vote(proposal_id, member_id, vote, rationale)
        except ValueError:
            logger.error(f"Invalid vote type: {vote_str}")
    
    def add_member(self, member_id: str, name: str, expertise: List[str],
                   weight: float = 1.0) -> bool:
        """Add a new parliament member."""
        if member_id in self.members:
            logger.warning(f"Member {member_id} already exists")
            return False
        
        self.members[member_id] = ParliamentMember(member_id, name, expertise, weight)
        logger.info(f"Added parliament member: {name} ({member_id})")
        return True
    
    def remove_member(self, member_id: str) -> bool:
        """Remove a parliament member."""
        if member_id not in self.members:
            logger.warning(f"Member {member_id} not found")
            return False
        
        self.members[member_id].active = False
        logger.info(f"Deactivated parliament member: {member_id}")
        return True
    
    def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get list of active proposals."""
        return [proposal.to_dict() for proposal in self.active_proposals.values()]
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific proposal."""
        if proposal_id not in self.active_proposals:
            return None
        
        proposal = self.active_proposals[proposal_id]
        return {
            **proposal.to_dict(),
            "votes": proposal.votes,
            "discussion": proposal.discussion
        }
    
    def get_member_stats(self) -> Dict[str, Any]:
        """Get parliament member statistics."""
        active_count = sum(1 for m in self.members.values() if m.active)
        total_votes = sum(len(m.vote_history) for m in self.members.values())
        
        return {
            "total_members": len(self.members),
            "active_members": active_count,
            "total_votes_cast": total_votes,
            "active_proposals": len(self.active_proposals),
            "completed_reviews": len(self.review_history)
        }