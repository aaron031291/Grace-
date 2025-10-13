"""
Quorum Consensus Schema - Defines data structures and validation for consensus operations.

This module provides schemas for the quorum consensus mechanism mentioned in the
Grace architecture, ensuring proper validation and structure for democratic decisions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class VoteType(str, Enum):
    """Types of votes in quorum consensus."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"


class QuorumType(str, Enum):
    """Types of quorum requirements."""

    SIMPLE_MAJORITY = "simple_majority"  # > 50%
    QUALIFIED_MAJORITY = "qualified_majority"  # >= 2/3
    UNANIMOUS = "unanimous"  # 100%
    SUPER_MAJORITY = "super_majority"  # >= 75%


class ConsensusStatus(str, Enum):
    """Status of consensus process."""

    PENDING = "pending"
    ACTIVE = "active"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class QuorumMember:
    """Represents a member of the quorum."""

    member_id: str
    name: str
    role: str
    weight: float = 1.0  # Voting weight
    trust_score: float = 1.0
    specialization: Optional[str] = None
    active: bool = True
    last_activity: Optional[datetime] = None


@dataclass
class Vote:
    """Represents a single vote in the consensus."""

    member_id: str
    vote_type: VoteType
    reasoning: Optional[str] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusProposal(BaseModel):
    """Schema for consensus proposals."""

    proposal_id: str = Field(..., description="Unique identifier for the proposal")
    title: str = Field(
        ..., min_length=1, max_length=200, description="Brief title of the proposal"
    )
    description: str = Field(
        ..., min_length=10, description="Detailed description of what's being proposed"
    )
    proposer_id: str = Field(..., description="ID of the member who proposed this")
    proposal_type: str = Field(
        ...,
        description="Type of proposal (policy_change, constitutional_amendment, etc.)",
    )

    # Voting configuration
    quorum_type: QuorumType = Field(
        default=QuorumType.SIMPLE_MAJORITY, description="Required quorum type"
    )
    min_participation: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Minimum participation rate"
    )
    voting_deadline: datetime = Field(..., description="When voting closes")

    # Content and context
    rationale: str = Field(
        ..., min_length=10, description="Justification for the proposal"
    )
    impact_assessment: Optional[str] = Field(
        None, description="Assessment of potential impacts"
    )
    alternatives_considered: Optional[List[str]] = Field(
        default_factory=list, description="Alternative approaches considered"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    priority: str = Field(
        default="normal", description="Priority level (low, normal, high, critical)"
    )

    # Constitutional compliance
    constitutional_review_required: bool = Field(
        default=True, description="Whether constitutional review is required"
    )
    transparency_level: str = Field(
        default="democratic_oversight", description="Audit transparency level"
    )

    @validator("voting_deadline")
    def voting_deadline_must_be_future(cls, v):
        if v <= datetime.now():
            raise ValueError("Voting deadline must be in the future")
        return v

    @validator("min_participation")
    def min_participation_reasonable(cls, v):
        if v < 0.1:
            raise ValueError("Minimum participation cannot be less than 10%")
        return v


class ConsensusResult(BaseModel):
    """Schema for consensus results."""

    proposal_id: str
    status: ConsensusStatus

    # Vote tallies
    total_members: int = Field(..., ge=1)
    participating_members: int = Field(..., ge=0)
    votes_approve: int = Field(default=0, ge=0)
    votes_reject: int = Field(default=0, ge=0)
    votes_abstain: int = Field(default=0, ge=0)
    votes_delegate: int = Field(default=0, ge=0)

    # Weighted results (considering member weights)
    weighted_approve: float = Field(default=0.0, ge=0.0)
    weighted_reject: float = Field(default=0.0, ge=0.0)
    weighted_abstain: float = Field(default=0.0, ge=0.0)

    # Participation metrics
    participation_rate: float = Field(..., ge=0.0, le=1.0)
    quorum_met: bool = Field(...)
    consensus_achieved: bool = Field(...)

    # Results
    approval_percentage: float = Field(..., ge=0.0, le=1.0)
    rejection_percentage: float = Field(..., ge=0.0, le=1.0)

    # Timing
    voting_started: datetime
    voting_ended: Optional[datetime] = None
    result_calculated: datetime = Field(default_factory=datetime.now)

    # Audit and rationale
    decision_rationale: Optional[str] = None
    dissenting_opinions: List[str] = Field(default_factory=list)
    implementation_notes: Optional[str] = None

    @validator("participation_rate")
    def participation_rate_valid(cls, v, values):
        total = values.get("total_members", 1)
        participating = values.get("participating_members", 0)
        expected_rate = participating / total if total > 0 else 0
        if abs(v - expected_rate) > 0.01:  # Allow small floating point differences
            raise ValueError(
                f"Participation rate {v} does not match calculated rate {expected_rate}"
            )
        return v


class QuorumConsensusEngine:
    """
    Engine for managing quorum consensus processes.

    This implements the consensus mechanism for democratic decision-making
    in the Grace governance system.
    """

    def __init__(self):
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.votes: Dict[str, List[Vote]] = {}  # proposal_id -> votes
        self.members: Dict[str, QuorumMember] = {}
        self.results: Dict[str, ConsensusResult] = {}

    def add_member(self, member: QuorumMember) -> None:
        """Add a member to the quorum."""
        self.members[member.member_id] = member
        logger.info(f"Added quorum member: {member.name} ({member.member_id})")

    def remove_member(self, member_id: str) -> bool:
        """Remove a member from the quorum."""
        if member_id in self.members:
            del self.members[member_id]
            logger.info(f"Removed quorum member: {member_id}")
            return True
        return False

    async def submit_proposal(self, proposal: ConsensusProposal) -> str:
        """Submit a new proposal for consensus."""
        # Validate proposal
        if proposal.proposal_id in self.active_proposals:
            raise ValueError(f"Proposal {proposal.proposal_id} already exists")

        # Initialize voting
        self.active_proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = []

        logger.info(
            f"Submitted proposal for consensus: {proposal.title} ({proposal.proposal_id})"
        )

        # Start voting period
        await self._notify_members_of_proposal(proposal)

        return proposal.proposal_id

    async def cast_vote(
        self,
        proposal_id: str,
        member_id: str,
        vote_type: VoteType,
        reasoning: Optional[str] = None,
        confidence: float = 1.0,
    ) -> bool:
        """Cast a vote on a proposal."""
        # Validate inputs
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")

        member = self.members[member_id]
        if not member.active:
            raise ValueError(f"Member {member_id} is not active")

        proposal = self.active_proposals[proposal_id]
        if datetime.now() > proposal.voting_deadline:
            raise ValueError(f"Voting deadline has passed for proposal {proposal_id}")

        # Check if member already voted
        existing_votes = [
            v for v in self.votes[proposal_id] if v.member_id == member_id
        ]
        if existing_votes:
            raise ValueError(
                f"Member {member_id} has already voted on proposal {proposal_id}"
            )

        # Create and store vote
        vote = Vote(
            member_id=member_id,
            vote_type=vote_type,
            reasoning=reasoning,
            confidence=confidence,
            weight=member.weight,
            timestamp=datetime.now(),
        )

        self.votes[proposal_id].append(vote)

        # Update member activity
        member.last_activity = datetime.now()

        logger.info(f"Vote cast by {member_id} on {proposal_id}: {vote_type.value}")

        # Check if we can close voting early (unanimous decision)
        await self._check_early_closure(proposal_id)

        return True

    async def calculate_result(self, proposal_id: str) -> ConsensusResult:
        """Calculate the consensus result for a proposal."""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.active_proposals[proposal_id]
        votes = self.votes.get(proposal_id, [])

        # Count votes
        vote_counts = {
            VoteType.APPROVE: 0,
            VoteType.REJECT: 0,
            VoteType.ABSTAIN: 0,
            VoteType.DELEGATE: 0,
        }

        weighted_counts = {
            VoteType.APPROVE: 0.0,
            VoteType.REJECT: 0.0,
            VoteType.ABSTAIN: 0.0,
            VoteType.DELEGATE: 0.0,
        }

        participating_members = set()

        for vote in votes:
            vote_counts[vote.vote_type] += 1
            weighted_counts[vote.vote_type] += vote.weight
            participating_members.add(vote.member_id)

        total_members = len([m for m in self.members.values() if m.active])
        participation_rate = (
            len(participating_members) / total_members if total_members > 0 else 0
        )

        # Calculate percentages
        total_votes = sum(vote_counts.values())
        total_weighted = sum(weighted_counts.values())

        approval_percentage = (
            weighted_counts[VoteType.APPROVE] / total_weighted
            if total_weighted > 0
            else 0
        )
        rejection_percentage = (
            weighted_counts[VoteType.REJECT] / total_weighted
            if total_weighted > 0
            else 0
        )

        # Determine if quorum is met
        quorum_met = participation_rate >= proposal.min_participation

        # Determine if consensus is achieved
        consensus_achieved = False
        status = ConsensusStatus.REJECTED

        if quorum_met:
            if proposal.quorum_type == QuorumType.SIMPLE_MAJORITY:
                consensus_achieved = approval_percentage > 0.5
            elif proposal.quorum_type == QuorumType.QUALIFIED_MAJORITY:
                consensus_achieved = approval_percentage >= 2 / 3
            elif proposal.quorum_type == QuorumType.SUPER_MAJORITY:
                consensus_achieved = approval_percentage >= 0.75
            elif proposal.quorum_type == QuorumType.UNANIMOUS:
                consensus_achieved = (
                    approval_percentage == 1.0 and vote_counts[VoteType.ABSTAIN] == 0
                )

            if consensus_achieved:
                status = ConsensusStatus.APPROVED
        else:
            status = ConsensusStatus.REJECTED  # Not enough participation

        # Create result
        result = ConsensusResult(
            proposal_id=proposal_id,
            status=status,
            total_members=total_members,
            participating_members=len(participating_members),
            votes_approve=vote_counts[VoteType.APPROVE],
            votes_reject=vote_counts[VoteType.REJECT],
            votes_abstain=vote_counts[VoteType.ABSTAIN],
            votes_delegate=vote_counts[VoteType.DELEGATE],
            weighted_approve=weighted_counts[VoteType.APPROVE],
            weighted_reject=weighted_counts[VoteType.REJECT],
            weighted_abstain=weighted_counts[VoteType.ABSTAIN],
            participation_rate=participation_rate,
            quorum_met=quorum_met,
            consensus_achieved=consensus_achieved,
            approval_percentage=approval_percentage,
            rejection_percentage=rejection_percentage,
            voting_started=proposal.created_at,
            voting_ended=datetime.now(),
        )

        # Store result and cleanup
        self.results[proposal_id] = result

        # Remove from active proposals
        if proposal_id in self.active_proposals:
            del self.active_proposals[proposal_id]

        logger.info(f"Consensus result calculated for {proposal_id}: {status.value}")

        return result

    async def _check_early_closure(self, proposal_id: str) -> bool:
        """Check if voting can be closed early due to unanimous decision."""
        proposal = self.active_proposals[proposal_id]
        votes = self.votes[proposal_id]

        active_members = [m for m in self.members.values() if m.active]

        # Check if all members have voted
        if len(votes) == len(active_members):
            # All members voted, can close early
            await self.calculate_result(proposal_id)
            return True

        # Check for unanimous approval/rejection
        if len(votes) > 0:
            all_approve = all(v.vote_type == VoteType.APPROVE for v in votes)
            all_reject = all(v.vote_type == VoteType.REJECT for v in votes)

            if (all_approve or all_reject) and len(votes) >= len(
                active_members
            ) * proposal.min_participation:
                # Early unanimous decision
                await self.calculate_result(proposal_id)
                return True

        return False

    async def _notify_members_of_proposal(self, proposal: ConsensusProposal) -> None:
        """Notify all active members of a new proposal."""
        # This would integrate with the notification system
        logger.info(
            f"Notifying {len(self.members)} members of new proposal: {proposal.title}"
        )

    def get_member_voting_history(self, member_id: str) -> Dict[str, Any]:
        """Get voting history for a member."""
        member_votes = []

        for proposal_id, votes in self.votes.items():
            member_vote = next((v for v in votes if v.member_id == member_id), None)
            if member_vote:
                member_votes.append(
                    {
                        "proposal_id": proposal_id,
                        "vote_type": member_vote.vote_type.value,
                        "timestamp": member_vote.timestamp,
                        "confidence": member_vote.confidence,
                    }
                )

        return {
            "member_id": member_id,
            "total_votes": len(member_votes),
            "votes": member_votes,
        }

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get overall consensus statistics."""
        total_proposals = len(self.results)
        approved = len(
            [r for r in self.results.values() if r.status == ConsensusStatus.APPROVED]
        )
        rejected = len(
            [r for r in self.results.values() if r.status == ConsensusStatus.REJECTED]
        )

        avg_participation = (
            sum(r.participation_rate for r in self.results.values()) / total_proposals
            if total_proposals > 0
            else 0
        )

        return {
            "total_proposals": total_proposals,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / total_proposals if total_proposals > 0 else 0,
            "average_participation": avg_participation,
            "active_members": len([m for m in self.members.values() if m.active]),
        }
