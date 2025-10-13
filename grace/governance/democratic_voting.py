"""
Democratic Parliament Voting Logic - Advanced democratic decision algorithms.

Implements sophisticated democratic voting mechanisms as specified in the missing
components requirements. Includes:
- Weighted voting systems
- Consensus building algorithms
- Multi-round deliberation
- Expertise-based weighting
- Conflict resolution mechanisms
- Transparent audit trails
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VotingMethod(Enum):
    """Democratic voting methods."""

    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    SUPERMAJORITY = "supermajority"
    CONSENSUS = "consensus"
    RANKED_CHOICE = "ranked_choice"
    EXPERTISE_WEIGHTED = "expertise_weighted"
    MULTI_ROUND = "multi_round"
    BORDA_COUNT = "borda_count"


class VoteValue(Enum):
    """Possible vote values."""

    STRONGLY_APPROVE = 2
    APPROVE = 1
    ABSTAIN = 0
    OPPOSE = -1
    STRONGLY_OPPOSE = -2


class ProposalCategory(Enum):
    """Categories of proposals requiring different voting thresholds."""

    OPERATIONAL = "operational"  # 50% threshold
    POLICY = "policy"  # 60% threshold
    CONSTITUTIONAL = "constitutional"  # 75% threshold
    EMERGENCY = "emergency"  # 40% threshold, expedited
    SECURITY = "security"  # 70% threshold
    ETHICAL = "ethical"  # 80% threshold


@dataclass
class DemocraticVote:
    """Represents a vote in the democratic process."""

    voter_id: str
    voter_name: str
    vote_value: VoteValue
    reasoning: str
    confidence: float  # 0.0 to 1.0
    expertise_relevance: float  # How relevant is voter's expertise to this topic
    timestamp: datetime
    deliberation_rounds: int = 0
    changed_mind: bool = False  # Did voter change their mind after deliberation

    def get_weighted_value(self, base_weight: float = 1.0) -> float:
        """Get vote value weighted by confidence and expertise."""
        expertise_multiplier = 0.5 + (self.expertise_relevance * 0.5)
        confidence_multiplier = 0.7 + (self.confidence * 0.3)

        return (
            float(self.vote_value.value)
            * base_weight
            * expertise_multiplier
            * confidence_multiplier
        )


@dataclass
class VotingRound:
    """Represents a round of voting in multi-round deliberation."""

    round_number: int
    votes: List[DemocraticVote]
    deliberation_notes: List[str]
    timestamp: datetime
    outcome: Optional[str] = None
    consensus_level: float = 0.0

    def get_vote_distribution(self) -> Dict[VoteValue, int]:
        """Get distribution of votes."""
        distribution = {vote_value: 0 for vote_value in VoteValue}
        for vote in self.votes:
            distribution[vote.vote_value] += 1
        return distribution

    def calculate_weighted_outcome(self) -> Tuple[float, float]:
        """Calculate weighted outcome score and confidence."""
        if not self.votes:
            return 0.0, 0.0

        total_weighted_score = sum(vote.get_weighted_value() for vote in self.votes)
        total_weight = sum(
            vote.confidence * vote.expertise_relevance * abs(vote.vote_value.value)
            for vote in self.votes
            if vote.vote_value != VoteValue.ABSTAIN
        )

        weighted_average = total_weighted_score / len(self.votes) if self.votes else 0.0
        confidence = total_weight / len(self.votes) if self.votes else 0.0

        return weighted_average, min(1.0, confidence)


@dataclass
class DemocraticProposal:
    """Enhanced proposal for democratic deliberation."""

    proposal_id: str
    title: str
    description: str
    category: ProposalCategory
    proposer: str
    required_threshold: float
    voting_method: VotingMethod

    # Deliberation process
    voting_rounds: List[VotingRound] = field(default_factory=list)
    eligible_voters: Set[str] = field(default_factory=set)
    current_round: int = 0
    max_rounds: int = 3

    # Context and metadata
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    expert_opinions: List[Dict[str, Any]] = field(default_factory=list)

    # Status tracking
    status: str = "pending"  # pending, deliberating, decided, withdrawn
    final_outcome: Optional[str] = None
    decision_rationale: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None

    def get_current_consensus_level(self) -> float:
        """Calculate current level of consensus."""
        if not self.voting_rounds:
            return 0.0

        latest_round = self.voting_rounds[-1]
        distribution = latest_round.get_vote_distribution()

        # Calculate consensus as inverse of vote spread
        total_votes = sum(distribution.values())
        if total_votes == 0:
            return 0.0

        # Find the dominant position
        max_votes = max(distribution.values())
        consensus = max_votes / total_votes

        # Adjust for abstentions (they don't count against consensus)
        non_abstain_votes = total_votes - distribution[VoteValue.ABSTAIN]
        if non_abstain_votes > 0:
            adjusted_consensus = max_votes / non_abstain_votes
            consensus = min(consensus, adjusted_consensus)

        return consensus


class DemocraticVotingEngine:
    """Advanced democratic voting engine with multiple algorithms."""

    def __init__(self, event_bus=None, memory_core=None):
        self.event_bus = event_bus
        self.memory_core = memory_core

        # Voter registry with expertise profiles
        self.voter_registry: Dict[str, Dict[str, Any]] = {}

        # Active proposals
        self.active_proposals: Dict[str, DemocraticProposal] = {}

        # Historical decisions for learning
        self.decision_history: List[Dict[str, Any]] = []

        # Voting thresholds by category
        self.category_thresholds = {
            ProposalCategory.OPERATIONAL: 0.50,
            ProposalCategory.POLICY: 0.60,
            ProposalCategory.CONSTITUTIONAL: 0.75,
            ProposalCategory.EMERGENCY: 0.40,
            ProposalCategory.SECURITY: 0.70,
            ProposalCategory.ETHICAL: 0.80,
        }

        # Default voting methods by category
        self.default_voting_methods = {
            ProposalCategory.OPERATIONAL: VotingMethod.SIMPLE_MAJORITY,
            ProposalCategory.POLICY: VotingMethod.WEIGHTED_MAJORITY,
            ProposalCategory.CONSTITUTIONAL: VotingMethod.SUPERMAJORITY,
            ProposalCategory.EMERGENCY: VotingMethod.EXPERTISE_WEIGHTED,
            ProposalCategory.SECURITY: VotingMethod.EXPERTISE_WEIGHTED,
            ProposalCategory.ETHICAL: VotingMethod.CONSENSUS,
        }

    async def register_voter(
        self,
        voter_id: str,
        name: str,
        expertise_areas: List[str],
        base_weight: float = 1.0,
        credentials: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a voter with their expertise profile."""
        voter_profile = {
            "voter_id": voter_id,
            "name": name,
            "expertise_areas": expertise_areas,
            "base_weight": base_weight,
            "credentials": credentials or {},
            "voting_history": [],
            "reputation_score": 0.5,  # Start with neutral reputation
            "registration_date": datetime.now(),
        }

        self.voter_registry[voter_id] = voter_profile
        logger.info(f"Registered voter {voter_id} with expertise in {expertise_areas}")

        return True

    async def submit_proposal(
        self,
        title: str,
        description: str,
        category: ProposalCategory,
        proposer: str,
        supporting_evidence: Optional[List[Dict[str, Any]]] = None,
        custom_threshold: Optional[float] = None,
        custom_voting_method: Optional[VotingMethod] = None,
        deadline_hours: int = 168,
    ) -> str:  # Default 1 week
        """Submit a proposal for democratic consideration."""

        proposal_id = f"prop_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Determine voting parameters
        threshold = custom_threshold or self.category_thresholds[category]
        voting_method = custom_voting_method or self.default_voting_methods[category]

        # Create proposal
        proposal = DemocraticProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            category=category,
            proposer=proposer,
            required_threshold=threshold,
            voting_method=voting_method,
            deadline=datetime.now() + timedelta(hours=deadline_hours),
        )

        if supporting_evidence:
            proposal.supporting_evidence = supporting_evidence

        # Identify eligible voters based on expertise relevance
        proposal.eligible_voters = await self._identify_eligible_voters(proposal)

        self.active_proposals[proposal_id] = proposal

        # Notify eligible voters
        await self._notify_eligible_voters(proposal)

        logger.info(
            f"Submitted proposal {proposal_id} ({category.value}) with {voting_method.value}"
        )

        return proposal_id

    async def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_value: VoteValue,
        reasoning: str,
        confidence: float = 1.0,
    ) -> bool:
        """Cast a vote on a proposal."""

        if proposal_id not in self.active_proposals:
            logger.warning(f"Proposal {proposal_id} not found")
            return False

        proposal = self.active_proposals[proposal_id]

        if voter_id not in proposal.eligible_voters:
            logger.warning(f"Voter {voter_id} not eligible for proposal {proposal_id}")
            return False

        if proposal.status != "pending" and proposal.status != "deliberating":
            logger.warning(f"Voting closed for proposal {proposal_id}")
            return False

        # Calculate expertise relevance
        expertise_relevance = await self._calculate_expertise_relevance(
            voter_id, proposal
        )

        # Create vote
        vote = DemocraticVote(
            voter_id=voter_id,
            voter_name=self.voter_registry[voter_id]["name"],
            vote_value=vote_value,
            reasoning=reasoning,
            confidence=confidence,
            expertise_relevance=expertise_relevance,
            timestamp=datetime.now(),
            deliberation_rounds=proposal.current_round,
        )

        # Add vote to current round or create new round
        if not proposal.voting_rounds or proposal.current_round == 0:
            proposal.voting_rounds.append(
                VotingRound(
                    round_number=1,
                    votes=[vote],
                    deliberation_notes=[],
                    timestamp=datetime.now(),
                )
            )
            proposal.current_round = 1
        else:
            # Check if voter already voted in this round
            current_round = proposal.voting_rounds[-1]
            existing_vote_idx = None
            for i, existing_vote in enumerate(current_round.votes):
                if existing_vote.voter_id == voter_id:
                    existing_vote_idx = i
                    break

            if existing_vote_idx is not None:
                # Replace existing vote
                old_vote = current_round.votes[existing_vote_idx]
                vote.changed_mind = old_vote.vote_value != vote_value
                current_round.votes[existing_vote_idx] = vote
            else:
                # Add new vote
                current_round.votes.append(vote)

        proposal.status = "deliberating"

        # Check if we can make a decision
        decision = await self._evaluate_proposal_outcome(proposal)
        if decision["can_decide"]:
            await self._finalize_proposal(proposal, decision)
        else:
            # Check if we need another round of deliberation
            await self._consider_new_deliberation_round(proposal)

        logger.info(
            f"Vote cast by {voter_id} on proposal {proposal_id}: {vote_value.name}"
        )

        return True

    async def _identify_eligible_voters(self, proposal: DemocraticProposal) -> Set[str]:
        """Identify voters eligible for this proposal based on expertise."""
        eligible = set()

        # All voters are eligible for constitutional and emergency matters
        if proposal.category in [
            ProposalCategory.CONSTITUTIONAL,
            ProposalCategory.EMERGENCY,
        ]:
            return set(self.voter_registry.keys())

        # For other categories, consider expertise relevance
        for voter_id, voter_profile in self.voter_registry.items():
            relevance = await self._calculate_proposal_voter_relevance(
                proposal, voter_profile
            )

            # Include voters with sufficient relevance
            if relevance > 0.2:  # 20% relevance threshold
                eligible.add(voter_id)

        # Ensure minimum participation
        if len(eligible) < 3 and len(self.voter_registry) > 3:
            # Add highest reputation voters to reach minimum
            remaining_voters = set(self.voter_registry.keys()) - eligible
            sorted_by_reputation = sorted(
                remaining_voters,
                key=lambda v: self.voter_registry[v]["reputation_score"],
                reverse=True,
            )
            eligible.update(sorted_by_reputation[: 3 - len(eligible)])

        return eligible

    async def _calculate_expertise_relevance(
        self, voter_id: str, proposal: DemocraticProposal
    ) -> float:
        """Calculate how relevant voter's expertise is to the proposal."""
        if voter_id not in self.voter_registry:
            return 0.0

        voter_profile = self.voter_registry[voter_id]
        return await self._calculate_proposal_voter_relevance(proposal, voter_profile)

    async def _calculate_proposal_voter_relevance(
        self, proposal: DemocraticProposal, voter_profile: Dict[str, Any]
    ) -> float:
        """Calculate relevance of voter's expertise to proposal."""
        expertise_areas = voter_profile.get("expertise_areas", [])

        if not expertise_areas:
            return 0.5  # Neutral relevance

        # Category-specific relevance mapping
        category_keywords = {
            ProposalCategory.OPERATIONAL: [
                "operations",
                "management",
                "process",
                "workflow",
            ],
            ProposalCategory.POLICY: ["policy", "governance", "strategy", "planning"],
            ProposalCategory.CONSTITUTIONAL: [
                "legal",
                "constitutional",
                "rights",
                "principles",
            ],
            ProposalCategory.EMERGENCY: ["security", "crisis", "emergency", "response"],
            ProposalCategory.SECURITY: ["security", "privacy", "cyber", "threat"],
            ProposalCategory.ETHICAL: ["ethics", "fairness", "moral", "responsibility"],
        }

        relevant_keywords = category_keywords.get(proposal.category, [])

        # Calculate relevance based on keyword matching
        relevance_score = 0.0
        for area in expertise_areas:
            area_lower = area.lower()
            for keyword in relevant_keywords:
                if keyword in area_lower or area_lower in keyword:
                    relevance_score += 0.25

        # Also check proposal text for expertise area mentions
        proposal_text = f"{proposal.title} {proposal.description}".lower()
        for area in expertise_areas:
            if area.lower() in proposal_text:
                relevance_score += 0.1

        # Normalize and bound
        return min(1.0, relevance_score)

    async def _evaluate_proposal_outcome(
        self, proposal: DemocraticProposal
    ) -> Dict[str, Any]:
        """Evaluate if proposal can be decided and what the outcome would be."""
        if not proposal.voting_rounds:
            return {"can_decide": False, "reason": "No votes cast"}

        current_round = proposal.voting_rounds[-1]

        # Check if minimum participation achieved
        min_participation = max(
            3, len(proposal.eligible_voters) * 0.5
        )  # 50% of eligible voters
        if len(current_round.votes) < min_participation:
            return {
                "can_decide": False,
                "reason": f"Insufficient participation ({len(current_round.votes)}/{min_participation})",
            }

        # Apply voting method
        voting_method = proposal.voting_method

        if voting_method == VotingMethod.SIMPLE_MAJORITY:
            result = await self._evaluate_simple_majority(proposal, current_round)
        elif voting_method == VotingMethod.WEIGHTED_MAJORITY:
            result = await self._evaluate_weighted_majority(proposal, current_round)
        elif voting_method == VotingMethod.SUPERMAJORITY:
            result = await self._evaluate_supermajority(proposal, current_round)
        elif voting_method == VotingMethod.CONSENSUS:
            result = await self._evaluate_consensus(proposal, current_round)
        elif voting_method == VotingMethod.EXPERTISE_WEIGHTED:
            result = await self._evaluate_expertise_weighted(proposal, current_round)
        else:
            result = await self._evaluate_weighted_majority(
                proposal, current_round
            )  # Default

        return result

    async def _evaluate_simple_majority(
        self, proposal: DemocraticProposal, round: VotingRound
    ) -> Dict[str, Any]:
        """Evaluate using simple majority rule."""
        approve_votes = sum(1 for vote in round.votes if vote.vote_value.value > 0)
        oppose_votes = sum(1 for vote in round.votes if vote.vote_value.value < 0)
        total_decisive = approve_votes + oppose_votes

        if total_decisive == 0:
            return {"can_decide": False, "reason": "All votes abstained"}

        approval_ratio = approve_votes / total_decisive

        return {
            "can_decide": True,
            "outcome": "approved" if approval_ratio > 0.5 else "rejected",
            "approval_ratio": approval_ratio,
            "method": "simple_majority",
            "approve_votes": approve_votes,
            "oppose_votes": oppose_votes,
            "abstain_votes": len(round.votes) - total_decisive,
        }

    async def _evaluate_weighted_majority(
        self, proposal: DemocraticProposal, round: VotingRound
    ) -> Dict[str, Any]:
        """Evaluate using weighted majority (considers vote strength and voter weight)."""
        weighted_score, confidence = round.calculate_weighted_outcome()

        # Normalize to approval ratio
        # Weighted score ranges from -2 to +2, normalize to 0-1
        approval_ratio = (weighted_score + 2) / 4

        meets_threshold = approval_ratio >= proposal.required_threshold

        return {
            "can_decide": confidence > 0.6,  # Require reasonable confidence
            "outcome": "approved" if meets_threshold else "rejected",
            "approval_ratio": approval_ratio,
            "confidence": confidence,
            "method": "weighted_majority",
            "weighted_score": weighted_score,
            "required_threshold": proposal.required_threshold,
        }

    async def _evaluate_supermajority(
        self, proposal: DemocraticProposal, round: VotingRound
    ) -> Dict[str, Any]:
        """Evaluate using supermajority rule (typically 2/3 or 3/4)."""
        approve_votes = sum(1 for vote in round.votes if vote.vote_value.value > 0)
        total_votes = len(
            [vote for vote in round.votes if vote.vote_value != VoteValue.ABSTAIN]
        )

        if total_votes == 0:
            return {"can_decide": False, "reason": "All votes abstained"}

        approval_ratio = approve_votes / total_votes
        supermajority_threshold = max(proposal.required_threshold, 0.67)  # At least 2/3

        return {
            "can_decide": True,
            "outcome": "approved"
            if approval_ratio >= supermajority_threshold
            else "rejected",
            "approval_ratio": approval_ratio,
            "method": "supermajority",
            "required_threshold": supermajority_threshold,
            "approve_votes": approve_votes,
            "total_decisive_votes": total_votes,
        }

    async def _evaluate_consensus(
        self, proposal: DemocraticProposal, round: VotingRound
    ) -> Dict[str, Any]:
        """Evaluate using consensus building approach."""
        consensus_level = proposal.get_current_consensus_level()
        distribution = round.get_vote_distribution()

        # Check for strong opposition
        strong_oppose = distribution[VoteValue.STRONGLY_OPPOSE]
        if strong_oppose > len(round.votes) * 0.1:  # More than 10% strongly oppose
            return {
                "can_decide": False,
                "reason": f"Strong opposition ({strong_oppose} votes) prevents consensus",
                "consensus_level": consensus_level,
            }

        # Consensus requires very high agreement
        consensus_threshold = 0.85
        can_decide = consensus_level >= consensus_threshold

        # Determine outcome based on dominant position
        approve_votes = (
            distribution[VoteValue.STRONGLY_APPROVE] + distribution[VoteValue.APPROVE]
        )
        oppose_votes = (
            distribution[VoteValue.STRONGLY_OPPOSE] + distribution[VoteValue.OPPOSE]
        )

        outcome = "approved" if approve_votes > oppose_votes else "rejected"

        return {
            "can_decide": can_decide,
            "outcome": outcome,
            "consensus_level": consensus_level,
            "method": "consensus",
            "required_consensus": consensus_threshold,
            "vote_distribution": distribution,
        }

    async def _evaluate_expertise_weighted(
        self, proposal: DemocraticProposal, round: VotingRound
    ) -> Dict[str, Any]:
        """Evaluate using expertise-weighted scoring."""
        total_weighted_score = 0.0
        total_expertise_weight = 0.0

        for vote in round.votes:
            if vote.vote_value == VoteValue.ABSTAIN:
                continue

            expertise_weight = vote.expertise_relevance * vote.confidence
            vote_contribution = float(vote.vote_value.value) * expertise_weight

            total_weighted_score += vote_contribution
            total_expertise_weight += expertise_weight

        if total_expertise_weight == 0:
            return {"can_decide": False, "reason": "No expert opinions available"}

        # Normalize score to 0-1 range
        normalized_score = (total_weighted_score / total_expertise_weight + 2) / 4

        meets_threshold = normalized_score >= proposal.required_threshold

        return {
            "can_decide": True,
            "outcome": "approved" if meets_threshold else "rejected",
            "approval_ratio": normalized_score,
            "method": "expertise_weighted",
            "total_expertise_weight": total_expertise_weight,
            "required_threshold": proposal.required_threshold,
        }

    async def _finalize_proposal(
        self, proposal: DemocraticProposal, decision: Dict[str, Any]
    ) -> None:
        """Finalize the proposal with the decision."""
        proposal.status = "decided"
        proposal.final_outcome = decision["outcome"]

        # Generate decision rationale
        rationale_parts = [
            f"Decision: {decision['outcome'].upper()}",
            f"Method: {decision['method']}",
            f"Approval ratio: {decision.get('approval_ratio', 0):.2%}",
            f"Required threshold: {proposal.required_threshold:.2%}",
        ]

        if "reason" in decision:
            rationale_parts.append(f"Reason: {decision['reason']}")

        proposal.decision_rationale = " | ".join(rationale_parts)

        # Record in history
        decision_record = {
            "proposal_id": proposal.proposal_id,
            "outcome": decision["outcome"],
            "decision_method": decision["method"],
            "category": proposal.category.value,
            "timestamp": datetime.now(),
            "voting_rounds": len(proposal.voting_rounds),
            "total_votes": sum(len(round.votes) for round in proposal.voting_rounds),
            "decision_data": decision,
        }

        self.decision_history.append(decision_record)

        # Remove from active proposals
        if proposal.proposal_id in self.active_proposals:
            del self.active_proposals[proposal.proposal_id]

        # Notify stakeholders
        await self._notify_decision(proposal, decision)

        logger.info(f"Finalized proposal {proposal.proposal_id}: {decision['outcome']}")

    async def _consider_new_deliberation_round(
        self, proposal: DemocraticProposal
    ) -> None:
        """Consider if a new deliberation round is needed."""
        if proposal.current_round >= proposal.max_rounds:
            # Force decision with current data
            decision = await self._evaluate_proposal_outcome(proposal)
            decision["can_decide"] = True  # Force decision
            await self._finalize_proposal(proposal, decision)
            return

        # Check if there are significant disagreements that could benefit from deliberation
        current_round = proposal.voting_rounds[-1]
        consensus_level = proposal.get_current_consensus_level()

        if consensus_level < 0.6:  # Low consensus might benefit from more discussion
            await self._initiate_deliberation_round(proposal)

    async def _initiate_deliberation_round(self, proposal: DemocraticProposal) -> None:
        """Initiate a new round of deliberation."""
        proposal.current_round += 1

        # Create new voting round
        new_round = VotingRound(
            round_number=proposal.current_round,
            votes=[],
            deliberation_notes=[],
            timestamp=datetime.now(),
        )

        proposal.voting_rounds.append(new_round)

        # Notify voters about new deliberation round
        await self._notify_deliberation_round(proposal)

        logger.info(
            f"Started deliberation round {proposal.current_round} for proposal {proposal.proposal_id}"
        )

    async def _notify_eligible_voters(self, proposal: DemocraticProposal) -> None:
        """Notify eligible voters about new proposal."""
        if self.event_bus:
            await self.event_bus.publish(
                "PARLIAMENT_NEW_PROPOSAL",
                {
                    "proposal_id": proposal.proposal_id,
                    "title": proposal.title,
                    "category": proposal.category.value,
                    "eligible_voters": list(proposal.eligible_voters),
                    "deadline": proposal.deadline.isoformat()
                    if proposal.deadline
                    else None,
                },
            )

    async def _notify_deliberation_round(self, proposal: DemocraticProposal) -> None:
        """Notify voters about new deliberation round."""
        if self.event_bus:
            await self.event_bus.publish(
                "PARLIAMENT_DELIBERATION_ROUND",
                {
                    "proposal_id": proposal.proposal_id,
                    "round_number": proposal.current_round,
                    "eligible_voters": list(proposal.eligible_voters),
                },
            )

    async def _notify_decision(
        self, proposal: DemocraticProposal, decision: Dict[str, Any]
    ) -> None:
        """Notify stakeholders about final decision."""
        if self.event_bus:
            await self.event_bus.publish(
                "PARLIAMENT_DECISION_MADE",
                {
                    "proposal_id": proposal.proposal_id,
                    "outcome": decision["outcome"],
                    "method": decision["method"],
                    "rationale": proposal.decision_rationale,
                },
            )

    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a proposal."""
        if proposal_id in self.active_proposals:
            proposal = self.active_proposals[proposal_id]
            return {
                "proposal_id": proposal_id,
                "status": proposal.status,
                "category": proposal.category.value,
                "voting_method": proposal.voting_method.value,
                "current_round": proposal.current_round,
                "total_rounds": len(proposal.voting_rounds),
                "consensus_level": proposal.get_current_consensus_level(),
                "eligible_voters": len(proposal.eligible_voters),
                "votes_cast": sum(len(round.votes) for round in proposal.voting_rounds),
                "deadline": proposal.deadline.isoformat()
                if proposal.deadline
                else None,
            }

        # Check decision history
        for record in self.decision_history:
            if record["proposal_id"] == proposal_id:
                return {
                    "proposal_id": proposal_id,
                    "status": "decided",
                    "outcome": record["outcome"],
                    "decision_method": record["decision_method"],
                    "timestamp": record["timestamp"].isoformat(),
                }

        return None

    def get_voting_statistics(self) -> Dict[str, Any]:
        """Get comprehensive voting statistics."""
        total_proposals = len(self.decision_history)
        if total_proposals == 0:
            return {"total_proposals": 0}

        outcomes = [record["outcome"] for record in self.decision_history]
        methods = [record["decision_method"] for record in self.decision_history]

        return {
            "total_proposals": total_proposals,
            "active_proposals": len(self.active_proposals),
            "approval_rate": outcomes.count("approved") / total_proposals,
            "rejection_rate": outcomes.count("rejected") / total_proposals,
            "method_distribution": {
                method: methods.count(method) for method in set(methods)
            },
            "average_rounds": sum(
                record["voting_rounds"] for record in self.decision_history
            )
            / total_proposals,
            "registered_voters": len(self.voter_registry),
        }


# Global instance for democratic voting
democratic_voting_engine = DemocraticVotingEngine()
