"""
Collective Consensus Engine - Group decision-making for distributed Grace instances
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConsensusAlgorithm(Enum):
    """Consensus algorithms available"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    RAFT = "raft"
    BYZANTINE_FAULT_TOLERANT = "bft"
    QUORUM = "quorum"
    DELEGATE_PROOF = "delegate_proof"


class VoteType(Enum):
    """Types of votes"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ProposalStatus(Enum):
    """Status of a proposal"""
    PENDING = "pending"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Vote:
    """Represents a vote from a node"""
    voter_id: str
    vote_type: VoteType
    weight: float = 1.0
    rationale: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """Represents a consensus proposal"""
    proposal_id: str
    proposer_id: str
    proposal_type: str
    content: Dict[str, Any]
    algorithm: ConsensusAlgorithm
    status: ProposalStatus = ProposalStatus.PENDING
    votes: List[Vote] = field(default_factory=list)
    required_votes: int = 3
    quorum_threshold: float = 0.66  # 2/3 majority
    created_at: datetime = field(default_factory=datetime.now)
    voting_deadline: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusRound:
    """Represents a round of consensus"""
    round_id: str
    proposals: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)


class CollectiveConsensusEngine:
    """
    Manages group decision-making across distributed Grace instances
    Implements multiple consensus algorithms for different scenarios
    """
    
    def __init__(self):
        self.proposals: Dict[str, Proposal] = {}
        self.consensus_rounds: Dict[str, ConsensusRound] = {}
        self.node_weights: Dict[str, float] = {}
        self.node_reputations: Dict[str, float] = defaultdict(lambda: 1.0)
        self.consensus_history: List[Dict[str, Any]] = []
        self.quorum_callbacks: Dict[str, List] = defaultdict(list)
        logger.info("CollectiveConsensusEngine initialized")
    
    def register_node(self, node_id: str, weight: float = 1.0, reputation: float = 1.0):
        """Register a node for consensus participation"""
        self.node_weights[node_id] = weight
        self.node_reputations[node_id] = reputation
        logger.info(f"Registered node for consensus: {node_id} (weight: {weight}, reputation: {reputation})")
    
    def create_proposal(
        self,
        proposer_id: str,
        proposal_type: str,
        content: Dict[str, Any],
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE,
        required_votes: Optional[int] = None,
        quorum_threshold: float = 0.66,
        voting_duration: Optional[timedelta] = None
    ) -> Proposal:
        """Create a new consensus proposal"""
        proposal = Proposal(
            proposal_id=str(uuid.uuid4()),
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            content=content,
            algorithm=algorithm,
            required_votes=required_votes or len(self.node_weights),
            quorum_threshold=quorum_threshold
        )
        
        if voting_duration:
            proposal.voting_deadline = datetime.now() + voting_duration
        
        self.proposals[proposal.proposal_id] = proposal
        proposal.status = ProposalStatus.VOTING
        
        logger.info(f"Created proposal: {proposal.proposal_id} (type: {proposal_type}, algorithm: {algorithm.value})")
        
        return proposal
    
    def submit_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_type: VoteType,
        rationale: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Submit a vote for a proposal"""
        if proposal_id not in self.proposals:
            logger.warning(f"Vote for unknown proposal: {proposal_id}")
            return False
        
        proposal = self.proposals[proposal_id]
        
        # Check if voting is still open
        if proposal.status != ProposalStatus.VOTING:
            logger.warning(f"Voting closed for proposal: {proposal_id}")
            return False
        
        # Check deadline
        if proposal.voting_deadline and datetime.now() > proposal.voting_deadline:
            self._close_proposal(proposal, ProposalStatus.EXPIRED)
            return False
        
        # Check if already voted
        if any(v.voter_id == voter_id for v in proposal.votes):
            logger.warning(f"Node {voter_id} already voted on {proposal_id}")
            return False
        
        # Get voter weight
        weight = self.node_weights.get(voter_id, 1.0)
        reputation = self.node_reputations.get(voter_id, 1.0)
        
        vote = Vote(
            voter_id=voter_id,
            vote_type=vote_type,
            weight=weight * reputation,
            rationale=rationale,
            metadata=metadata or {}
        )
        
        proposal.votes.append(vote)
        
        logger.info(f"Vote recorded: {voter_id} -> {vote_type.value} on {proposal_id}")
        
        # Check if consensus reached
        self._check_consensus(proposal)
        
        return True
    
    def _check_consensus(self, proposal: Proposal):
        """Check if consensus has been reached"""
        if proposal.algorithm == ConsensusAlgorithm.MAJORITY_VOTE:
            self._check_majority_consensus(proposal)
        elif proposal.algorithm == ConsensusAlgorithm.WEIGHTED_VOTE:
            self._check_weighted_consensus(proposal)
        elif proposal.algorithm == ConsensusAlgorithm.QUORUM:
            self._check_quorum_consensus(proposal)
        elif proposal.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
            self._check_bft_consensus(proposal)
        else:
            logger.warning(f"Unknown consensus algorithm: {proposal.algorithm}")
    
    def _check_majority_consensus(self, proposal: Proposal):
        """Check simple majority consensus"""
        if len(proposal.votes) < proposal.required_votes:
            return
        
        approve_count = sum(1 for v in proposal.votes if v.vote_type == VoteType.APPROVE)
        reject_count = sum(1 for v in proposal.votes if v.vote_type == VoteType.REJECT)
        total_votes = approve_count + reject_count
        
        if total_votes == 0:
            return
        
        approval_ratio = approve_count / total_votes
        
        if approval_ratio >= proposal.quorum_threshold:
            self._close_proposal(proposal, ProposalStatus.APPROVED, {
                'approval_ratio': approval_ratio,
                'approve_count': approve_count,
                'reject_count': reject_count
            })
        elif approve_count + reject_count >= proposal.required_votes:
            self._close_proposal(proposal, ProposalStatus.REJECTED, {
                'approval_ratio': approval_ratio,
                'approve_count': approve_count,
                'reject_count': reject_count
            })
    
    def _check_weighted_consensus(self, proposal: Proposal):
        """Check weighted vote consensus"""
        total_weight = sum(v.weight for v in proposal.votes)
        approve_weight = sum(v.weight for v in proposal.votes if v.vote_type == VoteType.APPROVE)
        reject_weight = sum(v.weight for v in proposal.votes if v.vote_type == VoteType.REJECT)
        
        if total_weight == 0:
            return
        
        approval_ratio = approve_weight / total_weight
        
        if approval_ratio >= proposal.quorum_threshold and len(proposal.votes) >= proposal.required_votes:
            self._close_proposal(proposal, ProposalStatus.APPROVED, {
                'approval_ratio': approval_ratio,
                'total_weight': total_weight,
                'approve_weight': approve_weight,
                'reject_weight': reject_weight
            })
        elif len(proposal.votes) >= proposal.required_votes:
            self._close_proposal(proposal, ProposalStatus.REJECTED, {
                'approval_ratio': approval_ratio,
                'total_weight': total_weight
            })
    
    def _check_quorum_consensus(self, proposal: Proposal):
        """Check quorum-based consensus"""
        total_nodes = len(self.node_weights)
        quorum_size = int(total_nodes * proposal.quorum_threshold)
        
        if len(proposal.votes) < quorum_size:
            return
        
        approve_count = sum(1 for v in proposal.votes if v.vote_type == VoteType.APPROVE)
        
        if approve_count >= quorum_size:
            self._close_proposal(proposal, ProposalStatus.APPROVED, {
                'quorum_size': quorum_size,
                'approve_count': approve_count,
                'total_votes': len(proposal.votes)
            })
        elif len(proposal.votes) >= proposal.required_votes:
            self._close_proposal(proposal, ProposalStatus.REJECTED, {
                'quorum_size': quorum_size,
                'approve_count': approve_count
            })
    
    def _check_bft_consensus(self, proposal: Proposal):
        """Check Byzantine Fault Tolerant consensus (simplified)"""
        # BFT requires (2f + 1) nodes where f is max faulty nodes
        total_nodes = len(self.node_weights)
        f = (total_nodes - 1) // 3  # Max faulty nodes
        required_honest = 2 * f + 1
        
        if len(proposal.votes) < required_honest:
            return
        
        approve_count = sum(1 for v in proposal.votes if v.vote_type == VoteType.APPROVE)
        
        if approve_count >= required_honest:
            self._close_proposal(proposal, ProposalStatus.APPROVED, {
                'bft_threshold': required_honest,
                'approve_count': approve_count,
                'max_faulty_nodes': f
            })
        elif len(proposal.votes) >= proposal.required_votes:
            self._close_proposal(proposal, ProposalStatus.REJECTED, {
                'bft_threshold': required_honest,
                'approve_count': approve_count
            })
    
    def _close_proposal(self, proposal: Proposal, status: ProposalStatus, result: Optional[Dict] = None):
        """Close a proposal with a status"""
        proposal.status = status
        proposal.result = result or {}
        
        # Update node reputations based on outcome
        if status == ProposalStatus.APPROVED:
            # Reward nodes that voted correctly
            for vote in proposal.votes:
                if vote.vote_type == VoteType.APPROVE:
                    self.node_reputations[vote.voter_id] *= 1.05
        
        # Record in history
        self.consensus_history.append({
            'proposal_id': proposal.proposal_id,
            'type': proposal.proposal_type,
            'status': status.value,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Proposal {proposal.proposal_id} closed: {status.value}")
        
        # Trigger callbacks
        self._trigger_quorum_callbacks(proposal)
    
    def _trigger_quorum_callbacks(self, proposal: Proposal):
        """Trigger callbacks for quorum integration"""
        for callback in self.quorum_callbacks.get(proposal.proposal_type, []):
            try:
                callback(proposal)
            except Exception as e:
                logger.error(f"Quorum callback error: {e}")
    
    def register_quorum_callback(self, proposal_type: str, callback):
        """Register callback for quorum integration"""
        self.quorum_callbacks[proposal_type].append(callback)
        logger.info(f"Registered quorum callback for {proposal_type}")
    
    def start_consensus_round(self, proposals: List[str]) -> ConsensusRound:
        """Start a new consensus round with multiple proposals"""
        round_obj = ConsensusRound(
            round_id=str(uuid.uuid4()),
            proposals=proposals
        )
        
        self.consensus_rounds[round_obj.round_id] = round_obj
        
        logger.info(f"Started consensus round: {round_obj.round_id} with {len(proposals)} proposals")
        
        return round_obj
    
    def complete_consensus_round(self, round_id: str) -> Dict[str, Any]:
        """Complete a consensus round and aggregate results"""
        if round_id not in self.consensus_rounds:
            return {}
        
        round_obj = self.consensus_rounds[round_id]
        round_obj.completed_at = datetime.now()
        
        results = {}
        for proposal_id in round_obj.proposals:
            if proposal_id in self.proposals:
                proposal = self.proposals[proposal_id]
                results[proposal_id] = {
                    'status': proposal.status.value,
                    'result': proposal.result
                }
        
        round_obj.results = results
        
        logger.info(f"Completed consensus round: {round_id}")
        
        return results
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a proposal"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        
        return {
            'proposal_id': proposal.proposal_id,
            'type': proposal.proposal_type,
            'status': proposal.status.value,
            'algorithm': proposal.algorithm.value,
            'votes': len(proposal.votes),
            'required_votes': proposal.required_votes,
            'result': proposal.result,
            'created_at': proposal.created_at.isoformat(),
            'voting_deadline': proposal.voting_deadline.isoformat() if proposal.voting_deadline else None
        }
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get consensus engine statistics"""
        total_proposals = len(self.proposals)
        approved = sum(1 for p in self.proposals.values() if p.status == ProposalStatus.APPROVED)
        rejected = sum(1 for p in self.proposals.values() if p.status == ProposalStatus.REJECTED)
        
        return {
            'total_proposals': total_proposals,
            'approved': approved,
            'rejected': rejected,
            'pending': sum(1 for p in self.proposals.values() if p.status == ProposalStatus.VOTING),
            'approval_rate': approved / total_proposals if total_proposals > 0 else 0,
            'registered_nodes': len(self.node_weights),
            'consensus_rounds': len(self.consensus_rounds),
            'avg_votes_per_proposal': sum(len(p.votes) for p in self.proposals.values()) / total_proposals if total_proposals > 0 else 0
        }
    
    def delegate_decision(
        self,
        decision_type: str,
        options: List[Dict[str, Any]],
        delegates: List[str]
    ) -> Dict[str, Any]:
        """Delegate a decision to specific expert nodes"""
        proposal = self.create_proposal(
            proposer_id="system",
            proposal_type=decision_type,
            content={'options': options},
            algorithm=ConsensusAlgorithm.DELEGATE_PROOF,
            required_votes=len(delegates)
        )
        
        # Only count votes from delegates
        proposal.metadata['delegates'] = delegates
        
        return {
            'proposal_id': proposal.proposal_id,
            'delegates': delegates,
            'options': options
        }
