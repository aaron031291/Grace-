"""
Class 8: Specialist Consensus - MLDL Specialist evaluation with quorum logic
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SpecialistType(Enum):
    """Types of specialist models"""
    REASONING = "reasoning"
    ETHICS = "ethics"
    SAFETY = "safety"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"


@dataclass
class SpecialistVote:
    """Vote from a specialist"""
    specialist_id: str
    specialist_type: SpecialistType
    decision: bool
    confidence: float
    rationale: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuorumResult:
    """Result of quorum consensus"""
    decision: bool
    confidence: float
    votes_for: int
    votes_against: int
    total_weight_for: float
    total_weight_against: float
    consensus_strength: float
    participating_specialists: List[str]
    rationales: List[str]


class MLDLSpecialist:
    """
    Multi-Layer Deep Learning Specialist with quorum evaluation
    Core implementation of Class 8
    """
    
    def __init__(self):
        self.specialists: Dict[str, Dict[str, Any]] = {}
        self.quorum_threshold = 0.66  # 2/3 majority
        self.evaluation_history: List[Dict[str, Any]] = []
        logger.info("MLDLSpecialist initialized")
    
    def register_specialist(
        self,
        specialist_id: str,
        specialist_type: SpecialistType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """Register a specialist for quorum"""
        self.specialists[specialist_id] = {
            'type': specialist_type,
            'weight': weight,
            'metadata': metadata or {},
            'evaluations': 0
        }
        logger.info(f"Registered specialist: {specialist_id} ({specialist_type.value})")
    
    def evaluate(
        self,
        proposal: Dict[str, Any],
        required_specialists: Optional[Set[SpecialistType]] = None,
        quorum_threshold: Optional[float] = None
    ) -> QuorumResult:
        """
        Evaluate proposal with specialist quorum logic
        Core implementation of Class 8
        """
        threshold = quorum_threshold or self.quorum_threshold
        votes: List[SpecialistVote] = []
        
        # Determine which specialists should vote
        if required_specialists:
            eligible_specialists = [
                (sid, sdata) for sid, sdata in self.specialists.items()
                if sdata['type'] in required_specialists
            ]
        else:
            eligible_specialists = list(self.specialists.items())
        
        if not eligible_specialists:
            logger.warning("No eligible specialists for evaluation")
            return QuorumResult(
                decision=False,
                confidence=0.0,
                votes_for=0,
                votes_against=0,
                total_weight_for=0.0,
                total_weight_against=0.0,
                consensus_strength=0.0,
                participating_specialists=[],
                rationales=[]
            )
        
        # Collect votes from specialists
        for specialist_id, specialist_data in eligible_specialists:
            vote = self._get_specialist_vote(
                specialist_id,
                specialist_data,
                proposal
            )
            votes.append(vote)
            specialist_data['evaluations'] += 1
        
        # Calculate quorum
        result = self._calculate_quorum(votes, threshold)
        
        # Record evaluation
        self.evaluation_history.append({
            'proposal': proposal,
            'result': result,
            'votes': votes,
            'timestamp': datetime.now(timezone.utc).isoformat()  # FIXED: timezone-aware
        })
        
        logger.info(
            f"Quorum evaluation: {result.decision} "
            f"(confidence: {result.confidence:.2f}, "
            f"consensus: {result.consensus_strength:.2f})"
        )
        
        return result
    
    def _get_specialist_vote(
        self,
        specialist_id: str,
        specialist_data: Dict[str, Any],
        proposal: Dict[str, Any]
    ) -> SpecialistVote:
        """Get vote from a specialist"""
        specialist_type = specialist_data['type']
        
        # Simulate specialist evaluation based on type
        decision, confidence, rationale = self._specialist_evaluation(
            specialist_type,
            proposal
        )
        
        return SpecialistVote(
            specialist_id=specialist_id,
            specialist_type=specialist_type,
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            weight=specialist_data['weight']
        )
    
    def _specialist_evaluation(
        self,
        specialist_type: SpecialistType,
        proposal: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        """Simulate specialist evaluation logic"""
        
        if specialist_type == SpecialistType.ETHICS:
            # Ethics specialist checks
            ethical_score = proposal.get('ethical_score', 0.5)
            decision = ethical_score >= 0.7
            confidence = abs(ethical_score - 0.5) * 2
            rationale = f"Ethical analysis: score {ethical_score:.2f}"
            
        elif specialist_type == SpecialistType.SAFETY:
            # Safety specialist checks
            safety_score = proposal.get('safety_score', 0.5)
            decision = safety_score >= 0.8
            confidence = abs(safety_score - 0.5) * 2
            rationale = f"Safety analysis: score {safety_score:.2f}"
            
        elif specialist_type == SpecialistType.REASONING:
            # Reasoning specialist checks
            logic_score = proposal.get('logic_score', 0.5)
            decision = logic_score >= 0.6
            confidence = abs(logic_score - 0.5) * 2
            rationale = f"Reasoning analysis: score {logic_score:.2f}"
            
        elif specialist_type == SpecialistType.CREATIVITY:
            # Creativity specialist checks
            novelty_score = proposal.get('novelty_score', 0.5)
            decision = novelty_score >= 0.5
            confidence = novelty_score
            rationale = f"Creativity analysis: novelty {novelty_score:.2f}"
            
        else:
            # Default evaluation
            decision = proposal.get('approved', True)
            confidence = 0.5
            rationale = f"{specialist_type.value} evaluation: default"
        
        return decision, confidence, rationale
    
    def _calculate_quorum(
        self,
        votes: List[SpecialistVote],
        threshold: float
    ) -> QuorumResult:
        """Calculate quorum result from votes"""
        votes_for = sum(1 for v in votes if v.decision)
        votes_against = len(votes) - votes_for
        
        total_weight_for = sum(v.weight for v in votes if v.decision)
        total_weight_against = sum(v.weight for v in votes if not v.decision)
        total_weight = total_weight_for + total_weight_against
        
        # Calculate consensus strength
        if total_weight > 0:
            consensus_strength = total_weight_for / total_weight
        else:
            consensus_strength = 0.0
        
        # Determine decision
        decision = consensus_strength >= threshold
        
        # Calculate confidence (weighted average of voting specialists)
        if votes:
            confidence = sum(v.confidence * v.weight for v in votes if v.decision == decision) / \
                        sum(v.weight for v in votes if v.decision == decision) if \
                        sum(v.weight for v in votes if v.decision == decision) > 0 else 0.0
        else:
            confidence = 0.0
        
        return QuorumResult(
            decision=decision,
            confidence=confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_weight_for=total_weight_for,
            total_weight_against=total_weight_against,
            consensus_strength=consensus_strength,
            participating_specialists=[v.specialist_id for v in votes],
            rationales=[v.rationale for v in votes]
        )
    
    def get_specialist_performance(self, specialist_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specialist"""
        if specialist_id not in self.specialists:
            return None
        
        specialist = self.specialists[specialist_id]
        
        # Analyze historical votes
        specialist_votes = [
            vote for eval_record in self.evaluation_history
            for vote in eval_record['votes']
            if vote.specialist_id == specialist_id
        ]
        
        if not specialist_votes:
            return {
                'specialist_id': specialist_id,
                'type': specialist['type'].value,
                'evaluations': 0
            }
        
        avg_confidence = sum(v.confidence for v in specialist_votes) / len(specialist_votes)
        approval_rate = sum(1 for v in specialist_votes if v.decision) / len(specialist_votes)
        
        return {
            'specialist_id': specialist_id,
            'type': specialist['type'].value,
            'evaluations': len(specialist_votes),
            'avg_confidence': avg_confidence,
            'approval_rate': approval_rate,
            'weight': specialist['weight']
        }
    
    def get_quorum_statistics(self) -> Dict[str, Any]:
        """Get quorum statistics"""
        if not self.evaluation_history:
            return {'total_evaluations': 0}
        
        total = len(self.evaluation_history)
        approved = sum(1 for e in self.evaluation_history if e['result'].decision)
        
        avg_consensus = sum(
            e['result'].consensus_strength for e in self.evaluation_history
        ) / total
        
        return {
            'total_evaluations': total,
            'approved': approved,
            'rejected': total - approved,
            'approval_rate': approved / total,
            'avg_consensus_strength': avg_consensus,
            'registered_specialists': len(self.specialists),
            'quorum_threshold': self.quorum_threshold
        }
