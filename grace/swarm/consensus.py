"""
Collective consensus engine for swarm decisions
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConsensusAlgorithm(Enum):
    """Available consensus algorithms"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    FEDERATED_AVERAGING = "federated_averaging"
    RAFT = "raft"
    BYZANTINE_FAULT_TOLERANT = "bft"


class CollectiveConsensusEngine:
    """
    Combines decisions and trust scores across swarm nodes
    
    Algorithms:
    - Majority voting
    - Weighted averaging (by trust scores)
    - Federated averaging (for ML models)
    - Byzantine fault tolerance
    """
    
    def __init__(self, algorithm: ConsensusAlgorithm = ConsensusAlgorithm.WEIGHTED_AVERAGE):
        self.algorithm = algorithm
        self.consensus_history: List[Dict[str, Any]] = []
        
        logger.info(f"CollectiveConsensusEngine initialized: {algorithm.value}")
    
    def compute_consensus(
        self,
        node_decisions: Dict[str, Any],
        node_trust_scores: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute collective consensus from node decisions
        
        Args:
            node_decisions: {node_id: decision}
            node_trust_scores: {node_id: trust_score}
            context: Additional context
            
        Returns:
            Consensus result with metadata
        """
        if not node_decisions:
            return {
                "consensus": None,
                "confidence": 0.0,
                "algorithm": self.algorithm.value,
                "participating_nodes": 0
            }
        
        # Select algorithm
        if self.algorithm == ConsensusAlgorithm.MAJORITY_VOTE:
            result = self._majority_vote(node_decisions, node_trust_scores)
        elif self.algorithm == ConsensusAlgorithm.WEIGHTED_AVERAGE:
            result = self._weighted_average(node_decisions, node_trust_scores)
        elif self.algorithm == ConsensusAlgorithm.FEDERATED_AVERAGING:
            result = self._federated_averaging(node_decisions, node_trust_scores)
        elif self.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
            result = self._byzantine_fault_tolerant(node_decisions, node_trust_scores)
        else:
            result = self._weighted_average(node_decisions, node_trust_scores)
        
        # Add metadata
        result["algorithm"] = self.algorithm.value
        result["participating_nodes"] = len(node_decisions)
        result["timestamp"] = context.get("timestamp") if context else None
        
        # Store in history
        self.consensus_history.append(result)
        
        logger.info(
            f"Consensus computed: {result['consensus']} "
            f"(confidence: {result['confidence']:.3f}, nodes: {len(node_decisions)})"
        )
        
        return result
    
    def _majority_vote(
        self,
        decisions: Dict[str, Any],
        trust_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Majority voting consensus"""
        votes: Dict[Any, float] = {}
        
        for node_id, decision in decisions.items():
            decision_key = str(decision)
            trust = trust_scores.get(node_id, 0.5)
            
            votes[decision_key] = votes.get(decision_key, 0) + trust
        
        if not votes:
            return {"consensus": None, "confidence": 0.0}
        
        # Winner
        winner = max(votes.items(), key=lambda x: x[1])
        total_trust = sum(votes.values())
        
        return {
            "consensus": winner[0],
            "confidence": winner[1] / total_trust if total_trust > 0 else 0.0,
            "vote_distribution": dict(votes)
        }
    
    def _weighted_average(
        self,
        decisions: Dict[str, Any],
        trust_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Weighted average consensus (for numeric decisions)"""
        numeric_decisions = {}
        
        for node_id, decision in decisions.items():
            try:
                if isinstance(decision, (int, float)):
                    numeric_decisions[node_id] = float(decision)
                elif isinstance(decision, dict) and "value" in decision:
                    numeric_decisions[node_id] = float(decision["value"])
            except:
                pass
        
        if not numeric_decisions:
            # Fallback to majority vote
            return self._majority_vote(decisions, trust_scores)
        
        # Compute weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for node_id, value in numeric_decisions.items():
            weight = trust_scores.get(node_id, 0.5)
            weighted_sum += value * weight
            total_weight += weight
        
        if total_weight == 0:
            return {"consensus": None, "confidence": 0.0}
        
        consensus = weighted_sum / total_weight
        
        # Calculate confidence based on variance
        variance = sum(
            trust_scores.get(nid, 0.5) * (val - consensus) ** 2
            for nid, val in numeric_decisions.items()
        ) / total_weight
        
        confidence = 1.0 / (1.0 + np.sqrt(variance))
        
        return {
            "consensus": consensus,
            "confidence": confidence,
            "variance": variance,
            "node_count": len(numeric_decisions)
        }
    
    def _federated_averaging(
        self,
        decisions: Dict[str, Any],
        trust_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Federated averaging (for ML model parameters)"""
        # Extract model updates
        model_updates = {}
        
        for node_id, decision in decisions.items():
            if isinstance(decision, dict) and "model_update" in decision:
                model_updates[node_id] = decision["model_update"]
            elif isinstance(decision, (list, np.ndarray)):
                model_updates[node_id] = decision
        
        if not model_updates:
            return self._weighted_average(decisions, trust_scores)
        
        # Federated averaging
        total_weight = sum(trust_scores.get(nid, 0.5) for nid in model_updates.keys())
        
        # Average updates weighted by trust
        averaged_update = None
        
        for node_id, update in model_updates.items():
            weight = trust_scores.get(node_id, 0.5) / total_weight
            
            if averaged_update is None:
                averaged_update = np.array(update) * weight
            else:
                averaged_update += np.array(update) * weight
        
        return {
            "consensus": averaged_update.tolist() if averaged_update is not None else None,
            "confidence": 0.8,  # High confidence for federated averaging
            "node_count": len(model_updates)
        }
    
    def _byzantine_fault_tolerant(
        self,
        decisions: Dict[str, Any],
        trust_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Byzantine fault tolerant consensus"""
        # Identify outliers and exclude them
        numeric_decisions = {}
        
        for node_id, decision in decisions.items():
            try:
                if isinstance(decision, (int, float)):
                    numeric_decisions[node_id] = float(decision)
            except:
                pass
        
        if len(numeric_decisions) < 4:
            # Need at least 3f+1 nodes for BFT (assuming f=1)
            return self._weighted_average(decisions, trust_scores)
        
        # Detect and remove outliers
        values = list(numeric_decisions.values())
        median = np.median(values)
        mad = np.median([abs(v - median) for v in values])
        
        # Filter outliers (beyond 3*MAD)
        threshold = 3 * mad
        filtered_decisions = {
            nid: val for nid, val in numeric_decisions.items()
            if abs(val - median) <= threshold
        }
        
        # Compute weighted average of non-outliers
        result = self._weighted_average(filtered_decisions, trust_scores)
        result["outliers_removed"] = len(numeric_decisions) - len(filtered_decisions)
        
        return result
    
    def reconcile_with_local(
        self,
        global_consensus: Dict[str, Any],
        local_decision: Any,
        local_trust: float
    ) -> Dict[str, Any]:
        """
        Reconcile global consensus with local decision
        
        Returns hybrid decision balancing global and local
        """
        global_value = global_consensus.get("consensus")
        global_confidence = global_consensus.get("confidence", 0.5)
        
        # Weight global vs local
        global_weight = global_confidence
        local_weight = local_trust
        
        total_weight = global_weight + local_weight
        
        if total_weight == 0:
            return {"decision": local_decision, "source": "local"}
        
        # If both are numeric, blend them
        try:
            if isinstance(global_value, (int, float)) and isinstance(local_decision, (int, float)):
                blended = (
                    global_value * global_weight +
                    local_decision * local_weight
                ) / total_weight
                
                return {
                    "decision": blended,
                    "source": "hybrid",
                    "global_weight": global_weight / total_weight,
                    "local_weight": local_weight / total_weight
                }
        except:
            pass
        
        # Otherwise, choose based on confidence
        if global_weight > local_weight:
            return {"decision": global_value, "source": "global"}
        else:
            return {"decision": local_decision, "source": "local"}
