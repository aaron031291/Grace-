"""
MLDL Quorum Aggregator - Real consensus from specialist outputs
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Methods for computing consensus"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"


@dataclass
class SpecialistOutput:
    """Output from a single specialist"""
    specialist_id: str
    specialist_type: str
    prediction: Any
    confidence: float
    uncertainty: Optional[Dict[str, float]] = None  # confidence intervals
    metadata: Optional[Dict[str, Any]] = None
    weight: float = 1.0


@dataclass
class QuorumResult:
    """Aggregated consensus result"""
    consensus_prediction: Any
    consensus_confidence: float
    agreement_score: float  # 0-1, how much specialists agree
    participating_specialists: List[str]
    method_used: ConsensusMethod
    individual_outputs: List[SpecialistOutput]
    uncertainty_bounds: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class MLDLQuorumAggregator:
    """
    Aggregates outputs from multiple MLDL specialists into consensus
    
    Features:
    - Weighted voting by specialist reliability
    - Confidence-based weighting
    - Uncertainty quantification
    - Fallback consensus methods
    """
    
    def __init__(self, default_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED):
        self.default_method = default_method
        self.specialist_weights: Dict[str, float] = {}
        self.specialist_history: Dict[str, List[float]] = {}
    
    def register_specialist(self, specialist_id: str, initial_weight: float = 1.0):
        """Register a specialist with initial weight"""
        self.specialist_weights[specialist_id] = initial_weight
        self.specialist_history[specialist_id] = []
        logger.info(f"Registered specialist: {specialist_id} with weight {initial_weight}")
    
    def update_specialist_weight(self, specialist_id: str, performance_score: float):
        """Update specialist weight based on performance"""
        if specialist_id not in self.specialist_weights:
            self.register_specialist(specialist_id)
        
        # Exponential moving average
        alpha = 0.3
        current_weight = self.specialist_weights[specialist_id]
        new_weight = alpha * performance_score + (1 - alpha) * current_weight
        
        self.specialist_weights[specialist_id] = np.clip(new_weight, 0.1, 2.0)
        self.specialist_history[specialist_id].append(performance_score)
        
        logger.debug(f"Updated weight for {specialist_id}: {current_weight:.3f} -> {new_weight:.3f}")
    
    def aggregate_outputs(
        self,
        outputs: List[SpecialistOutput],
        method: Optional[ConsensusMethod] = None,
        min_agreement: float = 0.5
    ) -> QuorumResult:
        """
        Aggregate specialist outputs into consensus
        
        Args:
            outputs: List of specialist outputs
            method: Consensus method (uses default if None)
            min_agreement: Minimum agreement threshold
            
        Returns:
            QuorumResult with consensus
        """
        if not outputs:
            raise ValueError("No specialist outputs provided")
        
        method = method or self.default_method
        
        # Apply specialist weights
        for output in outputs:
            if output.specialist_id in self.specialist_weights:
                output.weight = self.specialist_weights[output.specialist_id]
        
        # Select aggregation method
        if method == ConsensusMethod.WEIGHTED_AVERAGE:
            result = self._weighted_average_consensus(outputs)
        elif method == ConsensusMethod.MAJORITY_VOTE:
            result = self._majority_vote_consensus(outputs)
        elif method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            result = self._confidence_weighted_consensus(outputs)
        elif method == ConsensusMethod.BAYESIAN:
            result = self._bayesian_consensus(outputs)
        else:
            result = self._ensemble_consensus(outputs)
        
        # Check agreement threshold
        if result.agreement_score < min_agreement:
            logger.warning(
                f"Low agreement score: {result.agreement_score:.3f} < {min_agreement} "
                f"(method: {method.value})"
            )
        
        return result
    
    def _weighted_average_consensus(self, outputs: List[SpecialistOutput]) -> QuorumResult:
        """Compute consensus as weighted average"""
        predictions = []
        weights = []
        
        for output in outputs:
            if isinstance(output.prediction, (int, float)):
                predictions.append(output.prediction)
                weights.append(output.weight)
        
        if not predictions:
            # Fallback for non-numeric predictions
            return self._majority_vote_consensus(outputs)
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        consensus_pred = np.average(predictions, weights=weights)
        
        # Calculate agreement (inverse of weighted std)
        weighted_std = np.sqrt(np.average((predictions - consensus_pred) ** 2, weights=weights))
        agreement = 1.0 / (1.0 + weighted_std)
        
        # Average confidence
        consensus_conf = np.average([o.confidence for o in outputs], weights=weights)
        
        # Uncertainty bounds
        uncertainty = {
            "lower": consensus_pred - 1.96 * weighted_std,
            "upper": consensus_pred + 1.96 * weighted_std,
            "std": weighted_std
        }
        
        return QuorumResult(
            consensus_prediction=float(consensus_pred),
            consensus_confidence=float(consensus_conf),
            agreement_score=float(agreement),
            participating_specialists=[o.specialist_id for o in outputs],
            method_used=ConsensusMethod.WEIGHTED_AVERAGE,
            individual_outputs=outputs,
            uncertainty_bounds=uncertainty
        )
    
    def _confidence_weighted_consensus(self, outputs: List[SpecialistOutput]) -> QuorumResult:
        """Weight by both specialist weight and prediction confidence"""
        predictions = []
        weights = []
        
        for output in outputs:
            if isinstance(output.prediction, (int, float)):
                # Weight is product of specialist weight and confidence
                combined_weight = output.weight * output.confidence
                predictions.append(output.prediction)
                weights.append(combined_weight)
        
        if not predictions:
            return self._majority_vote_consensus(outputs)
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        consensus_pred = np.average(predictions, weights=weights)
        consensus_conf = np.average([o.confidence for o in outputs], weights=weights)
        
        # Agreement based on confidence-weighted variance
        weighted_var = np.average((predictions - consensus_pred) ** 2, weights=weights)
        agreement = 1.0 / (1.0 + np.sqrt(weighted_var))
        
        uncertainty = {
            "lower": consensus_pred - 1.96 * np.sqrt(weighted_var),
            "upper": consensus_pred + 1.96 * np.sqrt(weighted_var),
            "std": np.sqrt(weighted_var)
        }
        
        return QuorumResult(
            consensus_prediction=float(consensus_pred),
            consensus_confidence=float(consensus_conf),
            agreement_score=float(agreement),
            participating_specialists=[o.specialist_id for o in outputs],
            method_used=ConsensusMethod.CONFIDENCE_WEIGHTED,
            individual_outputs=outputs,
            uncertainty_bounds=uncertainty
        )
    
    def _majority_vote_consensus(self, outputs: List[SpecialistOutput]) -> QuorumResult:
        """Majority voting for classification tasks"""
        votes: Dict[Any, float] = {}
        
        for output in outputs:
            pred = str(output.prediction)  # Convert to string for hashing
            weight = output.weight * output.confidence
            votes[pred] = votes.get(pred, 0) + weight
        
        if not votes:
            raise ValueError("No valid votes")
        
        # Winner
        consensus_pred = max(votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(votes.values())
        consensus_conf = votes[consensus_pred] / total_weight
        
        # Agreement: how dominant is the winner
        agreement = votes[consensus_pred] / total_weight
        
        return QuorumResult(
            consensus_prediction=consensus_pred,
            consensus_confidence=float(consensus_conf),
            agreement_score=float(agreement),
            participating_specialists=[o.specialist_id for o in outputs],
            method_used=ConsensusMethod.MAJORITY_VOTE,
            individual_outputs=outputs
        )
    
    def _bayesian_consensus(self, outputs: List[SpecialistOutput]) -> QuorumResult:
        """Bayesian model averaging"""
        # Prior: uniform distribution
        prior_weight = 1.0 / len(outputs)
        
        predictions = []
        posterior_weights = []
        
        for output in outputs:
            if isinstance(output.prediction, (int, float)):
                # Posterior weight proportional to confidence and specialist weight
                posterior = output.confidence * output.weight
                predictions.append(output.prediction)
                posterior_weights.append(posterior)
        
        if not predictions:
            return self._majority_vote_consensus(outputs)
        
        predictions = np.array(predictions)
        posterior_weights = np.array(posterior_weights)
        posterior_weights = posterior_weights / posterior_weights.sum()
        
        # Bayesian model average
        consensus_pred = np.sum(predictions * posterior_weights)
        
        # Uncertainty from weighted variance
        weighted_var = np.sum(posterior_weights * (predictions - consensus_pred) ** 2)
        agreement = 1.0 / (1.0 + np.sqrt(weighted_var))
        
        consensus_conf = float(np.max(posterior_weights))
        
        uncertainty = {
            "lower": consensus_pred - 1.96 * np.sqrt(weighted_var),
            "upper": consensus_pred + 1.96 * np.sqrt(weighted_var),
            "std": np.sqrt(weighted_var)
        }
        
        return QuorumResult(
            consensus_prediction=float(consensus_pred),
            consensus_confidence=consensus_conf,
            agreement_score=float(agreement),
            participating_specialists=[o.specialist_id for o in outputs],
            method_used=ConsensusMethod.BAYESIAN,
            individual_outputs=outputs,
            uncertainty_bounds=uncertainty
        )
    
    def _ensemble_consensus(self, outputs: List[SpecialistOutput]) -> QuorumResult:
        """Ensemble method combining multiple approaches"""
        # Try confidence-weighted first
        try:
            conf_result = self._confidence_weighted_consensus(outputs)
            if conf_result.agreement_score > 0.7:
                return conf_result
        except:
            pass
        
        # Try weighted average
        try:
            avg_result = self._weighted_average_consensus(outputs)
            if avg_result.agreement_score > 0.6:
                return avg_result
        except:
            pass
        
        # Fallback to majority vote
        return self._majority_vote_consensus(outputs)
    
    def get_specialist_stats(self) -> Dict[str, Any]:
        """Get statistics about specialists"""
        stats = {}
        
        for spec_id, weight in self.specialist_weights.items():
            history = self.specialist_history.get(spec_id, [])
            stats[spec_id] = {
                "current_weight": weight,
                "num_predictions": len(history),
                "avg_performance": np.mean(history) if history else 0.0,
                "reliability": np.std(history) if len(history) > 1 else 1.0
            }
        
        return stats
