"""
ML/DL Consensus Engine - Layer 2

Aggregates predictions from multiple specialists and reaches consensus
through weighted voting, confidence scoring, and trust metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .base_specialist import (
    BaseMLDLSpecialist,
    SpecialistPrediction,
    SpecialistCapability
)


@dataclass
class ConsensusResult:
    """Result of consensus process"""
    final_prediction: Any
    confidence: float
    consensus_score: float  # 0-1, how much specialists agree
    participating_specialists: List[str]
    specialist_predictions: Dict[str, SpecialistPrediction]
    weighted_contributions: Dict[str, float]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    constitutional_compliance: bool = True
    trust_score: float = 0.8
    

class MLDLConsensusEngine:
    """
    Consensus Engine for aggregating specialist predictions
    
    Features:
    - Weighted voting based on trust scores
    - Confidence-based filtering
    - Capability-aware routing
    - Constitutional compliance validation
    - KPI tracking
    
    Integration with Grace:
    - Reports to governance_bridge
    - Logs to immutable audit trail
    - Updates KPI monitors
    - Stores in memory_bridge
    """
    
    def __init__(
        self,
        governance_bridge=None,
        kpi_monitor=None,
        immutable_logs=None,
        memory_bridge=None,
        min_specialists_required=2,
        consensus_threshold=0.6
    ):
        self.governance_bridge = governance_bridge
        self.kpi_monitor = kpi_monitor
        self.immutable_logs = immutable_logs
        self.memory_bridge = memory_bridge
        
        self.min_specialists_required = min_specialists_required
        self.consensus_threshold = consensus_threshold
        
        self.specialists: Dict[str, BaseMLDLSpecialist] = {}
        self.consensus_history: List[ConsensusResult] = []
        
    def register_specialist(self, specialist: BaseMLDLSpecialist):
        """Register a specialist with the consensus engine"""
        self.specialists[specialist.specialist_id] = specialist
        
    def unregister_specialist(self, specialist_id: str):
        """Remove a specialist from the engine"""
        if specialist_id in self.specialists:
            del self.specialists[specialist_id]
    
    def get_specialists_by_capability(
        self,
        capability: SpecialistCapability
    ) -> List[BaseMLDLSpecialist]:
        """Get all specialists that have a specific capability"""
        return [
            specialist for specialist in self.specialists.values()
            if capability in specialist.capabilities
        ]
    
    async def reach_consensus(
        self,
        X: np.ndarray,
        required_capability: SpecialistCapability,
        specialist_predictions: Optional[List[SpecialistPrediction]] = None,
        **kwargs
    ) -> ConsensusResult:
        """
        Reach consensus among specialists
        
        Args:
            X: Input data
            required_capability: The capability needed for this task
            specialist_predictions: Pre-computed predictions (optional)
            **kwargs: Additional parameters
            
        Returns:
            ConsensusResult with aggregated prediction
        """
        start_time = datetime.now()
        
        # Get relevant specialists
        relevant_specialists = self.get_specialists_by_capability(required_capability)
        
        if len(relevant_specialists) < self.min_specialists_required:
            raise ValueError(
                f"Need at least {self.min_specialists_required} specialists, "
                f"but only {len(relevant_specialists)} available for {required_capability}"
            )
        
        # Get predictions from each specialist if not provided
        if specialist_predictions is None:
            specialist_predictions = []
            for specialist in relevant_specialists:
                try:
                    prediction = await specialist.predict(X)
                    specialist_predictions.append(prediction)
                except Exception as e:
                    # Log error but continue with other specialists
                    if self.immutable_logs:
                        await self.immutable_logs.log_event({
                            "type": "specialist_prediction_error",
                            "specialist_id": specialist.specialist_id,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
        
        # Build prediction dictionary
        pred_dict = {
            pred.specialist_id: pred
            for pred in specialist_predictions
        }
        
        # Calculate weights based on trust score and confidence
        weights = {}
        total_weight = 0.0
        
        for pred in specialist_predictions:
            # Weight = trust_score * confidence * constitutional_compliance
            compliance_factor = 1.0 if pred.constitutional_compliance else 0.5
            weight = pred.trust_score * pred.confidence * compliance_factor
            weights[pred.specialist_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate predictions based on task type
        final_prediction, consensus_score = self._aggregate_predictions(
            specialist_predictions,
            weights,
            required_capability
        )
        
        # Calculate overall confidence (weighted average)
        overall_confidence = sum(
            pred.confidence * weights.get(pred.specialist_id, 0)
            for pred in specialist_predictions
        )
        
        # Validate constitutional compliance (all must comply)
        all_compliant = all(
            pred.constitutional_compliance
            for pred in specialist_predictions
        )
        
        # Calculate trust score (weighted average)
        overall_trust = sum(
            pred.trust_score * weights.get(pred.specialist_id, 0)
            for pred in specialist_predictions
        )
        
        # Build reasoning
        reasoning = self._build_consensus_reasoning(
            specialist_predictions,
            weights,
            consensus_score
        )
        
        result = ConsensusResult(
            final_prediction=final_prediction,
            confidence=overall_confidence,
            consensus_score=consensus_score,
            participating_specialists=[p.specialist_id for p in specialist_predictions],
            specialist_predictions=pred_dict,
            weighted_contributions=weights,
            reasoning=reasoning,
            constitutional_compliance=all_compliant,
            trust_score=overall_trust
        )
        
        # Store in history
        self.consensus_history.append(result)
        
        # Log to immutable trail
        if self.immutable_logs:
            await self.immutable_logs.log_event({
                "type": "consensus_reached",
                "timestamp": datetime.now().isoformat(),
                "capability": required_capability.value,
                "n_specialists": len(specialist_predictions),
                "consensus_score": consensus_score,
                "confidence": overall_confidence,
                "compliance": all_compliant,
                "participating_specialists": result.participating_specialists
            })
        
        # Report to KPI monitor
        if self.kpi_monitor:
            await self.kpi_monitor.record_metric({
                "metric_type": "consensus",
                "consensus_score": consensus_score,
                "confidence": overall_confidence,
                "n_specialists": len(specialist_predictions),
                "timestamp": datetime.now().isoformat()
            })
        
        # Store in memory
        if self.memory_bridge:
            await self.memory_bridge.store({
                "type": "consensus_result",
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
        
        return result
    
    def _aggregate_predictions(
        self,
        predictions: List[SpecialistPrediction],
        weights: Dict[str, float],
        capability: SpecialistCapability
    ) -> tuple[Any, float]:
        """
        Aggregate predictions based on capability type
        
        Returns:
            (final_prediction, consensus_score)
        """
        if capability == SpecialistCapability.CLASSIFICATION:
            return self._aggregate_classification(predictions, weights)
        elif capability == SpecialistCapability.REGRESSION:
            return self._aggregate_regression(predictions, weights)
        elif capability == SpecialistCapability.CLUSTERING:
            return self._aggregate_clustering(predictions, weights)
        elif capability == SpecialistCapability.DIMENSIONALITY_REDUCTION:
            return self._aggregate_dimensionality_reduction(predictions, weights)
        else:
            # Default: weighted voting
            return self._aggregate_classification(predictions, weights)
    
    def _aggregate_classification(
        self,
        predictions: List[SpecialistPrediction],
        weights: Dict[str, float]
    ) -> tuple[Any, float]:
        """Aggregate classification predictions via weighted voting"""
        vote_counts = {}
        
        for pred in predictions:
            prediction_val = pred.prediction
            if isinstance(prediction_val, list):
                prediction_val = tuple(prediction_val)  # Make hashable
            
            weight = weights.get(pred.specialist_id, 0)
            
            if prediction_val not in vote_counts:
                vote_counts[prediction_val] = 0
            vote_counts[prediction_val] += weight
        
        # Get winner
        if not vote_counts:
            return None, 0.0
        
        winner = max(vote_counts, key=vote_counts.get)
        winner_votes = vote_counts[winner]
        total_votes = sum(vote_counts.values())
        
        # Consensus score = proportion of votes for winner
        consensus_score = winner_votes / total_votes if total_votes > 0 else 0
        
        # Convert back to list if needed
        final_prediction = list(winner) if isinstance(winner, tuple) else winner
        
        return final_prediction, consensus_score
    
    def _aggregate_regression(
        self,
        predictions: List[SpecialistPrediction],
        weights: Dict[str, float]
    ) -> tuple[Any, float]:
        """Aggregate regression predictions via weighted average"""
        weighted_sum = 0.0
        
        for pred in predictions:
            prediction_val = pred.prediction
            if isinstance(prediction_val, list):
                prediction_val = np.array(prediction_val)
            
            weight = weights.get(pred.specialist_id, 0)
            weighted_sum += prediction_val * weight
        
        final_prediction = weighted_sum
        
        # Consensus score based on variance
        # Lower variance = higher consensus
        pred_values = []
        for pred in predictions:
            val = pred.prediction
            if isinstance(val, list):
                val = np.array(val)
            pred_values.append(val)
        
        pred_array = np.array(pred_values)
        variance = np.var(pred_array)
        mean = np.mean(pred_array)
        
        # Normalized variance (coefficient of variation)
        cv = np.sqrt(variance) / (abs(mean) + 1e-6)
        consensus_score = max(0, 1.0 - min(cv, 1.0))
        
        return final_prediction, consensus_score
    
    def _aggregate_clustering(
        self,
        predictions: List[SpecialistPrediction],
        weights: Dict[str, float]
    ) -> tuple[Any, float]:
        """Aggregate clustering predictions (use most trusted)"""
        # For clustering, use prediction from most trusted specialist
        best_specialist = max(
            predictions,
            key=lambda p: weights.get(p.specialist_id, 0)
        )
        
        # Consensus score = weight of best specialist
        consensus_score = weights.get(best_specialist.specialist_id, 0)
        
        return best_specialist.prediction, consensus_score
    
    def _aggregate_dimensionality_reduction(
        self,
        predictions: List[SpecialistPrediction],
        weights: Dict[str, float]
    ) -> tuple[Any, float]:
        """Aggregate dimensionality reduction (weighted average of embeddings)"""
        # Average the reduced representations
        weighted_sum = None
        
        for pred in predictions:
            prediction_val = np.array(pred.prediction)
            weight = weights.get(pred.specialist_id, 0)
            
            if weighted_sum is None:
                weighted_sum = prediction_val * weight
            else:
                weighted_sum += prediction_val * weight
        
        final_prediction = weighted_sum
        
        # Consensus score based on similarity of embeddings
        # Calculate pairwise cosine similarities
        embeddings = [np.array(pred.prediction) for pred in predictions]
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity
                emb_i = embeddings[i].flatten()
                emb_j = embeddings[j].flatten()
                
                dot_product = np.dot(emb_i, emb_j)
                norm_i = np.linalg.norm(emb_i)
                norm_j = np.linalg.norm(emb_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                    similarities.append(similarity)
        
        consensus_score = np.mean(similarities) if similarities else 0.5
        
        return final_prediction.tolist(), consensus_score
    
    def _build_consensus_reasoning(
        self,
        predictions: List[SpecialistPrediction],
        weights: Dict[str, float],
        consensus_score: float
    ) -> str:
        """Build human-readable reasoning for consensus"""
        reasoning_parts = [
            f"Consensus from {len(predictions)} specialists:",
            f"Agreement: {consensus_score:.1%}"
        ]
        
        # Top 3 contributors
        top_contributors = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        reasoning_parts.append("Top contributors:")
        for specialist_id, weight in top_contributors:
            pred = next(p for p in predictions if p.specialist_id == specialist_id)
            reasoning_parts.append(
                f"  - {pred.specialist_type} ({weight:.1%}): "
                f"confidence={pred.confidence:.2f}, trust={pred.trust_score:.2f}"
            )
        
        return "\n".join(reasoning_parts)
    
    async def validate_consensus_governance(
        self,
        consensus_result: ConsensusResult
    ) -> bool:
        """Validate consensus result against governance rules"""
        if not self.governance_bridge:
            return True
        
        # Check consensus threshold
        if consensus_result.consensus_score < self.consensus_threshold:
            return False
        
        # Check constitutional compliance
        if not consensus_result.constitutional_compliance:
            return False
        
        # Delegate to governance bridge for full validation
        try:
            validation_result = await self.governance_bridge.validate({
                "type": "consensus_result",
                "data": consensus_result,
                "timestamp": datetime.now().isoformat()
            })
            return validation_result.get("approved", True)
        except Exception:
            # Default to True if governance check fails
            return True
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get statistics about consensus history"""
        if not self.consensus_history:
            return {
                "total_consensus": 0,
                "avg_consensus_score": 0,
                "avg_confidence": 0,
                "avg_specialists": 0
            }
        
        return {
            "total_consensus": len(self.consensus_history),
            "avg_consensus_score": np.mean([r.consensus_score for r in self.consensus_history]),
            "avg_confidence": np.mean([r.confidence for r in self.consensus_history]),
            "avg_specialists": np.mean([len(r.participating_specialists) for r in self.consensus_history]),
            "compliance_rate": np.mean([r.constitutional_compliance for r in self.consensus_history])
        }
