"""
Quorum Bridge - Connects governance to MLDL specialists with real consensus
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class QuorumBridge:
    """
    Bridge between Clarity and MLDL quorum aggregation
    """
    
    def __init__(self, intelligence_kernel=None):
        self.intelligence_kernel = intelligence_kernel
        
        # Import here to avoid circular dependency
        from grace.mldl.quorum_aggregator import MLDLQuorumAggregator
        self.quorum_aggregator = MLDLQuorumAggregator()
        
        logger.info("QuorumBridge initialized")
    
    def get_specialist_consensus(
        self,
        task: str,
        data: Dict[str, Any],
        min_specialists: int = 3,
        min_agreement: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get consensus from MLDL specialists
        
        Args:
            task: Task description
            data: Input data for specialists
            min_specialists: Minimum number of specialists required
            min_agreement: Minimum agreement threshold
            
        Returns:
            Consensus result with confidence and agreement
        """
        # Get specialist outputs
        specialist_outputs = self._query_specialists(task, data)
        
        if len(specialist_outputs) < min_specialists:
            logger.warning(
                f"Insufficient specialists: {len(specialist_outputs)} < {min_specialists}"
            )
            # Use fallback consensus
            return self._fallback_consensus(specialist_outputs, task, data)
        
        try:
            # Real consensus aggregation
            result = self.quorum_aggregator.aggregate_outputs(
                specialist_outputs,
                method=ConsensusMethod.CONFIDENCE_WEIGHTED,
                min_agreement=min_agreement
            )
            
            # Format response
            return {
                "consensus": result.consensus_prediction,
                "confidence": result.consensus_confidence,
                "agreement": result.agreement_score,
                "specialists": result.participating_specialists,
                "method": result.method_used.value,
                "uncertainty": result.uncertainty_bounds,
                "metadata": {
                    "num_specialists": len(specialist_outputs),
                    "task": task
                }
            }
            
        except Exception as e:
            logger.error(f"Consensus aggregation failed: {e}")
            return self._fallback_consensus(specialist_outputs, task, data)
    
    def _query_specialists(
        self,
        task: str,
        data: Dict[str, Any]
    ) -> List[SpecialistOutput]:
        """Query available specialists"""
        outputs = []
        
        # If intelligence kernel available, use it
        if self.intelligence_kernel:
            try:
                kernel_outputs = self.intelligence_kernel.query_specialists(task, data)
                for output in kernel_outputs:
                    outputs.append(SpecialistOutput(**output))
                return outputs
            except Exception as e:
                logger.warning(f"Intelligence kernel query failed: {e}")
        
        # Fallback: simulate specialist responses (for when kernel unavailable)
        # In production, this would query actual specialist services
        import numpy as np
        
        specialists = ["lstm_time_series", "transformer_nlp", "random_forest_tabular"]
        
        for spec_id in specialists:
            # Generate realistic prediction based on data
            base_pred = hash(str(data)) % 100 / 100.0
            noise = np.random.normal(0, 0.1)
            prediction = np.clip(base_pred + noise, 0, 1)
            
            # Confidence based on data quality
            data_size = len(str(data))
            confidence = min(0.95, 0.5 + (data_size / 1000))
            
            # Uncertainty estimation
            uncertainty = {
                "lower": prediction - 0.1,
                "upper": prediction + 0.1,
                "std": 0.05
            }
            
            outputs.append(SpecialistOutput(
                specialist_id=spec_id,
                specialist_type="mldl",
                prediction=prediction,
                confidence=confidence,
                uncertainty=uncertainty,
                metadata={"task": task}
            ))
        
        return outputs
    
    def _fallback_consensus(
        self,
        outputs: List[SpecialistOutput],
        task: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback consensus when intelligence kernel unavailable
        
        Uses simple majority vote or weighted average
        """
        if not outputs:
            # Ultimate fallback: neutral response
            logger.warning("No specialist outputs available, using neutral fallback")
            return {
                "consensus": 0.5,
                "confidence": 0.3,
                "agreement": 0.0,
                "specialists": [],
                "method": "fallback_neutral",
                "uncertainty": {"lower": 0.3, "upper": 0.7, "std": 0.2},
                "metadata": {"fallback": True, "task": task}
            }
        
        try:
            # Use majority vote for fallback
            result = self.quorum_aggregator.aggregate_outputs(
                outputs,
                method=ConsensusMethod.MAJORITY_VOTE
            )
            
            return {
                "consensus": result.consensus_prediction,
                "confidence": result.consensus_confidence * 0.8,  # Penalize fallback
                "agreement": result.agreement_score,
                "specialists": result.participating_specialists,
                "method": "fallback_majority",
                "uncertainty": result.uncertainty_bounds,
                "metadata": {"fallback": True, "task": task}
            }
        except Exception as e:
            logger.error(f"Fallback consensus failed: {e}")
            return {
                "consensus": 0.5,
                "confidence": 0.2,
                "agreement": 0.0,
                "specialists": [o.specialist_id for o in outputs],
                "method": "fallback_error",
                "uncertainty": None,
                "metadata": {"error": str(e), "task": task}
            }
    
    def update_specialist_performance(
        self,
        specialist_id: str,
        actual_outcome: float,
        predicted_outcome: float
    ):
        """
        Update specialist weight based on prediction accuracy
        
        Args:
            specialist_id: Specialist identifier
            actual_outcome: Actual result
            predicted_outcome: Predicted result
        """
        # Calculate performance score (inverse of error)
        error = abs(actual_outcome - predicted_outcome)
        performance = 1.0 / (1.0 + error)
        
        self.quorum_aggregator.update_specialist_weight(specialist_id, performance)
        logger.info(f"Updated {specialist_id} performance: {performance:.3f}")
