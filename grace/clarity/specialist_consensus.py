"""
Specialist Consensus - Final MLDL quorum with consensus IDs (Class 8)
"""

from typing import Dict, Any, List, Optional
import hashlib
import json
import logging

from grace.mldl.quorum_aggregator import MLDLQuorumAggregator, SpecialistOutput

logger = logging.getLogger(__name__)


class SpecialistConsensus:
    """
    Finalizes specialist consensus with consensus IDs
    
    Produces consensus that can be fed into Unified Logic
    """
    
    def __init__(self, quorum_aggregator: Optional[MLDLQuorumAggregator] = None):
        self.quorum_aggregator = quorum_aggregator or MLDLQuorumAggregator()
        self.consensus_history: List[Dict[str, Any]] = []
        
        logger.info("SpecialistConsensus initialized")
    
    def generate_consensus(
        self,
        specialist_outputs: List[SpecialistOutput],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate consensus from specialist outputs
        
        Returns consensus with unique ID for tracking
        """
        # Get consensus from aggregator
        result = self.quorum_aggregator.aggregate_outputs(specialist_outputs)
        
        # Generate consensus ID
        consensus_id = self._generate_consensus_id(specialist_outputs, result)
        
        # Package consensus
        consensus = {
            "consensus_id": consensus_id,
            "prediction": result.consensus_prediction,
            "confidence": result.consensus_confidence,
            "agreement": result.agreement_score,
            "method": result.method_used.value,
            "specialists": result.participating_specialists,
            "uncertainty": result.uncertainty_bounds,
            "individual_outputs": [
                {
                    "specialist_id": o.specialist_id,
                    "prediction": o.prediction,
                    "confidence": o.confidence,
                    "weight": o.weight
                }
                for o in result.individual_outputs
            ],
            "context": context,
            "timestamp": context.get("timestamp")
        }
        
        # Store in history
        self.consensus_history.append(consensus)
        
        logger.info(
            f"Generated consensus {consensus_id}: "
            f"prediction={result.consensus_prediction}, "
            f"confidence={result.consensus_confidence:.3f}, "
            f"agreement={result.agreement_score:.3f}"
        )
        
        return consensus
    
    def _generate_consensus_id(
        self,
        outputs: List[SpecialistOutput],
        result
    ) -> str:
        """Generate unique consensus ID"""
        # Create deterministic hash from outputs
        data_str = json.dumps({
            "outputs": [
                {
                    "specialist_id": o.specialist_id,
                    "prediction": str(o.prediction),
                    "confidence": o.confidence
                }
                for o in outputs
            ],
            "consensus": str(result.consensus_prediction),
            "timestamp": str(result.individual_outputs[0].metadata.get("timestamp") if result.individual_outputs else "")
        }, sort_keys=True)
        
        consensus_id = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"consensus_{consensus_id}"
    
    def get_consensus_by_id(self, consensus_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve consensus by ID"""
        for consensus in self.consensus_history:
            if consensus["consensus_id"] == consensus_id:
                return consensus
        return None
    
    def feed_to_unified_logic(
        self,
        consensus: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare consensus for Unified Logic consumption
        
        Formats consensus in the expected input format
        """
        unified_input = {
            "consensus_id": consensus["consensus_id"],
            "prediction": consensus["prediction"],
            "confidence": consensus["confidence"],
            "agreement_level": consensus["agreement"],
            "specialist_count": len(consensus["specialists"]),
            "uncertainty": consensus.get("uncertainty"),
            "context": {
                **consensus.get("context", {}),
                **(additional_context or {})
            },
            "metadata": {
                "method": consensus["method"],
                "specialists": consensus["specialists"],
                "timestamp": consensus.get("timestamp")
            }
        }
        
        return unified_input
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        if not self.consensus_history:
            return {"total_consensus": 0}
        
        import numpy as np
        
        confidences = [c["confidence"] for c in self.consensus_history]
        agreements = [c["agreement"] for c in self.consensus_history]
        
        return {
            "total_consensus": len(self.consensus_history),
            "avg_confidence": np.mean(confidences),
            "avg_agreement": np.mean(agreements),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "specialist_stats": self.quorum_aggregator.get_specialist_stats()
        }
