"""
Quorum Integration - Connects swarm consensus to MLDL Quorum module
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QuorumIntegration:
    """
    Integrates swarm consensus with MLDL Quorum module
    Provides feedback loop for collective decision-making
    """
    
    def __init__(self):
        self.model_preferences: Dict[str, float] = {}
        self.parameter_consensus: Dict[str, Any] = {}
        self.strategy_adoptions: List[Dict[str, Any]] = []
        self.consensus_history: List[Dict[str, Any]] = []
        logger.info("QuorumIntegration initialized")
    
    def update_model_preference(
        self,
        model_id: str,
        confidence: float,
        consensus_data: Optional[Dict] = None
    ):
        """Update model preference based on swarm consensus"""
        self.model_preferences[model_id] = confidence
        
        self.consensus_history.append({
            'type': 'model_preference',
            'model_id': model_id,
            'confidence': confidence,
            'consensus_data': consensus_data or {}
        })
        
        logger.info(f"Updated model preference: {model_id} (confidence: {confidence:.2f})")
    
    def apply_parameter_consensus(
        self,
        parameters: Dict[str, Any],
        consensus_strength: float
    ):
        """Apply parameter settings from swarm consensus"""
        for param_name, param_value in parameters.items():
            self.parameter_consensus[param_name] = {
                'value': param_value,
                'consensus_strength': consensus_strength
            }
        
        self.consensus_history.append({
            'type': 'parameter_consensus',
            'parameters': parameters,
            'consensus_strength': consensus_strength
        })
        
        logger.info(f"Applied parameter consensus: {len(parameters)} parameters")
    
    def adopt_strategy(
        self,
        strategy: Dict[str, Any],
        support_level: float
    ):
        """Adopt strategy based on swarm consensus"""
        adoption = {
            'strategy': strategy,
            'support_level': support_level,
            'adopted': support_level >= 0.66
        }
        
        self.strategy_adoptions.append(adoption)
        
        self.consensus_history.append({
            'type': 'strategy_adoption',
            **adoption
        })
        
        logger.info(f"Strategy adoption: {strategy.get('name', 'unnamed')} (support: {support_level:.2f})")
    
    def get_preferred_model(self) -> Optional[str]:
        """Get the most preferred model based on consensus"""
        if not self.model_preferences:
            return None
        
        return max(self.model_preferences.items(), key=lambda x: x[1])[0]
    
    def get_consensus_parameters(self) -> Dict[str, Any]:
        """Get parameters agreed upon by consensus"""
        return {
            param: data['value']
            for param, data in self.parameter_consensus.items()
            if data['consensus_strength'] >= 0.5
        }
    
    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get strategies adopted by consensus"""
        return [
            adoption for adoption in self.strategy_adoptions
            if adoption['adopted']
        ]
    
    def get_quorum_status(self) -> Dict[str, Any]:
        """Get quorum integration status"""
        return {
            'model_preferences_count': len(self.model_preferences),
            'preferred_model': self.get_preferred_model(),
            'consensus_parameters': len(self.parameter_consensus),
            'active_strategies': len(self.get_active_strategies()),
            'total_consensus_decisions': len(self.consensus_history)
        }
