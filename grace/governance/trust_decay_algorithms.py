"""
Trust Decay Algorithms - Mathematical models for trust degradation over time.

Implements sophisticated mathematical models for trust decay as specified in 
the missing components requirements. Includes:
- Exponential decay models
- Temporal stability analysis  
- Context-aware decay rates
- Multi-factor trust erosion
- Trust recovery mechanisms
"""

import asyncio
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DecayModel(Enum):
    """Types of trust decay models."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    STEPPED = "stepped"
    ADAPTIVE = "adaptive"


class TrustFactor(Enum):
    """Factors that influence trust decay."""
    TIME_ELAPSED = "time_elapsed"
    USAGE_FREQUENCY = "usage_frequency"
    PERFORMANCE_VARIANCE = "performance_variance"
    CONTEXT_CHANGES = "context_changes"
    ERROR_RATE = "error_rate"
    DOMAIN_DRIFT = "domain_drift"
    EXTERNAL_VALIDATION = "external_validation"


@dataclass
class TrustDecayParameters:
    """Parameters for trust decay calculations."""
    base_decay_rate: float = 0.01  # Base daily decay rate
    half_life_days: float = 30.0   # Days for trust to reach half value
    minimum_trust: float = 0.1     # Minimum trust floor
    maximum_decay_per_day: float = 0.05  # Maximum daily decay
    recovery_factor: float = 1.2   # Multiplier for trust recovery
    variance_penalty: float = 0.02  # Penalty for high performance variance
    inactivity_threshold_days: int = 7  # Days before inactivity decay kicks in
    critical_error_penalty: float = 0.1  # Trust loss for critical errors
    
    # Contextual factors
    high_risk_multiplier: float = 1.5  # Faster decay for high-risk contexts
    low_risk_multiplier: float = 0.7   # Slower decay for low-risk contexts
    domain_stability_bonus: float = 0.9  # Slower decay for stable domains


@dataclass
class TrustDecayState:
    """Current state of trust decay for an entity."""
    entity_id: str
    current_trust: float
    base_trust: float
    last_update: datetime
    last_interaction: datetime
    decay_model: DecayModel
    parameters: TrustDecayParameters = field(default_factory=TrustDecayParameters)
    
    # Historical tracking
    trust_history: List[Tuple[datetime, float]] = field(default_factory=list)
    interaction_frequency: List[datetime] = field(default_factory=list)
    performance_variance_history: List[float] = field(default_factory=list)
    
    # Context tracking
    domain_changes: int = 0
    error_count: int = 0
    critical_errors: int = 0
    successful_interactions: int = 0
    total_interactions: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_interactions == 0:
            return 0.5  # Neutral for no interactions
        return self.successful_interactions / self.total_interactions
    
    def get_recent_variance(self, days: int = 7) -> float:
        """Get performance variance over recent period."""
        if len(self.performance_variance_history) < 2:
            return 0.0
        
        recent_history = self.performance_variance_history[-days:]
        return np.std(recent_history) if len(recent_history) > 1 else 0.0


class TrustDecayEngine:
    """Advanced trust decay engine with multiple mathematical models."""
    
    def __init__(self):
        self.decay_states: Dict[str, TrustDecayState] = {}
        self.global_parameters = TrustDecayParameters()
        self.decay_functions = {
            DecayModel.EXPONENTIAL: self._exponential_decay,
            DecayModel.LINEAR: self._linear_decay,
            DecayModel.LOGARITHMIC: self._logarithmic_decay,
            DecayModel.SIGMOID: self._sigmoid_decay,
            DecayModel.STEPPED: self._stepped_decay,
            DecayModel.ADAPTIVE: self._adaptive_decay
        }
        
    async def initialize_entity(self, entity_id: str, 
                              initial_trust: float = 0.5,
                              decay_model: DecayModel = DecayModel.EXPONENTIAL) -> TrustDecayState:
        """Initialize trust decay tracking for an entity."""
        now = datetime.now()
        
        state = TrustDecayState(
            entity_id=entity_id,
            current_trust=initial_trust,
            base_trust=initial_trust,
            last_update=now,
            last_interaction=now,
            decay_model=decay_model
        )
        
        self.decay_states[entity_id] = state
        logger.info(f"Initialized trust decay for entity {entity_id} with model {decay_model.value}")
        
        return state
    
    async def update_trust_with_decay(self, entity_id: str, 
                                    performance_score: float,
                                    context: Optional[Dict[str, Any]] = None) -> float:
        """Update trust considering both performance and decay."""
        if entity_id not in self.decay_states:
            await self.initialize_entity(entity_id)
        
        state = self.decay_states[entity_id]
        now = datetime.now()
        
        # First apply time-based decay
        decayed_trust = await self._apply_time_decay(state, now)
        
        # Then apply performance-based adjustment
        adjusted_trust = await self._apply_performance_adjustment(
            state, decayed_trust, performance_score, context
        )
        
        # Update state
        state.current_trust = adjusted_trust
        state.last_update = now
        state.last_interaction = now
        state.total_interactions += 1
        
        if performance_score > 0.6:
            state.successful_interactions += 1
        if performance_score < 0.3:
            state.error_count += 1
        if performance_score < 0.1:
            state.critical_errors += 1
        
        # Record history
        state.trust_history.append((now, adjusted_trust))
        state.interaction_frequency.append(now)
        state.performance_variance_history.append(performance_score)
        
        # Trim history to reasonable size
        if len(state.trust_history) > 1000:
            state.trust_history = state.trust_history[-500:]
        if len(state.interaction_frequency) > 1000:
            state.interaction_frequency = state.interaction_frequency[-500:]
        if len(state.performance_variance_history) > 100:
            state.performance_variance_history = state.performance_variance_history[-50:]
        
        logger.debug(f"Updated trust for {entity_id}: {adjusted_trust:.3f} (performance: {performance_score:.3f})")
        
        return adjusted_trust
    
    async def get_current_trust(self, entity_id: str) -> float:
        """Get current trust score with decay applied."""
        if entity_id not in self.decay_states:
            return 0.5  # Neutral trust for unknown entities
        
        state = self.decay_states[entity_id]
        now = datetime.now()
        
        # Apply decay without updating the state (read-only)
        decayed_trust = await self._calculate_decayed_trust(state, now)
        
        return decayed_trust
    
    async def _apply_time_decay(self, state: TrustDecayState, current_time: datetime) -> float:
        """Apply time-based trust decay."""
        time_elapsed = current_time - state.last_update
        days_elapsed = time_elapsed.total_seconds() / (24 * 3600)
        
        if days_elapsed <= 0:
            return state.current_trust
        
        # Get the appropriate decay function
        decay_function = self.decay_functions[state.decay_model]
        decayed_trust = await decay_function(state, days_elapsed)
        
        # Apply minimum trust floor
        decayed_trust = max(decayed_trust, state.parameters.minimum_trust)
        
        return decayed_trust
    
    async def _calculate_decayed_trust(self, state: TrustDecayState, current_time: datetime) -> float:
        """Calculate decayed trust without updating state."""
        time_elapsed = current_time - state.last_update
        days_elapsed = time_elapsed.total_seconds() / (24 * 3600)
        
        if days_elapsed <= 0:
            return state.current_trust
        
        decay_function = self.decay_functions[state.decay_model]
        decayed_trust = await decay_function(state, days_elapsed)
        
        return max(decayed_trust, state.parameters.minimum_trust)
    
    async def _exponential_decay(self, state: TrustDecayState, days_elapsed: float) -> float:
        """Exponential decay model: trust = initial * e^(-rate * time)"""
        # Calculate decay rate from half-life
        decay_rate = math.log(2) / state.parameters.half_life_days
        
        # Apply contextual modifiers
        effective_rate = self._get_contextual_decay_rate(state, decay_rate)
        
        # Apply exponential decay
        decayed_trust = state.current_trust * math.exp(-effective_rate * days_elapsed)
        
        return decayed_trust
    
    async def _linear_decay(self, state: TrustDecayState, days_elapsed: float) -> float:
        """Linear decay model: trust = initial - rate * time"""
        daily_decay = state.parameters.base_decay_rate
        
        # Apply contextual modifiers
        effective_rate = self._get_contextual_decay_rate(state, daily_decay)
        
        decayed_trust = state.current_trust - (effective_rate * days_elapsed)
        
        return max(decayed_trust, state.parameters.minimum_trust)
    
    async def _logarithmic_decay(self, state: TrustDecayState, days_elapsed: float) -> float:
        """Logarithmic decay model: trust = initial - rate * log(1 + time)"""
        base_rate = state.parameters.base_decay_rate
        effective_rate = self._get_contextual_decay_rate(state, base_rate)
        
        decay_amount = effective_rate * math.log(1 + days_elapsed)
        decayed_trust = state.current_trust - decay_amount
        
        return max(decayed_trust, state.parameters.minimum_trust)
    
    async def _sigmoid_decay(self, state: TrustDecayState, days_elapsed: float) -> float:
        """Sigmoid decay model: smooth transition with stabilization periods."""
        # Sigmoid function parameters
        midpoint = state.parameters.half_life_days
        steepness = 0.1
        
        # Sigmoid decay factor (0 to 1)
        sigmoid_factor = 1 / (1 + math.exp(-steepness * (days_elapsed - midpoint)))
        
        # Apply contextual modifiers
        base_rate = state.parameters.base_decay_rate
        effective_rate = self._get_contextual_decay_rate(state, base_rate)
        
        max_decay = effective_rate * days_elapsed
        actual_decay = max_decay * sigmoid_factor
        
        decayed_trust = state.current_trust - actual_decay
        
        return max(decayed_trust, state.parameters.minimum_trust)
    
    async def _stepped_decay(self, state: TrustDecayState, days_elapsed: float) -> float:
        """Stepped decay model: discrete decay at intervals."""
        decay_interval = 7  # Decay every 7 days
        intervals_passed = int(days_elapsed / decay_interval)
        
        if intervals_passed == 0:
            return state.current_trust
        
        base_rate = state.parameters.base_decay_rate
        effective_rate = self._get_contextual_decay_rate(state, base_rate)
        
        decay_per_interval = effective_rate * decay_interval
        total_decay = decay_per_interval * intervals_passed
        
        decayed_trust = state.current_trust - total_decay
        
        return max(decayed_trust, state.parameters.minimum_trust)
    
    async def _adaptive_decay(self, state: TrustDecayState, days_elapsed: float) -> float:
        """Adaptive decay model: adjusts based on entity behavior patterns."""
        # Start with exponential decay as base
        base_decayed = await self._exponential_decay(state, days_elapsed)
        
        # Adjust based on interaction frequency
        interaction_freq = await self._calculate_interaction_frequency(state)
        if interaction_freq > 1.0:  # More than daily interaction
            # Reduce decay for active entities
            base_decayed = state.current_trust - (state.current_trust - base_decayed) * 0.5
        elif interaction_freq < 0.1:  # Less than weekly interaction
            # Increase decay for inactive entities
            base_decayed = state.current_trust - (state.current_trust - base_decayed) * 1.5
        
        # Adjust based on performance variance
        variance = state.get_recent_variance()
        if variance > 0.3:  # High variance
            variance_penalty = variance * state.parameters.variance_penalty * days_elapsed
            base_decayed -= variance_penalty
        
        # Adjust based on success rate
        success_rate = state.get_success_rate()
        if success_rate > 0.8:  # High success rate
            # Slower decay for reliable entities
            base_decayed = state.current_trust - (state.current_trust - base_decayed) * 0.8
        elif success_rate < 0.4:  # Low success rate
            # Faster decay for unreliable entities
            base_decayed = state.current_trust - (state.current_trust - base_decayed) * 1.3
        
        return max(base_decayed, state.parameters.minimum_trust)
    
    def _get_contextual_decay_rate(self, state: TrustDecayState, base_rate: float) -> float:
        """Calculate effective decay rate based on context."""
        effective_rate = base_rate
        
        # Adjust for error rates
        if state.total_interactions > 0:
            error_rate = state.error_count / state.total_interactions
            if error_rate > 0.2:
                effective_rate *= (1 + error_rate)
        
        # Adjust for critical errors
        if state.critical_errors > 0:
            critical_penalty = state.critical_errors * state.parameters.critical_error_penalty
            effective_rate += critical_penalty
        
        # Adjust for domain changes (instability)
        if state.domain_changes > 3:
            domain_instability = 1 + (state.domain_changes * 0.1)
            effective_rate *= domain_instability
        
        # Cap the maximum decay rate
        return min(effective_rate, state.parameters.maximum_decay_per_day)
    
    async def _calculate_interaction_frequency(self, state: TrustDecayState) -> float:
        """Calculate interaction frequency (interactions per day)."""
        if len(state.interaction_frequency) < 2:
            return 0.0
        
        # Look at last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_interactions = [
            dt for dt in state.interaction_frequency 
            if dt >= cutoff_date
        ]
        
        if len(recent_interactions) < 2:
            return 0.0
        
        # Calculate average daily frequency
        time_span = (recent_interactions[-1] - recent_interactions[0]).total_seconds() / (24 * 3600)
        if time_span <= 0:
            return 0.0
        
        frequency = len(recent_interactions) / time_span
        return frequency
    
    async def _apply_performance_adjustment(self, state: TrustDecayState, 
                                          current_trust: float,
                                          performance_score: float,
                                          context: Optional[Dict[str, Any]]) -> float:
        """Apply performance-based trust adjustment."""
        # Calculate adjustment magnitude
        if performance_score > 0.8:
            # Strong positive performance
            adjustment = (performance_score - 0.8) * state.parameters.recovery_factor * 0.1
        elif performance_score > 0.6:
            # Good performance - small positive adjustment
            adjustment = (performance_score - 0.6) * 0.05
        elif performance_score > 0.4:
            # Mediocre performance - minimal change
            adjustment = (performance_score - 0.5) * 0.02
        else:
            # Poor performance - negative adjustment
            adjustment = (performance_score - 0.4) * 0.15
        
        # Apply contextual modifiers
        if context:
            risk_level = context.get("risk_level", "medium")
            if risk_level == "high":
                # More sensitive to performance in high-risk contexts
                adjustment *= 1.5
            elif risk_level == "low":
                # Less sensitive in low-risk contexts
                adjustment *= 0.7
        
        adjusted_trust = current_trust + adjustment
        
        # Bound the result
        return max(0.0, min(1.0, adjusted_trust))
    
    async def get_trust_projection(self, entity_id: str, days_ahead: int = 30) -> List[Tuple[datetime, float]]:
        """Project trust decay over future time period."""
        if entity_id not in self.decay_states:
            return []
        
        state = self.decay_states[entity_id]
        projections = []
        
        current_time = datetime.now()
        current_trust = await self.get_current_trust(entity_id)
        
        for day in range(1, days_ahead + 1):
            future_time = current_time + timedelta(days=day)
            
            # Create temporary state for projection
            temp_state = TrustDecayState(
                entity_id=state.entity_id,
                current_trust=current_trust,
                base_trust=state.base_trust,
                last_update=current_time,
                last_interaction=state.last_interaction,
                decay_model=state.decay_model,
                parameters=state.parameters
            )
            
            projected_trust = await self._calculate_decayed_trust(temp_state, future_time)
            projections.append((future_time, projected_trust))
        
        return projections
    
    def get_trust_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive trust statistics for an entity."""
        if entity_id not in self.decay_states:
            return {}
        
        state = self.decay_states[entity_id]
        
        # Calculate trends
        if len(state.trust_history) >= 2:
            recent_trend = state.trust_history[-1][1] - state.trust_history[-2][1]
        else:
            recent_trend = 0.0
        
        return {
            "entity_id": entity_id,
            "current_trust": state.current_trust,
            "base_trust": state.base_trust,
            "decay_model": state.decay_model.value,
            "total_interactions": state.total_interactions,
            "successful_interactions": state.successful_interactions,
            "success_rate": state.get_success_rate(),
            "error_count": state.error_count,
            "critical_errors": state.critical_errors,
            "recent_variance": state.get_recent_variance(),
            "domain_changes": state.domain_changes,
            "days_since_last_interaction": (datetime.now() - state.last_interaction).days,
            "recent_trend": recent_trend,
            "interaction_frequency": asyncio.create_task(self._calculate_interaction_frequency(state))
        }


# Global instance for the trust decay engine
trust_decay_engine = TrustDecayEngine()