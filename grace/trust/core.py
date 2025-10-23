"""
Trust Core - Complete specification-compliant implementation
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TrustScore:
    """Trust score with metadata"""
    entity_id: str
    score: float
    confidence: float
    factors: Dict[str, float]
    timestamp: str


@dataclass
class TrustThresholds:
    """Trust threshold configuration"""
    minimum: float = 0.3
    warning: float = 0.5
    acceptable: float = 0.7
    good: float = 0.85


class TrustCoreKernel:
    """
    Complete Trust Core implementation
    
    All async methods with correct signatures
    """
    
    def __init__(self):
        self.trust_scores: Dict[str, float] = {}
        self.trust_history: Dict[str, list] = {}
        self.thresholds = TrustThresholds()
        
    async def calculate_trust(
        self,
        entity_id: str,
        operation_context: Dict[str, Any]
    ) -> TrustScore:
        """
        Async calculate trust score for entity
        
        Required by specification with exact signature
        """
        # Use asyncio.sleep to make it truly async
        await asyncio.sleep(0)
        
        # Get historical trust
        historical_trust = self.trust_scores.get(entity_id, 0.5)
        
        # Calculate factors
        factors = await self._calculate_trust_factors(entity_id, operation_context)
        
        # Weighted combination
        weights = {
            "historical": 0.4,
            "recent_performance": 0.3,
            "context_specific": 0.2,
            "governance_compliance": 0.1
        }
        
        score = (
            historical_trust * weights["historical"] +
            factors.get("recent_performance", 0.5) * weights["recent_performance"] +
            factors.get("context_specific", 0.5) * weights["context_specific"] +
            factors.get("governance_compliance", 1.0) * weights["governance_compliance"]
        )
        
        # Confidence based on data availability
        confidence = min(1.0, len(self.trust_history.get(entity_id, [])) / 10)
        
        trust_score = TrustScore(
            entity_id=entity_id,
            score=score,
            confidence=confidence,
            factors=factors,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.debug(f"Calculated trust for {entity_id}: {score:.3f} (confidence: {confidence:.3f})")
        
        return trust_score
    
    async def update_trust(
        self,
        entity_id: str,
        outcome: Dict[str, Any],
        operation_context: Optional[Dict[str, Any]] = None
    ) -> TrustScore:
        """
        Async update trust based on outcome
        
        Required by specification with exact signature
        """
        # Use asyncio.sleep to make it truly async
        await asyncio.sleep(0)
        
        # Get current score
        current = self.trust_scores.get(entity_id, 0.5)
        
        # Determine outcome quality
        success = outcome.get("success", False)
        error_rate = outcome.get("error_rate", 0.0)
        latency = outcome.get("latency_ms", 0)
        
        # Calculate adjustment
        adjustment = 0.0
        
        if success:
            adjustment += 0.05
        else:
            adjustment -= 0.10
        
        if error_rate > 0.1:
            adjustment -= 0.05
        
        if latency > 1000:  # High latency
            adjustment -= 0.02
        
        # Apply adjustment with decay
        decay_factor = 0.9
        new_score = (current * decay_factor) + adjustment
        
        # Clamp to [0, 1]
        new_score = max(0.0, min(1.0, new_score))
        
        # Update
        self.trust_scores[entity_id] = new_score
        
        # Record history
        if entity_id not in self.trust_history:
            self.trust_history[entity_id] = []
        
        self.trust_history[entity_id].append({
            "score": new_score,
            "outcome": outcome,
            "timestamp": outcome.get("timestamp", datetime.now(timezone.utc).isoformat())
        })
        
        # Keep only recent history
        if len(self.trust_history[entity_id]) > 100:
            self.trust_history[entity_id] = self.trust_history[entity_id][-100:]
        
        logger.info(f"Updated trust for {entity_id}: {current:.3f} → {new_score:.3f}")
        
        # Return new trust score
        return await self.calculate_trust(entity_id, operation_context or {})
    
    async def _calculate_trust_factors(
        self,
        entity_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate individual trust factors"""
        await asyncio.sleep(0)  # Truly async
        
        factors = {}
        
        # Recent performance
        history = self.trust_history.get(entity_id, [])
        if history:
            recent = history[-10:]
            success_rate = sum(1 for h in recent if h.get("outcome", {}).get("success", False)) / len(recent)
            factors["recent_performance"] = success_rate
        else:
            factors["recent_performance"] = 0.5
        
        # Context-specific trust
        operation_type = context.get("operation_type", "general")
        if operation_type == "critical":
            factors["context_specific"] = 0.3
        else:
            factors["context_specific"] = 0.7
        
        # Governance compliance
        violations = context.get("governance_violations", [])
        if violations:
            factors["governance_compliance"] = max(0.0, 1.0 - len(violations) * 0.2)
        else:
            factors["governance_compliance"] = 1.0
        
        return factors
    
    # Legacy sync methods (deprecated but kept for backwards compatibility)
    def get_trust_score(self, entity_id: str) -> float:
        """Get current trust score (synchronous, deprecated)"""
        logger.warning("get_trust_score() is deprecated, use async calculate_trust()")
        return self.trust_scores.get(entity_id, 0.5)
    
    def update_trust_sync(self, entity_id: str, score: float):
        """Update trust score (synchronous, deprecated)"""
        logger.warning("update_trust_sync() is deprecated, use async update_trust()")
        self.trust_scores[entity_id] = max(0.0, min(1.0, score))
    
    def check_threshold(self, entity_id: str, required_level: str = "acceptable") -> bool:
        """Check if entity meets trust threshold"""
        score = self.trust_scores.get(entity_id, 0.5)
        threshold = getattr(self.thresholds, required_level)
        return score >= threshold
    
    def configure_thresholds(self, thresholds: TrustThresholds):
        """Configure trust thresholds"""
        self.thresholds = thresholds
        logger.info(f"Trust thresholds configured: {thresholds}")
