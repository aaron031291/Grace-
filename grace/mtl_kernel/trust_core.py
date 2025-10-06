"""Trust Core - Real-time trust calculation and management."""

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrustEntity:
    """Entity registered in trust system."""
    entity_id: str
    entity_type: str  # component, data_source, agent
    trust_score: float
    created_at: datetime
    last_updated: datetime
    history: List[Dict[str, Any]]


@dataclass
class TrustUpdate:
    """Trust score update record."""
    entity_id: str
    old_score: float
    new_score: float
    delta: float
    reason: str
    timestamp: datetime


class TrustCore:
    """
    Real-time trust calculation and entity registry.
    
    Features:
    - Trust score calculation (0.0-1.0)
    - Entity trust registry
    - Performance-based trust updates
    - Trust decay algorithms
    - Context-aware trust thresholds
    - Trust history tracking
    """
    
    # Trust decay parameters
    DECAY_RATE = 0.01  # 1% decay per day without activity
    MIN_TRUST_SCORE = 0.0
    MAX_TRUST_SCORE = 1.0
    DEFAULT_TRUST = 0.5
    
    def __init__(self):
        self._entities: Dict[str, TrustEntity] = {}
        self._lock = asyncio.Lock()
        
        # Trust factors configuration
        self._trust_factors = {
            "historical_performance": 0.30,
            "constitutional_compliance": 0.25,
            "error_frequency": 0.20,
            "response_time": 0.15,
            "governance_approval": 0.10
        }
        
        # Statistics
        self._stats = {
            "total_entities": 0,
            "total_updates": 0,
            "total_calculations": 0,
            "decay_applications": 0,
            "start_time": time.time()
        }
        
        logger.info("Trust Core initialized")
    
    async def register_entity(
        self,
        entity_id: str,
        entity_type: str = "component",
        initial_trust: float = DEFAULT_TRUST
    ) -> bool:
        """Register a new entity in the trust system."""
        async with self._lock:
            if entity_id in self._entities:
                logger.warning(f"Entity already registered: {entity_id}")
                return False
            
            # Clamp initial trust
            initial_trust = max(self.MIN_TRUST_SCORE, min(self.MAX_TRUST_SCORE, initial_trust))
            
            entity = TrustEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                trust_score=initial_trust,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                history=[]
            )
            
            self._entities[entity_id] = entity
            self._stats["total_entities"] += 1
            
            logger.info(f"Registered entity: {entity_id} with trust {initial_trust}")
            return True
    
    async def calculate_trust(
        self,
        entity_id: str,
        operation_context: Dict[str, Any]
    ) -> float:
        """
        Calculate trust score based on context and history.
        
        Args:
            entity_id: Entity to calculate trust for
            operation_context: Context including sensitivity, risk level, etc.
        
        Returns:
            Trust score (0.0-1.0)
        """
        async with self._lock:
            if entity_id not in self._entities:
                # Auto-register with default trust
                await self.register_entity(entity_id)
            
            entity = self._entities[entity_id]
            base_score = entity.trust_score
            
            # Apply context modifiers
            sensitivity = operation_context.get("sensitivity", 0.5)
            risk_level = operation_context.get("risk_level", 0.5)
            
            # Higher sensitivity or risk reduces effective trust
            context_modifier = 1.0 - (sensitivity * 0.2) - (risk_level * 0.2)
            
            final_score = base_score * context_modifier
            final_score = max(self.MIN_TRUST_SCORE, min(self.MAX_TRUST_SCORE, final_score))
            
            self._stats["total_calculations"] += 1
            
            return final_score
    
    async def update_trust(
        self,
        entity_id: str,
        performance_data: Dict[str, Any]
    ) -> float:
        """
        Update trust score based on performance data.
        
        Args:
            performance_data: Dict containing:
                - success: bool
                - error_count: int
                - response_time_ms: float
                - constitutional_compliant: bool
                - governance_approved: bool
        
        Returns:
            New trust score
        """
        async with self._lock:
            if entity_id not in self._entities:
                await self.register_entity(entity_id)
            
            entity = self._entities[entity_id]
            old_score = entity.trust_score
            
            # Calculate trust delta based on performance factors
            delta = 0.0
            
            # Success/failure impact
            if performance_data.get("success", True):
                delta += 0.05
            else:
                delta -= 0.10
            
            # Error frequency impact
            error_count = performance_data.get("error_count", 0)
            if error_count == 0:
                delta += 0.02
            else:
                delta -= 0.05 * min(error_count, 5)
            
            # Response time impact
            response_time = performance_data.get("response_time_ms", 0)
            if response_time < 100:
                delta += 0.03
            elif response_time > 1000:
                delta -= 0.05
            
            # Constitutional compliance
            if performance_data.get("constitutional_compliant", True):
                delta += 0.05
            else:
                delta -= 0.15  # Strong penalty for non-compliance
            
            # Governance approval
            if performance_data.get("governance_approved", True):
                delta += 0.03
            
            # Apply weighted delta
            new_score = old_score + (delta * 0.5)  # Dampen changes
            new_score = max(self.MIN_TRUST_SCORE, min(self.MAX_TRUST_SCORE, new_score))
            
            # Update entity
            entity.trust_score = new_score
            entity.last_updated = datetime.utcnow()
            
            # Record update
            update = TrustUpdate(
                entity_id=entity_id,
                old_score=old_score,
                new_score=new_score,
                delta=new_score - old_score,
                reason=str(performance_data),
                timestamp=datetime.utcnow()
            )
            entity.history.append(update.__dict__)
            
            self._stats["total_updates"] += 1
            
            logger.debug(f"Trust updated for {entity_id}: {old_score:.3f} -> {new_score:.3f}")
            
            return new_score
    
    async def get_trust_score(self, entity_id: str) -> float:
        """Get current trust score for entity."""
        async with self._lock:
            if entity_id not in self._entities:
                return self.DEFAULT_TRUST
            
            return self._entities[entity_id].trust_score
    
    async def apply_decay(self, entity_id: str) -> float:
        """
        Apply time-based trust decay.
        
        Trust decays when there's no recent activity.
        """
        async with self._lock:
            if entity_id not in self._entities:
                return self.DEFAULT_TRUST
            
            entity = self._entities[entity_id]
            
            # Calculate days since last update
            days_inactive = (datetime.utcnow() - entity.last_updated).days
            
            if days_inactive > 0:
                # Exponential decay
                decay_factor = math.exp(-self.DECAY_RATE * days_inactive)
                new_score = entity.trust_score * decay_factor
                new_score = max(self.MIN_TRUST_SCORE, new_score)
                
                old_score = entity.trust_score
                entity.trust_score = new_score
                entity.last_updated = datetime.utcnow()
                
                # Record decay
                entity.history.append({
                    "entity_id": entity_id,
                    "old_score": old_score,
                    "new_score": new_score,
                    "delta": new_score - old_score,
                    "reason": f"decay_{days_inactive}_days",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                self._stats["decay_applications"] += 1
                
                logger.debug(f"Applied decay to {entity_id}: {old_score:.3f} -> {new_score:.3f}")
                
                return new_score
            
            return entity.trust_score
    
    async def get_trust_history(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get trust history for entity."""
        async with self._lock:
            if entity_id not in self._entities:
                return []
            
            return self._entities[entity_id].history[-50:]  # Last 50 updates
    
    async def validate_trust_threshold(
        self,
        entity_id: str,
        required_trust: float,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate if entity meets required trust threshold.
        
        Args:
            entity_id: Entity to validate
            required_trust: Minimum required trust score
            operation_context: Optional context for calculation
        
        Returns:
            True if trust meets or exceeds threshold
        """
        if operation_context:
            trust_score = await self.calculate_trust(entity_id, operation_context)
        else:
            trust_score = await self.get_trust_score(entity_id)
        
        return trust_score >= required_trust
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get trust system statistics."""
        async with self._lock:
            uptime = time.time() - self._stats["start_time"]
            
            # Calculate average trust score
            avg_trust = 0.0
            if self._entities:
                avg_trust = sum(e.trust_score for e in self._entities.values()) / len(self._entities)
            
            # Trust distribution
            high_trust = sum(1 for e in self._entities.values() if e.trust_score >= 0.7)
            medium_trust = sum(1 for e in self._entities.values() if 0.4 <= e.trust_score < 0.7)
            low_trust = sum(1 for e in self._entities.values() if e.trust_score < 0.4)
            
            return {
                "total_entities": self._stats["total_entities"],
                "total_updates": self._stats["total_updates"],
                "total_calculations": self._stats["total_calculations"],
                "decay_applications": self._stats["decay_applications"],
                "avg_trust_score": round(avg_trust, 3),
                "trust_distribution": {
                    "high": high_trust,
                    "medium": medium_trust,
                    "low": low_trust
                },
                "uptime_seconds": round(uptime, 1)
            }
