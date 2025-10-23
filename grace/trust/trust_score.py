"""
Trust Score Management - Tracks reliability and trustworthiness
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust level classifications"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


@dataclass
class TrustScore:
    """Trust score for an entity"""
    entity_id: str
    entity_type: str  # 'component', 'node', 'decision', 'actor'
    score: float  # 0.0 to 1.0
    level: TrustLevel
    successes: int = 0
    failures: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrustScoreManager:
    """
    Manages trust scores across the system
    """
    
    def __init__(self):
        self.trust_scores: Dict[str, TrustScore] = {}
        self.decay_rate = 0.99  # Daily decay
        self.boost_threshold = 0.95
        self.penalty_threshold = 0.5
        logger.info("TrustScoreManager initialized")
    
    def initialize_trust(
        self,
        entity_id: str,
        entity_type: str,
        initial_score: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> TrustScore:
        """Initialize trust score for new entity"""
        level = self._score_to_level(initial_score)
        
        trust = TrustScore(
            entity_id=entity_id,
            entity_type=entity_type,
            score=initial_score,
            level=level,
            metadata=metadata or {}
        )
        
        self.trust_scores[entity_id] = trust
        logger.info(f"Initialized trust for {entity_id}: {initial_score:.2f} ({level.name})")
        
        return trust
    
    def record_success(
        self,
        entity_id: str,
        weight: float = 1.0,
        context: Optional[Dict] = None
    ):
        """Record successful action"""
        if entity_id not in self.trust_scores:
            self.initialize_trust(entity_id, "unknown")
        
        trust = self.trust_scores[entity_id]
        trust.successes += 1
        
        # Update score
        old_score = trust.score
        trust.score = min(1.0, trust.score + (0.1 * weight))
        trust.level = self._score_to_level(trust.score)
        trust.last_updated = datetime.now()
        
        # Record in history
        trust.history.append({
            'type': 'success',
            'weight': weight,
            'old_score': old_score,
            'new_score': trust.score,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        })
        
        logger.debug(f"Success recorded for {entity_id}: {old_score:.2f} -> {trust.score:.2f}")
    
    def record_failure(
        self,
        entity_id: str,
        severity: float = 1.0,
        context: Optional[Dict] = None
    ):
        """Record failed action"""
        if entity_id not in self.trust_scores:
            self.initialize_trust(entity_id, "unknown")
        
        trust = self.trust_scores[entity_id]
        trust.failures += 1
        
        # Update score
        old_score = trust.score
        trust.score = max(0.0, trust.score - (0.2 * severity))
        trust.level = self._score_to_level(trust.score)
        trust.last_updated = datetime.now()
        
        # Record in history
        trust.history.append({
            'type': 'failure',
            'severity': severity,
            'old_score': old_score,
            'new_score': trust.score,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        })
        
        logger.warning(f"Failure recorded for {entity_id}: {old_score:.2f} -> {trust.score:.2f}")
    
    def apply_decay(self):
        """Apply time-based decay to all scores"""
        now = datetime.now()
        
        for trust in self.trust_scores.values():
            # Calculate days since last update
            days_elapsed = (now - trust.last_updated).total_seconds() / 86400
            
            if days_elapsed > 0:
                # Apply decay
                decay_factor = self.decay_rate ** days_elapsed
                old_score = trust.score
                trust.score *= decay_factor
                trust.level = self._score_to_level(trust.score)
                trust.last_updated = now
                
                logger.debug(f"Applied decay to {trust.entity_id}: {old_score:.2f} -> {trust.score:.2f}")
    
    def get_trust_score(self, entity_id: str) -> Optional[TrustScore]:
        """Get current trust score"""
        return self.trust_scores.get(entity_id)
    
    def is_trusted(self, entity_id: str, min_level: TrustLevel = TrustLevel.MEDIUM) -> bool:
        """Check if entity meets minimum trust level"""
        trust = self.get_trust_score(entity_id)
        if not trust:
            return False
        
        return trust.level.value >= min_level.value
    
    def get_trusted_entities(
        self,
        entity_type: Optional[str] = None,
        min_score: float = 0.5
    ) -> List[TrustScore]:
        """Get list of trusted entities"""
        results = []
        
        for trust in self.trust_scores.values():
            if entity_type and trust.entity_type != entity_type:
                continue
            
            if trust.score >= min_score:
                results.append(trust)
        
        return sorted(results, key=lambda t: t.score, reverse=True)
    
    def _score_to_level(self, score: float) -> TrustLevel:
        """Convert numeric score to trust level"""
        if score >= 0.9:
            return TrustLevel.VERIFIED
        elif score >= 0.7:
            return TrustLevel.HIGH
        elif score >= 0.5:
            return TrustLevel.MEDIUM
        elif score >= 0.3:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
    
    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get trust system statistics"""
        if not self.trust_scores:
            return {'total_entities': 0}
        
        by_level = {}
        for level in TrustLevel:
            by_level[level.name] = sum(
                1 for t in self.trust_scores.values() if t.level == level
            )
        
        total_actions = sum(
            t.successes + t.failures for t in self.trust_scores.values()
        )
        total_successes = sum(t.successes for t in self.trust_scores.values())
        
        return {
            'total_entities': len(self.trust_scores),
            'by_level': by_level,
            'avg_score': sum(t.score for t in self.trust_scores.values()) / len(self.trust_scores),
            'total_actions': total_actions,
            'success_rate': total_successes / total_actions if total_actions > 0 else 0
        }
