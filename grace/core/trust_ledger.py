"""
Grace AI - Trust Ledger
Dynamic trust scoring and tracking for entities, sources, and memory fragments

The Trust Ledger maintains a running trust score for all entities Grace interacts with:
- Data sources (APIs, sensors, users, external systems)
- Memory fragments (claims, facts, reasoning chains)
- LLM models and agents
- External services

Trust scores are adjusted based on:
- Verification outcomes from VWX
- Consistency with known truths
- Historical accuracy
- Temporal decay
- Contradiction detection
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrustRecord:
    """A single trust score record for an entity"""
    entity_id: str
    entity_type: str  # "source", "memory_fragment", "model", "service"
    trust_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0 (confidence in the trust score itself)
    last_updated: float  # timestamp
    total_interactions: int
    successful_verifications: int
    failed_verifications: int
    quarantine_count: int
    metadata: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate verification success rate"""
        if self.total_interactions == 0:
            return 0.5  # Neutral for new entities
        return self.successful_verifications / self.total_interactions
    
    @property
    def trust_level(self) -> str:
        """Categorical trust level"""
        if self.trust_score >= 0.9:
            return "HIGHLY_TRUSTED"
        elif self.trust_score >= 0.7:
            return "TRUSTED"
        elif self.trust_score >= 0.5:
            return "NEUTRAL"
        elif self.trust_score >= 0.3:
            return "UNTRUSTED"
        else:
            return "QUARANTINED"


class TrustLedger:
    """
    Maintains dynamic trust scores for all entities in Grace's ecosystem.
    
    Features:
    - Real-time trust score updates based on VWX verification
    - Temporal decay (trust erodes over time without reinforcement)
    - Contradiction penalties
    - Quarantine mechanisms for low-trust entities
    - Historical tracking and audit trail
    """
    
    def __init__(self, persistence_path: str = "grace_data/trust_ledger.jsonl"):
        self.persistence_path = persistence_path
        self.trust_records: Dict[str, TrustRecord] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Configuration
        self.default_trust = 0.5  # New entities start neutral
        self.decay_rate = 0.01  # Trust decay per day without reinforcement
        self.decay_interval = 86400  # 24 hours in seconds
        
        # Load existing ledger
        self._load_ledger()
        
        logger.info(f"Trust Ledger initialized with {len(self.trust_records)} entities")
    
    def _load_ledger(self):
        """Load trust ledger from persistent storage"""
        try:
            with open(self.persistence_path, 'r') as f:
                for line in f:
                    record_data = json.loads(line.strip())
                    record = TrustRecord(**record_data)
                    self.trust_records[record.entity_id] = record
            logger.info(f"Loaded {len(self.trust_records)} trust records from {self.persistence_path}")
        except FileNotFoundError:
            logger.info(f"No existing trust ledger found at {self.persistence_path}")
        except Exception as e:
            logger.error(f"Error loading trust ledger: {e}")
    
    def _save_record(self, record: TrustRecord):
        """Append a trust record update to persistent storage"""
        try:
            with open(self.persistence_path, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
        except Exception as e:
            logger.error(f"Error saving trust record: {e}")
    
    def get_trust_score(self, entity_id: str, entity_type: str = "source") -> float:
        """Get current trust score for an entity"""
        if entity_id not in self.trust_records:
            # New entity - return default trust
            return self.default_trust
        
        record = self.trust_records[entity_id]
        
        # Apply temporal decay
        now = time.time()
        time_since_update = now - record.last_updated
        decay_periods = time_since_update / self.decay_interval
        decay_amount = decay_periods * self.decay_rate
        
        # Trust decays towards neutral (0.5)
        decayed_score = record.trust_score
        if record.trust_score > 0.5:
            decayed_score = max(0.5, record.trust_score - decay_amount)
        elif record.trust_score < 0.5:
            decayed_score = min(0.5, record.trust_score + decay_amount)
        
        return decayed_score
    
    def get_trust_record(self, entity_id: str) -> Optional[TrustRecord]:
        """Get full trust record for an entity"""
        return self.trust_records.get(entity_id)
    
    def update_trust(
        self, 
        entity_id: str, 
        entity_type: str,
        delta: float,
        event_id: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrustRecord:
        """
        Update trust score for an entity based on verification outcome.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity ("source", "memory_fragment", "model", "service")
            delta: Trust delta (-1.0 to +1.0)
            event_id: Event that triggered this update
            reason: Reason for the trust update
            metadata: Additional context
        
        Returns:
            Updated TrustRecord
        """
        now = time.time()
        
        # Get or create record
        if entity_id in self.trust_records:
            record = self.trust_records[entity_id]
            
            # Apply temporal decay before update
            current_trust = self.get_trust_score(entity_id, entity_type)
            record.trust_score = current_trust
        else:
            record = TrustRecord(
                entity_id=entity_id,
                entity_type=entity_type,
                trust_score=self.default_trust,
                confidence=0.5,
                last_updated=now,
                total_interactions=0,
                successful_verifications=0,
                failed_verifications=0,
                quarantine_count=0,
                metadata=metadata or {}
            )
        
        # Apply delta with bounds checking
        new_trust = max(0.0, min(1.0, record.trust_score + delta))
        record.trust_score = new_trust
        record.last_updated = now
        record.total_interactions += 1
        
        # Update verification counts
        if delta > 0:
            record.successful_verifications += 1
        elif delta < 0:
            record.failed_verifications += 1
        
        # Quarantine check
        if new_trust < 0.3:
            record.quarantine_count += 1
            logger.warning(f"Entity {entity_id} QUARANTINED (trust={new_trust:.2f})")
        
        # Update confidence based on interaction count
        # More interactions = higher confidence in the score
        record.confidence = min(1.0, record.total_interactions / 100.0)
        
        # Update metadata
        if metadata:
            record.metadata.update(metadata)
        record.metadata["last_event_id"] = event_id
        record.metadata["last_reason"] = reason
        
        # Store updated record
        self.trust_records[entity_id] = record
        self._save_record(record)
        
        # Log to history
        history_entry = {
            "timestamp": now,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "delta": delta,
            "new_trust": new_trust,
            "event_id": event_id,
            "reason": reason
        }
        self.history.append(history_entry)
        
        logger.info(
            f"TRUST_UPDATE: entity={entity_id}, type={entity_type}, "
            f"delta={delta:+.2f}, new_trust={new_trust:.2f}, level={record.trust_level}"
        )
        
        return record
    
    def quarantine_entity(self, entity_id: str, reason: str, event_id: str):
        """Immediately quarantine an entity (set trust to 0.0)"""
        return self.update_trust(
            entity_id=entity_id,
            entity_type=self.trust_records.get(entity_id, TrustRecord(
                entity_id=entity_id,
                entity_type="unknown",
                trust_score=0.5,
                confidence=0.0,
                last_updated=time.time(),
                total_interactions=0,
                successful_verifications=0,
                failed_verifications=0,
                quarantine_count=0,
                metadata={}
            )).entity_type,
            delta=-1.0,
            event_id=event_id,
            reason=f"QUARANTINE: {reason}"
        )
    
    def restore_entity(self, entity_id: str, new_trust: float, reason: str, event_id: str):
        """Restore a quarantined entity to a specific trust level"""
        if entity_id not in self.trust_records:
            logger.warning(f"Cannot restore unknown entity: {entity_id}")
            return None
        
        record = self.trust_records[entity_id]
        delta = new_trust - record.trust_score
        
        return self.update_trust(
            entity_id=entity_id,
            entity_type=record.entity_type,
            delta=delta,
            event_id=event_id,
            reason=f"RESTORE: {reason}"
        )
    
    def get_trusted_entities(self, entity_type: Optional[str] = None, min_trust: float = 0.7) -> List[TrustRecord]:
        """Get all entities above a trust threshold"""
        results = []
        for record in self.trust_records.values():
            if entity_type and record.entity_type != entity_type:
                continue
            current_trust = self.get_trust_score(record.entity_id, record.entity_type)
            if current_trust >= min_trust:
                results.append(record)
        return sorted(results, key=lambda r: r.trust_score, reverse=True)
    
    def get_quarantined_entities(self) -> List[TrustRecord]:
        """Get all quarantined entities"""
        results = []
        for record in self.trust_records.values():
            current_trust = self.get_trust_score(record.entity_id, record.entity_type)
            if current_trust < 0.3:
                results.append(record)
        return sorted(results, key=lambda r: r.trust_score)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trust ledger statistics"""
        total = len(self.trust_records)
        by_type = defaultdict(int)
        by_level = defaultdict(int)
        
        for record in self.trust_records.values():
            by_type[record.entity_type] += 1
            by_level[record.trust_level] += 1
        
        return {
            "total_entities": total,
            "by_type": dict(by_type),
            "by_level": dict(by_level),
            "total_interactions": sum(r.total_interactions for r in self.trust_records.values()),
            "quarantined_count": by_level.get("QUARANTINED", 0)
        }
