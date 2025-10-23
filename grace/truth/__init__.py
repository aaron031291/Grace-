"""
Grace AI L1 - Truth & Audit Layer (Source of Truth)
Immutable logging, crypto keys, KPIs, trust, health, clarity logs
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json
from enum import Enum

logger = logging.getLogger(__name__)

class ClarityLevel(Enum):
    """Clarity framework logging levels."""
    TRANSPARENT = "transparent"
    EXPLAINABLE = "explainable"
    VERIFIABLE = "verifiable"
    AUDITABLE = "auditable"

class TruthEntry:
    """Single immutable entry in the truth ledger."""
    
    def __init__(self, entry_id: str, event_type: str, component: str, data: Dict[str, Any]):
        self.entry_id = entry_id
        self.event_type = event_type
        self.component = component
        self.data = data
        self.timestamp = datetime.now().isoformat()
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute immutable hash."""
        entry_str = json.dumps({
            "event_type": self.event_type,
            "component": self.component,
            "data": self.data,
            "timestamp": self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(entry_str.encode()).hexdigest()[:16]

class TruthLayer:
    """
    L1 - Truth & Audit Layer
    Immutable source of truth for all system state
    """
    
    def __init__(self):
        self.entries: List[TruthEntry] = []
        self.crypto_keys: Dict[str, str] = {}
        self.kpis: Dict[str, float] = {
            "stability": 99.0,
            "performance": 98.0,
            "reliability": 99.5,
            "security": 98.0,
            "availability": 99.9,
            "clarity": 95.0
        }
        self.trust_scores: Dict[str, float] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.clarity_logs: List[Dict[str, Any]] = []
    
    async def record_event(self, event_type: str, component: str, data: Dict[str, Any]) -> TruthEntry:
        """Record immutable event."""
        import uuid
        entry_id = str(uuid.uuid4())[:8]
        entry = TruthEntry(entry_id, event_type, component, data)
        self.entries.append(entry)
        logger.info(f"Truth recorded: {event_type} ({entry_id})")
        return entry
    
    def generate_crypto_key(self, key_id: str) -> str:
        """Generate cryptographic key."""
        key = hashlib.sha256(f"{key_id}{datetime.now().isoformat()}".encode()).hexdigest()
        self.crypto_keys[key_id] = key
        logger.info(f"Generated crypto key: {key_id}")
        return key
    
    async def update_kpi(self, kpi_name: str, value: float) -> bool:
        """Update KPI (canonical metric)."""
        if kpi_name in self.kpis:
            self.kpis[kpi_name] = value
            logger.info(f"KPI updated: {kpi_name} = {value}")
            return True
        return False
    
    async def set_trust_score(self, component: str, score: float) -> bool:
        """Set trust score."""
        self.trust_scores[component] = max(0.0, min(100.0, score))
        logger.info(f"Trust score set: {component} = {score}")
        return True
    
    async def log_clarity_event(self, event: str, level: ClarityLevel, details: Dict[str, Any]):
        """Log clarity framework event."""
        clarity_entry = {
            "event": event,
            "level": level.value,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.clarity_logs.append(clarity_entry)
        logger.info(f"Clarity logged: {event} ({level.value})")
    
    def get_canonical_state(self) -> Dict[str, Any]:
        """Get canonical system state."""
        return {
            "kpis": self.kpis.copy(),
            "trust_scores": self.trust_scores.copy(),
            "total_events": len(self.entries),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history."""
        return [
            {
                "entry_id": e.entry_id,
                "event_type": e.event_type,
                "component": e.component,
                "timestamp": e.timestamp,
                "hash": e.hash
            }
            for e in self.entries[-limit:]
        ]
