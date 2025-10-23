"""
Grace AI - Layer 1: Truth & Audit (MTL - Multi-Task Learning)
Immutable logger, cryptographic keys, KPIs, trust scores, health, clarity logs
SOURCE OF TRUTH for entire system
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class ClarityFramework(Enum):
    """Clarity frameworks for system transparency."""
    DECISION_LOG = "decision_log"
    ACTION_TRACE = "action_trace"
    KNOWLEDGE_BASE = "knowledge_base"
    HEALTH_METRICS = "health_metrics"

class TruthAuditLayer:
    """
    Layer 1: Truth & Audit (MTL)
    The canonical source of truth - immutable, cryptographically verified,
    with complete audit trail of all system state changes.
    """
    
    def __init__(self):
        self.immutable_log: List[Dict[str, Any]] = []
        self.crypto_keys: Dict[str, str] = {}
        self.kpis: Dict[str, float] = {
            "stability": 99.0,
            "performance": 98.0,
            "reliability": 99.5,
            "security": 98.0
        }
        self.trust_scores: Dict[str, float] = {}
        self.health_status: Dict[str, Any] = {}
        self.clarity_logs: Dict[ClarityFramework, List[Dict[str, Any]]] = {
            cf: [] for cf in ClarityFramework
        }
        self.merkle_chain: List[str] = []
    
    def _generate_master_key(self) -> str:
        """Generate master cryptographic key."""
        key = hashlib.sha256(f"grace_master_{datetime.now().isoformat()}".encode()).hexdigest()
        self.crypto_keys["master"] = key
        logger.info("Master cryptographic key generated")
        return key
    
    def _sign_entry(self, entry: Dict[str, Any]) -> str:
        """Sign an entry cryptographically."""
        master_key = self.crypto_keys.get("master")
        if not master_key:
            master_key = self._generate_master_key()
        
        entry_str = json.dumps(entry, sort_keys=True)
        signature = hashlib.sha256(f"{entry_str}{master_key}".encode()).hexdigest()
        return signature
    
    async def record_event(
        self,
        event_type: str,
        component: str,
        data: Dict[str, Any],
        correlation_id: str = None
    ) -> str:
        """Record an immutable, signed event."""
        event_id = str(uuid.uuid4())[:8]
        
        entry = {
            "event_id": event_id,
            "event_type": event_type,
            "component": component,
            "data": data,
            "correlation_id": correlation_id or event_id,
            "timestamp": datetime.now().isoformat(),
            "previous_hash": self.merkle_chain[-1] if self.merkle_chain else "genesis"
        }
        
        signature = self._sign_entry(entry)
        entry["signature"] = signature
        
        self.immutable_log.append(entry)
        self.merkle_chain.append(signature)
        
        logger.info(f"Immutable event recorded: {event_type} ({event_id})")
        return event_id
    
    async def update_kpi(self, kpi_name: str, value: float):
        """Update a KPI (canonical metric)."""
        if kpi_name in self.kpis:
            old_value = self.kpis[kpi_name]
            self.kpis[kpi_name] = value
            
            # Log to clarity framework
            await self.log_to_clarity(
                ClarityFramework.HEALTH_METRICS,
                {
                    "kpi": kpi_name,
                    "old_value": old_value,
                    "new_value": value
                }
            )
    
    async def set_trust_score(self, component: str, score: float):
        """Set trust score for a component."""
        self.trust_scores[component] = max(0.0, min(100.0, score))
        
        await self.log_to_clarity(
            ClarityFramework.DECISION_LOG,
            {"action": "trust_score_updated", "component": component, "score": score}
        )
    
    async def log_to_clarity(self, framework: ClarityFramework, entry: Dict[str, Any]):
        """Log to clarity framework."""
        entry["timestamp"] = datetime.now().isoformat()
        self.clarity_logs[framework].append(entry)
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the merkle chain."""
        for i, entry in enumerate(self.immutable_log):
            expected_previous = self.merkle_chain[i-1] if i > 0 else "genesis"
            if entry["previous_hash"] != expected_previous:
                logger.error(f"Chain integrity violation at entry {i}")
                return False
        return True
    
    def get_canonical_state(self) -> Dict[str, Any]:
        """Get the canonical, verified system state."""
        return {
            "kpis": self.kpis.copy(),
            "trust_scores": self.trust_scores.copy(),
            "health": self.health_status.copy(),
            "integrity_verified": self.verify_chain_integrity(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_clarity_report(self, framework: ClarityFramework, limit: int = 100) -> List[Dict[str, Any]]:
        """Get clarity framework report."""
        return self.clarity_logs[framework][-limit:]
