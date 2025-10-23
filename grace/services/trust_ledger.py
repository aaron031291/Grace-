"""
Grace AI Trust Ledger - Immutable trust scoring and verification
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)

class TrustAction(Enum):
    """Types of actions that affect trust."""
    CODE_EXECUTED = "code_executed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    POLICY_VIOLATION = "policy_violation"
    IMPROVEMENT_APPLIED = "improvement_applied"
    SELF_HEALING_SUCCESS = "self_healing_success"
    SELF_HEALING_FAILURE = "self_healing_failure"

class TrustLedgerEntry:
    """An immutable entry in the trust ledger."""
    
    def __init__(self, action: TrustAction, trust_delta: float, component: str, details: Dict[str, Any]):
        self.action = action
        self.trust_delta = trust_delta
        self.component = component
        self.details = details
        self.timestamp = datetime.now().isoformat()
        self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute a hash of this entry for immutability."""
        import hashlib
        entry_str = json.dumps({
            "action": self.action.value,
            "trust_delta": self.trust_delta,
            "component": self.component,
            "timestamp": self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(entry_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "trust_delta": self.trust_delta,
            "component": self.component,
            "details": self.details,
            "timestamp": self.timestamp,
            "hash": self.entry_hash
        }

class TrustLedger:
    """Immutable ledger for tracking trust scores and decisions."""
    
    def __init__(self):
        self.entries: List[TrustLedgerEntry] = []
        self.component_trust: Dict[str, float] = {}
        self.overall_trust = 50.0
    
    def record_action(self, action: TrustAction, trust_delta: float, component: str, details: Dict[str, Any] = None) -> TrustLedgerEntry:
        """Record an action and its impact on trust."""
        entry = TrustLedgerEntry(action, trust_delta, component, details or {})
        self.entries.append(entry)
        
        # Update component trust
        current_trust = self.component_trust.get(component, 50.0)
        new_trust = max(0.0, min(100.0, current_trust + trust_delta))
        self.component_trust[component] = new_trust
        
        # Update overall trust
        if self.component_trust:
            self.overall_trust = sum(self.component_trust.values()) / len(self.component_trust)
        
        logger.info(f"Trust action recorded: {action.value} ({trust_delta:+.1f}) for {component}")
        return entry
    
    def get_component_trust(self, component: str) -> float:
        """Get the trust score for a component."""
        return self.component_trust.get(component, 50.0)
    
    def get_overall_trust(self) -> float:
        """Get the overall system trust score."""
        return self.overall_trust
    
    def get_ledger_entries(self, component: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get ledger entries, optionally filtered by component."""
        entries = self.entries
        if component:
            entries = [e for e in entries if e.component == component]
        
        return [e.to_dict() for e in entries[-limit:]]
    
    def get_trust_history(self, component: str) -> List[float]:
        """Get the trust history for a component."""
        history = []
        current_trust = 50.0
        
        for entry in self.entries:
            if entry.component == component:
                current_trust = max(0.0, min(100.0, current_trust + entry.trust_delta))
                history.append(current_trust)
        
        return history
