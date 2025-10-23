"""
Grace AI - Core Truth Layer
MTL, Immutable Logging, Cryptographic Keys, KPIs, Trust Scores
This is the canonical source of truth for the entire system
"""
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class DataIntegrity:
    """Cryptographic integrity verification for all system data."""
    
    def __init__(self):
        self.key_registry: Dict[str, str] = {}
        self.hash_chain: List[str] = []
    
    def generate_key(self, key_id: str) -> str:
        """Generate a cryptographic key."""
        key = hashlib.sha256(f"{key_id}{datetime.now().isoformat()}".encode()).hexdigest()
        self.key_registry[key_id] = key
        logger.info(f"Generated cryptographic key: {key_id}")
        return key
    
    def sign_data(self, data: Dict[str, Any], key_id: str) -> str:
        """Sign data with a cryptographic key."""
        key = self.key_registry.get(key_id)
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        data_str = json.dumps(data, sort_keys=True)
        signature = hashlib.sha256(f"{data_str}{key}".encode()).hexdigest()
        
        self.hash_chain.append(signature)
        return signature
    
    def verify_data(self, data: Dict[str, Any], signature: str, key_id: str) -> bool:
        """Verify data integrity."""
        key = self.key_registry.get(key_id)
        if not key:
            return False
        
        data_str = json.dumps(data, sort_keys=True)
        expected_signature = hashlib.sha256(f"{data_str}{key}".encode()).hexdigest()
        
        return signature == expected_signature

class MTLKernelCore:
    """
    Multi-Task Learning Kernel - The canonical source of truth for all learned knowledge.
    All system intelligence flows through and is validated by this component.
    """
    
    def __init__(self, data_integrity: DataIntegrity):
        self.data_integrity = data_integrity
        self.learned_models: Dict[str, Dict[str, Any]] = {}
        self.task_knowledge: Dict[str, Dict[str, Any]] = {}
        self.model_versioning: Dict[str, List[str]] = {}
    
    async def register_learned_model(self, model_id: str, model_data: Dict[str, Any]) -> str:
        """Register a learned model as canonical knowledge."""
        signature = self.data_integrity.sign_data(model_data, "master_key")
        
        self.learned_models[model_id] = {
            "data": model_data,
            "signature": signature,
            "registered_at": datetime.now().isoformat(),
            "version": 1
        }
        
        logger.info(f"Registered learned model: {model_id}")
        return signature
    
    async def update_model(self, model_id: str, new_data: Dict[str, Any]) -> bool:
        """Update a learned model (creates new version)."""
        if model_id not in self.learned_models:
            logger.warning(f"Model not found: {model_id}")
            return False
        
        old_version = self.learned_models[model_id]["version"]
        signature = self.data_integrity.sign_data(new_data, "master_key")
        
        self.learned_models[model_id] = {
            "data": new_data,
            "signature": signature,
            "updated_at": datetime.now().isoformat(),
            "version": old_version + 1
        }
        
        if model_id not in self.model_versioning:
            self.model_versioning[model_id] = []
        self.model_versioning[model_id].append(signature)
        
        logger.info(f"Updated model {model_id} to version {old_version + 1}")
        return True
    
    def get_canonical_knowledge(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get the canonical, verified knowledge for a model."""
        model = self.learned_models.get(model_id)
        if not model:
            return None
        
        # Verify integrity
        if self.data_integrity.verify_data(model["data"], model["signature"], "master_key"):
            return model["data"]
        
        logger.error(f"Integrity check failed for model {model_id}")
        return None

class ImmutableTruthLog:
    """
    Immutable Ledger - The canonical source of truth for all system events.
    Every decision, change, and action is recorded here immutably.
    """
    
    def __init__(self, data_integrity: DataIntegrity):
        self.data_integrity = data_integrity
        self.entries: List[Dict[str, Any]] = []
        self.merkle_tree: List[str] = []
    
    async def record_event(
        self,
        event_type: str,
        component: str,
        data: Dict[str, Any],
        correlation_id: str = None
    ) -> str:
        """Record an immutable event in the truth log."""
        event_id = str(uuid.uuid4())[:8]
        
        entry = {
            "event_id": event_id,
            "event_type": event_type,
            "component": component,
            "data": data,
            "correlation_id": correlation_id or event_id,
            "timestamp": datetime.now().isoformat(),
            "previous_hash": self.merkle_tree[-1] if self.merkle_tree else "genesis"
        }
        
        # Sign the entry
        signature = self.data_integrity.sign_data(entry, "master_key")
        entry["signature"] = signature
        
        self.entries.append(entry)
        self.merkle_tree.append(signature)
        
        logger.info(f"Recorded immutable event: {event_type} ({event_id})")
        return event_id
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the entire chain."""
        for i, entry in enumerate(self.entries):
            expected_previous = self.merkle_tree[i-1] if i > 0 else "genesis"
            if entry["previous_hash"] != expected_previous:
                logger.error(f"Chain integrity violation at entry {i}")
                return False
        
        return True
    
    def get_entries_by_correlation(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all entries for a correlation ID."""
        return [e for e in self.entries if e["correlation_id"] == correlation_id]

class SystemMetrics:
    """
    KPIs and Trust Scores - The canonical measurement of system health.
    All other components read from but only this component writes to metrics.
    """
    
    def __init__(self):
        self.kpis: Dict[str, float] = {
            "stability": 99.0,
            "performance": 98.0,
            "reliability": 99.5,
            "security": 98.0,
            "availability": 99.9,
            "clarity": 95.0
        }
        self.trust_scores: Dict[str, float] = {}
        self.metric_history: List[Dict[str, Any]] = []
    
    async def update_kpi(self, kpi_name: str, value: float):
        """Update a KPI (canonical metric)."""
        if kpi_name in self.kpis:
            old_value = self.kpis[kpi_name]
            self.kpis[kpi_name] = value
            
            self.metric_history.append({
                "kpi": kpi_name,
                "old_value": old_value,
                "new_value": value,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"KPI updated: {kpi_name} = {value}")
    
    async def set_trust_score(self, component: str, score: float):
        """Set trust score for a component (canonical truth)."""
        self.trust_scores[component] = max(0.0, min(100.0, score))
        logger.info(f"Trust score set: {component} = {score}")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        avg_kpi = sum(self.kpis.values()) / len(self.kpis) if self.kpis else 0
        avg_trust = sum(self.trust_scores.values()) / len(self.trust_scores) if self.trust_scores else 50.0
        
        return {
            "average_kpi": avg_kpi,
            "average_trust": avg_trust,
            "system_health": "healthy" if avg_kpi > 90 and avg_trust > 60 else "degraded",
            "timestamp": datetime.now().isoformat()
        }

class CoreTruthLayer:
    """
    The Core Truth Layer - Central canonical source of truth.
    All other kernels and services depend on and validate against this layer.
    """
    
    def __init__(self):
        self.data_integrity = DataIntegrity()
        self.mtl_kernel = MTLKernelCore(self.data_integrity)
        self.immutable_log = ImmutableTruthLog(self.data_integrity)
        self.system_metrics = SystemMetrics()
        
        # Generate master key
        self.data_integrity.generate_key("master_key")
        
        logger.info("Core Truth Layer initialized - System of record established")
    
    async def get_canonical_state(self) -> Dict[str, Any]:
        """Get the canonical system state from the source of truth."""
        return {
            "metrics": self.system_metrics.kpis,
            "trust": self.system_metrics.trust_scores,
            "health": await self.system_metrics.get_overall_health(),
            "timestamp": datetime.now().isoformat()
        }
