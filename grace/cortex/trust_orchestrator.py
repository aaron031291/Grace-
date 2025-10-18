"""
Trust Orchestrator - Pod-level trust management (OLD Cortex style)
Enhanced with timezone-aware timestamps
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
import json
import logging
import os

logger = logging.getLogger(__name__)


class TrustOrchestrator:
    """
    Manages trust scores for pods with detailed component tracking
    Original Cortex implementation with timezone fixes
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = logging.getLogger("grace.cortex.trust")
        self.trust_scores: Dict[str, Dict[str, Any]] = {}
        self.trust_lock = Lock()
        
        # Setup storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "cortex_trust"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Trust orchestration initialized at {self.storage_path}")
        
        self._load_trust_scores()
    
    def _load_trust_scores(self) -> None:
        """Load trust scores from storage"""
        try:
            trust_files = list(self.storage_path.glob("*.json"))
            for trust_file in trust_files:
                try:
                    with open(trust_file, "r") as f:
                        trust_data = json.load(f)
                    
                    pod_id = trust_data.get("pod_id")
                    if pod_id:
                        with self.trust_lock:
                            self.trust_scores[pod_id] = trust_data
                
                except Exception as e:
                    self.logger.error(f"Error loading trust from {trust_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error loading trust scores: {e}")
    
    def _save_trust_score(self, pod_id: str) -> bool:
        """Save trust score to storage"""
        try:
            with self.trust_lock:
                if pod_id not in self.trust_scores:
                    return False
                
                trust_data = self.trust_scores[pod_id]
                file_path = self.storage_path / f"{pod_id}.json"
                
                with open(file_path, "w") as f:
                    json.dump(trust_data, f, indent=2)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error saving trust score for {pod_id}: {e}")
            return False
    
    def initialize_trust_score(
        self,
        pod_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Initialize trust score for new pod"""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()  # FIXED
            
            trust_record = {
                "pod_id": pod_id,
                "trust_score": 0.5,
                "confidence": 0.3,
                "components": {
                    "history": 0.5,
                    "verification": 0.5,
                    "consistency": 0.5,
                    "context": 0.5,
                    "source": 0.5
                },
                "history": [],
                "created_at": timestamp,
                "updated_at": timestamp,
                "metadata": metadata or {}
            }
            
            with self.trust_lock:
                self.trust_scores[pod_id] = trust_record
            
            self._save_trust_score(pod_id)
            self.logger.info(f"Initialized trust score for pod {pod_id}")
            
            return trust_record
        
        except Exception as e:
            self.logger.error(f"Error initializing trust score: {e}")
            raise ValueError(f"Failed to initialize trust score: {e}")
    
    def update_trust_score(
        self,
        pod_id: str,
        components: Dict[str, float],
        reason: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update trust score with component details"""
        try:
            with self.trust_lock:
                if pod_id not in self.trust_scores:
                    trust_record = self.initialize_trust_score(pod_id)
                else:
                    trust_record = self.trust_scores[pod_id]
                
                # Update components
                for component, score in components.items():
                    if component in trust_record["components"]:
                        trust_record["components"][component] = max(0.0, min(1.0, score))
                
                # Calculate weighted average
                weights = {
                    "history": 0.25,
                    "verification": 0.2,
                    "consistency": 0.2,
                    "context": 0.15,
                    "source": 0.2
                }
                
                weighted_sum = sum(
                    trust_record["components"][comp] * weight
                    for comp, weight in weights.items()
                    if comp in trust_record["components"]
                )
                weight_total = sum(weights.values())
                
                trust_record["trust_score"] = weighted_sum / weight_total
                trust_record["confidence"] = min(0.95, trust_record["confidence"] + 0.05)
                
                # Record in history
                timestamp = datetime.now(timezone.utc).isoformat()  # FIXED
                trust_record["updated_at"] = timestamp
                
                history_entry = {
                    "timestamp": timestamp,
                    "previous_score": trust_record.get("trust_score", 0.5),
                    "new_score": trust_record["trust_score"],
                    "reason": reason,
                    "components": components.copy(),
                    "evidence": evidence or {}
                }
                
                trust_record["history"].append(history_entry)
                
                # Limit history
                if len(trust_record["history"]) > 100:
                    trust_record["history"] = trust_record["history"][-100:]
                
                self._save_trust_score(pod_id)
                
                return trust_record
        
        except Exception as e:
            self.logger.error(f"Error updating trust score: {e}")
            raise ValueError(f"Failed to update trust score: {e}")
    
    def get_trust_score(self, pod_id: str) -> Dict[str, Any]:
        """Get trust score for pod"""
        with self.trust_lock:
            if pod_id not in self.trust_scores:
                raise ValueError(f"No trust record for pod {pod_id}")
            return self.trust_scores[pod_id].copy()
    
    def evaluate_trust_threshold(
        self,
        pod_id: str,
        threshold: float
    ) -> tuple[bool, Dict[str, Any]]:
        """Evaluate if pod meets trust threshold"""
        try:
            with self.trust_lock:
                if pod_id not in self.trust_scores:
                    return False, {"error": f"No trust record for pod {pod_id}"}
                
                trust_record = self.trust_scores[pod_id].copy()
                trust_score = trust_record["trust_score"]
                meets_threshold = trust_score >= threshold
                
                return meets_threshold, {
                    "meets_threshold": meets_threshold,
                    "trust_score": trust_score,
                    "threshold": threshold,
                    "margin": trust_score - threshold,
                    "trust_record": trust_record
                }
        
        except Exception as e:
            self.logger.error(f"Error evaluating trust threshold: {e}")
            return False, {"error": str(e)}
    
    def calculate_system_trust(self) -> Dict[str, Any]:
        """Calculate system-wide trust metrics"""
        with self.trust_lock:
            if not self.trust_scores:
                return {
                    "average_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "pod_count": 0
                }
            
            scores = [record["trust_score"] for record in self.trust_scores.values()]
            
            return {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pod_count": len(scores),
                "high_trust_count": sum(1 for s in scores if s >= 0.7),
                "medium_trust_count": sum(1 for s in scores if 0.4 <= s < 0.7),
                "low_trust_count": sum(1 for s in scores if s < 0.4)
            }
