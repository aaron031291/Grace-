"""
MemoryCore - Persistent storage and retrieval system for governance precedents and decisions.
"""
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
import sqlite3
from pathlib import Path
import logging

from .contracts import UnifiedDecision, GovernanceSnapshot, Experience


logger = logging.getLogger(__name__)


class MemoryCore:
    """
    Central memory system for storing and retrieving governance decisions,
    precedents, snapshots, and learning experiences.
    """
    
    def __init__(self, db_path: str = "grace_governance.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Governance decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS governance_decisions (
                    decision_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    inputs_hash TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    trust_score REAL NOT NULL,
                    outcome TEXT,
                    instance_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    raw_data TEXT NOT NULL
                )
            """)
            
            # Governance snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS governance_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    instance_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    thresholds_json TEXT NOT NULL,
                    policies_json TEXT NOT NULL,
                    model_weights_json TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Shadow mode deltas table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS governance_deltas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    instance_a TEXT NOT NULL,
                    instance_b TEXT NOT NULL,
                    a_vs_b_diff TEXT NOT NULL,
                    latency_diff REAL NOT NULL,
                    compliance_diff REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Experiences table for meta-learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    success_score REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Precedents table for case-based reasoning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_hash TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    similarity_keywords TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.commit()
            logger.info("Database schema initialized")
    
    def store_decision(self, decision: UnifiedDecision, outcome: Optional[str] = None,
                      instance_id: str = "default", version: str = "1.0.0"):
        """Store a governance decision."""
        inputs_hash = self._hash_dict(decision.inputs)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO governance_decisions 
                (decision_id, subject, inputs_hash, recommendation, rationale, 
                 confidence, trust_score, outcome, instance_id, version, timestamp, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id, decision.topic, inputs_hash,
                decision.recommendation, decision.rationale,
                decision.confidence, decision.trust_score, outcome,
                instance_id, version, decision.timestamp.isoformat(),
                json.dumps(decision.to_dict())
            ))
            conn.commit()
        
        # Create precedent entry
        self._create_precedent(decision.topic, decision.decision_id, 
                              decision.recommendation, decision.confidence)
        
        logger.info(f"Stored decision {decision.decision_id}")
    
    def store_snapshot(self, snapshot: GovernanceSnapshot):
        """Store a governance snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO governance_snapshots
                (snapshot_id, instance_id, version, thresholds_json, policies_json,
                 model_weights_json, state_hash, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id, snapshot.instance_id, snapshot.version,
                json.dumps(snapshot.thresholds), json.dumps(snapshot.policies),
                json.dumps(snapshot.model_weights), snapshot.state_hash,
                snapshot.created_at.isoformat()
            ))
            conn.commit()
        
        logger.info(f"Stored snapshot {snapshot.snapshot_id}")
    
    def store_experience(self, experience: Experience):
        """Store a learning experience."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiences
                (type, component_id, context_json, outcome_json, success_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experience.type, experience.component_id,
                json.dumps(experience.context), json.dumps(experience.outcome),
                experience.success_score, experience.timestamp.isoformat()
            ))
            conn.commit()
        
        logger.info(f"Stored experience: {experience.type} from {experience.component_id}")
    
    def get_similar_decisions(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar decisions for precedent-based reasoning."""
        keywords = self._extract_keywords(topic)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Simple keyword matching for now - could be enhanced with embeddings
            cursor.execute("""
                SELECT * FROM precedents 
                WHERE similarity_keywords LIKE ? 
                ORDER BY confidence DESC, timestamp DESC
                LIMIT ?
            """, (f"%{keywords}%", limit))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def get_decision_history(self, instance_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent decision history for an instance."""
        cutoff = iso_format()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM governance_decisions 
                WHERE instance_id = ? AND timestamp >= date(?, '-{} days')
                ORDER BY timestamp DESC
            """.format(days), (instance_id, cutoff))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def get_latest_snapshot(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot for an instance."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM governance_snapshots 
                WHERE instance_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (instance_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def store_shadow_delta(self, decision_id: str, instance_a: str, instance_b: str,
                          diff_data: Dict[str, Any], latency_diff: float, compliance_diff: float):
        """Store comparison data between shadow instances."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO governance_deltas
                (decision_id, instance_a, instance_b, a_vs_b_diff, latency_diff, compliance_diff, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id, instance_a, instance_b, json.dumps(diff_data),
                latency_diff, compliance_diff, iso_format()
            ))
            conn.commit()
    
    def get_shadow_performance_metrics(self, instance_a: str, instance_b: str, 
                                     limit: int = 1000) -> Dict[str, Any]:
        """Get performance comparison metrics between shadow instances."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT latency_diff, compliance_diff FROM governance_deltas
                WHERE instance_a = ? AND instance_b = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (instance_a, instance_b, limit))
            
            rows = cursor.fetchall()
            if not rows:
                return {"average_latency_diff": 0, "average_compliance_diff": 0, "count": 0}
            
            latency_diffs = [row[0] for row in rows]
            compliance_diffs = [row[1] for row in rows]
            
            return {
                "average_latency_diff": sum(latency_diffs) / len(latency_diffs),
                "average_compliance_diff": sum(compliance_diffs) / len(compliance_diffs),
                "count": len(rows),
                "latest_latency_diff": latency_diffs[0] if latency_diffs else 0,
                "latest_compliance_diff": compliance_diffs[0] if compliance_diffs else 0
            }
    
    def _create_precedent(self, topic: str, decision_id: str, outcome: str, confidence: float):
        """Create a precedent entry for case-based reasoning."""
        topic_hash = hashlib.md5(topic.encode()).hexdigest()
        keywords = self._extract_keywords(topic)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO precedents
                (topic_hash, decision_id, similarity_keywords, outcome, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                topic_hash, decision_id, keywords, outcome, confidence,
                iso_format()
            ))
            conn.commit()
    
    def _extract_keywords(self, text: str) -> str:
        """Simple keyword extraction - could be enhanced with NLP."""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(keywords[:10])  # Take up to 10 keywords
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create a hash of dictionary data for deduplication."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def close(self):
        """Close database connections."""
        # For SQLite, connections are closed automatically with context manager
        logger.info("MemoryCore connections closed")