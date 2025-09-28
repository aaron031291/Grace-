"""Trust service for managing trust attestations and scores."""

from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
import json
from pathlib import Path

from .schemas import TrustRecord


class TrustService:
    """Service for managing trust attestations."""
    
    def __init__(self, db_path: str = "grace_trust.db"):
        """Initialize trust service with database."""
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trust_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    delta TEXT,
                    attestor TEXT,
                    timestamp TEXT,
                    trust_score REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trust_memory_id 
                ON trust_records(memory_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trust_attestor 
                ON trust_records(attestor)
            """)
    
    def init_trust(self, memory_id: str) -> str:
        """Initialize trust record for a memory entry."""
        record = TrustRecord(
            memory_id=memory_id,
            delta={},
            attestor="system",
            trust_score=0.5  # neutral starting score
        )
        return self._store_record(record)
    
    def attest(self, memory_id: str, delta: Dict[str, Any], attestor: str = "user") -> str:
        """Add trust attestation."""
        # Calculate trust score based on delta
        trust_score = self._calculate_trust_score(delta)
        
        record = TrustRecord(
            memory_id=memory_id,
            delta=delta,
            attestor=attestor,
            trust_score=trust_score
        )
        
        return self._store_record(record)
    
    def get_trust_score(self, memory_id: str) -> float:
        """Get aggregated trust score for memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT AVG(trust_score) as avg_score, COUNT(*) as count
                FROM trust_records 
                WHERE memory_id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row and row[1] > 0:  # count > 0
                return row[0]  # avg_score
            
            return 0.5  # neutral default
    
    def get_trust_records(self, memory_id: str) -> List[TrustRecord]:
        """Get all trust records for a memory entry."""
        records = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("""
                SELECT * FROM trust_records 
                WHERE memory_id = ? 
                ORDER BY timestamp DESC
            """, (memory_id,)):
                records.append(TrustRecord(
                    memory_id=row['memory_id'],
                    delta=json.loads(row['delta'] or '{}'),
                    attestor=row['attestor'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    trust_score=row['trust_score']
                ))
        
        return records
    
    def _store_record(self, record: TrustRecord) -> str:
        """Store a trust record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trust_records 
                (memory_id, delta, attestor, timestamp, trust_score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record.memory_id,
                json.dumps(record.delta),
                record.attestor,
                record.timestamp.isoformat(),
                record.trust_score
            ))
            
            return f"trust_{cursor.lastrowid}"
    
    def _calculate_trust_score(self, delta: Dict[str, Any]) -> float:
        """Calculate trust score from delta."""
        # Simple scoring algorithm - can be enhanced
        score = 0.5  # neutral base
        
        if delta.get('verified', False):
            score += 0.2
        
        if delta.get('accuracy'):
            accuracy = delta['accuracy']
            if accuracy >= 0.8:
                score += 0.3
            elif accuracy >= 0.6:
                score += 0.1
            else:
                score -= 0.1
        
        if delta.get('source_credibility'):
            credibility = delta['source_credibility']
            score += (credibility - 0.5) * 0.2
        
        if delta.get('contradictions'):
            score -= len(delta['contradictions']) * 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))