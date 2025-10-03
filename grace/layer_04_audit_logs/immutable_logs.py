"""
Immutable Logs - Tamper-proof audit trail system for Grace governance kernel.
"""
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3
from pathlib import Path
import logging

from ..core.contracts import Experience


logger = logging.getLogger(__name__)


class LogEntry:
    """Represents an immutable log entry."""
    
    def __init__(self, entry_id: str, category: str, data: Dict[str, Any],
                 transparency_level: str = "democratic_oversight"):
        self.entry_id = entry_id
        self.category = category
        self.data = data
        self.transparency_level = transparency_level
        self.timestamp = datetime.now()
        self.hash = self._calculate_hash()
        self.previous_hash = None  # Set by ImmutableLogs
        self.chain_hash = None     # Set by ImmutableLogs
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the entry."""
        hash_data = {
            "entry_id": self.entry_id,
            "category": self.category,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "transparency_level": self.transparency_level
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def set_chain_info(self, previous_hash: str, chain_position: int):
        """Set blockchain-like chain information."""
        self.previous_hash = previous_hash
        chain_data = f"{self.hash}:{previous_hash}:{chain_position}"
        self.chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of this log entry."""
        expected_hash = self._calculate_hash()
        return self.hash == expected_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "category": self.category,
            "data": self.data,
            "transparency_level": self.transparency_level,
            "timestamp": self.timestamp.isoformat(),
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "chain_hash": self.chain_hash
        }


class ImmutableLogs:
    """
    Tamper-proof audit trail system with constitutional transparency controls.
    Provides immutable logging with blockchain-like verification for governance actions.
    """
    
    def __init__(self, db_path: str = "governance_audit.db"):
        self.db_path = db_path
        self.log_chain = []  # In-memory chain for recent entries
        self.transparency_levels = self._define_transparency_levels()
        self._init_database()
        self._load_recent_chain()
    
    def _define_transparency_levels(self) -> Dict[str, Dict[str, Any]]:
        """Define transparency levels and access controls."""
        return {
            "public": {
                "description": "Fully public, accessible to all",
                "retention_days": 2555,  # ~7 years
                "access_level": 0
            },
            "democratic_oversight": {
                "description": "Available to democratic oversight bodies",
                "retention_days": 1825,  # ~5 years
                "access_level": 1
            },
            "governance_internal": {
                "description": "Internal governance operations",
                "retention_days": 365,   # 1 year
                "access_level": 2
            },
            "audit_only": {
                "description": "Available only for audit purposes",
                "retention_days": 2555,  # ~7 years
                "access_level": 3
            },
            "security_sensitive": {
                "description": "Security-sensitive information",
                "retention_days": 90,    # 90 days
                "access_level": 4
            }
        }
    
    def _init_database(self):
        """Initialize the audit database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    entry_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    transparency_level TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    previous_hash TEXT,
                    chain_hash TEXT,
                    chain_position INTEGER,
                    verified BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Chain verification table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chain_verification (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_position INTEGER NOT NULL,
                    end_position INTEGER NOT NULL,
                    chain_hash TEXT NOT NULL,
                    verified_at TEXT NOT NULL,
                    verification_result TEXT NOT NULL
                )
            """)
            
            # Log categories table for easier querying
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS log_categories (
                    category TEXT PRIMARY KEY,
                    entry_count INTEGER DEFAULT 0,
                    last_entry_timestamp TEXT,
                    transparency_level TEXT
                )
            """)
            
            conn.commit()
    
    def _load_recent_chain(self, limit: int = 1000):
        """Load recent entries into memory for chain verification."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_id, category, data_json, transparency_level, timestamp, 
                       hash, previous_hash, chain_hash, chain_position
                FROM audit_logs 
                ORDER BY chain_position DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            # Reconstruct log entries
            self.log_chain = []
            for row in reversed(rows):  # Reverse to maintain chronological order
                entry = LogEntry(row[0], row[1], json.loads(row[2]), row[3])
                entry.timestamp = datetime.fromisoformat(row[4])
                entry.hash = row[5]
                entry.previous_hash = row[6]
                entry.chain_hash = row[7]
                self.log_chain.append(entry)
    
    async def log_governance_action(self, action_type: str, data: Dict[str, Any],
                                  transparency_level: str = "democratic_oversight") -> str:
        """
        Log a governance action with immutable audit trail.
        
        Args:
            action_type: Type of governance action
            data: Action data to log
            transparency_level: Transparency level for access control
            
        Returns:
            Entry ID of the logged action
        """
        return await self._create_log_entry(
            category="governance_action",
            data={
                "action_type": action_type,
                "action_data": data,
                "logged_by": "governance_engine"
            },
            transparency_level=transparency_level
        )
    
    async def log_decision(self, decision_data: Dict[str, Any],
                          transparency_level: str = "democratic_oversight") -> str:
        """Log a governance decision."""
        return await self._create_log_entry(
            category="governance_decision",
            data=decision_data,
            transparency_level=transparency_level
        )
    
    async def log_policy_change(self, change_data: Dict[str, Any],
                               transparency_level: str = "public") -> str:
        """Log a policy change."""
        return await self._create_log_entry(
            category="policy_change",
            data=change_data,
            transparency_level=transparency_level
        )
    
    async def log_threshold_change(self, threshold_changes: Dict[str, Any],
                                  transparency_level: str = "democratic_oversight") -> str:
        """Log threshold changes."""
        return await self._create_log_entry(
            category="threshold_change",
            data=threshold_changes,
            transparency_level=transparency_level
        )
    
    async def log_constitutional_event(self, event_data: Dict[str, Any],
                                     transparency_level: str = "public") -> str:
        """Log constitutional events."""
        return await self._create_log_entry(
            category="constitutional_operations",
            data=event_data,
            transparency_level=transparency_level
        )
    
    async def log_rollback(self, rollback_data: Dict[str, Any],
                          transparency_level: str = "democratic_oversight") -> str:
        """Log governance rollback events."""
        return await self._create_log_entry(
            category="rollback_operation",
            data=rollback_data,
            transparency_level=transparency_level
        )
    
    async def log_experience(self, experience: Experience,
                           transparency_level: str = "governance_internal") -> str:
        """Log learning experiences."""
        return await self._create_log_entry(
            category="learning_experience",
            data=experience.to_dict(),
            transparency_level=transparency_level
        )
    
    async def _create_log_entry(self, category: str, data: Dict[str, Any],
                              transparency_level: str) -> str:
        """Create and store an immutable log entry."""
        # Generate unique entry ID
        entry_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{len(self.log_chain):06d}"
        
        # Validate transparency level
        if transparency_level not in self.transparency_levels:
            transparency_level = "governance_internal"
            logger.warning(f"Invalid transparency level, using: {transparency_level}")
        
        # Create log entry
        entry = LogEntry(entry_id, category, data, transparency_level)
        
        # Set chain information
        previous_hash = self.log_chain[-1].chain_hash if self.log_chain else "genesis"
        chain_position = len(self.log_chain)
        entry.set_chain_info(previous_hash, chain_position)
        
        # Add to chain
        self.log_chain.append(entry)
        
        # Persist to database
        await self._persist_entry(entry, chain_position)
        
        # Update category statistics
        await self._update_category_stats(category, transparency_level)
        
        logger.info(f"Logged {category} entry: {entry_id} (level: {transparency_level})")
        
        return entry_id
    
    async def _persist_entry(self, entry: LogEntry, chain_position: int):
        """Persist log entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs 
                (entry_id, category, data_json, transparency_level, timestamp, 
                 hash, previous_hash, chain_hash, chain_position, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id, entry.category, json.dumps(entry.data),
                entry.transparency_level, entry.timestamp.isoformat(),
                entry.hash, entry.previous_hash, entry.chain_hash,
                chain_position, True
            ))
            conn.commit()
    
    async def _update_category_stats(self, category: str, transparency_level: str):
        """Update category statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO log_categories 
                (category, entry_count, last_entry_timestamp, transparency_level)
                VALUES (
                    ?, 
                    COALESCE((SELECT entry_count FROM log_categories WHERE category = ?), 0) + 1,
                    ?,
                    ?
                )
            """, (category, category, datetime.now().isoformat(), transparency_level))
            conn.commit()
    
    async def verify_chain_integrity(self, start_position: int = 0,
                                   end_position: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify the integrity of the log chain.
        
        Args:
            start_position: Starting position for verification
            end_position: Ending position (None for all)
            
        Returns:
            Verification results
        """
        if end_position is None:
            end_position = len(self.log_chain)
        
        verification_results = {
            "verified": True,
            "start_position": start_position,
            "end_position": end_position,
            "entries_checked": 0,
            "integrity_errors": [],
            "chain_breaks": []
        }
        
        for i in range(start_position, min(end_position, len(self.log_chain))):
            entry = self.log_chain[i]
            verification_results["entries_checked"] += 1
            
            # Verify entry integrity
            if not entry.verify_integrity():
                verification_results["verified"] = False
                verification_results["integrity_errors"].append({
                    "position": i,
                    "entry_id": entry.entry_id,
                    "error": "Hash mismatch"
                })
            
            # Verify chain linkage
            if i > 0:
                previous_entry = self.log_chain[i - 1]
                if entry.previous_hash != previous_entry.chain_hash:
                    verification_results["verified"] = False
                    verification_results["chain_breaks"].append({
                        "position": i,
                        "entry_id": entry.entry_id,
                        "expected_previous": previous_entry.chain_hash,
                        "actual_previous": entry.previous_hash
                    })
        
        # Record verification in database
        await self._record_verification(verification_results)
        
        return verification_results
    
    async def _record_verification(self, results: Dict[str, Any]):
        """Record chain verification results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chain_verification 
                (start_position, end_position, chain_hash, verified_at, verification_result)
                VALUES (?, ?, ?, ?, ?)
            """, (
                results["start_position"], results["end_position"],
                self.log_chain[-1].chain_hash if self.log_chain else "empty",
                datetime.now().isoformat(),
                json.dumps(results)
            ))
            conn.commit()
    
    def query_logs(self, category: Optional[str] = None,
                   transparency_level: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 100,
                   access_level: int = 0) -> List[Dict[str, Any]]:
        """
        Query log entries with access control.
        
        Args:
            category: Filter by category
            transparency_level: Filter by transparency level
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of entries to return
            access_level: Requester's access level
            
        Returns:
            List of log entries matching criteria
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query with access control
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            # Apply access control
            allowed_levels = [
                level for level, config in self.transparency_levels.items()
                if config["access_level"] <= access_level
            ]
            
            if allowed_levels:
                placeholders = ",".join(["?" for _ in allowed_levels])
                query += f" AND transparency_level IN ({placeholders})"
                params.extend(allowed_levels)
            else:
                # No access to any levels
                return []
            
            # Apply filters
            if category:
                query += " AND category = ?"
                params.append(category)
            
            if transparency_level and transparency_level in allowed_levels:
                query += " AND transparency_level = ?"
                params.append(transparency_level)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in rows:
                entry_dict = dict(zip(columns, row))
                entry_dict["data"] = json.loads(entry_dict["data_json"])
                del entry_dict["data_json"]
                results.append(entry_dict)
            
            return results
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute("SELECT COUNT(*) FROM audit_logs")
            total_entries = cursor.fetchone()[0]
            
            # Category breakdown
            cursor.execute("""
                SELECT category, COUNT(*) as count, 
                       MIN(timestamp) as earliest, MAX(timestamp) as latest
                FROM audit_logs 
                GROUP BY category
            """)
            category_stats = cursor.fetchall()
            
            # Transparency level breakdown
            cursor.execute("""
                SELECT transparency_level, COUNT(*) as count
                FROM audit_logs 
                GROUP BY transparency_level
            """)
            transparency_stats = cursor.fetchall()
            
            # Recent verification results
            cursor.execute("""
                SELECT verified_at, verification_result 
                FROM chain_verification 
                ORDER BY verified_at DESC 
                LIMIT 5
            """)
            recent_verifications = cursor.fetchall()
            
            return {
                "total_entries": total_entries,
                "chain_length": len(self.log_chain),
                "categories": {row[0]: {"count": row[1], "earliest": row[2], "latest": row[3]} 
                             for row in category_stats},
                "transparency_levels": {row[0]: row[1] for row in transparency_stats},
                "recent_verifications": [
                    {"verified_at": row[0], "results": json.loads(row[1])}
                    for row in recent_verifications
                ],
                "last_entry_hash": self.log_chain[-1].chain_hash if self.log_chain else None
            }
    
    async def cleanup_old_entries(self):
        """Clean up old entries based on retention policies."""
        current_time = datetime.now()
        cleanup_results = {"removed_count": 0, "categories": {}}
        
        for level, config in self.transparency_levels.items():
            retention_days = config["retention_days"]
            cutoff_date = current_time - timedelta(days=retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count entries to be removed
                cursor.execute("""
                    SELECT COUNT(*) FROM audit_logs 
                    WHERE transparency_level = ? AND timestamp < ?
                """, (level, cutoff_date.isoformat()))
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Remove old entries
                    cursor.execute("""
                        DELETE FROM audit_logs 
                        WHERE transparency_level = ? AND timestamp < ?
                    """, (level, cutoff_date.isoformat()))
                    
                    cleanup_results["removed_count"] += count
                    cleanup_results["categories"][level] = count
                    
                    logger.info(f"Cleaned up {count} old {level} entries")
                
                conn.commit()
        
        # Rebuild in-memory chain after cleanup
        self._load_recent_chain()
        
        return cleanup_results
    
    def export_audit_trail(self, category: Optional[str] = None,
                          transparency_level: Optional[str] = None,
                          access_level: int = 0) -> str:
        """Export audit trail to JSON format."""
        entries = self.query_logs(
            category=category,
            transparency_level=transparency_level,
            limit=10000,  # Large limit for export
            access_level=access_level
        )
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_entries": len(entries),
            "filters": {
                "category": category,
                "transparency_level": transparency_level,
                "access_level": access_level
            },
            "entries": entries
        }
        
        return json.dumps(export_data, indent=2)