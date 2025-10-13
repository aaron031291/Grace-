"""Memory Snapshots Manager - Manages versioned snapshots of Memory Kernel state for rollback capabilities."""

import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class MemorySnapshotManager:
    """Manages versioned snapshots of Memory Kernel state for rollback capabilities."""

    def __init__(self, db_path: str, storage_path: str = "/tmp/memory_snapshots"):
        self.db_path = db_path
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize snapshot database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for snapshot management."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    lightning_state_json TEXT NOT NULL,
                    fusion_index_json TEXT NOT NULL,
                    librarian_config_json TEXT NOT NULL,
                    access_patterns_json TEXT NOT NULL,
                    statistics_json TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    size_bytes INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize memory snapshots database: {e}")

    def create_snapshot(self, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a snapshot of current Memory Kernel state."""
        try:
            snapshot_id = (
                f"memory_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            # Collect current state
            lightning_state = self._collect_lightning_state()
            fusion_index = self._collect_fusion_index()
            librarian_config = self._collect_librarian_config()
            access_patterns = self._collect_access_patterns()
            statistics = self._collect_memory_statistics()

            # Create snapshot data
            snapshot_data = {
                "snapshot_id": snapshot_id,
                "lightning_state": lightning_state,
                "fusion_index": fusion_index,
                "librarian_config": librarian_config,
                "access_patterns": access_patterns,
                "statistics": statistics,
                "created_at": datetime.utcnow().isoformat(),
                "description": description or f"Memory snapshot {snapshot_id}",
            }

            # Generate hash
            hash_content = json.dumps(snapshot_data, sort_keys=True)
            snapshot_hash = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
            snapshot_data["hash"] = snapshot_hash

            # Store snapshot
            self._store_snapshot(snapshot_data)

            logger.info(f"Created memory snapshot: {snapshot_id}")
            return snapshot_data

        except Exception as e:
            logger.error(f"Failed to create memory snapshot: {e}")
            return {}

    def _collect_lightning_state(self) -> Dict[str, Any]:
        """Collect current lightning memory state."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT key, content_type, ttl_seconds, access_count, created_at, last_accessed_at
                FROM lightning_memory 
                WHERE expires_at > CURRENT_TIMESTAMP OR expires_at IS NULL
                ORDER BY last_accessed_at DESC
            """)

            entries = []
            for row in cursor.fetchall():
                entries.append(
                    {
                        "key": row["key"],
                        "content_type": row["content_type"],
                        "ttl_seconds": row["ttl_seconds"],
                        "access_count": row["access_count"],
                        "created_at": row["created_at"],
                        "last_accessed_at": row["last_accessed_at"],
                    }
                )

            conn.close()

            return {
                "active_entries": len(entries),
                "entries_summary": entries[
                    :100
                ],  # Store summary to avoid large snapshots
            }

        except Exception as e:
            logger.error(f"Error collecting lightning state: {e}")
            return {"error": str(e)}

    def _collect_fusion_index(self) -> Dict[str, Any]:
        """Collect fusion memory index."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT entry_id, key, content_type, size_bytes, checksum, 
                       compressed, compression_type, created_at, accessed_count
                FROM fusion_memory
                ORDER BY last_accessed_at DESC
            """)

            entries = []
            total_size = 0
            for row in cursor.fetchall():
                entry_data = {
                    "entry_id": row["entry_id"],
                    "key": row["key"],
                    "content_type": row["content_type"],
                    "size_bytes": row["size_bytes"],
                    "checksum": row["checksum"],
                    "compressed": row["compressed"],
                    "compression_type": row["compression_type"],
                    "created_at": row["created_at"],
                    "accessed_count": row["accessed_count"],
                }
                entries.append(entry_data)
                total_size += row["size_bytes"] or 0

            conn.close()

            return {
                "total_entries": len(entries),
                "total_size_bytes": total_size,
                "entries_index": entries,
            }

        except Exception as e:
            logger.error(f"Error collecting fusion index: {e}")
            return {"error": str(e)}

    def _collect_librarian_config(self) -> Dict[str, Any]:
        """Collect librarian configuration and index state."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Get index statistics
            cursor = conn.execute("""
                SELECT index_type, COUNT(*) as count, AVG(weight) as avg_weight
                FROM librarian_index
                GROUP BY index_type
            """)

            index_stats = {}
            for row in cursor.fetchall():
                index_stats[row["index_type"]] = {
                    "count": row["count"],
                    "avg_weight": row["avg_weight"],
                }

            conn.close()

            return {
                "index_statistics": index_stats,
                "config_version": "1.0.0",
                "last_rebuild": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error collecting librarian config: {e}")
            return {"error": str(e)}

    def _collect_access_patterns(self) -> Dict[str, Any]:
        """Collect recent memory access patterns."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Get access pattern summary for last 24 hours
            cursor = conn.execute("""
                SELECT access_type, COUNT(*) as count, AVG(response_time_ms) as avg_response_time,
                       SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
                FROM memory_access_patterns
                WHERE access_timestamp >= datetime('now', '-24 hours')
                GROUP BY access_type
            """)

            patterns = {}
            for row in cursor.fetchall():
                patterns[row["access_type"]] = {
                    "count": row["count"],
                    "avg_response_time_ms": row["avg_response_time"],
                    "cache_hits": row["cache_hits"],
                    "cache_hit_rate": row["cache_hits"] / row["count"]
                    if row["count"] > 0
                    else 0,
                }

            conn.close()

            return {
                "patterns_24h": patterns,
                "collected_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error collecting access patterns: {e}")
            return {"error": str(e)}

    def _collect_memory_statistics(self) -> Dict[str, Any]:
        """Collect current memory system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Get latest statistics for each component
            cursor = conn.execute("""
                SELECT component, stat_type, value, unit
                FROM memory_stats
                WHERE recorded_at >= datetime('now', '-1 hour')
                ORDER BY recorded_at DESC
            """)

            stats = {}
            for row in cursor.fetchall():
                if row["component"] not in stats:
                    stats[row["component"]] = {}
                stats[row["component"]][row["stat_type"]] = {
                    "value": row["value"],
                    "unit": row["unit"],
                }

            conn.close()

            return {
                "component_stats": stats,
                "collected_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error collecting memory statistics: {e}")
            return {"error": str(e)}

    def _store_snapshot(self, snapshot_data: Dict[str, Any]):
        """Store snapshot in database and file system."""
        snapshot_id = snapshot_data["snapshot_id"]

        try:
            # Store in database
            conn = sqlite3.connect(self.db_path)

            snapshot_json = json.dumps(snapshot_data)
            snapshot_size = len(snapshot_json.encode("utf-8"))

            conn.execute(
                """
                INSERT INTO memory_snapshots (
                    snapshot_id, lightning_state_json, fusion_index_json, 
                    librarian_config_json, access_patterns_json, statistics_json,
                    hash, description, size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    snapshot_id,
                    json.dumps(snapshot_data["lightning_state"]),
                    json.dumps(snapshot_data["fusion_index"]),
                    json.dumps(snapshot_data["librarian_config"]),
                    json.dumps(snapshot_data["access_patterns"]),
                    json.dumps(snapshot_data["statistics"]),
                    snapshot_data["hash"],
                    snapshot_data["description"],
                    snapshot_size,
                ),
            )

            conn.commit()
            conn.close()

            # Store full snapshot as file
            snapshot_file = self.storage_path / f"{snapshot_id}.json"
            with open(snapshot_file, "w") as f:
                json.dump(snapshot_data, f, indent=2)

            logger.info(f"Stored memory snapshot {snapshot_id} ({snapshot_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to store snapshot: {e}")
            raise

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT snapshot_id, hash, created_at, description, size_bytes
                FROM memory_snapshots
                ORDER BY created_at DESC
            """)

            snapshots = []
            for row in cursor.fetchall():
                snapshots.append(
                    {
                        "snapshot_id": row["snapshot_id"],
                        "hash": row["hash"],
                        "created_at": row["created_at"],
                        "description": row["description"],
                        "size_bytes": row["size_bytes"],
                    }
                )

            conn.close()
            return snapshots

        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific snapshot."""
        try:
            # Try loading from file first (complete data)
            snapshot_file = self.storage_path / f"{snapshot_id}.json"
            if snapshot_file.exists():
                with open(snapshot_file, "r") as f:
                    return json.load(f)

            # Fallback to database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT * FROM memory_snapshots WHERE snapshot_id = ?
            """,
                (snapshot_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "snapshot_id": row["snapshot_id"],
                    "lightning_state": json.loads(row["lightning_state_json"]),
                    "fusion_index": json.loads(row["fusion_index_json"]),
                    "librarian_config": json.loads(row["librarian_config_json"]),
                    "access_patterns": json.loads(row["access_patterns_json"]),
                    "statistics": json.loads(row["statistics_json"]),
                    "hash": row["hash"],
                    "created_at": row["created_at"],
                    "description": row["description"],
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get snapshot {snapshot_id}: {e}")
            return None

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        try:
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "DELETE FROM memory_snapshots WHERE snapshot_id = ?", (snapshot_id,)
            )
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()

            # Remove file if exists
            snapshot_file = self.storage_path / f"{snapshot_id}.json"
            if snapshot_file.exists():
                snapshot_file.unlink()

            if deleted_rows > 0:
                logger.info(f"Deleted memory snapshot: {snapshot_id}")
                return True
            else:
                logger.warning(f"Snapshot not found: {snapshot_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
            return False
