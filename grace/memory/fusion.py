"""
Fusion Memory - Long-term storage with append-only writes and efficient querying.

Features:
- SQLite/Parquet storage backends
- Append-only write semantics
- Efficient indexing and querying
- Data compression and archival
- Integrity verification
"""

import gzip
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from ..utils.time import now_utc, iso_now_utc
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import uuid

logger = logging.getLogger(__name__)


class FusionEntry:
    """Entry in Fusion long-term storage."""

    def __init__(
        self,
        key: str,
        value: Any,
        content_type: str = "application/json",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.entry_id = str(uuid.uuid4())
        self.key = key
        self.value = value
        self.content_type = content_type
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_at = now_utc()

        # Calculate size and hash
        self.value_json = json.dumps(value, default=str)
        self.size_bytes = len(self.value_json.encode())
        self.content_hash = hashlib.sha256(self.value_json.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "value": self.value,
            "content_type": self.content_type,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionEntry":
        """Create entry from dictionary."""
        entry = cls(
            key=data["key"],
            value=data["value"],
            content_type=data.get("content_type", "application/json"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

        # Override calculated fields
        entry.entry_id = data["entry_id"]
        entry.created_at = datetime.fromisoformat(data["created_at"])
        entry.size_bytes = data.get("size_bytes", entry.size_bytes)
        entry.content_hash = data.get("content_hash", entry.content_hash)

        return entry


class FusionMemory:
    """
    Long-term storage with append-only writes.

    Provides durable, queryable storage with compression and archival capabilities.
    """

    def __init__(
        self,
        storage_path: str = "/tmp/fusion_storage",
        db_name: str = "fusion.db",
        enable_compression: bool = True,
        archive_threshold_days: int = 90,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_path / db_name
        self.enable_compression = enable_compression
        self.archive_threshold_days = archive_threshold_days

        # Thread safety
        self._lock = threading.RLock()

        # Initialize database
        self._init_database()

        # Statistics
        self._stats = {
            "total_writes": 0,
            "total_reads": 0,
            "total_size_bytes": 0,
            "compressed_size_bytes": 0,
            "archived_entries": 0,
            "start_time": now_utc(),
        }

        logger.info(f"Fusion Memory initialized at {storage_path}")

    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Main entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    entry_id TEXT PRIMARY KEY,
                    key TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    archived BOOLEAN DEFAULT FALSE,
                    archive_path TEXT NULL,
                    tags_json TEXT NULL,
                    metadata_json TEXT NULL,
                    value_json TEXT NOT NULL
                )
            """)

            # Indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_key ON entries(key)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_created_at ON entries(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_content_hash ON entries(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_archived ON entries(archived)"
            )

            # Tags table for tag-based queries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entry_tags (
                    entry_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (entry_id, tag),
                    FOREIGN KEY (entry_id) REFERENCES entries(entry_id)
                )
            """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entry_tags_tag ON entry_tags(tag)"
            )

            # Statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    stat_date DATE PRIMARY KEY,
                    total_entries INTEGER,
                    total_size_bytes INTEGER,
                    writes_count INTEGER,
                    reads_count INTEGER
                )
            """)

            conn.commit()

    def write(
        self,
        key: str,
        value: Any,
        content_type: str = "application/json",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Write entry to long-term storage (append-only).

        Returns:
            Entry ID of the written entry
        """
        try:
            with self._lock:
                # Create entry
                entry = FusionEntry(key, value, content_type, tags, metadata)

                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    # Insert main entry
                    conn.execute(
                        """
                        INSERT INTO entries (
                            entry_id, key, content_type, size_bytes, content_hash,
                            created_at, archived, tags_json, metadata_json, value_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            entry.entry_id,
                            entry.key,
                            entry.content_type,
                            entry.size_bytes,
                            entry.content_hash,
                            entry.created_at.isoformat(),
                            False,
                            json.dumps(entry.tags),
                            json.dumps(entry.metadata),
                            entry.value_json,
                        ),
                    )

                    # Insert tags
                    for tag in entry.tags:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO entry_tags (entry_id, tag)
                            VALUES (?, ?)
                        """,
                            (entry.entry_id, tag),
                        )

                    conn.commit()

                # Update statistics
                self._stats["total_writes"] += 1
                self._stats["total_size_bytes"] += entry.size_bytes

                logger.debug(f"Wrote entry {entry.entry_id} for key '{key}'")
                return entry.entry_id

        except Exception as e:
            logger.error(f"Failed to write entry for key '{key}': {e}")
            raise

    def read(self, entry_id: str) -> Optional[FusionEntry]:
        """Read entry by ID."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    cursor = conn.execute(
                        """
                        SELECT * FROM entries WHERE entry_id = ?
                    """,
                        (entry_id,),
                    )

                    row = cursor.fetchone()
                    if not row:
                        return None

                    # Parse stored data
                    entry_data = {
                        "entry_id": row["entry_id"],
                        "key": row["key"],
                        "value": json.loads(row["value_json"]),
                        "content_type": row["content_type"],
                        "tags": json.loads(row["tags_json"])
                        if row["tags_json"]
                        else [],
                        "metadata": json.loads(row["metadata_json"])
                        if row["metadata_json"]
                        else {},
                        "created_at": row["created_at"],
                        "size_bytes": row["size_bytes"],
                        "content_hash": row["content_hash"],
                    }

                    entry = FusionEntry.from_dict(entry_data)

                    self._stats["total_reads"] += 1
                    return entry

        except Exception as e:
            logger.error(f"Failed to read entry {entry_id}: {e}")
            return None

    def search(
        self,
        key_pattern: str = None,
        tags: List[str] = None,
        content_type: str = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[FusionEntry]:
        """Search entries with various filters."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    # Build query
                    query = "SELECT DISTINCT e.* FROM entries e"
                    params = []
                    conditions = []

                    # Join with tags table if needed
                    if tags:
                        query += " JOIN entry_tags et ON e.entry_id = et.entry_id"
                        placeholders = ",".join("?" * len(tags))
                        conditions.append(f"et.tag IN ({placeholders})")
                        params.extend(tags)

                    # Add filters
                    if key_pattern:
                        conditions.append("e.key LIKE ?")
                        params.append(f"%{key_pattern}%")

                    if content_type:
                        conditions.append("e.content_type = ?")
                        params.append(content_type)

                    if start_time:
                        conditions.append("e.created_at >= ?")
                        params.append(start_time.isoformat())

                    if end_time:
                        conditions.append("e.created_at <= ?")
                        params.append(end_time.isoformat())

                    # Add non-archived filter by default
                    conditions.append("e.archived = FALSE")

                    # Combine conditions
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)

                    query += " ORDER BY e.created_at DESC LIMIT ?"
                    params.append(limit)

                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()

                    # Convert to entries
                    entries = []
                    for row in rows:
                        entry_data = {
                            "entry_id": row["entry_id"],
                            "key": row["key"],
                            "value": json.loads(row["value_json"]),
                            "content_type": row["content_type"],
                            "tags": json.loads(row["tags_json"])
                            if row["tags_json"]
                            else [],
                            "metadata": json.loads(row["metadata_json"])
                            if row["metadata_json"]
                            else {},
                            "created_at": row["created_at"],
                            "size_bytes": row["size_bytes"],
                            "content_hash": row["content_hash"],
                        }
                        entries.append(FusionEntry.from_dict(entry_data))

                    self._stats["total_reads"] += len(entries)
                    return entries

        except Exception as e:
            logger.error(f"Failed to search entries: {e}")
            return []

    def get_latest(self, key: str) -> Optional[FusionEntry]:
        """Get the latest entry for a key."""
        entries = self.search(key_pattern=key, limit=1)
        return entries[0] if entries else None

    def archive_old_entries(self, older_than_days: int = None) -> int:
        """Archive old entries to compressed storage."""
        if older_than_days is None:
            older_than_days = self.archive_threshold_days

        try:
            with self._lock:
                cutoff_date = now_utc() - timedelta(days=older_than_days)

                with sqlite3.connect(self.db_path) as conn:
                    # Get entries to archive
                    cursor = conn.execute(
                        """
                        SELECT entry_id, key, value_json FROM entries
                        WHERE created_at < ? AND archived = FALSE
                    """,
                        (cutoff_date.isoformat(),),
                    )

                    entries_to_archive = cursor.fetchall()

                    if not entries_to_archive:
                        return 0

                    # Create archive file
                    archive_filename = (
                        f"archive_{now_utc().strftime('%Y%m%d_%H%M%S')}.json.gz"
                    )
                    archive_path = self.storage_path / "archives" / archive_filename
                    archive_path.parent.mkdir(exist_ok=True)

                    # Compress and archive entries
                    archive_data = []
                    for row in entries_to_archive:
                        archive_data.append(
                            {
                                "entry_id": row[0],
                                "key": row[1],
                                "value_json": row[2],
                                "archived_at": iso_now_utc(),
                            }
                        )

                    with gzip.open(archive_path, "wt") as f:
                        json.dump(archive_data, f)

                    # Update database entries
                    entry_ids = [row[0] for row in entries_to_archive]
                    placeholders = ",".join("?" * len(entry_ids))

                    conn.execute(
                        f"""
                        UPDATE entries 
                        SET archived = TRUE, archive_path = ?, value_json = ''
                        WHERE entry_id IN ({placeholders})
                    """,
                        [str(archive_path)] + entry_ids,
                    )

                    conn.commit()

                    archived_count = len(entries_to_archive)
                    self._stats["archived_entries"] += archived_count

                    logger.info(
                        f"Archived {archived_count} entries to {archive_filename}"
                    )
                    return archived_count

        except Exception as e:
            logger.error(f"Failed to archive entries: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size,
                        COUNT(CASE WHEN archived = TRUE THEN 1 END) as archived_count,
                        COUNT(CASE WHEN archived = FALSE THEN 1 END) as active_count
                    FROM entries
                """)

                db_stats = cursor.fetchone()

                # Add runtime stats
                uptime = (datetime.utcnow() - self._stats["start_time"]).total_seconds()

                return {
                    "total_entries": db_stats[0] or 0,
                    "total_size_bytes": db_stats[1] or 0,
                    "active_entries": db_stats[3] or 0,
                    "archived_entries": db_stats[2] or 0,
                    "uptime_seconds": round(uptime, 1),
                    "compression_enabled": self.enable_compression,
                    "archive_threshold_days": self.archive_threshold_days,
                    "runtime_stats": self._stats.copy(),
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def verify_integrity(self, entry_id: str = None) -> Dict[str, Any]:
        """Verify integrity of entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if entry_id:
                    cursor = conn.execute(
                        """
                        SELECT entry_id, content_hash, value_json FROM entries
                        WHERE entry_id = ? AND archived = FALSE
                    """,
                        (entry_id,),
                    )
                else:
                    cursor = conn.execute("""
                        SELECT entry_id, content_hash, value_json FROM entries
                        WHERE archived = FALSE LIMIT 1000
                    """)

                entries = cursor.fetchall()

                verified_count = 0
                corrupted_entries = []

                for row in entries:
                    expected_hash = row["content_hash"]
                    actual_hash = hashlib.sha256(row["value_json"].encode()).hexdigest()

                    if expected_hash == actual_hash:
                        verified_count += 1
                    else:
                        corrupted_entries.append(
                            {
                                "entry_id": row["entry_id"],
                                "expected_hash": expected_hash,
                                "actual_hash": actual_hash,
                            }
                        )

                return {
                    "verified_entries": verified_count,
                    "corrupted_entries": len(corrupted_entries),
                    "corruption_details": corrupted_entries,
                    "verified_at": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to verify integrity: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup and close resources."""
        logger.info("Fusion Memory cleanup completed")
