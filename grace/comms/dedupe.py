"""
Grace Message Deduplication - Prevent duplicate message processing using content hashing and time windows.
"""

import logging
import hashlib
import json
import sqlite3
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .envelope import GraceMessageEnvelope


logger = logging.getLogger(__name__)


class DeduplicationEntry:
    """Represents a deduplicated message entry."""

    def __init__(self, message_hash: str, message_id: str, created_at: datetime):
        self.message_hash = message_hash
        self.message_id = message_id
        self.created_at = created_at
        self.processed_at = datetime.utcnow()
        self.access_count = 1


class MessageDeduplicator:
    """Grace message deduplicator with configurable time windows and storage."""

    def __init__(
        self,
        db_path: str = "/tmp/grace_dedupe.db",
        window_minutes: int = 60,
        max_memory_entries: int = 10000,
    ):
        self.db_path = db_path
        self.window_minutes = window_minutes
        self.max_memory_entries = max_memory_entries

        # In-memory cache for recent messages
        self.memory_cache: Dict[str, DeduplicationEntry] = {}
        self.hash_to_id: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "total_checked": 0,
            "duplicates_found": 0,
            "unique_messages": 0,
            "cache_hits": 0,
            "db_hits": 0,
            "cleanup_runs": 0,
        }

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for deduplication persistence."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_dedupe (
                    message_hash TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dedupe_created 
                ON message_dedupe(created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dedupe_processed 
                ON message_dedupe(processed_at)
            """)

            conn.commit()
            conn.close()

            logger.info("Deduplicator database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize deduplicator database: {e}")

    def _generate_content_hash(self, envelope: GraceMessageEnvelope) -> str:
        """Generate content hash for message deduplication."""
        try:
            # Include key fields that determine message uniqueness
            hash_data = {
                "kind": envelope.kind,
                "topic": envelope.routing.topic,
                "payload": envelope.payload,
                "source_component": envelope.source_component,
                "created_at_minute": envelope.created_at.replace(
                    second=0, microsecond=0
                ).isoformat(),
            }

            # Sort keys to ensure consistent hashing
            content_str = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        except Exception as e:
            logger.error(f"Failed to generate content hash: {e}")
            # Fallback to simple hash
            return hashlib.sha256(str(envelope.payload).encode("utf-8")).hexdigest()

    def _generate_strict_hash(self, envelope: GraceMessageEnvelope) -> str:
        """Generate strict hash including timestamp for exact duplicate detection."""
        try:
            # Include exact timestamp for strict deduplication
            hash_data = {
                "kind": envelope.kind,
                "topic": envelope.routing.topic,
                "payload": envelope.payload,
                "source_component": envelope.source_component,
                "created_at": envelope.created_at.isoformat(),
            }

            content_str = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        except Exception as e:
            logger.error(f"Failed to generate strict hash: {e}")
            return self._generate_content_hash(envelope)

    async def is_duplicate(
        self, envelope: GraceMessageEnvelope, strict_mode: bool = False
    ) -> bool:
        """Check if message is a duplicate within the configured time window."""
        try:
            self.stats["total_checked"] += 1

            # Generate appropriate hash
            message_hash = (
                self._generate_strict_hash(envelope)
                if strict_mode
                else self._generate_content_hash(envelope)
            )

            # Check memory cache first
            if message_hash in self.memory_cache:
                entry = self.memory_cache[message_hash]
                if self._within_window(entry.created_at):
                    entry.access_count += 1
                    self.stats["cache_hits"] += 1
                    self.stats["duplicates_found"] += 1
                    logger.debug(f"Duplicate found in cache: {envelope.id}")
                    return True
                else:
                    # Expired from window, remove from cache
                    del self.memory_cache[message_hash]
                    if message_hash in self.hash_to_id:
                        del self.hash_to_id[message_hash]

            # Check database
            duplicate_found = await self._check_db_duplicate(message_hash)
            if duplicate_found:
                self.stats["db_hits"] += 1
                self.stats["duplicates_found"] += 1

                # Add to memory cache for faster future lookups
                self._add_to_cache(message_hash, envelope.id, envelope.created_at)
                logger.debug(f"Duplicate found in database: {envelope.id}")
                return True

            # Not a duplicate, record it
            await self._record_message(message_hash, envelope)
            self.stats["unique_messages"] += 1
            return False

        except Exception as e:
            logger.error(f"Error checking for duplicate: {e}")
            # On error, assume not duplicate to avoid blocking legitimate messages
            return False

    def _within_window(self, message_time: datetime) -> bool:
        """Check if message is within the deduplication time window."""
        window_start = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        return message_time >= window_start

    async def _check_db_duplicate(self, message_hash: str) -> bool:
        """Check database for duplicate message hash within time window."""
        try:
            window_start = datetime.utcnow() - timedelta(minutes=self.window_minutes)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """
                SELECT message_id, created_at, access_count 
                FROM message_dedupe 
                WHERE message_hash = ? AND created_at >= ?
            """,
                (message_hash, window_start.isoformat()),
            )

            result = cursor.fetchone()

            if result:
                # Update access count
                conn.execute(
                    """
                    UPDATE message_dedupe 
                    SET access_count = access_count + 1, processed_at = ?
                    WHERE message_hash = ?
                """,
                    (datetime.utcnow().isoformat(), message_hash),
                )
                conn.commit()

            conn.close()
            return result is not None

        except Exception as e:
            logger.error(f"Error checking database for duplicates: {e}")
            return False

    async def _record_message(self, message_hash: str, envelope: GraceMessageEnvelope):
        """Record new message hash in both cache and database."""
        try:
            # Add to memory cache
            self._add_to_cache(message_hash, envelope.id, envelope.created_at)

            # Store in database
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO message_dedupe 
                (message_hash, message_id, created_at, processed_at, access_count)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    message_hash,
                    envelope.id,
                    envelope.created_at.isoformat(),
                    datetime.utcnow().isoformat(),
                    1,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording message: {e}")

    def _add_to_cache(self, message_hash: str, message_id: str, created_at: datetime):
        """Add message to memory cache with size limit management."""
        try:
            # Check cache size limit
            if len(self.memory_cache) >= self.max_memory_entries:
                # Remove oldest entries (simple LRU-like behavior)
                oldest_hash = min(
                    self.memory_cache.keys(),
                    key=lambda h: self.memory_cache[h].processed_at,
                )
                del self.memory_cache[oldest_hash]
                if oldest_hash in self.hash_to_id:
                    del self.hash_to_id[oldest_hash]

            # Add new entry
            entry = DeduplicationEntry(message_hash, message_id, created_at)
            self.memory_cache[message_hash] = entry
            self.hash_to_id[message_hash] = message_id

        except Exception as e:
            logger.error(f"Error adding to cache: {e}")

    async def cleanup_expired(self) -> int:
        """Clean up expired entries from database and memory."""
        try:
            self.stats["cleanup_runs"] += 1
            window_start = datetime.utcnow() - timedelta(minutes=self.window_minutes)

            # Clean up database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """
                DELETE FROM message_dedupe WHERE created_at < ?
            """,
                (window_start.isoformat(),),
            )

            db_deleted = cursor.rowcount
            conn.commit()
            conn.close()

            # Clean up memory cache
            expired_hashes = []
            for message_hash, entry in self.memory_cache.items():
                if not self._within_window(entry.created_at):
                    expired_hashes.append(message_hash)

            for message_hash in expired_hashes:
                del self.memory_cache[message_hash]
                if message_hash in self.hash_to_id:
                    del self.hash_to_id[message_hash]

            memory_deleted = len(expired_hashes)
            total_deleted = db_deleted + memory_deleted

            if total_deleted > 0:
                logger.info(
                    f"Cleaned up {total_deleted} expired dedupe entries "
                    f"(DB: {db_deleted}, Memory: {memory_deleted})"
                )

            return total_deleted

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def get_duplicate_info(
        self, envelope: GraceMessageEnvelope, strict_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get information about when this message was first seen (if duplicate)."""
        try:
            message_hash = (
                self._generate_strict_hash(envelope)
                if strict_mode
                else self._generate_content_hash(envelope)
            )

            # Check memory cache
            if message_hash in self.memory_cache:
                entry = self.memory_cache[message_hash]
                if self._within_window(entry.created_at):
                    return {
                        "is_duplicate": True,
                        "original_message_id": entry.message_id,
                        "first_seen": entry.created_at.isoformat(),
                        "access_count": entry.access_count,
                        "source": "cache",
                    }

            # Check database
            window_start = datetime.utcnow() - timedelta(minutes=self.window_minutes)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """
                SELECT message_id, created_at, access_count 
                FROM message_dedupe 
                WHERE message_hash = ? AND created_at >= ?
            """,
                (message_hash, window_start.isoformat()),
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    "is_duplicate": True,
                    "original_message_id": result[0],
                    "first_seen": result[1],
                    "access_count": result[2],
                    "source": "database",
                }

            return {"is_duplicate": False, "message_hash": message_hash}

        except Exception as e:
            logger.error(f"Error getting duplicate info: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get database stats
            cursor = conn.execute("SELECT COUNT(*) FROM message_dedupe")
            db_count = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM message_dedupe 
                WHERE created_at >= ?
            """,
                (
                    (
                        datetime.utcnow() - timedelta(minutes=self.window_minutes)
                    ).isoformat(),
                ),
            )
            active_window_count = cursor.fetchone()[0]

            conn.close()

            return {
                **self.stats,
                "memory_cache_size": len(self.memory_cache),
                "database_entries": db_count,
                "active_window_entries": active_window_count,
                "window_minutes": self.window_minutes,
                "max_memory_entries": self.max_memory_entries,
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return self.stats

    async def reset(self):
        """Reset deduplicator state (for testing/maintenance)."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.hash_to_id.clear()

            # Clear database
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM message_dedupe")
            conn.commit()
            conn.close()

            # Reset stats
            self.stats = {
                "total_checked": 0,
                "duplicates_found": 0,
                "unique_messages": 0,
                "cache_hits": 0,
                "db_hits": 0,
                "cleanup_runs": 0,
            }

            logger.info("Deduplicator state reset")

        except Exception as e:
            logger.error(f"Error resetting deduplicator: {e}")
