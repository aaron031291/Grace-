"""
Lightning Memory - High-speed in-memory cache with TTL, LRU eviction, and SHA256 integrity.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    """Individual cache entry with metadata."""

    def __init__(
        self, key: str, value: Any, ttl_seconds: int = 3600, tags: List[str] = None
    ):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 1
        self.ttl_seconds = ttl_seconds
        self.tags = tags or []

        # Calculate integrity hash
        self.value_hash = self._calculate_hash(value)

    def _calculate_hash(self, value: Any) -> str:
        """Calculate SHA256 hash of value."""
        if isinstance(value, (str, bytes)):
            content = value if isinstance(value, bytes) else value.encode()
        else:
            content = json.dumps(value, sort_keys=True).encode()

        return hashlib.sha256(content).hexdigest()

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def verify_integrity(self) -> bool:
        """Verify entry integrity."""
        current_hash = self._calculate_hash(self.value)
        return current_hash == self.value_hash

    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class LightningMemory:
    """High-speed in-memory cache with TTL, LRU eviction, and integrity checking."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl

        # Thread-safe ordered dictionary for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_writes": 0,
            "total_reads": 0,
            "start_time": time.time(),
        }

        logger.info(
            f"Lightning Memory initialized: max_size={max_size}, ttl={default_ttl}s"
        )

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: List[str] = None,
    ) -> bool:
        """Store value in cache."""
        try:
            with self._lock:
                ttl = ttl_seconds or self.default_ttl
                entry = CacheEntry(key, value, ttl, tags)

                # Remove existing entry if present
                if key in self._cache:
                    del self._cache[key]

                # Add new entry
                self._cache[key] = entry

                # Check size and evict if necessary
                self._check_and_evict()

                self._stats["total_writes"] += 1
                return True

        except Exception as e:
            logger.error(f"Failed to store key '{key}': {e}")
            return False

    def get(self, key: str, verify_integrity: bool = True) -> Optional[Any]:
        """Retrieve value from cache."""
        try:
            with self._lock:
                self._stats["total_reads"] += 1

                if key not in self._cache:
                    self._stats["misses"] += 1
                    return None

                entry = self._cache[key]

                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None

                # Verify integrity if requested
                if verify_integrity and not entry.verify_integrity():
                    del self._cache[key]
                    self._stats["misses"] += 1
                    logger.warning(f"Integrity check failed for key '{key}'")
                    return None

                # Update access info and move to end (most recently used)
                entry.touch()
                self._cache.move_to_end(key)

                self._stats["hits"] += 1
                return entry.value

        except Exception as e:
            logger.error(f"Failed to retrieve key '{key}': {e}")
            self._stats["misses"] += 1
            return None

    def _check_and_evict(self):
        """Check memory usage and evict entries if necessary."""
        # Size-based eviction (LRU)
        while len(self._cache) > self.max_size:
            # Remove least recently used (first item in OrderedDict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            uptime = time.time() - self._stats["start_time"]
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total_requests) if total_requests > 0 else 0
            )

            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 3),
                "uptime_seconds": round(uptime, 1),
                "stats": self._stats.copy(),
            }
