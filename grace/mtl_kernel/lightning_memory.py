"""Lightning Memory - High-speed in-memory cache (<1ms operations)."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LightningMemory:
    """
    High-speed in-memory cache with Redis-like semantics.
    
    Performance target: <1ms for all operations
    Features:
    - TTL management (default 1 hour)
    - Hot data caching
    - Prefix-based key organization
    - JSON serialization
    - Pipeline support for batch ops
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Thread-safe ordered dictionary for LRU
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_writes": 0,
            "total_reads": 0,
            "start_time": time.time()
        }
        
        logger.info(f"Lightning Memory initialized: max_size={max_size}, ttl={default_ttl}s")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (<1ms target)."""
        start = time.time()
        
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None
                
                # Update access info
                entry["accessed_at"] = time.time()
                entry["access_count"] += 1
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                
                self._stats["hits"] += 1
                self._stats["total_reads"] += 1
                
                elapsed = (time.time() - start) * 1000
                if elapsed > 1:
                    logger.warning(f"Lightning get exceeded 1ms: {elapsed:.2f}ms")
                
                return entry["value"]
            
            self._stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache (<1ms target)."""
        start = time.time()
        
        try:
            async with self._lock:
                ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
                
                # Serialize complex objects
                if isinstance(value, (dict, list)):
                    serialized = json.dumps(value, default=str)
                else:
                    serialized = value
                
                entry = {
                    "value": value,
                    "created_at": time.time(),
                    "accessed_at": time.time(),
                    "access_count": 1,
                    "ttl_seconds": ttl,
                    "value_hash": hashlib.sha256(
                        str(serialized).encode()
                    ).hexdigest()
                }
                
                # Remove existing entry if present
                if key in self._cache:
                    del self._cache[key]
                
                # Add new entry
                self._cache[key] = entry
                
                # Check size and evict if necessary
                await self._check_and_evict()
                
                self._stats["total_writes"] += 1
                
                elapsed = (time.time() - start) * 1000
                if elapsed > 1:
                    logger.warning(f"Lightning set exceeded 1ms: {elapsed:.2f}ms")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not self._is_expired(entry):
                    return True
                del self._cache[key]
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys (pipeline operation)."""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_many(self, items: Dict[str, Any], ttl_seconds: Optional[int] = None) -> int:
        """Set multiple key-value pairs (pipeline operation)."""
        count = 0
        for key, value in items.items():
            if await self.set(key, value, ttl_seconds):
                count += 1
        return count
    
    async def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all keys with given prefix."""
        result = {}
        async with self._lock:
            for key in self._cache.keys():
                if key.startswith(prefix):
                    entry = self._cache[key]
                    if not self._is_expired(entry):
                        result[key] = entry["value"]
        return result
    
    async def clear(self) -> int:
        """Clear all entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            uptime = time.time() - self._stats["start_time"]
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests 
                if total_requests > 0 else 0
            )
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 3),
                "evictions": self._stats["evictions"],
                "total_writes": self._stats["total_writes"],
                "total_reads": self._stats["total_reads"],
                "uptime_seconds": round(uptime, 1)
            }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry has expired."""
        return time.time() - entry["created_at"] > entry["ttl_seconds"]
    
    async def _check_and_evict(self):
        """Check size and evict oldest entries if necessary (LRU)."""
        while len(self._cache) > self.max_size:
            # Remove oldest (first) item
            key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"Evicted key: {key}")
