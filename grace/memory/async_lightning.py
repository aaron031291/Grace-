"""
Async Lightning Memory - Redis-backed high-speed cache
"""

from typing import Any, Optional, Dict
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class AsyncLightningMemory:
    """
    Async Redis-backed cache for ultra-fast retrieval
    
    Specification-compliant with:
    - Async operations
    - TTL management
    - Pattern storage
    - Connection pooling
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        max_size: int = 10000
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.redis = None
        self._connected = False
        
    async def connect(self):
        """Connect to Redis"""
        try:
            import aioredis
            self.redis = await aioredis.create_redis_pool(self.redis_url)
            self._connected = True
            logger.info(f"Connected to Redis: {self.redis_url}")
        except ImportError:
            logger.warning("aioredis not installed, using in-memory fallback")
            self.redis = {}
            self._connected = True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}, using in-memory fallback")
            self.redis = {}
            self._connected = True
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis and hasattr(self.redis, 'close'):
            self.redis.close()
            await self.redis.wait_closed()
        self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        if not self._connected:
            await self.connect()
        
        if isinstance(self.redis, dict):
            # In-memory fallback
            entry = self.redis.get(key)
            if entry and entry['expires_at'] > datetime.now(timezone.utc):
                return entry['value']
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get failed for {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Store value in cache with TTL"""
        if not self._connected:
            await self.connect()
        
        ttl = ttl or self.default_ttl
        
        if isinstance(self.redis, dict):
            # In-memory fallback
            self.redis[key] = {
                'value': value,
                'expires_at': datetime.now(timezone.utc) + timedelta(seconds=ttl)
            }
            
            # Enforce max size (LRU eviction)
            if len(self.redis) > self.max_size:
                oldest_key = min(self.redis.keys(), key=lambda k: self.redis[k]['expires_at'])
                del self.redis[oldest_key]
            
            return True
        
        try:
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set failed for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self._connected:
            await self.connect()
        
        if isinstance(self.redis, dict):
            if key in self.redis:
                del self.redis[key]
            return True
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete failed for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        value = await self.get(key)
        return value is not None
    
    async def clear(self):
        """Clear all cache entries"""
        if isinstance(self.redis, dict):
            self.redis.clear()
        elif self.redis:
            await self.redis.flushdb()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if isinstance(self.redis, dict):
            return {
                "type": "in_memory",
                "size": len(self.redis),
                "max_size": self.max_size
            }
        
        if self.redis:
            info = await self.redis.info()
            return {
                "type": "redis",
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_keys": await self.redis.dbsize()
            }
        
        return {"type": "not_connected"}
