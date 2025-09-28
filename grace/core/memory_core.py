"""
MemoryCore - Enhanced persistent storage and retrieval system for governance.
Includes Redis caching, PostgreSQL integration, and structured memory operations.
Part of Phase 3: Memory Core Production implementation.
"""
import json
import hashlib
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import sqlite3
import logging
from pathlib import Path

from .contracts import UnifiedDecision, GovernanceSnapshot, Experience
from ..config.environment import get_grace_config

# Optional dependencies for production features
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    asyncpg = None


logger = logging.getLogger(__name__)


class MemoryCore:
    """
    Enhanced central memory system with production-ready features:
    - Redis caching for fast access
    - PostgreSQL support for scalable storage
    - Structured memory operations
    - Event-driven recall/store/update workflows
    """
    
    def __init__(self, db_path: str = "grace_governance.db", 
                 event_publisher: Optional[Any] = None):
        self.config = get_grace_config()
        self.db_path = db_path
        self.event_publisher = event_publisher
        
        # Database connections
        self.postgres_pool: Optional[Any] = None
        self.redis_client: Optional[Any] = None
        
        # Configuration flags
        self.use_postgres = self.config["database_config"]["use_postgres"] and POSTGRES_AVAILABLE
        self.use_redis_cache = self.config["database_config"]["use_redis_cache"] and REDIS_AVAILABLE
        
        # Memory statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        
        # Initialize database
        if not self.use_postgres:
            self._init_sqlite_database()
        
        logger.info(f"MemoryCore initialized (postgres: {self.use_postgres}, redis: {self.use_redis_cache})")
    
    async def start(self):
        """Start the MemoryCore with async database connections."""
        logger.info("Starting MemoryCore...")
        
        # Initialize PostgreSQL pool
        if self.use_postgres:
            try:
                self.postgres_pool = await asyncpg.create_pool(
                    self.config["database_config"]["postgres_url"],
                    min_size=1,
                    max_size=10,
                    command_timeout=60
                )
                await self._init_postgres_schema()
                logger.info("PostgreSQL connection pool established")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.use_postgres = False
                self._init_sqlite_database()
        
        # Initialize Redis client
        if self.use_redis_cache:
            try:
                redis_url = self.config["database_config"]["redis_url"]
                self.redis_client = redis.from_url(redis_url)
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.use_redis_cache = False
        
        # Log startup event
        if self.event_publisher:
            await self.event_publisher("memory_core_started", {
                "postgres_enabled": self.use_postgres,
                "redis_enabled": self.use_redis_cache,
                "instance_id": self.config["environment_config"]["instance_id"]
            })
    
    async def stop(self):
        """Stop the MemoryCore and close connections."""
        logger.info("Stopping MemoryCore...")
        
        # Close PostgreSQL pool
        if self.postgres_pool:
            await self.postgres_pool.close()
            self.postgres_pool = None
        
        # Close Redis client
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None
        
        # Log shutdown event
        if self.event_publisher:
            await self.event_publisher("memory_core_stopped", {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_queries": self.total_queries
            })
    
    async def store_structured_memory(self, memory_type: str, content: Dict[str, Any],
                                    metadata: Optional[Dict[str, Any]] = None,
                                    importance_score: float = 0.5) -> str:
        """
        Store structured memory with automatic deduplication and caching.
        Returns the memory ID.
        """
        memory_id = f"mem_{int(datetime.now().timestamp() * 1000000)}"
        content_hash = self._hash_dict(content)
        
        # Check for existing content
        existing_memory = await self._get_memory_by_hash(content_hash)
        if existing_memory:
            # Update access count and timestamp
            await self._update_memory_access(existing_memory["memory_id"])
            return existing_memory["memory_id"]
        
        # Create new memory entry
        memory_entry = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "content_hash": content_hash,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "accessed_at": datetime.now(),
            "access_count": 0,
            "importance_score": importance_score
        }
        
        # Store in database
        if self.use_postgres:
            await self._store_memory_postgres(memory_entry)
        else:
            await self._store_memory_sqlite(memory_entry)
        
        # Cache in Redis
        if self.use_redis_cache:
            await self._cache_memory(memory_id, memory_entry)
        
        # Publish event
        if self.event_publisher:
            await self.event_publisher("memory_stored", {
                "memory_id": memory_id,
                "memory_type": memory_type,
                "content_hash": content_hash,
                "importance_score": importance_score
            })
        
        self.total_queries += 1
        logger.debug(f"Stored structured memory: {memory_id} ({memory_type})")
        
        return memory_id
    
    async def update_structured_memory(self, memory_id: str, content: Dict[str, Any],
                                     metadata: Optional[Dict[str, Any]] = None,
                                     importance_score: Optional[float] = None) -> bool:
        """
        Update existing structured memory entry.
        Returns True if successful, False if memory not found.
        """
        # Get existing memory
        existing_memory = await self.recall_structured_memory(memory_id)
        if not existing_memory:
            return False
        
        # Update fields
        updated_content = content
        updated_metadata = metadata if metadata is not None else existing_memory.get("metadata", {})
        updated_importance = importance_score if importance_score is not None else existing_memory.get("importance_score", 0.5)
        
        # Calculate new hash
        content_hash = self._hash_dict(updated_content)
        
        # Update entry
        update_data = {
            "content": updated_content,
            "metadata": updated_metadata,
            "importance_score": updated_importance,
            "content_hash": content_hash,
            "accessed_at": datetime.now()
        }
        
        # Update in database
        if self.use_postgres:
            await self._update_memory_postgres(memory_id, update_data)
        else:
            await self._update_memory_sqlite(memory_id, update_data)
        
        # Update cache
        if self.use_redis_cache:
            # Invalidate old cache
            await self.redis_client.delete(f"memory:{memory_id}")
            # Store updated entry
            updated_entry = existing_memory.copy()
            updated_entry.update(update_data)
            await self._cache_memory(memory_id, updated_entry)
        
        # Publish event
        if self.event_publisher:
            await self.event_publisher("memory_updated", {
                "memory_id": memory_id,
                "content_hash": content_hash,
                "importance_score": updated_importance
            })
        
        self.total_queries += 1
        logger.debug(f"Updated structured memory: {memory_id}")
        
        return True
    
    async def recall_structured_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Recall structured memory by ID with caching.
        Returns the memory entry or None if not found.
        """
        self.total_queries += 1
        
        # Try Redis cache first
        if self.use_redis_cache:
            cached_memory = await self._get_cached_memory(memory_id)
            if cached_memory:
                self.cache_hits += 1
                await self._update_memory_access(memory_id)
                return cached_memory
            self.cache_misses += 1
        
        # Query database
        if self.use_postgres:
            memory = await self._get_memory_postgres(memory_id)
        else:
            memory = await self._get_memory_sqlite(memory_id)
        
        if memory:
            # Cache the result
            if self.use_redis_cache:
                await self._cache_memory(memory_id, memory)
            
            # Update access tracking
            await self._update_memory_access(memory_id)
            
            # Publish recall event
            if self.event_publisher:
                await self.event_publisher("memory_recalled", {
                    "memory_id": memory_id,
                    "memory_type": memory.get("memory_type", "unknown"),
                    "cache_hit": False
                })
        
        return memory
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        cache_hit_rate = self.cache_hits / max(1, self.total_queries)
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "postgres_enabled": self.use_postgres,
            "redis_enabled": self.use_redis_cache,
            "database_path": self.db_path if not self.use_postgres else "postgres"
        }
    
    # Legacy compatibility methods (simplified)
    async def store_decision(self, decision: UnifiedDecision, outcome: Optional[str] = None,
                           instance_id: str = "default", version: str = "1.0.0"):
        """Store a governance decision with caching support."""
        decision_content = decision.to_dict()
        decision_content.update({
            "outcome": outcome,
            "instance_id": instance_id,
            "version": version
        })
        
        memory_id = await self.store_structured_memory(
            memory_type="governance_decision",
            content=decision_content,
            metadata={
                "decision_id": decision.decision_id,
                "subject": decision.topic
            },
            importance_score=decision.confidence
        )
        
        logger.info(f"Stored decision {decision.decision_id}")
        return memory_id
    
    # Private helper methods
    def _init_sqlite_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Structured memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structured_memory (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5
                )
            """)
            
            # Legacy compatibility tables (simplified)
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
            
            conn.commit()
            logger.info("SQLite database schema initialized")
    
    async def _init_postgres_schema(self):
        """Initialize PostgreSQL schema (assumes schema created via init script)."""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Verify schemas exist
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name IN ('memory', 'governance', 'audit')"
                )
                if result < 3:
                    logger.warning("PostgreSQL schemas not fully initialized")
                else:
                    logger.info("PostgreSQL schemas verified")
        except Exception as e:
            logger.error(f"Failed to verify PostgreSQL schema: {e}")
    
    async def _get_memory_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get memory entry by content hash to check for duplicates."""
        if self.use_postgres:
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM memory.structured_memory WHERE content_hash = $1",
                    content_hash
                )
                if row:
                    result = dict(row)
                    result["content"] = json.loads(result["content"])
                    result["metadata"] = json.loads(result["metadata"])
                    return result
        else:
            # SQLite implementation
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM structured_memory WHERE content_hash = ?",
                    (content_hash,)
                )
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    result = dict(zip(columns, row))
                    result["content"] = json.loads(result["content"])
                    result["metadata"] = json.loads(result["metadata"])
                    return result
        return None
    
    async def _store_memory_postgres(self, memory_entry: Dict[str, Any]):
        """Store memory entry in PostgreSQL."""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory.structured_memory 
                (memory_id, memory_type, content_hash, content, metadata, 
                 created_at, accessed_at, access_count, importance_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
                memory_entry["memory_id"],
                memory_entry["memory_type"],
                memory_entry["content_hash"],
                json.dumps(memory_entry["content"]),
                json.dumps(memory_entry["metadata"]),
                memory_entry["created_at"],
                memory_entry["accessed_at"],
                memory_entry["access_count"],
                memory_entry["importance_score"]
            )
    
    async def _store_memory_sqlite(self, memory_entry: Dict[str, Any]):
        """Store memory entry in SQLite."""
        def _store():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO structured_memory 
                    (memory_id, memory_type, content_hash, content, metadata,
                     created_at, accessed_at, access_count, importance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_entry["memory_id"],
                    memory_entry["memory_type"],
                    memory_entry["content_hash"],
                    json.dumps(memory_entry["content"]),
                    json.dumps(memory_entry["metadata"]),
                    memory_entry["created_at"].isoformat(),
                    memory_entry["accessed_at"].isoformat(),
                    memory_entry["access_count"],
                    memory_entry["importance_score"]
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _store)
    
    async def _update_memory_postgres(self, memory_id: str, update_data: Dict[str, Any]):
        """Update memory entry in PostgreSQL."""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE memory.structured_memory 
                SET content = $2, metadata = $3, importance_score = $4, 
                    content_hash = $5, accessed_at = $6
                WHERE memory_id = $1
            """,
                memory_id,
                json.dumps(update_data["content"]),
                json.dumps(update_data["metadata"]),
                update_data["importance_score"],
                update_data["content_hash"],
                update_data["accessed_at"]
            )
    
    async def _update_memory_sqlite(self, memory_id: str, update_data: Dict[str, Any]):
        """Update memory entry in SQLite."""
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE structured_memory 
                    SET content = ?, metadata = ?, importance_score = ?, 
                        content_hash = ?, accessed_at = ?
                    WHERE memory_id = ?
                """, (
                    json.dumps(update_data["content"]),
                    json.dumps(update_data["metadata"]),
                    update_data["importance_score"],
                    update_data["content_hash"],
                    update_data["accessed_at"].isoformat(),
                    memory_id
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def _get_memory_postgres(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory entry from PostgreSQL."""
        async with self.postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory.structured_memory WHERE memory_id = $1",
                memory_id
            )
            if row:
                result = dict(row)
                result["content"] = json.loads(result["content"])
                result["metadata"] = json.loads(result["metadata"])
                return result
        return None
    
    async def _get_memory_sqlite(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory entry from SQLite."""
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM structured_memory WHERE memory_id = ?",
                    (memory_id,)
                )
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    result = dict(zip(columns, row))
                    result["content"] = json.loads(result["content"])
                    result["metadata"] = json.loads(result["metadata"])
                    return result
            return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def _update_memory_access(self, memory_id: str):
        """Update access count and timestamp for a memory entry."""
        if self.use_postgres:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE memory.structured_memory SET accessed_at = NOW(), access_count = access_count + 1 WHERE memory_id = $1",
                    memory_id
                )
        else:
            def _update():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE structured_memory SET accessed_at = ?, access_count = access_count + 1 WHERE memory_id = ?",
                        (datetime.now().isoformat(), memory_id)
                    )
                    conn.commit()
            
            await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def _cache_memory(self, memory_id: str, memory_entry: Dict[str, Any]):
        """Cache a memory entry in Redis."""
        if not self.use_redis_cache:
            return
        
        try:
            cache_key = f"memory:{memory_id}"
            # Convert datetime objects to strings for JSON serialization
            cacheable_entry = json.loads(json.dumps(memory_entry, default=str))
            await self.redis_client.setex(
                cache_key, 3600,  # Cache for 1 hour
                json.dumps(cacheable_entry)
            )
        except Exception as e:
            logger.error(f"Failed to cache memory {memory_id}: {e}")
    
    async def _get_cached_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory entry from Redis cache."""
        if not self.use_redis_cache:
            return None
        
        try:
            cache_key = f"memory:{memory_id}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Failed to get cached memory {memory_id}: {e}")
        
        return None
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create a hash of dictionary data for deduplication."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    async def store_experience(self, experience: Experience) -> str:
        """
        Store an Experience object in memory for learning purposes.
        This method is required for governance learning systems.
        """
        memory_id = await self.store_structured_memory(
            memory_type="experience",
            content=experience.to_dict(),
            metadata={
                "component_id": experience.component_id,
                "type": experience.type,
                "success_score": experience.success_score,
                "source": "governance_learning"
            },
            importance_score=min(1.0, experience.success_score + 0.1)  # Boost importance slightly
        )
        
        logger.debug(f"Stored experience: {memory_id}")
        return memory_id
    
    async def close(self):
        """Close database connections."""
        await self.stop()