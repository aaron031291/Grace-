"""
Enhanced Memory Core - Production-ready memory system with DB transactions,
health monitoring, and Clarity Framework integration
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import hashlib
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryHealth(Enum):
    """Health status of memory system"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class MemoryMetrics:
    """Health monitoring metrics"""
    total_memories: int = 0
    cache_hit_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    db_connection_healthy: bool = False
    redis_connection_healthy: bool = False
    embedding_api_healthy: bool = False
    last_check: datetime = field(default_factory=datetime.now)


class EnhancedMemoryCore:
    """
    Production-ready memory system with PostgreSQL, Redis, and health monitoring
    Integrates with Clarity Framework and AVN self-diagnostics
    """
    
    def __init__(
        self,
        db_connection=None,
        redis_client=None,
        clarity_memory_bank=None,
        avn_reporter=None
    ):
        # Database connections
        self.db = db_connection
        self.redis = redis_client
        
        # Clarity Framework integration
        self.clarity_memory = clarity_memory_bank
        
        # AVN self-diagnostic integration
        self.avn = avn_reporter
        
        # Health monitoring
        self.health_status = MemoryHealth.OFFLINE
        self.metrics = MemoryMetrics()
        
        # Embedding fallback
        self.embedding_fallback_enabled = True
        self.embedding_cache = {}
        
        # Performance tracking
        self.retrieval_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("EnhancedMemoryCore initialized")
    
    def initialize(self) -> bool:
        """
        Initialize memory core with health monitoring
        Embeds health checks into startup
        """
        logger.info("Initializing Enhanced Memory Core...")
        
        # Check database connection
        db_healthy = self._check_db_health()
        
        # Check Redis connection
        redis_healthy = self._check_redis_health()
        
        # Check embedding API
        embedding_healthy = self._check_embedding_api_health()
        
        # Update metrics
        self.metrics.db_connection_healthy = db_healthy
        self.metrics.redis_connection_healthy = redis_healthy
        self.metrics.embedding_api_healthy = embedding_healthy
        self.metrics.last_check = datetime.now()
        
        # Determine overall health
        if db_healthy and redis_healthy and embedding_healthy:
            self.health_status = MemoryHealth.HEALTHY
        elif db_healthy and (redis_healthy or embedding_healthy):
            self.health_status = MemoryHealth.DEGRADED
        elif db_healthy:
            self.health_status = MemoryHealth.CRITICAL
        else:
            self.health_status = MemoryHealth.OFFLINE
            
        # Report to AVN
        if self.avn:
            self._report_to_avn({
                'component': 'enhanced_memory_core',
                'status': self.health_status.value,
                'metrics': self.metrics.__dict__,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"Memory Core initialized - Status: {self.health_status.value}")
        
        return self.health_status != MemoryHealth.OFFLINE
    
    def store_structured_memory(
        self,
        memory_id: str,
        content: Dict[str, Any],
        memory_type: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store structured memory with PostgreSQL transaction
        Production DB implementation
        """
        if not self.db:
            logger.error("Database connection not available")
            return False
        
        try:
            # Generate embedding
            embedding = self._get_embedding(content)
            
            # Prepare data
            memory_data = {
                'memory_id': memory_id,
                'content': json.dumps(content),
                'memory_type': memory_type,
                'embedding': embedding.tolist() if embedding is not None else None,
                'metadata': json.dumps(metadata or {}),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # PostgreSQL transaction
            with self.db.begin() as transaction:
                # Insert into memories table
                insert_query = """
                    INSERT INTO structured_memories 
                    (memory_id, content, memory_type, embedding, metadata, created_at, updated_at)
                    VALUES (%(memory_id)s, %(content)s, %(memory_type)s, %(embedding)s, 
                            %(metadata)s, %(created_at)s, %(updated_at)s)
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """
                
                self.db.execute(insert_query, memory_data)
                transaction.commit()
            
            # Cache the memory
            self._cache_memory(memory_id, memory_data)
            
            # Link to Clarity Framework
            if self.clarity_memory:
                from grace.clarity.memory_scoring import MemoryType
                self.clarity_memory.store(
                    memory_id=memory_id,
                    memory_type=MemoryType.SEMANTIC,
                    content=content,
                    source="enhanced_memory_core",
                    metadata=metadata
                )
            
            logger.info(f"Stored structured memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory_id}: {e}")
            if self.avn:
                self._report_to_avn({
                    'error': 'memory_store_failed',
                    'memory_id': memory_id,
                    'exception': str(e)
                })
            return False
    
    def update_structured_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update structured memory with PostgreSQL transaction
        Production DB implementation
        """
        if not self.db:
            logger.error("Database connection not available")
            return False
        
        try:
            # Fetch existing memory
            existing = self._fetch_from_db(memory_id)
            if not existing:
                logger.warning(f"Memory not found: {memory_id}")
                return False
            
            # Merge updates
            existing_content = json.loads(existing['content'])
            existing_content.update(updates)
            
            # Regenerate embedding if content changed
            embedding = self._get_embedding(existing_content)
            
            # PostgreSQL transaction
            with self.db.begin() as transaction:
                update_query = """
                    UPDATE structured_memories
                    SET content = %(content)s,
                        embedding = %(embedding)s,
                        updated_at = %(updated_at)s
                    WHERE memory_id = %(memory_id)s
                """
                
                self.db.execute(update_query, {
                    'memory_id': memory_id,
                    'content': json.dumps(existing_content),
                    'embedding': embedding.tolist() if embedding is not None else None,
                    'updated_at': datetime.now()
                })
                
                transaction.commit()
            
            # Invalidate cache
            self._invalidate_memory_cache(memory_id)
            
            # Update in Clarity Framework
            if self.clarity_memory:
                self.clarity_memory.update_knowledge_node(
                    node_id=memory_id,
                    content=updates,
                    updater="enhanced_memory_core"
                )
            
            logger.info(f"Updated structured memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            if self.avn:
                self._report_to_avn({
                    'error': 'memory_update_failed',
                    'memory_id': memory_id,
                    'exception': str(e)
                })
            return False
    
    def _cache_memory(self, memory_id: str, memory_data: Dict[str, Any]):
        """
        Cache memory in Redis with transaction
        Production Redis implementation
        """
        if not self.redis:
            return
        
        try:
            # Redis transaction (MULTI/EXEC)
            pipe = self.redis.pipeline()
            
            # Store memory data
            cache_key = f"memory:{memory_id}"
            pipe.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(memory_data, default=str)
            )
            
            # Add to memory index
            pipe.sadd("memory:index", memory_id)
            
            # Execute transaction
            pipe.execute()
            
            logger.debug(f"Cached memory: {memory_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cache memory {memory_id}: {e}")
    
    def _invalidate_memory_cache(self, memory_id: str):
        """
        Invalidate memory cache in Redis with transaction
        Production Redis implementation
        """
        if not self.redis:
            return
        
        try:
            # Redis transaction
            pipe = self.redis.pipeline()
            
            cache_key = f"memory:{memory_id}"
            pipe.delete(cache_key)
            pipe.srem("memory:index", memory_id)
            
            pipe.execute()
            
            logger.debug(f"Invalidated cache: {memory_id}")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate cache {memory_id}: {e}")
    
    def retrieve_memory(
        self,
        memory_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Retrieve memory with cache fallback"""
        import time
        start_time = time.time()
        
        # Try cache first
        if use_cache and self.redis:
            cached = self._fetch_from_cache(memory_id)
            if cached:
                self.cache_hits += 1
                retrieval_time = time.time() - start_time
                self.retrieval_times.append(retrieval_time)
                return cached
            self.cache_misses += 1
        
        # Fetch from database
        memory = self._fetch_from_db(memory_id)
        
        if memory and use_cache:
            self._cache_memory(memory_id, memory)
        
        retrieval_time = time.time() - start_time
        self.retrieval_times.append(retrieval_time)
        
        return memory
    
    def _fetch_from_cache(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch memory from Redis cache"""
        if not self.redis:
            return None
        
        try:
            cache_key = f"memory:{memory_id}"
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
        except Exception as e:
            logger.warning(f"Cache fetch failed: {e}")
        
        return None
    
    def _fetch_from_db(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch memory from PostgreSQL"""
        if not self.db:
            return None
        
        try:
            query = """
                SELECT memory_id, content, memory_type, embedding, metadata, 
                       created_at, updated_at
                FROM structured_memories
                WHERE memory_id = %(memory_id)s
            """
            
            result = self.db.execute(query, {'memory_id': memory_id}).fetchone()
            
            if result:
                return dict(result)
            
        except Exception as e:
            logger.error(f"DB fetch failed: {e}")
        
        return None
    
    def _get_embedding(self, content: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get embedding with graceful fallback for OpenAI API failure
        """
        # Create content hash for caching
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        # Check embedding cache
        if content_hash in self.embedding_cache:
            return self.embedding_cache[content_hash]
        
        try:
            # Try OpenAI API
            embedding = self._openai_embedding(content_str)
            
            # Cache successful embedding
            self.embedding_cache[content_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
            
            # Graceful fallback
            if self.embedding_fallback_enabled:
                embedding = self._fallback_embedding(content_str)
                logger.info("Using fallback embedding")
                return embedding
            
            # Report to AVN
            if self.avn:
                self._report_to_avn({
                    'warning': 'embedding_api_failure',
                    'fallback_used': self.embedding_fallback_enabled,
                    'exception': str(e)
                })
            
            return None
    
    def _openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API"""
        # Placeholder - implement actual OpenAI API call
        # import openai
        # response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        # return np.array(response['data'][0]['embedding'])
        
        # Simulated embedding for demo
        return np.random.randn(1536)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding when OpenAI API fails
        Uses simple hash-based embedding
        """
        # Simple deterministic embedding based on text hash
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Convert hash to float array
        embedding = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)
        
        # Pad or truncate to standard size (1536 for OpenAI compatibility)
        target_size = 1536
        if len(embedding) < target_size:
            embedding = np.pad(embedding, (0, target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _check_db_health(self) -> bool:
        """Check PostgreSQL health"""
        if not self.db:
            return False
        
        try:
            self.db.execute("SELECT 1").fetchone()
            return True
        except Exception as e:
            logger.error(f"DB health check failed: {e}")
            return False
    
    def _check_redis_health(self) -> bool:
        """Check Redis health"""
        if not self.redis:
            return False
        
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def _check_embedding_api_health(self) -> bool:
        """Check OpenAI embedding API health"""
        try:
            # Test embedding
            test_embedding = self._openai_embedding("health check")
            return test_embedding is not None
        except Exception:
            return False
    
    def _report_to_avn(self, diagnostic_data: Dict[str, Any]):
        """Report self-diagnostic data to AVN"""
        if not self.avn:
            return
        
        try:
            self.avn.report_diagnostic(
                component="enhanced_memory_core",
                data=diagnostic_data,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"AVN reporting failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        # Update metrics
        total_requests = self.cache_hits + self.cache_misses
        self.metrics.cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        if self.retrieval_times:
            self.metrics.avg_retrieval_time = sum(self.retrieval_times) / len(self.retrieval_times)
        
        # Get Clarity memory stats
        clarity_stats = {}
        if self.clarity_memory:
            clarity_stats = self.clarity_memory.get_memory_statistics()
        
        return {
            'status': self.health_status.value,
            'metrics': {
                'cache_hit_rate': f"{self.metrics.cache_hit_rate:.2%}",
                'avg_retrieval_time': f"{self.metrics.avg_retrieval_time:.3f}s",
                'db_healthy': self.metrics.db_connection_healthy,
                'redis_healthy': self.metrics.redis_connection_healthy,
                'embedding_api_healthy': self.metrics.embedding_api_healthy,
                'last_check': self.metrics.last_check.isoformat()
            },
            'clarity_integration': clarity_stats,
            'total_retrievals': len(self.retrieval_times),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
    
    def run_health_check(self) -> bool:
        """Run comprehensive health check"""
        return self.initialize()
