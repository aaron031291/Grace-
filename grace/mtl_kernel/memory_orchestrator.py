"""Memory Orchestrator - Unified API for all memory operations."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .lightning_memory import LightningMemory
from .fusion_memory import FusionMemory
from .vector_memory import VectorMemory

logger = logging.getLogger(__name__)


class MemoryOrchestrator:
    """
    Unified API for Lightning/Fusion/Vector/Librarian memory operations.
    
    Features:
    - Smart routing logic (Lightning → Fusion → Vector)
    - Performance targets: <1ms Lightning, <50ms Fusion
    - Batch operations support
    - Connection pooling for all backends
    - Async/await throughout
    - Fallback chain for resilience
    """
    
    def __init__(self):
        # Initialize memory backends
        self.lightning = LightningMemory(max_size=10000, default_ttl=3600)
        self.fusion = FusionMemory()
        self.vector = VectorMemory()
        
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_gets": 0,
            "total_sets": 0,
            "total_queries": 0,
            "lightning_hits": 0,
            "fusion_hits": 0,
            "vector_hits": 0,
            "fallback_uses": 0,
            "start_time": time.time()
        }
        
        logger.info("Memory Orchestrator initialized")
    
    async def get(self, key: str, fallback_chain: bool = True) -> Optional[Any]:
        """
        Get value with automatic fallback chain.
        
        1. Check Lightning (hot cache) - <1ms
        2. Check Fusion (structured storage) - <50ms if fallback enabled
        3. Return None if not found
        
        Args:
            key: Key to retrieve
            fallback_chain: Enable fallback to slower tiers
        
        Returns:
            Value if found, None otherwise
        """
        start = time.time()
        self._stats["total_gets"] += 1
        
        # 1. Try Lightning first (hot cache)
        value = await self.lightning.get(key)
        if value is not None:
            self._stats["lightning_hits"] += 1
            elapsed = (time.time() - start) * 1000
            logger.debug(f"Lightning hit for '{key}' in {elapsed:.2f}ms")
            return value
        
        # 2. Try Fusion if fallback enabled
        if fallback_chain:
            # Query Fusion for key
            results = await self.fusion.query(
                "learned_patterns",  # Default table
                filters={"pattern_type": key},
                limit=1
            )
            
            if results:
                value = results[0]["data"]
                self._stats["fusion_hits"] += 1
                self._stats["fallback_uses"] += 1
                
                # Promote to Lightning for future fast access
                await self.lightning.set(key, value)
                
                elapsed = (time.time() - start) * 1000
                logger.debug(f"Fusion hit for '{key}' in {elapsed:.2f}ms")
                return value
        
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Miss for '{key}' in {elapsed:.2f}ms")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        storage_tier: str = "auto",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set value with tier selection.
        
        Args:
            key: Key to store
            value: Value to store
            storage_tier: "lightning", "fusion", "both", or "auto"
            ttl_seconds: TTL for Lightning tier
        
        Returns:
            True if successful
        """
        start = time.time()
        self._stats["total_sets"] += 1
        
        success = False
        
        # Determine storage tier
        if storage_tier == "auto":
            # Auto-select based on value size
            value_size = len(str(value))
            if value_size < 10000:  # < 10KB -> Lightning
                storage_tier = "lightning"
            else:  # >= 10KB -> Both
                storage_tier = "both"
        
        # Store in Lightning
        if storage_tier in ["lightning", "both"]:
            success = await self.lightning.set(key, value, ttl_seconds)
        
        # Store in Fusion
        if storage_tier in ["fusion", "both"]:
            await self.fusion.insert(
                "learned_patterns",
                {
                    "pattern_type": key,
                    "pattern_data": value,
                    "confidence": 0.8,
                    "usage_count": 0
                }
            )
            success = True
        
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Set '{key}' in {storage_tier} ({elapsed:.2f}ms)")
        
        return success
    
    async def query(self, pattern: str, search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Query across memory tiers.
        
        Args:
            pattern: Search pattern
            search_type: "lightning", "fusion", "vector", or "hybrid"
        
        Returns:
            List of matching results
        """
        start = time.time()
        self._stats["total_queries"] += 1
        
        results = []
        
        if search_type in ["lightning", "hybrid"]:
            # Search Lightning by prefix
            lightning_results = await self.lightning.get_by_prefix(pattern)
            for key, value in lightning_results.items():
                results.append({
                    "source": "lightning",
                    "key": key,
                    "value": value
                })
        
        if search_type in ["fusion", "hybrid"]:
            # Search Fusion (would use full-text search in production)
            fusion_results = await self.fusion.query(
                "learned_patterns",
                limit=20
            )
            for result in fusion_results:
                if pattern.lower() in str(result["data"]).lower():
                    results.append({
                        "source": "fusion",
                        "id": result["id"],
                        "data": result["data"]
                    })
        
        if search_type in ["vector", "hybrid"]:
            # Generate embedding and search vectors
            query_embedding = VectorMemory.generate_mock_embedding(pattern)
            vector_results = await self.vector.search(
                "knowledge",
                query_embedding,
                top_k=10
            )
            for result in vector_results:
                results.append({
                    "source": "vector",
                    "id": result["id"],
                    "similarity": result["similarity"],
                    "metadata": result["metadata"]
                })
        
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Query '{pattern}' ({search_type}) returned {len(results)} results in {elapsed:.2f}ms")
        
        return results
    
    async def store_with_trust(
        self,
        data: Dict[str, Any],
        trust_score: float,
        tier: str = "auto"
    ) -> str:
        """
        Store data with trust score.
        
        Args:
            data: Data to store
            trust_score: Trust score (0.0-1.0)
            tier: Storage tier
        
        Returns:
            Entry ID
        """
        # Store in Fusion trust ledger
        entry_id = await self.fusion.insert(
            "trust_ledger",
            {
                "entity_id": data.get("entity_id", "unknown"),
                "trust_score": trust_score,
                "event_type": data.get("event_type", "unknown"),
                "delta": 0.0,
                "reason": data.get("reason", "")
            },
            trust_score=trust_score
        )
        
        # Also cache in Lightning if tier allows
        if tier in ["auto", "both", "lightning"]:
            await self.lightning.set(
                f"trust_{entry_id}",
                {
                    "trust_score": trust_score,
                    "data": data
                }
            )
        
        return entry_id
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "knowledge"
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using vector memory.
        
        Args:
            query: Search query
            top_k: Number of results
            collection: Vector collection to search
        
        Returns:
            Semantically similar results
        """
        # Generate query embedding
        query_embedding = VectorMemory.generate_mock_embedding(query)
        
        # Search vector memory
        results = await self.vector.search(
            collection,
            query_embedding,
            top_k=top_k
        )
        
        self._stats["vector_hits"] += len(results)
        
        return results
    
    async def health_check(self) -> Dict[str, str]:
        """
        Check health of all memory backends.
        
        Returns:
            Health status for each backend
        """
        health = {}
        
        # Check Lightning
        try:
            await self.lightning.set("_health_check", "ok", ttl_seconds=1)
            health["lightning"] = "healthy"
        except Exception as e:
            health["lightning"] = f"unhealthy: {e}"
        
        # Check Fusion
        try:
            await self.fusion.query("learned_patterns", limit=1)
            health["fusion"] = "healthy"
        except Exception as e:
            health["fusion"] = f"unhealthy: {e}"
        
        # Check Vector
        try:
            await self.vector.get_stats()
            health["vector"] = "healthy"
        except Exception as e:
            health["vector"] = f"unhealthy: {e}"
        
        return health
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        lightning_stats = await self.lightning.get_stats()
        fusion_stats = await self.fusion.get_stats()
        vector_stats = await self.vector.get_stats()
        
        uptime = time.time() - self._stats["start_time"]
        
        total_requests = self._stats["total_gets"] + self._stats["total_queries"]
        hit_rate = 0.0
        if total_requests > 0:
            hits = self._stats["lightning_hits"] + self._stats["fusion_hits"]
            hit_rate = hits / total_requests
        
        return {
            "orchestrator": {
                "total_gets": self._stats["total_gets"],
                "total_sets": self._stats["total_sets"],
                "total_queries": self._stats["total_queries"],
                "hit_rate": round(hit_rate, 3),
                "lightning_hits": self._stats["lightning_hits"],
                "fusion_hits": self._stats["fusion_hits"],
                "vector_hits": self._stats["vector_hits"],
                "fallback_uses": self._stats["fallback_uses"],
                "uptime_seconds": round(uptime, 1)
            },
            "lightning": lightning_stats,
            "fusion": fusion_stats,
            "vector": vector_stats
        }
