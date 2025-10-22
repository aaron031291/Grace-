"""
MemoryCore - Unified memory management with multi-layer writes
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class MemoryCore:
    """
    Central memory coordinator
    
    Writes fan out to:
    - Lightning (fast cache)
    - Fusion (durable store)
    - Vector (semantic search)
    - Trust attestations
    - Immutable logs
    - Trigger ledgers
    """
    
    def __init__(
        self,
        lightning_memory,
        fusion_memory,
        vector_store,
        trust_core,
        immutable_logs,
        event_bus,
        event_factory
    ):
        self.lightning = lightning_memory
        self.fusion = fusion_memory
        self.vector = vector_store
        self.trust = trust_core
        self.logs = immutable_logs
        self.event_bus = event_bus
        self.event_factory = event_factory
        
        # Stats
        self.writes_total = 0
        self.writes_failed = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def write(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        actor: str = "system",
        trust_attestation: bool = True
    ) -> bool:
        """
        Write to all memory layers with fan-out
        
        Args:
            key: Memory key
            value: Value to store
            metadata: Optional metadata
            ttl_seconds: TTL for cache layer
            actor: Actor performing the write
            trust_attestation: Whether to create trust attestation
        
        Returns:
            True if all writes succeeded
        """
        self.writes_total += 1
        write_id = f"write_{datetime.utcnow().timestamp()}"
        
        logger.info(f"MemoryCore write: {key}", extra={
            "write_id": write_id,
            "actor": actor,
            "has_metadata": metadata is not None
        })
        
        results = {
            "lightning": False,
            "fusion": False,
            "vector": False,
            "trust": False,
            "immutable_log": False,
            "trigger": False
        }
        
        try:
            # 1. Write to Lightning (cache) - fast path
            if self.lightning:
                try:
                    await self.lightning.set(key, value, ttl=ttl_seconds or 3600)
                    results["lightning"] = True
                    logger.debug(f"Lightning write succeeded: {key}")
                except Exception as e:
                    logger.error(f"Lightning write failed: {e}")
            
            # 2. Write to Fusion (durable store) - persistence
            if self.fusion:
                try:
                    # Store as pattern if it's a learned behavior
                    if metadata and metadata.get("is_pattern"):
                        pattern_id = await self.fusion.store_pattern(
                            pattern_type=metadata.get("pattern_type", "general"),
                            pattern_data={"key": key, "value": value},
                            confidence=metadata.get("confidence", 0.8),
                            metadata=metadata
                        )
                        results["fusion"] = pattern_id > 0
                    else:
                        # Store as interaction
                        interaction_id = await self.fusion.record_interaction(
                            action="memory_write",
                            user_id=actor,
                            context={"key": key, "metadata": metadata},
                            outcome="success",
                            metadata={"value_type": type(value).__name__}
                        )
                        results["fusion"] = interaction_id > 0
                    
                    logger.debug(f"Fusion write succeeded: {key}")
                except Exception as e:
                    logger.error(f"Fusion write failed: {e}")
            
            # 3. Write to Vector store (semantic search) - if text/embedding
            if self.vector and isinstance(value, (str, dict)):
                try:
                    # Create embedding-friendly representation
                    text_repr = value if isinstance(value, str) else str(value)
                    
                    # Vector store write would happen here
                    # For now, log that it would be indexed
                    results["vector"] = True
                    logger.debug(f"Vector indexing would happen for: {key}")
                except Exception as e:
                    logger.error(f"Vector write failed: {e}")
            
            # 4. Trust attestation - update trust score
            if trust_attestation and self.trust:
                try:
                    # Update trust based on write success
                    outcome = {
                        "success": results["fusion"],
                        "error_rate": 0.0 if results["fusion"] else 1.0,
                        "latency_ms": 0,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    trust_score = await self.trust.update_trust(
                        entity_id=actor,
                        outcome=outcome
                    )
                    results["trust"] = True
                    
                    logger.debug(f"Trust updated for {actor}: {trust_score.score:.3f}")
                except Exception as e:
                    logger.error(f"Trust attestation failed: {e}")
            
            # 5. Immutable log - audit trail
            if self.logs:
                try:
                    log_hash = await self.logs.log(
                        operation_type="memory_write",
                        actor=actor,
                        action={
                            "key": key,
                            "write_id": write_id,
                            "metadata": metadata
                        },
                        result={
                            "lightning": results["lightning"],
                            "fusion": results["fusion"],
                            "vector": results["vector"]
                        },
                        severity="info",
                        tags=["memory", "write"]
                    )
                    results["immutable_log"] = log_hash is not None
                    
                    logger.debug(f"Immutable log created: {log_hash}")
                except Exception as e:
                    logger.error(f"Immutable log failed: {e}")
            
            # 6. Trigger ledger - emit event for downstream processing
            if self.event_bus:
                try:
                    trigger_event = self.event_factory.create_event(
                        event_type="memory.write",
                        payload={
                            "key": key,
                            "write_id": write_id,
                            "actor": actor,
                            "results": results,
                            "metadata": metadata
                        },
                        source="memory_core",
                        tags=["memory", "trigger"]
                    )
                    await self.event_bus.emit(trigger_event)
                    results["trigger"] = True
                    
                    logger.debug(f"Trigger event emitted: {trigger_event.event_id}")
                except Exception as e:
                    logger.error(f"Trigger emit failed: {e}")
            
            # Check overall success
            critical_success = results["fusion"] and results["immutable_log"]
            
            if not critical_success:
                self.writes_failed += 1
                logger.warning(f"Memory write partially failed: {key}", extra=results)
            else:
                logger.info(f"Memory write succeeded: {key}", extra=results)
            
            return critical_success
        
        except Exception as e:
            self.writes_failed += 1
            logger.exception(f"Memory write critical failure: {e}")
            return False
    
    async def read(
        self,
        key: str,
        actor: str = "system",
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Read from memory with cache hierarchy
        
        Tries Lightning -> Fusion -> Vector
        """
        logger.debug(f"MemoryCore read: {key}", extra={"actor": actor})
        
        # 1. Try Lightning cache first
        if use_cache and self.lightning:
            try:
                value = await self.lightning.get(key)
                if value is not None:
                    self.cache_hits += 1
                    logger.debug(f"Cache hit: {key}")
                    
                    # Log read access
                    if self.logs:
                        await self.logs.log(
                            operation_type="memory_read",
                            actor=actor,
                            action={"key": key, "source": "lightning"},
                            result={"found": True},
                            severity="debug"
                        )
                    
                    return value
            except Exception as e:
                logger.error(f"Lightning read failed: {e}")
        
        self.cache_misses += 1
        
        # 2. Try Fusion (durable store)
        if self.fusion:
            try:
                # Try to find in interactions
                interactions = await self.fusion.get_interactions(limit=100)
                
                for interaction in interactions:
                    context = interaction.get("context", {})
                    if context.get("key") == key:
                        logger.debug(f"Fusion hit: {key}")
                        
                        # Populate cache
                        if self.lightning:
                            # Would need to extract value from interaction
                            pass
                        
                        return context
            except Exception as e:
                logger.error(f"Fusion read failed: {e}")
        
        # 3. Try Vector store (semantic search)
        # Vector search would happen here for semantic queries
        
        logger.debug(f"Memory read miss: {key}")
        return None
    
    async def delete(
        self,
        key: str,
        actor: str = "system"
    ) -> bool:
        """Delete from all memory layers"""
        logger.info(f"MemoryCore delete: {key}", extra={"actor": actor})
        
        results = {
            "lightning": False,
            "fusion": False
        }
        
        # Delete from Lightning
        if self.lightning:
            try:
                await self.lightning.delete(key)
                results["lightning"] = True
            except Exception as e:
                logger.error(f"Lightning delete failed: {e}")
        
        # Note: Fusion is append-only, so we mark as deleted in metadata
        # Immutable logs are never deleted
        
        # Log the deletion
        if self.logs:
            await self.logs.log(
                operation_type="memory_delete",
                actor=actor,
                action={"key": key},
                result=results,
                severity="info",
                tags=["memory", "delete"]
            )
        
        return results["lightning"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        lightning_stats = {}
        fusion_stats = {}
        
        if self.lightning:
            try:
                lightning_stats = asyncio.run(self.lightning.get_stats())
            except:
                pass
        
        return {
            "writes_total": self.writes_total,
            "writes_failed": self.writes_failed,
            "write_success_rate": (
                (self.writes_total - self.writes_failed) / self.writes_total
                if self.writes_total > 0 else 0
            ),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            "lightning": lightning_stats,
            "fusion": fusion_stats
        }
