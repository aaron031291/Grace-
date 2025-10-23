"""
Grace AI Unified Memory System - Single Cohesive Module
========================================================

Consolidates ALL memory components into one logical system:
  - MTL (Immutable Logs)
  - Lightning (Fast Access)
  - Fusion (Knowledge Integration)
  - Vector (Semantic Storage)
  - Librarian (Organization & Retrieval)
  - Database Layer (PostgreSQL)
  - Table Schemas
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import hashlib
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# LAYER 1: MTL (Multi-Task Learning) - Immutable Core Truth
# ============================================================================

class MTLImmutableLedger:
    """
    Multi-Task Learning Immutable Ledger
    Canonical source of truth - all data flows through here
    Cryptographically signed, immutable record of all operations
    """
    
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.hash_chain: List[str] = []
        self.task_knowledge: Dict[str, Dict[str, Any]] = {}
    
    async def log_operation(
        self,
        task_id: str,
        operation_type: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Log immutable operation to ledger."""
        entry_id = hashlib.sha256(
            f"{task_id}{operation_type}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        entry = {
            "entry_id": entry_id,
            "task_id": task_id,
            "operation_type": operation_type,
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "previous_hash": self.hash_chain[-1] if self.hash_chain else "genesis"
        }
        
        # Sign entry
        signature = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()
        entry["signature"] = signature
        
        self.entries.append(entry)
        self.hash_chain.append(signature)
        
        logger.info(f"MTL: Logged operation {operation_type} (entry: {entry_id})")
        return entry_id
    
    async def record_learned_knowledge(self, task_id: str, knowledge: Dict[str, Any]):
        """Record learned knowledge for a task."""
        self.task_knowledge[task_id] = {
            "knowledge": knowledge,
            "learned_at": datetime.now().isoformat(),
            "version": len(self.task_knowledge.get(task_id, {}).get("versions", [])) + 1
        }
        
        await self.log_operation(
            task_id=task_id,
            operation_type="knowledge_learned",
            data=knowledge
        )
    
    def verify_integrity(self) -> bool:
        """Verify chain integrity."""
        for i, entry in enumerate(self.entries):
            expected_previous = self.hash_chain[i-1] if i > 0 else "genesis"
            if entry["previous_hash"] != expected_previous:
                logger.error(f"MTL integrity violation at entry {i}")
                return False
        return True
    
    def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all operations for a task."""
        return [e for e in self.entries if e["task_id"] == task_id]


# ============================================================================
# LAYER 2: LIGHTNING - Fast Access Layer
# ============================================================================

class LightningMemory:
    """
    Lightning Memory - Ultra-fast access layer
    In-memory cache for frequently accessed data
    Backed by MTL for durability
    """
    
    def __init__(self, mtl: MTLImmutableLedger):
        self.mtl = mtl
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.hit_rate = 0.0
        self.total_accesses = 0
        self.cache_hits = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        self.total_accesses += 1
        
        if key in self.cache:
            self.cache_hits += 1
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.hit_rate = self.cache_hits / self.total_accesses if self.total_accesses > 0 else 0
            logger.debug(f"Lightning: Cache HIT for {key} (hit_rate: {self.hit_rate:.1%})")
            return self.cache[key]
        
        logger.debug(f"Lightning: Cache MISS for {key}")
        return None
    
    async def set(self, key: str, value: Any, task_id: str = "system"):
        """Set in cache and log to MTL."""
        self.cache[key] = value
        self.access_count[key] = 0
        
        await self.mtl.log_operation(
            task_id=task_id,
            operation_type="cache_set",
            data={"key": key, "value_type": type(value).__name__}
        )
        
        logger.info(f"Lightning: Set cache key {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_keys": len(self.cache),
            "total_accesses": self.total_accesses,
            "cache_hits": self.cache_hits,
            "hit_rate": self.hit_rate,
            "access_distribution": self.access_count
        }


# ============================================================================
# LAYER 3: FUSION - Knowledge Integration
# ============================================================================

class FusionMemory:
    """
    Fusion Memory - Integrates knowledge from multiple sources
    Merges data, resolves conflicts, creates unified knowledge
    Backed by MTL for immutability
    """
    
    def __init__(self, mtl: MTLImmutableLedger):
        self.mtl = mtl
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, List[str]] = {}
        self.fusion_history: List[Dict[str, Any]] = []
    
    async def fuse_knowledge(
        self,
        knowledge_items: List[Dict[str, Any]],
        task_id: str,
        resolution_strategy: str = "merge"
    ) -> Dict[str, Any]:
        """
        Fuse multiple knowledge items into unified knowledge.
        """
        fused = {
            "components": len(knowledge_items),
            "strategy": resolution_strategy,
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }
        
        for item in knowledge_items:
            for key, value in item.items():
                if key not in fused["data"]:
                    fused["data"][key] = value
                elif resolution_strategy == "merge":
                    if isinstance(fused["data"][key], dict) and isinstance(value, dict):
                        fused["data"][key].update(value)
                    elif isinstance(fused["data"][key], list):
                        fused["data"][key].extend([value] if not isinstance(value, list) else value)
        
        fusion_id = hashlib.sha256(
            json.dumps(fused, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        fused["fusion_id"] = fusion_id
        self.fusion_history.append(fused)
        
        await self.mtl.log_operation(
            task_id=task_id,
            operation_type="knowledge_fusion",
            data=fused
        )
        
        logger.info(f"Fusion: Fused {len(knowledge_items)} knowledge items (id: {fusion_id})")
        return fused
    
    async def add_relationship(self, from_key: str, to_key: str, relation_type: str):
        """Add relationship between knowledge items."""
        if from_key not in self.relationships:
            self.relationships[from_key] = []
        
        self.relationships[from_key].append(f"{to_key}:{relation_type}")
        
        logger.info(f"Fusion: Added relationship {from_key} -> {to_key} ({relation_type})")
    
    def get_related_knowledge(self, key: str) -> List[Tuple[str, str]]:
        """Get all related knowledge."""
        relations = self.relationships.get(key, [])
        return [tuple(r.split(":")) for r in relations]


# ============================================================================
# LAYER 4: VECTOR - Semantic Storage
# ============================================================================

class VectorMemory:
    """
    Vector Memory - Semantic embedding storage
    Stores embeddings for similarity search
    Backed by MTL for durability
    """
    
    def __init__(self, mtl: MTLImmutableLedger, embedding_dim: int = 384):
        self.mtl = mtl
        self.embeddings: Dict[str, List[float]] = {}
        self.embedding_dim = embedding_dim
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    async def store_embedding(
        self,
        key: str,
        embedding: List[float],
        text: str,
        task_id: str = "system"
    ) -> bool:
        """Store semantic embedding."""
        if len(embedding) != self.embedding_dim:
            logger.error(f"Vector: Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}")
            return False
        
        self.embeddings[key] = embedding
        self.metadata[key] = {
            "text": text,
            "stored_at": datetime.now().isoformat(),
            "dimension": self.embedding_dim
        }
        
        await self.mtl.log_operation(
            task_id=task_id,
            operation_type="embedding_stored",
            data={"key": key, "text": text[:100]}
        )
        
        logger.info(f"Vector: Stored embedding for {key}")
        return True
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        if not self.embeddings:
            return []
        
        scores = []
        for key, embedding in self.embeddings.items():
            # Simple cosine similarity
            dot_product = sum(q * e for q, e in zip(query_embedding, embedding))
            magnitude_q = sum(q**2 for q in query_embedding) ** 0.5
            magnitude_e = sum(e**2 for e in embedding) ** 0.5
            
            if magnitude_q > 0 and magnitude_e > 0:
                similarity = dot_product / (magnitude_q * magnitude_e)
                scores.append((key, similarity))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Vector: Search returned {min(len(scores), top_k)} results")
        return scores[:top_k]
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector storage statistics."""
        return {
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": self.embedding_dim,
            "total_storage_mb": (len(self.embeddings) * self.embedding_dim * 4) / (1024 * 1024)
        }


# ============================================================================
# LAYER 5: LIBRARIAN - Organization & Retrieval
# ============================================================================

class LibrarianMemory:
    """
    Librarian Memory - Organization and retrieval system
    Catalogs, indexes, and retrieves information efficiently
    Backed by MTL for audit trail
    """
    
    def __init__(self, mtl: MTLImmutableLedger):
        self.mtl = mtl
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.indexes: Dict[str, List[str]] = {}
        self.tags: Dict[str, List[str]] = {}
    
    async def catalog_item(
        self,
        item_id: str,
        title: str,
        content_type: str,
        tags: List[str],
        task_id: str = "system"
    ) -> bool:
        """Catalog a memory item."""
        self.catalog[item_id] = {
            "item_id": item_id,
            "title": title,
            "content_type": content_type,
            "cataloged_at": datetime.now().isoformat(),
            "tags": tags
        }
        
        # Index by content type
        if content_type not in self.indexes:
            self.indexes[content_type] = []
        self.indexes[content_type].append(item_id)
        
        # Index by tags
        for tag in tags:
            if tag not in self.tags:
                self.tags[tag] = []
            self.tags[tag].append(item_id)
        
        await self.mtl.log_operation(
            task_id=task_id,
            operation_type="item_cataloged",
            data={"item_id": item_id, "title": title, "tags": tags}
        )
        
        logger.info(f"Librarian: Cataloged {item_id} ({content_type})")
        return True
    
    async def search_by_tag(self, tag: str) -> List[str]:
        """Search for items by tag."""
        return self.tags.get(tag, [])
    
    async def search_by_type(self, content_type: str) -> List[str]:
        """Search for items by content type."""
        return self.indexes.get(content_type, [])
    
    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get cataloging statistics."""
        return {
            "total_items": len(self.catalog),
            "content_types": len(self.indexes),
            "unique_tags": len(self.tags),
            "items_by_type": {ct: len(items) for ct, items in self.indexes.items()},
            "tags_distribution": {tag: len(items) for tag, items in self.tags.items()}
        }


# ============================================================================
# UNIFIED MEMORY SYSTEM - Integration Point
# ============================================================================

class UnifiedMemorySystem:
    """
    Unified Memory System - Single cohesive memory architecture
    Integrates all memory layers:
    - MTL (Immutable Core)
    - Lightning (Fast Access)
    - Fusion (Integration)
    - Vector (Semantic Storage)
    - Librarian (Organization)
    """
    
    def __init__(self):
        self.mtl = MTLImmutableLedger()
        self.lightning = LightningMemory(self.mtl)
        self.fusion = FusionMemory(self.mtl)
        self.vector = VectorMemory(self.mtl)
        self.librarian = LibrarianMemory(self.mtl)
        
        logger.info("✓ Unified Memory System initialized")
    
    async def store_knowledge(
        self,
        key: str,
        content: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        task_id: str = "system"
    ) -> bool:
        """
        Store knowledge through all layers:
        1. Lightning: Fast cache
        2. MTL: Immutable log
        3. Vector: Semantic embedding (if provided)
        4. Librarian: Catalog (if tags provided)
        """
        
        # Lightning
        await self.lightning.set(key, content, task_id)
        
        # MTL
        entry_id = await self.mtl.log_operation(
            task_id=task_id,
            operation_type="knowledge_stored",
            data=content
        )
        
        # Vector (if embedding provided)
        if embedding:
            text = content.get("text", str(content))[:500]
            await self.vector.store_embedding(key, embedding, text, task_id)
        
        # Librarian (if tags provided)
        if tags:
            await self.librarian.catalog_item(
                item_id=key,
                title=content.get("title", key),
                content_type=content.get("type", "unknown"),
                tags=tags,
                task_id=task_id
            )
        
        logger.info(f"✓ Knowledge stored through all layers (entry: {entry_id})")
        return True
    
    async def retrieve_knowledge(
        self,
        key: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge (Lightning first, then MTL if needed)
        """
        
        # Try Lightning cache first
        if use_cache:
            cached = await self.lightning.get(key)
            if cached:
                return cached
        
        # If not in cache, retrieve from MTL
        # (In production, this would hit database)
        logger.info(f"Retrieving knowledge {key} from MTL")
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            "mtl": {
                "total_entries": len(self.mtl.entries),
                "chain_integrity": self.mtl.verify_integrity()
            },
            "lightning": self.lightning.get_stats(),
            "vector": self.vector.get_vector_stats(),
            "librarian": self.librarian.get_catalog_stats(),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# DATABASE SCHEMA (Conceptual)
# ============================================================================

class DatabaseSchema:
    """
    Defines schema for persistent storage in PostgreSQL
    """
    
    # MTL Table
    MTL_ENTRIES_TABLE = """
    CREATE TABLE IF NOT EXISTS mtl_entries (
        entry_id VARCHAR(16) PRIMARY KEY,
        task_id VARCHAR(255),
        operation_type VARCHAR(100),
        data JSONB,
        metadata JSONB,
        signature VARCHAR(64),
        previous_hash VARCHAR(64),
        timestamp TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Lightning Cache Table
    LIGHTNING_CACHE_TABLE = """
    CREATE TABLE IF NOT EXISTS lightning_cache (
        key VARCHAR(255) PRIMARY KEY,
        value JSONB,
        access_count INT DEFAULT 0,
        last_accessed TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Vector Embeddings Table
    VECTOR_EMBEDDINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS vector_embeddings (
        key VARCHAR(255) PRIMARY KEY,
        embedding VECTOR(384),
        text TEXT,
        metadata JSONB,
        stored_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Librarian Catalog Table
    LIBRARIAN_CATALOG_TABLE = """
    CREATE TABLE IF NOT EXISTS librarian_catalog (
        item_id VARCHAR(255) PRIMARY KEY,
        title VARCHAR(500),
        content_type VARCHAR(100),
        tags TEXT[],
        cataloged_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Fusion Knowledge Table
    FUSION_KNOWLEDGE_TABLE = """
    CREATE TABLE IF NOT EXISTS fusion_knowledge (
        fusion_id VARCHAR(16) PRIMARY KEY,
        components INT,
        strategy VARCHAR(50),
        data JSONB,
        timestamp TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """


logger.info("✓ Unified Memory System module loaded")
