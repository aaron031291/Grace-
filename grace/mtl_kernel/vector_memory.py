"""Vector Memory - Semantic search with ChromaDB-like interface."""

import asyncio
import hashlib
import json
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """Entry in vector store."""
    id: str
    collection: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime


class VectorMemory:
    """
    Semantic search with ChromaDB-like semantics.
    
    Features:
    - Collections for: patterns, precedents, knowledge
    - Semantic similarity search (cosine similarity)
    - Metadata filtering
    - Batch operations
    - Distance metrics
    """
    
    # Pre-defined collections
    COLLECTIONS = ["patterns", "precedents", "knowledge", "interactions"]
    
    def __init__(self):
        # In-memory storage (ChromaDB would be used in production)
        self._collections: Dict[str, List[VectorEntry]] = {
            name: [] for name in self.COLLECTIONS
        }
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_adds": 0,
            "total_searches": 0,
            "total_deletes": 0,
            "start_time": time.time()
        }
        
        logger.info(f"Vector Memory initialized with collections: {self.COLLECTIONS}")
    
    async def add(
        self,
        collection: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        entry_id: Optional[str] = None
    ) -> str:
        """Add embedding to collection."""
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection '{collection}' not found")
        
        async with self._lock:
            # Generate ID if not provided
            if not entry_id:
                entry_id = f"{collection}_{len(self._collections[collection])}_{int(time.time() * 1000)}"
            
            entry = VectorEntry(
                id=entry_id,
                collection=collection,
                embedding=embedding,
                metadata=metadata,
                created_at=datetime.utcnow()
            )
            
            self._collections[collection].append(entry)
            self._stats["total_adds"] += 1
            
            return entry_id
    
    async def add_many(
        self,
        collection: str,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Add multiple embeddings (batch operation)."""
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadatas")
        
        ids = []
        for embedding, metadata in zip(embeddings, metadatas):
            entry_id = await self.add(collection, embedding, metadata)
            ids.append(entry_id)
        
        return ids
    
    async def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search collection by semantic similarity."""
        start = time.time()
        
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection '{collection}' not found")
        
        async with self._lock:
            results = []
            
            for entry in self._collections[collection]:
                # Apply metadata filters
                if filters:
                    match = all(
                        entry.metadata.get(key) == value
                        for key, value in filters.items()
                    )
                    if not match:
                        continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                
                results.append({
                    "id": entry.id,
                    "similarity": similarity,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at.isoformat()
                })
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            self._stats["total_searches"] += 1
            
            elapsed = (time.time() - start) * 1000
            logger.debug(f"Vector search completed in {elapsed:.2f}ms")
            
            return results[:top_k]
    
    async def delete(self, collection: str, ids: List[str]) -> int:
        """Delete entries by IDs."""
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection '{collection}' not found")
        
        async with self._lock:
            deleted_count = 0
            
            self._collections[collection] = [
                entry for entry in self._collections[collection]
                if entry.id not in ids or (deleted_count := deleted_count + 1) and False
            ]
            
            self._stats["total_deletes"] += deleted_count
            
            return deleted_count
    
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Collection '{collection}' not found")
        
        async with self._lock:
            entries = self._collections[collection]
            
            # Calculate average embedding dimension
            avg_dim = 0
            if entries:
                avg_dim = sum(len(e.embedding) for e in entries) / len(entries)
            
            return {
                "name": collection,
                "count": len(entries),
                "avg_embedding_dim": round(avg_dim, 1)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        async with self._lock:
            collection_stats = {}
            total_vectors = 0
            
            for name in self.COLLECTIONS:
                stats = await self.get_collection_stats(name)
                collection_stats[name] = stats
                total_vectors += stats["count"]
            
            uptime = time.time() - self._stats["start_time"]
            
            return {
                "total_vectors": total_vectors,
                "collections": collection_stats,
                "total_adds": self._stats["total_adds"],
                "total_searches": self._stats["total_searches"],
                "total_deletes": self._stats["total_deletes"],
                "uptime_seconds": round(uptime, 1)
            }
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    @staticmethod
    def generate_mock_embedding(text: str, dimensions: int = 384) -> List[float]:
        """
        Generate a deterministic mock embedding from text.
        
        In production, this would use a real embedding model.
        """
        # Use hash to generate deterministic values
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        
        # Generate normalized vector
        values = [
            math.sin(hash_val / (i + 1)) 
            for i in range(dimensions)
        ]
        
        # Normalize
        magnitude = math.sqrt(sum(v * v for v in values))
        if magnitude > 0:
            values = [v / magnitude for v in values]
        
        return values
