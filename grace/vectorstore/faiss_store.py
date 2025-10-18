"""
FAISS vector store implementation with production features
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging
import pickle
from pathlib import Path
import uuid
import threading
import time

from .base import VectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store with:
    - Thread-safe operations
    - Automatic persistence
    - Metadata support
    - Index optimization
    """
    
    def __init__(
        self,
        dimension: int,
        index_path: Optional[str] = None,
        index_type: str = "Flat",
        metric: str = "L2"
    ):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save/load index
            index_type: FAISS index type (Flat, IVF, HNSW)
            metric: Distance metric (L2, IP for cosine)
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.index_type = index_type
        self.metric = metric
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu")
        
        # Initialize index
        self.index = self._create_index()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metadata storage
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Auto-save settings
        self.auto_save = True
        self.save_interval = 100  # Save every 100 operations
        self.ops_since_save = 0
        
        # Load existing index if available
        if self.index_path and self.index_path.exists():
            self.load()
    
    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == "Flat":
            if self.metric == "IP":  # Inner product (cosine similarity)
                index = self.faiss.IndexFlatIP(self.dimension)
            else:  # L2 distance
                index = self.faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted file index for larger datasets
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            index = self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            # Will need training before use
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World for fast search
            index = self.faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type} with {self.metric} metric")
        return index
    
    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata to FAISS index"""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        with self.lock:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            elif len(ids) != len(vectors):
                raise ValueError("Number of IDs must match number of vectors")
            
            # Convert to numpy array and normalize if using cosine similarity
            vectors_array = np.array(vectors).astype('float32')
            
            if self.metric == "IP":
                # Normalize for cosine similarity
                norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                vectors_array = vectors_array / norms
            
            # Add to FAISS index
            start_idx = self.next_idx
            self.index.add(vectors_array)
            
            # Store metadata
            result_ids = []
            for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
                idx = start_idx + i
                
                # Add indexing metadata
                meta = meta.copy()
                meta['_indexed_at'] = time.time()
                meta['_vector_id'] = vector_id
                
                self.metadata_store[idx] = meta
                self.id_to_idx[vector_id] = idx
                self.idx_to_id[idx] = vector_id
                result_ids.append(vector_id)
            
            self.next_idx += len(vectors)
            self.ops_since_save += len(vectors)
            
            # Auto-save if needed
            if self.auto_save and self.ops_since_save >= self.save_interval:
                self.save()
                self.ops_since_save = 0
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index (total: {self.index.ntotal})")
            return result_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in FAISS index"""
        with self.lock:
            if self.index.ntotal == 0:
                logger.warning("Index is empty, returning no results")
                return []
            
            # Prepare query vector
            query_array = np.array([query_vector]).astype('float32')
            
            if self.metric == "IP":
                # Normalize for cosine similarity
                norm = np.linalg.norm(query_array)
                if norm > 0:
                    query_array = query_array / norm
            
            # Search with more results for filtering
            search_k = min(k * 3, self.index.ntotal) if filter else k
            
            try:
                distances, indices = self.index.search(query_array, search_k)
            except Exception as e:
                logger.error(f"FAISS search error: {e}")
                return []
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                idx = int(idx)
                if idx not in self.idx_to_id:
                    continue
                
                vector_id = self.idx_to_id[idx]
                metadata = self.metadata_store.get(idx, {})
                
                # Apply filter if provided
                if filter:
                    match = all(
                        metadata.get(key) == value
                        for key, value in filter.items()
                    )
                    if not match:
                        continue
                
                # Convert distance to similarity score
                if self.metric == "IP":
                    # Cosine similarity (already normalized)
                    score = float(distance)
                else:
                    # L2 distance -> similarity
                    score = float(1.0 / (1.0 + distance))
                
                results.append((vector_id, score, metadata))
                
                if len(results) >= k:
                    break
            
            logger.debug(f"Search returned {len(results)} results")
            return results
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs
        Note: FAISS doesn't support true deletion, we mark as deleted
        """
        with self.lock:
            deleted_count = 0
            for vector_id in ids:
                if vector_id in self.id_to_idx:
                    idx = self.id_to_idx[vector_id]
                    
                    # Remove metadata
                    if idx in self.metadata_store:
                        del self.metadata_store[idx]
                    
                    # Remove mappings
                    del self.id_to_idx[vector_id]
                    del self.idx_to_id[idx]
                    deleted_count += 1
            
            if deleted_count > 0:
                self.ops_since_save += deleted_count
                if self.auto_save and self.ops_since_save >= self.save_interval:
                    self.save()
                    self.ops_since_save = 0
            
            logger.info(f"Deleted {deleted_count} vectors from FAISS index")
            return deleted_count > 0
    
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get vector and metadata by ID"""
        with self.lock:
            if id not in self.id_to_idx:
                return None
            
            idx = self.id_to_idx[id]
            if idx not in self.metadata_store:
                return None
            
            # FAISS doesn't allow direct vector retrieval
            # Return None for vector, metadata only
            return None, self.metadata_store[idx]
    
    def count(self) -> int:
        """Get total number of active vectors"""
        with self.lock:
            return len(self.id_to_idx)
    
    def save(self):
        """Save index and metadata to disk"""
        if not self.index_path:
            return
        
        with self.lock:
            try:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save FAISS index
                self.faiss.write_index(self.index, str(self.index_path))
                
                # Save metadata
                metadata_path = self.index_path.with_suffix('.metadata')
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata_store': self.metadata_store,
                        'id_to_idx': self.id_to_idx,
                        'idx_to_id': self.idx_to_id,
                        'next_idx': self.next_idx,
                        'dimension': self.dimension,
                        'index_type': self.index_type,
                        'metric': self.metric
                    }, f)
                
                logger.info(f"Saved FAISS index to {self.index_path}")
                
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
                raise
    
    def load(self):
        """Load index and metadata from disk"""
        if not self.index_path or not self.index_path.exists():
            return
        
        with self.lock:
            try:
                # Load FAISS index
                self.index = self.faiss.read_index(str(self.index_path))
                
                # Load metadata
                metadata_path = self.index_path.with_suffix('.metadata')
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        data = pickle.load(f)
                        self.metadata_store = data['metadata_store']
                        self.id_to_idx = data['id_to_idx']
                        self.idx_to_id = data['idx_to_id']
                        self.next_idx = data['next_idx']
                        
                        # Validate dimension
                        if data['dimension'] != self.dimension:
                            logger.warning(
                                f"Dimension mismatch: expected {self.dimension}, "
                                f"got {data['dimension']}"
                            )
                
                logger.info(
                    f"Loaded FAISS index from {self.index_path} "
                    f"({self.count()} vectors)"
                )
                
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                raise
    
    def optimize(self):
        """Optimize index for search performance"""
        with self.lock:
            if self.index_type == "IVF" and not self.index.is_trained:
                # Train IVF index if not already trained
                vectors = []
                for idx in range(self.index.ntotal):
                    if idx in self.idx_to_id:
                        # Note: Can't easily extract vectors from FAISS
                        # Would need to keep separate storage
                        pass
                
                logger.info("IVF index training would require vector storage")
