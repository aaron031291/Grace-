"""
FAISS vector store implementation
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging
import pickle
from pathlib import Path
import uuid

from .base import VectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with metadata support"""
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save/load index
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu")
            raise
        
        # Initialize index
        self.index = self.faiss.IndexFlatL2(dimension)
        
        # Metadata storage
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Load existing index if path provided
        if self.index_path and self.index_path.exists():
            self.load()
    
    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata to FAISS index"""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")
        
        # Convert to numpy array
        vectors_array = np.array(vectors).astype('float32')
        
        # Add to FAISS index
        self.index.add(vectors_array)
        
        # Store metadata
        result_ids = []
        for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
            idx = self.next_idx + i
            self.metadata_store[idx] = meta
            self.id_to_idx[vector_id] = idx
            self.idx_to_id[idx] = vector_id
            result_ids.append(vector_id)
        
        self.next_idx += len(vectors)
        
        # Save if path provided
        if self.index_path:
            self.save()
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
        return result_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in FAISS index"""
        query_array = np.array([query_vector]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_array, min(k * 2, self.index.ntotal))
        
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
            
            # Convert L2 distance to similarity score (lower is better)
            score = float(1.0 / (1.0 + distance))
            
            results.append((vector_id, score, metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (FAISS doesn't support efficient deletion)"""
        # Note: FAISS doesn't support efficient deletion
        # We mark as deleted in metadata
        deleted_count = 0
        for vector_id in ids:
            if vector_id in self.id_to_idx:
                idx = self.id_to_idx[vector_id]
                if idx in self.metadata_store:
                    del self.metadata_store[idx]
                del self.id_to_idx[vector_id]
                del self.idx_to_id[idx]
                deleted_count += 1
        
        if deleted_count > 0 and self.index_path:
            self.save()
        
        logger.info(f"Deleted {deleted_count} vectors from FAISS index")
        return deleted_count > 0
    
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get vector and metadata by ID"""
        if id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[id]
        if idx not in self.metadata_store:
            return None
        
        # FAISS doesn't allow direct vector retrieval, return None for vector
        return None, self.metadata_store[idx]
    
    def count(self) -> int:
        """Get total number of active vectors"""
        return len(self.id_to_idx)
    
    def save(self):
        """Save index and metadata to disk"""
        if not self.index_path:
            return
        
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
                'next_idx': self.next_idx
            }, f)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load(self):
        """Load index and metadata from disk"""
        if not self.index_path or not self.index_path.exists():
            return
        
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
        
        logger.info(f"Loaded FAISS index from {self.index_path} ({self.count()} vectors)")
