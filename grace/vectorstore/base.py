"""
Base vector store interface
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors, returns (id, score, metadata)"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get vector and metadata by ID"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of vectors"""
        pass
