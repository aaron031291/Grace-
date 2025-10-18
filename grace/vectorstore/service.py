"""
Vector store service - Main interface
"""

from typing import Optional
from grace.config import get_settings

import logging
import os

from .base import VectorStore
from .faiss_store import FAISSVectorStore
from .pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector store with automatic selection"""
    
    def __init__(
        self,
        dimension: int = 384,
        index_path: Optional[str] = None
    ):
        settings = get_settings()
        self.index_path = index_path or settings.vector_store.faiss_index_path
        self.store_type = settings.vector_store.store_type or os.getenv("VECTOR_STORE", "faiss")
        self.store = self._initialize_store()
    
    def _initialize_store(self) -> VectorStore:
        """Initialize the vector store"""
        
        if self.store_type == "pgvector":
            if not db_session:
                raise ValueError("db_session required for pgvector store")
            
            try:
                store = PgVectorStore(db_session, self.dimension)
                logger.info("Using PostgreSQL pgvector store")
                return store
            except Exception as e:
                logger.warning(f"Failed to initialize pgvector: {e}, falling back to FAISS")
        
        # Default to FAISS
        index_path = self.index_path or os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
        store = FAISSVectorStore(self.dimension, index_path)
        logger.info("Using FAISS vector store")
        return store
    
    def get_store(self) -> VectorStore:
        """Get the underlying vector store"""
        return self.store
    
    def get_store_info(self) -> dict:
        """Get information about the vector store"""
        return {
            "store_type": self.store.__class__.__name__,
            "dimension": self.dimension,
            "count": self.store.count()
        }
