"""
Vector store service - Main interface
"""

from typing import Optional
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
        dimension: int,
        store_type: Optional[str] = None,
        db_session = None,
        index_path: Optional[str] = None
    ):
        """
        Initialize vector store service
        
        Args:
            dimension: Embedding dimension
            store_type: 'faiss', 'pgvector', or None for auto-detect
            db_session: SQLAlchemy session (for pgvector)
            index_path: Path to FAISS index (for faiss)
        """
        self.dimension = dimension
        self.store_type = store_type or os.getenv("VECTOR_STORE", "faiss")
        self.store = self._initialize_store(db_session, index_path)
    
    def _initialize_store(self, db_session, index_path) -> VectorStore:
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
        index_path = index_path or os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
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
