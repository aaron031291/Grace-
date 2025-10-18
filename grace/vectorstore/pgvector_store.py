"""
PostgreSQL pgvector store implementation
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging
import json
import uuid

from sqlalchemy import Column, String, Integer, text, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session

from .base import VectorStore
from grace.auth.models import Base

logger = logging.getLogger(__name__)


class VectorDocument(Base):
    """Document with vector embedding in PostgreSQL"""
    __tablename__ = 'vector_documents'
    
    id = Column(String(36), primary_key=True)
    vector = Column(ARRAY(Float))  # pgvector type
    metadata = Column(JSON)


class PgVectorStore(VectorStore):
    """PostgreSQL pgvector-based vector store"""
    
    def __init__(self, db_session: Session, dimension: int):
        """
        Initialize pgvector store
        
        Args:
            db_session: SQLAlchemy session
            dimension: Embedding dimension
        """
        self.db = db_session
        self.dimension = dimension
        
        # Enable pgvector extension
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension: {e}")
    
    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata to PostgreSQL"""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        result_ids = []
        for vector, meta, vector_id in zip(vectors, metadata, ids):
            doc = VectorDocument(
                id=vector_id,
                vector=vector.tolist(),
                metadata=meta
            )
            self.db.add(doc)
            result_ids.append(vector_id)
        
        self.db.commit()
        logger.info(f"Added {len(vectors)} vectors to pgvector")
        return result_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors using pgvector"""
        # Build query with cosine similarity
        query = text("""
            SELECT id, metadata, 
                   1 - (vector <=> :query_vector) as similarity
            FROM vector_documents
            ORDER BY vector <=> :query_vector
            LIMIT :k
        """)
        
        result = self.db.execute(
            query,
            {
                "query_vector": query_vector.tolist(),
                "k": k
            }
        )
        
        results = []
        for row in result:
            if filter:
                match = all(
                    row.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue
            
            results.append((row.id, float(row.similarity), row.metadata))
        
        return results[:k]
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        deleted = self.db.query(VectorDocument).filter(
            VectorDocument.id.in_(ids)
        ).delete(synchronize_session=False)
        
        self.db.commit()
        logger.info(f"Deleted {deleted} vectors from pgvector")
        return deleted > 0
    
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get vector and metadata by ID"""
        doc = self.db.query(VectorDocument).filter(VectorDocument.id == id).first()
        if not doc:
            return None
        
        return np.array(doc.vector), doc.metadata
    
    def count(self) -> int:
        """Get total number of vectors"""
        return self.db.query(VectorDocument).count()
