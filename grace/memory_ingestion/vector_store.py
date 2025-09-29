"""
Vector store integration for Grace memory system.

Handles vector storage and similarity search using Qdrant.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Upsert vectors with metadata."""
        pass
    
    @abstractmethod
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, 
                              vector_dimension: int) -> bool:
        """Create a new collection."""
        pass


class QdrantVectorStore(VectorStore):
    """Vector store implementation using Qdrant."""
    
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "grace_memory"):
        self.url = url
        self.collection_name = collection_name
        self.client = None
        self._initialized = False
    
    async def _initialize(self):
        """Initialize Qdrant client."""
        if self._initialized:
            return
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            
            self.client = QdrantClient(url=self.url)
            
            # Check if collection exists, create if not
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                await self._create_default_collection()
            
            self._initialized = True
            logger.info(f"Qdrant client initialized: {self.url}")
            
        except ImportError:
            logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    async def _create_default_collection(self):
        """Create default collection with 384-dimensional vectors."""
        await self.create_collection(self.collection_name, 384)  # Default for MiniLM
    
    async def create_collection(self, collection_name: str, vector_dimension: int) -> bool:
        """Create a new collection."""
        try:
            from qdrant_client.models import VectorParams, Distance
            
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection_name} with dimension {vector_dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Upsert vectors with metadata to Qdrant."""
        await self._initialize()
        
        if not vectors:
            return True
        
        try:
            from qdrant_client.models import PointStruct
            
            points = []
            for vector_data in vectors:
                point = PointStruct(
                    id=vector_data['id'],
                    vector=vector_data['vector'],
                    payload=vector_data.get('metadata', {})
                )
                points.append(point)
            
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} vectors to {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        await self._initialize()
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter if provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            search_results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    'id': str(hit.id),
                    'score': hit.score,
                    'metadata': hit.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        await self._initialize()
        
        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=ids
            )
            
            logger.info(f"Deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False


class MockVectorStore(VectorStore):
    """Mock vector store for testing."""
    
    def __init__(self):
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self.collection_created = False
    
    async def create_collection(self, collection_name: str, vector_dimension: int) -> bool:
        """Create mock collection."""
        self.collection_created = True
        logger.info(f"Mock collection created: {collection_name}")
        return True
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Store vectors in memory."""
        for vector_data in vectors:
            self.vectors[vector_data['id']] = vector_data
        logger.info(f"Mock upserted {len(vectors)} vectors")
        return True
    
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Mock search using simple dot product."""
        import numpy as np
        
        results = []
        query_vec = np.array(query_vector)
        
        for vector_id, vector_data in self.vectors.items():
            stored_vec = np.array(vector_data['vector'])
            
            # Simple cosine similarity
            if np.linalg.norm(query_vec) > 0 and np.linalg.norm(stored_vec) > 0:
                similarity = np.dot(query_vec, stored_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                )
            else:
                similarity = 0.0
            
            # Apply filters if provided
            if filters:
                metadata = vector_data.get('metadata', {})
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            results.append({
                'id': vector_id,
                'score': float(similarity),
                'metadata': vector_data.get('metadata', {})
            })
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from memory."""
        for vector_id in ids:
            self.vectors.pop(vector_id, None)
        logger.info(f"Mock deleted {len(ids)} vectors")
        return True


class VectorSearchService:
    """High-level vector search service."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None, vector_url: Optional[str] = None):
        if vector_store:
            self.vector_store = vector_store
        elif vector_url:
            try:
                self.vector_store = QdrantVectorStore(url=vector_url)
            except Exception:
                logger.warning("Failed to initialize Qdrant, using mock store")
                self.vector_store = MockVectorStore()
        else:
            self.vector_store = MockVectorStore()
    
    async def index_content(self, content_id: str, vector: List[float], 
                          metadata: Dict[str, Any]) -> bool:
        """Index content with vector and metadata."""
        vector_data = {
            'id': content_id,
            'vector': vector,
            'metadata': metadata
        }
        
        return await self.vector_store.upsert_vectors([vector_data])
    
    async def search_content(self, query_vector: List[float], 
                           filters: Optional[Dict[str, Any]] = None,
                           trust_threshold: float = 0.5,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar content with filtering.
        
        Args:
            query_vector: Query vector for similarity search
            filters: Optional metadata filters
            trust_threshold: Minimum trust score for results
            limit: Maximum number of results
            
        Returns:
            List of search results with content and metadata
        """
        # Add trust threshold to filters if specified
        search_filters = filters.copy() if filters else {}
        
        # Perform vector search
        results = await self.vector_store.search_vectors(
            query_vector=query_vector,
            limit=limit * 2,  # Get more results to filter by trust
            filters=search_filters
        )
        
        # Post-filter by trust score if needed
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            trust_score = metadata.get('trust_score', 1.0)  # Default high trust
            
            if trust_score >= trust_threshold:
                filtered_results.append(result)
            
            if len(filtered_results) >= limit:
                break
        
        logger.info(f"Search returned {len(filtered_results)} results (trust >= {trust_threshold})")
        return filtered_results
    
    async def delete_content(self, content_ids: List[str]) -> bool:
        """Delete content by IDs."""
        return await self.vector_store.delete_vectors(content_ids)


# Global service instance
_search_service = None

def get_vector_search_service(vector_url: Optional[str] = None) -> VectorSearchService:
    """Get global vector search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = VectorSearchService(vector_url=vector_url)
    return _search_service