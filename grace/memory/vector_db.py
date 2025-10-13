"""
Next-generation Vector Database Integration for Grace Memory Infrastructure.

Provides:
- Milvus vector database support for semantic search
- Pinecone v3 integration for scalable vector operations
- Hybrid search combining vector and traditional approaches
- Enhanced recall and reasoning capabilities
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib

# Optional vector database dependencies
try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    import pinecone

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Optional embedding model support
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorDBInterface(ABC):
    """Abstract interface for vector database implementations."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Create a new collection with specified dimension."""
        pass

    @abstractmethod
    async def insert_vectors(
        self, collection_name: str, vectors: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert vectors into the collection."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from the collection."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass


class MilvusVectorDB(VectorDBInterface):
    """Milvus vector database implementation."""

    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.collections = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Milvus."""
        if not MILVUS_AVAILABLE:
            logger.error("Milvus not available. Install pymilvus package.")
            return False

        try:
            connections.connect("default", host=self.host, port=self.port)
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus")

    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Create a Milvus collection."""
        if not self.connected:
            return False

        try:
            # Define collection schema
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True
                ),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="trust_score", dtype=DataType.FLOAT),
                FieldSchema(name="constitutional_score", dtype=DataType.FLOAT),
            ]

            schema = CollectionSchema(
                fields, f"Grace memory collection: {collection_name}"
            )
            collection = Collection(collection_name, schema)

            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index("vector", index_params)

            self.collections[collection_name] = collection
            logger.info(f"Created Milvus collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Milvus collection {collection_name}: {e}")
            return False

    async def insert_vectors(
        self, collection_name: str, vectors: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert vectors into Milvus collection."""
        if collection_name not in self.collections:
            return []

        try:
            collection = self.collections[collection_name]

            # Prepare data for insertion
            data = [
                [v["id"] for v in vectors],  # ids
                [v["vector"] for v in vectors],  # vectors
                [v["content"] for v in vectors],  # content
                [json.dumps(v.get("metadata", {})) for v in vectors],  # metadata
                [
                    v.get("timestamp", datetime.now().isoformat()) for v in vectors
                ],  # timestamp
                [v.get("trust_score", 0.5) for v in vectors],  # trust_score
                [
                    v.get("constitutional_score", 0.5) for v in vectors
                ],  # constitutional_score
            ]

            collection.insert(data)
            collection.load()  # Load collection to memory

            inserted_ids = [v["id"] for v in vectors]
            logger.info(f"Inserted {len(inserted_ids)} vectors into {collection_name}")
            return inserted_ids

        except Exception as e:
            logger.error(f"Failed to insert vectors into {collection_name}: {e}")
            return []

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus."""
        if collection_name not in self.collections:
            return []

        try:
            collection = self.collections[collection_name]

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

            # Build filter expression if provided
            expr = None
            if filters:
                expr_parts = []
                if "min_trust_score" in filters:
                    expr_parts.append(f"trust_score >= {filters['min_trust_score']}")
                if "min_constitutional_score" in filters:
                    expr_parts.append(
                        f"constitutional_score >= {filters['min_constitutional_score']}"
                    )
                if expr_parts:
                    expr = " and ".join(expr_parts)

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[
                    "content",
                    "metadata",
                    "timestamp",
                    "trust_score",
                    "constitutional_score",
                ],
            )

            # Format results
            formatted_results = []
            if results:
                for hit in results[0]:
                    formatted_results.append(
                        {
                            "id": hit.id,
                            "score": float(hit.distance),
                            "content": hit.entity.get("content"),
                            "metadata": json.loads(hit.entity.get("metadata", "{}")),
                            "timestamp": hit.entity.get("timestamp"),
                            "trust_score": float(hit.entity.get("trust_score", 0.0)),
                            "constitutional_score": float(
                                hit.entity.get("constitutional_score", 0.0)
                            ),
                        }
                    )

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vectors in {collection_name}: {e}")
            return []

    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Milvus collection."""
        if collection_name not in self.collections:
            return False

        try:
            collection = self.collections[collection_name]
            expr = f"id in {vector_ids}"
            collection.delete(expr)
            logger.info(f"Deleted {len(vector_ids)} vectors from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Milvus statistics."""
        stats = {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "collections": list(self.collections.keys()),
            "total_collections": len(self.collections),
        }

        # Get collection stats
        collection_stats = {}
        for name, collection in self.collections.items():
            try:
                collection.load()
                collection_stats[name] = {
                    "num_entities": collection.num_entities,
                    "is_loaded": collection.is_loaded,
                }
            except Exception as e:
                collection_stats[name] = {"error": str(e)}

        stats["collection_stats"] = collection_stats
        return stats


class PineconeVectorDB(VectorDBInterface):
    """Pinecone v3 vector database implementation."""

    def __init__(self, api_key: str, environment: str = "us-west1-gcp"):
        self.api_key = api_key
        self.environment = environment
        self.client = None
        self.indexes = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Pinecone."""
        if not PINECONE_AVAILABLE:
            logger.error("Pinecone not available. Install pinecone-client package.")
            return False

        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            self.client = pinecone
            self.connected = True
            logger.info(f"Connected to Pinecone in {self.environment}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self.connected = False
        self.client = None
        logger.info("Disconnected from Pinecone")

    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Create a Pinecone index (collection)."""
        if not self.connected:
            return False

        try:
            # Check if index already exists
            if collection_name in pinecone.list_indexes():
                self.indexes[collection_name] = pinecone.Index(collection_name)
                logger.info(f"Using existing Pinecone index: {collection_name}")
                return True

            # Create new index
            pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric="cosine",
                pods=1,
                replicas=1,
                pod_type="p1.x1",
            )

            self.indexes[collection_name] = pinecone.Index(collection_name)
            logger.info(f"Created Pinecone index: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Pinecone index {collection_name}: {e}")
            return False

    async def insert_vectors(
        self, collection_name: str, vectors: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert vectors into Pinecone index."""
        if collection_name not in self.indexes:
            return []

        try:
            index = self.indexes[collection_name]

            # Prepare vectors for Pinecone
            upsert_data = []
            for v in vectors:
                metadata = v.get("metadata", {})
                metadata.update(
                    {
                        "content": v["content"],
                        "timestamp": v.get("timestamp", datetime.now().isoformat()),
                        "trust_score": v.get("trust_score", 0.5),
                        "constitutional_score": v.get("constitutional_score", 0.5),
                    }
                )

                upsert_data.append(
                    {"id": v["id"], "values": v["vector"], "metadata": metadata}
                )

            # Upsert in batches of 100
            batch_size = 100
            inserted_ids = []
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i : i + batch_size]
                index.upsert(vectors=batch)
                inserted_ids.extend([v["id"] for v in batch])

            logger.info(f"Inserted {len(inserted_ids)} vectors into {collection_name}")
            return inserted_ids

        except Exception as e:
            logger.error(f"Failed to insert vectors into {collection_name}: {e}")
            return []

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        if collection_name not in self.indexes:
            return []

        try:
            index = self.indexes[collection_name]

            # Build filter
            filter_dict = {}
            if filters:
                if "min_trust_score" in filters:
                    filter_dict["trust_score"] = {"$gte": filters["min_trust_score"]}
                if "min_constitutional_score" in filters:
                    filter_dict["constitutional_score"] = {
                        "$gte": filters["min_constitutional_score"]
                    }

            # Perform search
            search_result = index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict if filter_dict else None,
                include_metadata=True,
            )

            # Format results
            formatted_results = []
            for match in search_result.matches:
                metadata = match.metadata or {}
                formatted_results.append(
                    {
                        "id": match.id,
                        "score": float(match.score),
                        "content": metadata.get("content", ""),
                        "metadata": {
                            k: v
                            for k, v in metadata.items()
                            if k
                            not in [
                                "content",
                                "timestamp",
                                "trust_score",
                                "constitutional_score",
                            ]
                        },
                        "timestamp": metadata.get("timestamp"),
                        "trust_score": float(metadata.get("trust_score", 0.0)),
                        "constitutional_score": float(
                            metadata.get("constitutional_score", 0.0)
                        ),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vectors in {collection_name}: {e}")
            return []

    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Pinecone index."""
        if collection_name not in self.indexes:
            return False

        try:
            index = self.indexes[collection_name]
            index.delete(ids=vector_ids)
            logger.info(f"Deleted {len(vector_ids)} vectors from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics."""
        stats = {
            "connected": self.connected,
            "environment": self.environment,
            "indexes": list(self.indexes.keys()),
            "total_indexes": len(self.indexes),
        }

        # Get index stats
        index_stats = {}
        for name, index in self.indexes.items():
            try:
                index_info = index.describe_index_stats()
                index_stats[name] = {
                    "total_vector_count": index_info.total_vector_count,
                    "dimension": index_info.dimension,
                }
            except Exception as e:
                index_stats[name] = {"error": str(e)}

        stats["index_stats"] = index_stats
        return stats


class EmbeddingService:
    """Service for generating text embeddings for vector search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None

    def initialize(self) -> bool:
        """Initialize the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error(
                "SentenceTransformers not available. Install sentence-transformers package."
            )
            return False

        try:
            self.model = SentenceTransformer(self.model_name)
            # Get dimension by encoding a test sentence
            test_embedding = self.model.encode("test")
            self.dimension = len(test_embedding)
            logger.info(
                f"Initialized embedding model {self.model_name} (dim: {self.dimension})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False

    def encode_text(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Encode text into embeddings."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        embeddings = self.model.encode(text)

        if isinstance(text, str):
            return embeddings.tolist()
        else:
            return [emb.tolist() for emb in embeddings]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension or 384  # Default dimension for MiniLM

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode texts in batches for efficiency."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(batch)
            all_embeddings.extend([emb.tolist() for emb in batch_embeddings])

        return all_embeddings


class VectorMemoryCore:
    """
    Enhanced Memory Core with vector database integration.

    Provides semantic search, enhanced recall, and reasoning capabilities
    through next-generation vector databases.
    """

    def __init__(
        self,
        vector_db: VectorDBInterface,
        embedding_service: Optional[EmbeddingService] = None,
        default_collection: str = "grace_memory",
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service or EmbeddingService()
        self.default_collection = default_collection
        self.initialized = False

        # Statistics
        self.search_count = 0
        self.insert_count = 0
        self.cache_hits = 0

    async def initialize(self) -> bool:
        """Initialize the vector memory core."""
        try:
            # Connect to vector database
            if not await self.vector_db.connect():
                return False

            # Initialize embedding service
            if not self.embedding_service.initialize():
                return False

            # Create default collection
            dimension = self.embedding_service.get_dimension()
            if not await self.vector_db.create_collection(
                self.default_collection, dimension
            ):
                return False

            self.initialized = True
            logger.info("VectorMemoryCore initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize VectorMemoryCore: {e}")
            return False

    async def store_content(
        self,
        content: str,
        content_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trust_score: float = 0.5,
        constitutional_score: float = 0.5,
    ) -> Optional[str]:
        """Store content with vector embedding."""
        if not self.initialized:
            return None

        try:
            # Generate content ID if not provided
            if not content_id:
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                content_id = f"mem_{content_hash}_{int(datetime.now().timestamp())}"

            # Generate embedding
            embedding = self.embedding_service.encode_text(content)

            # Prepare vector data
            vector_data = {
                "id": content_id,
                "vector": embedding,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "trust_score": trust_score,
                "constitutional_score": constitutional_score,
            }

            # Insert into vector database
            inserted_ids = await self.vector_db.insert_vectors(
                self.default_collection, [vector_data]
            )

            if inserted_ids:
                self.insert_count += 1
                logger.debug(f"Stored content with vector embedding: {content_id}")
                return content_id

            return None

        except Exception as e:
            logger.error(f"Failed to store content: {e}")
            return None

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        min_trust_score: float = 0.0,
        min_constitutional_score: float = 0.0,
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings."""
        if not self.initialized:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_text(query)

            # Build filters
            filters = {}
            if min_trust_score > 0:
                filters["min_trust_score"] = min_trust_score
            if min_constitutional_score > 0:
                filters["min_constitutional_score"] = min_constitutional_score

            # Search in vector database
            results = await self.vector_db.search_vectors(
                collection or self.default_collection,
                query_embedding,
                top_k=top_k,
                filters=filters,
            )

            self.search_count += 1
            logger.debug(f"Semantic search completed: {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        keyword_results: List[Dict[str, Any]],
        top_k: int = 10,
        semantic_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results."""
        try:
            # Get semantic results
            semantic_results = await self.semantic_search(query, top_k=top_k * 2)

            # Create hybrid scoring
            semantic_scores = {r["id"]: r["score"] for r in semantic_results}
            keyword_scores = {
                r.get("id", ""): r.get("score", 0.0) for r in keyword_results
            }

            # Combine scores
            combined_results = {}
            all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())

            for content_id in all_ids:
                sem_score = semantic_scores.get(content_id, 0.0)
                kw_score = keyword_scores.get(content_id, 0.0)

                # Normalize and combine scores
                hybrid_score = (
                    semantic_weight * sem_score + (1 - semantic_weight) * kw_score
                )

                # Get the full result object
                result = None
                for r in semantic_results:
                    if r["id"] == content_id:
                        result = r.copy()
                        break

                if not result:
                    for r in keyword_results:
                        if r.get("id") == content_id:
                            result = r.copy()
                            break

                if result:
                    result["hybrid_score"] = hybrid_score
                    result["semantic_score"] = sem_score
                    result["keyword_score"] = kw_score
                    combined_results[content_id] = result

            # Sort by hybrid score and return top_k
            sorted_results = sorted(
                combined_results.values(), key=lambda x: x["hybrid_score"], reverse=True
            )

            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def shutdown(self) -> None:
        """Shutdown the vector memory core."""
        if self.vector_db:
            await self.vector_db.disconnect()

        self.initialized = False
        logger.info("VectorMemoryCore shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector memory statistics."""
        stats = {
            "initialized": self.initialized,
            "search_count": self.search_count,
            "insert_count": self.insert_count,
            "cache_hits": self.cache_hits,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.get_dimension(),
            "vector_db_stats": self.vector_db.get_stats(),
        }

        return stats
