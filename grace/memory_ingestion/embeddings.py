"""
Embedding generation for text chunks.

Generates vector embeddings for text using various embedding models.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import hashlib
import json

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = None
        
    async def _load_model(self):
        """Load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Load in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: SentenceTransformer(self.model_name)
                )
                logger.info(f"Loaded local embedding model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using local model."""
        await self._load_model()
        
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.get_embedding_dimension()
        
        # Generate embedding in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode([text], convert_to_numpy=True)[0]
        )
        
        # Convert numpy array to list
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Common dimensions for popular models
            dimension_map = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-distilroberta-v1": 768,
                "paraphrase-MiniLM-L6-v2": 384,
            }
            self._dimension = dimension_map.get(self.model_name, 384)  # Default
        return self._dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "mock-embedding-model"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash."""
        # Create deterministic "embedding" based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hex to numbers and normalize
        embedding = []
        for i in range(0, min(len(text_hash), self.dimension * 8), 8):
            hex_chunk = text_hash[i:i+8]
            if len(hex_chunk) == 8:
                # Convert hex to float between -1 and 1
                value = int(hex_chunk, 16) / (16**8 / 2) - 1
                embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        
        return embedding[:self.dimension]
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class EmbeddingGenerator:
    """Main embedding generator that coordinates providers."""
    
    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """
        Initialize embedding generator.
        
        Args:
            provider: Embedding provider to use. If None, will try to create LocalEmbeddingProvider
        """
        self.provider = provider
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the embedding provider."""
        if self.provider is None:
            try:
                # Try to use local embedding provider
                self.provider = LocalEmbeddingProvider()
            except ImportError:
                logger.warning("Local embedding provider not available, using mock provider")
                self.provider = MockEmbeddingProvider()
    
    async def generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding results with metadata
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                embedding = await self.provider.generate_embedding(text)
                
                result = {
                    'text': text,
                    'embedding': embedding,
                    'embedding_dimension': len(embedding),
                    'model_name': self.provider.get_model_name(),
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'index': i
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for text {i}: {e}")
                # Add empty result to maintain indexing
                results.append({
                    'text': text,
                    'embedding': [0.0] * self.provider.get_embedding_dimension(),
                    'embedding_dimension': self.provider.get_embedding_dimension(),
                    'model_name': self.provider.get_model_name(),
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'index': i,
                    'error': str(e)
                })
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return results
    
    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding result with metadata
        """
        results = await self.generate_embeddings([text])
        return results[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        return self.provider.get_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        return self.provider.get_model_name()


# Vector similarity functions
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same dimension")
    
    # Convert to numpy for efficient computation
    vec_a = np.array(a)
    vec_b = np.array(b)
    
    # Compute cosine similarity
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same dimension")
    
    vec_a = np.array(a)
    vec_b = np.array(b)
    
    return float(np.linalg.norm(vec_a - vec_b))


# Global embedding generator
_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Get global embedding generator instance."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator