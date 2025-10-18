"""
Embedding providers - OpenAI, HuggingFace, and local models
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings using text-embedding-ada-002"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._dimension = 1536  # ada-002 dimension
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            self.client = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using OpenAI"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using OpenAI"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [np.array(item.embedding) for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace sentence transformers embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._dimension = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded HuggingFace model: {model_name} (dim: {self._dimension})")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using HuggingFace"""
        if not self.model:
            raise RuntimeError("HuggingFace model not initialized")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using HuggingFace"""
        if not self.model:
            raise RuntimeError("HuggingFace model not initialized")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
        except Exception as e:
            logger.error(f"HuggingFace batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension if self._dimension else 384  # Default for MiniLM


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings using lightweight models (fallback)"""
    
    def __init__(self):
        self._dimension = 384
        
        try:
            from sentence_transformers import SentenceTransformer
            # Use the smallest, fastest model as fallback
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded local embedding model (all-MiniLM-L6-v2)")
        except ImportError:
            logger.warning("sentence-transformers not available, using random embeddings (NOT FOR PRODUCTION)")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text locally"""
        if self.model:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback to deterministic random (NOT FOR PRODUCTION)
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            seed = int.from_bytes(hash_obj.digest()[:4], 'little')
            np.random.seed(seed)
            return np.random.randn(self._dimension).astype(np.float32)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts locally"""
        if self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
        else:
            return [self.embed_text(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension
