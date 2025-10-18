"""
Embedding providers - OpenAI, HuggingFace, and local models
"""

from typing import List, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import logging
import os
import time
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


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
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Embed texts in batches for efficiency"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)
            embeddings.extend(batch_embeddings)
        return embeddings


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings using text-embedding-ada-002 or text-embedding-3-small"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self._dimension = 1536 if "ada-002" in model else 1536  # text-embedding-3-small also 1536
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI embeddings with model: {model}")
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    @retry_on_error(max_retries=3, delay=1.0)
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using OpenAI"""
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self._dimension, dtype=np.float32)
        
        try:
            # Truncate if too long (8191 tokens max for ada-002)
            if len(text) > 8000:
                text = text[:8000]
                logger.warning("Text truncated to 8000 characters")
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using OpenAI (batch API)"""
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text[:8000])  # Truncate
                valid_indices.append(i)
        
        if not valid_texts:
            return [np.zeros(self._dimension, dtype=np.float32) for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                input=valid_texts,
                model=self.model
            )
            
            embeddings = [None] * len(texts)
            for i, idx in enumerate(valid_indices):
                embedding = np.array(response.data[i].embedding, dtype=np.float32)
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings[idx] = embedding
            
            # Fill empty texts with zero vectors
            for i, emb in enumerate(embeddings):
                if emb is None:
                    embeddings[i] = np.zeros(self._dimension, dtype=np.float32)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace sentence transformers embeddings"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self._dimension = None
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Initialized HuggingFace model: {model_name} "
                f"(dimension: {self._dimension})"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using HuggingFace"""
        if not text or not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using HuggingFace"""
        if not texts:
            return []
        
        # Filter and prepare texts
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32
            )
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"HuggingFace batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension if self._dimension else 384


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings using lightweight models (fallback)"""
    
    def __init__(self):
        self._dimension = 384
        
        try:
            from sentence_transformers import SentenceTransformer
            # Use smallest model for fallback
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded local embedding model (all-MiniLM-L6-v2)")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers: {e}")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text locally"""
        if not text or not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)
        
        if self.model:
            try:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embedding.astype(np.float32)
            except Exception as e:
                logger.error(f"Local embedding error: {e}")
        
        # Fallback to deterministic random (for testing only)
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        seed = int.from_bytes(hash_obj.digest()[:4], 'little')
        np.random.seed(seed)
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts locally"""
        if not texts:
            return []
        
        if self.model:
            try:
                valid_texts = [t if t and t.strip() else " " for t in texts]
                embeddings = self.model.encode(
                    valid_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                )
                return [emb.astype(np.float32) for emb in embeddings]
            except Exception as e:
                logger.error(f"Local batch embedding error: {e}")
        
        # Fallback
        return [self.embed_text(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension
