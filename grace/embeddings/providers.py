"""
Embedding providers for Grace - Open Source Only
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


class HuggingFaceEmbeddings(EmbeddingProvider):
    """
    HuggingFace embedding provider using sentence-transformers
    
    Free, open-source, and runs locally without API keys
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded HuggingFace model: {model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using HuggingFace"""
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
        
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
        return self.dimension if self.dimension else 384


class LocalEmbeddings(EmbeddingProvider):
    """
    Local embedding provider using transformers library directly
    
    Completely self-contained, no external API calls
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.dimension = self.model.config.hidden_size
            
            logger.info(f"Loaded local model: {model_name}")
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers torch")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text locally"""
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
        
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
        embedding = np.random.randn(self.dimension).astype(np.float32)
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
        return self.dimension
