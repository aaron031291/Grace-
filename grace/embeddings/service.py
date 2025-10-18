"""
Embedding service - Main interface for embeddings
"""

from typing import List, Optional
import numpy as np
import logging
import os

from .providers import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    LocalEmbeddings
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing embeddings with automatic provider selection"""
    
    def __init__(self, provider: Optional[str] = None, **kwargs):
        """
        Initialize embedding service
        
        Args:
            provider: 'openai', 'huggingface', 'local', or None for auto-detect
            **kwargs: Provider-specific arguments
        """
        self.provider_name = provider or os.getenv("EMBEDDING_PROVIDER", "auto")
        self.provider = self._initialize_provider(**kwargs)
    
    def _initialize_provider(self, **kwargs) -> EmbeddingProvider:
        """Initialize the embedding provider with fallback chain"""
        
        if self.provider_name == "openai":
            try:
                provider = OpenAIEmbeddings(**kwargs)
                logger.info("Using OpenAI embeddings")
                return provider
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}, falling back")
        
        elif self.provider_name == "huggingface":
            try:
                provider = HuggingFaceEmbeddings(**kwargs)
                logger.info("Using HuggingFace embeddings")
                return provider
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace: {e}, falling back")
        
        elif self.provider_name == "local":
            provider = LocalEmbeddings()
            logger.info("Using local embeddings")
            return provider
        
        # Auto-detect with fallback chain
        # Try OpenAI first if API key available
        if os.getenv("OPENAI_API_KEY"):
            try:
                provider = OpenAIEmbeddings(**kwargs)
                logger.info("Auto-detected: Using OpenAI embeddings")
                return provider
            except Exception:
                pass
        
        # Try HuggingFace next
        try:
            provider = HuggingFaceEmbeddings(**kwargs)
            logger.info("Auto-detected: Using HuggingFace embeddings")
            return provider
        except Exception:
            pass
        
        # Fallback to local
        provider = LocalEmbeddings()
        logger.info("Auto-detected: Using local embeddings (fallback)")
        return provider
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text with error handling"""
        try:
            return self.provider.embed_text(text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts with error handling"""
        try:
            return self.provider.embed_texts(texts)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.provider.dimension
    
    def get_provider_info(self) -> dict:
        """Get information about the current provider"""
        return {
            "provider": self.provider.__class__.__name__,
            "dimension": self.dimension,
            "requested": self.provider_name
        }
