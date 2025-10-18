"""
Embeddings service - Support for multiple embedding providers
"""

from .providers import EmbeddingProvider, OpenAIEmbeddings, HuggingFaceEmbeddings, LocalEmbeddings
from .service import EmbeddingService

__all__ = [
    'EmbeddingProvider',
    'OpenAIEmbeddings',
    'HuggingFaceEmbeddings',
    'LocalEmbeddings',
    'EmbeddingService'
]
