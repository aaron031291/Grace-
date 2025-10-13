"""LLM Kernel - embeddings, reranking, and distillation utilities."""

import hashlib
import random
from typing import Any, Dict, List, Optional

from ...contracts.dto_common import MemoryEntry


class LLMKernel:
    """LLM utilities for embeddings, reranking, and distillation."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.embedding_dim = 384  # Mock embedding dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts (mock implementation)."""
        embeddings = []

        for text in texts:
            # Create deterministic mock embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            random.seed(seed)

            # Generate normalized mock embedding
            embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings

    def rerank(
        self, query: str, documents: List[MemoryEntry], top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Rerank documents by relevance to query (mock implementation)."""
        if not documents:
            return []

        # Mock reranking based on simple text similarity
        scored_docs = []
        query_words = set(query.lower().split())

        for doc in documents:
            content_words = set(doc.content.lower().split())
            # Simple Jaccard similarity
            intersection = len(query_words.intersection(content_words))
            union = len(query_words.union(content_words))
            score = intersection / union if union > 0 else 0

            scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top documents
        if top_k:
            scored_docs = scored_docs[:top_k]

        return [doc for score, doc in scored_docs]

    def distill(self, chunks: List[str], context: str = "") -> str:
        """Distill multiple text chunks into a summary (mock implementation)."""
        if not chunks:
            return ""

        # Mock distillation - extract key sentences
        all_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk.split(".") if s.strip()]
            all_sentences.extend(sentences)

        # Select diverse sentences (mock logic)
        if len(all_sentences) <= 3:
            result = ". ".join(all_sentences)
        else:
            # Pick first, middle, and last sentences
            indices = [0, len(all_sentences) // 2, len(all_sentences) - 1]
            selected = [all_sentences[i] for i in indices]
            result = ". ".join(selected)

        if context:
            result = f"Context: {context}\n\nSummary: {result}"

        return result

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if len(embedding1) != len(embedding2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_stats(self) -> Dict[str, Any]:
        """Get LLM kernel statistics."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "capabilities": ["embed", "rerank", "distill"],
            "status": "active",
        }
