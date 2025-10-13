"""Librarian - document ingestion and processing."""

import hashlib
from typing import Dict, List, Optional

from ..contracts.dto_common import MemoryEntry
from .w5h_indexer import W5HIndexer
from .memory_service import MemoryService
from .llm.kernel import LLMKernel


class Librarian:
    """Document librarian for ingesting and organizing content."""

    def __init__(
        self, memory_service: MemoryService, llm_kernel: Optional[LLMKernel] = None
    ):
        self.memory_service = memory_service
        self.w5h_indexer = W5HIndexer()
        self.llm_kernel = llm_kernel or LLMKernel()

    def ingest(
        self,
        content: str,
        content_type: str = "text/plain",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Ingest content into memory with full processing pipeline."""
        # Create base memory entry
        entry = MemoryEntry(
            content=content, content_type=content_type, metadata=metadata
        )

        # Enhance with W5H indexing
        entry = self.w5h_indexer.enhance_entry(entry)

        # Add embeddings if LLM kernel available
        try:
            embeddings = self.llm_kernel.embed([content])
            if embeddings:
                entry.embedding = embeddings[0]
        except Exception:
            # Embeddings optional for now
            pass

        # Calculate content hash
        entry.sha256 = hashlib.sha256(content.encode()).hexdigest()

        # Store in memory service (will trigger MTL fan-out)
        memory_id = self.memory_service.store(entry)

        return memory_id

    def batch_ingest(self, documents: List[Dict]) -> List[str]:
        """Batch ingest multiple documents."""
        results = []

        for doc in documents:
            memory_id = self.ingest(
                content=doc.get("content", ""),
                content_type=doc.get("content_type", "text/plain"),
                metadata=doc.get("metadata"),
            )
            results.append(memory_id)

        return results

    def search_and_rank(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search and rank results using LLM reranking."""
        # Get initial results from memory service
        results = self.memory_service.query(
            query, limit=limit * 2
        )  # Get more for reranking

        if not results or not self.llm_kernel:
            return results[:limit]

        # Rerank using LLM
        try:
            ranked_results = self.llm_kernel.rerank(query, results)
            return ranked_results[:limit]
        except Exception:
            # Fallback to original results
            return results[:limit]

    def distill_content(self, entries: List[MemoryEntry], context: str = "") -> str:
        """Distill multiple entries into a summary."""
        if not entries or not self.llm_kernel:
            return ""

        try:
            chunks = [entry.content for entry in entries]
            return self.llm_kernel.distill(chunks, context)
        except Exception:
            # Fallback to simple concatenation
            return "\n\n".join(entry.content for entry in entries[:3])

    def get_stats(self) -> Dict:
        """Get librarian statistics."""
        total_entries = len(self.memory_service.store.entries)

        # Calculate content types distribution
        content_types = {}
        for entry in self.memory_service.store.entries.values():
            ct = entry.content_type
            content_types[ct] = content_types.get(ct, 0) + 1

        return {
            "total_entries": total_entries,
            "content_types": content_types,
            "indexer_active": True,
            "llm_active": bool(self.llm_kernel),
        }
