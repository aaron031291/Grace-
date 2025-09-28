"""RAG query contracts."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .dto_common import BaseDTO, MemoryEntry


class RAGQuery(BaseModel):
    """Retrieval-Augmented Generation query."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(default=10, ge=1, le=100)
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    include_embeddings: bool = False
    rerank: bool = True


class RAGResult(BaseDTO):
    """RAG query result."""
    query: str
    items: List[MemoryEntry] = Field(default_factory=list)
    total_found: int = 0
    processing_time_ms: float = 0.0
    distilled_summary: Optional[str] = None