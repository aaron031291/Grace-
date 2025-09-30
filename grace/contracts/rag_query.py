"""
rag_contracts.py
Production-grade RAG (Retrieval-Augmented Generation) request/response contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from .dto_common import BaseDTO, MemoryEntry


# ---------------------------
# Common helpers
# ---------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "use_enum_values": True,
        "ser_json_bytes": "utf8",
        "populate_by_name": True,
    }


# ---------------------------
# Query models
# ---------------------------

class SearchType(StrEnum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RAGQuery(StrictModel):
    """Retrieval-Augmented Generation query."""
    query: str
    filters: Optional[Mapping[str, Any]] = None          # attribute filters, e.g. {"w5h_index.who": "alice"}
    limit: int = Field(default=10, ge=1, le=100)
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    include_embeddings: bool = False
    rerank: bool = True
    search_type: SearchType = SearchType.HYBRID
    namespace: Optional[str] = Field(
        default=None, description="Optional logical namespace/collection"
    )
    cursor: Optional[str] = Field(
        default=None, description="Opaque pagination cursor from a previous response"
    )

    @field_validator("query")
    @classmethod
    def _nonempty_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty")
        return v


# ---------------------------
# Result models
# ---------------------------

class RAGHit(StrictModel):
    """One retrieved item with ranking metadata."""
    entry: MemoryEntry
    score: float = Field(ge=0.0, le=1.0, description="Normalized relevance score")
    rank: int = Field(ge=1)
    distance: Optional[float] = Field(
        default=None, ge=0.0, description="Raw vector distance if available"
    )
    highlights: List[str] = Field(
        default_factory=list, description="Extracted snippets/matches"
    )

    @field_validator("highlights", mode="after")
    @classmethod
    def _norm_highlights(cls, v: List[str]) -> List[str]:
        out, seen = [], set()
        for s in v:
            if isinstance(s, str):
                t = s.strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
        return out


class RAGTiming(StrictModel):
    """Optional timing breakdown (ms)."""
    total_ms: float = Field(ge=0.0, default=0.0)
    search_ms: Optional[float] = Field(default=None, ge=0.0)
    rerank_ms: Optional[float] = Field(default=None, ge=0.0)
    hydrate_ms: Optional[float] = Field(default=None, ge=0.0)


class RAGQueryParams(StrictModel):
    """Echo of the effective query parameters actually used by the engine."""
    query: str
    limit: int
    min_relevance: float
    include_embeddings: bool
    rerank: bool
    search_type: SearchType
    namespace: Optional[str] = None
    filters: Optional[Mapping[str, Any]] = None


class RAGResult(BaseDTO):
    """RAG query result."""
    query: str
    items: List[RAGHit] = Field(default_factory=list)
    total_found: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    distilled_summary: Optional[str] = None
    next_cursor: Optional[str] = Field(
        default=None, description="Opaque cursor to fetch the next page"
    )
    params: Optional[RAGQueryParams] = None
    timing: Optional[RAGTiming] = None
    generated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("generated_at", mode="before")
    @classmethod
    def _ensure_tz(cls, dt: Optional[datetime]) -> datetime:
        if isinstance(dt, datetime) and dt.tzinfo:
            return dt
        return _utcnow()

    @model_validator(mode="after")
    def _sanity(self) -> "RAGResult":
        if self.total_found < len(self.items):
            # keep consistent; total_found should be >= returned items
            object.__setattr__(self, "total_found", len(self.items))
        # enforce monotonic ranks starting at 1
        if self.items:
            expected = list(range(1, len(self.items) + 1))
            ranks = [h.rank for h in self.items]
            if ranks != expected:
                # re-rank in-place by score desc, stable
                sorted_hits = sorted(self.items, key=lambda h: (-h.score, h.entry.created_at))
                for i, h in enumerate(sorted_hits, start=1):
                    object.__setattr__(h, "rank", i)
                object.__setattr__(self, "items", sorted_hits)
        return self
