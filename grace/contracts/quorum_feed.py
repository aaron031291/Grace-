"""
quorum_contracts.py
Production-grade Quorum feed + result contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import Field, field_validator
from .dto_common import BaseDTO


class QuorumFeedItem(BaseDTO):
    """A single item in a quorum feed."""
    memory_id: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id")
    @classmethod
    def _nonempty_memid(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("memory_id cannot be empty")
        return v


class QuorumResult(BaseDTO):
    """Result of quorum consensus."""
    consensus: str
    confidence: float = Field(ge=0.0, le=1.0)
    participant_count: int = Field(ge=1)
    agreement_level: float = Field(ge=0.0, le=1.0)
    dissenting_views: List[str] = Field(default_factory=list)
    evidence: List[QuorumFeedItem] = Field(default_factory=list)

    @field_validator("consensus")
    @classmethod
    def _nonempty_consensus(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("consensus cannot be empty")
        return v

    @field_validator("dissenting_views", mode="after")
    @classmethod
    def _norm_dissent(cls, v: List[str]) -> List[str]:
        out, seen = [], set()
        for s in v:
            if isinstance(s, str):
                s2 = s.strip()
                if s2 and s2 not in seen:
                    seen.add(s2)
                    out.append(s2)
        return out
