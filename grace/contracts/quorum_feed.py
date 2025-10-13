"""Quorum feed contracts."""

from typing import Any, Dict, List
from pydantic import Field
from .dto_common import BaseDTO


class QuorumFeedItem(BaseDTO):
    """A single item in a quorum feed."""

    memory_id: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)


class QuorumResult(BaseDTO):
    """Result of quorum consensus."""

    consensus: str
    confidence: float = Field(ge=0.0, le=1.0)
    participant_count: int
    agreement_level: float = Field(ge=0.0, le=1.0)
    dissenting_views: List[str] = Field(default_factory=list)
    evidence: List[QuorumFeedItem] = Field(default_factory=list)
