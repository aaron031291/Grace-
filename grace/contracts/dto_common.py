"""Common DTOs and base types."""

from datetime import datetime
from grace.utils.time import now_utc
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class BaseDTO(BaseModel):
    """Base DTO with common fields."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=now_utc)
    metadata: Optional[Dict[str, Any]] = None


class W5HIndex(BaseModel):
    """Who/What/When/Where/Why/How indexing."""

    who: List[str] = Field(default_factory=list)
    what: List[str] = Field(default_factory=list)
    when: Optional[datetime] = None
    where: List[str] = Field(default_factory=list)
    why: List[str] = Field(default_factory=list)
    how: List[str] = Field(default_factory=list)


class MemoryEntry(BaseDTO):
    """Core memory entry."""

    content: str
    content_type: str = "text/plain"
    w5h_index: W5HIndex = Field(default_factory=W5HIndex)
    embedding: Optional[List[float]] = None
    sha256: Optional[str] = None


class TrustAttestation(BaseDTO):
    """Trust attestation record."""

    memory_id: str
    delta: Dict[str, Any]
    attestor: str
    confidence: float = Field(ge=0.0, le=1.0)


class TriggerEvent(BaseDTO):
    """Trigger event record."""

    event_type: str
    source: str
    target_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
