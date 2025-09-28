"""Core data schemas for the MTL kernel."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import hashlib
import json


class MemoryEntry(BaseModel):
    """Core memory entry structure."""
    id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    w5h_index: Dict[str, Any] = Field(default_factory=dict)
    
    def generate_id(self) -> str:
        """Generate a deterministic ID from content."""
        content_hash = hashlib.sha256(
            json.dumps({
                "content": self.content,
                "timestamp": self.timestamp.isoformat(),
                "source": self.source
            }, sort_keys=True).encode()
        ).hexdigest()
        return f"mem_{content_hash[:16]}"

    def model_post_init(self, __context: Any) -> None:
        """Set ID after validation."""
        if not self.id:
            self.id = self.generate_id()


class TrustRecord(BaseModel):
    """Trust attestation record."""
    memory_id: str
    delta: Dict[str, Any]
    attestor: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trust_score: float = Field(ge=0.0, le=1.0)
    
    
class ImmutableLogEntry(BaseModel):
    """Immutable audit log entry."""
    id: str
    memory_id: str
    operation: str  # 'create', 'attest', 'recall'
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hash_chain: Optional[str] = None
    
    
class TriggerEvent(BaseModel):
    """Event trigger record."""
    memory_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)


class W5HIndex(BaseModel):
    """Who, What, When, Where, Why, How indexing structure."""
    who: List[str] = Field(default_factory=list)
    what: List[str] = Field(default_factory=list)
    when: Optional[datetime] = None
    where: List[str] = Field(default_factory=list)
    why: List[str] = Field(default_factory=list)
    how: List[str] = Field(default_factory=list)