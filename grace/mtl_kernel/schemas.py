"""MTL Kernel schemas and data models."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import hashlib
import json

from ..contracts.dto_common import MemoryEntry, W5HIndex, TrustAttestation, TriggerEvent


class MerkleProof(BaseModel):
    """Merkle proof for immutable log verification."""
    leaf_hash: str
    tree_root: str
    proof_path: List[str]
    leaf_index: int
    tree_size: int


class AuditRecord(BaseModel):
    """Immutable audit record."""
    id: str
    memory_id: str
    action: str  # write, attest, recall, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor: str = "system"
    payload_hash: str
    merkle_proof: Optional[MerkleProof] = None


class MemoryStore(BaseModel):
    """In-memory store for development."""
    entries: Dict[str, MemoryEntry] = Field(default_factory=dict)
    trust_records: Dict[str, List[TrustAttestation]] = Field(default_factory=dict)
    audit_log: List[AuditRecord] = Field(default_factory=list)
    trigger_events: List[TriggerEvent] = Field(default_factory=list)
    
    def get_merkle_root(self) -> str:
        """Calculate current Merkle root of audit log."""
        if not self.audit_log:
            return "empty_tree_root"
        
        # Simple hash chain for development (not a true Merkle tree)
        combined = "".join(record.payload_hash for record in self.audit_log)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def create_proof(self, audit_id: str) -> Optional[MerkleProof]:
        """Create Merkle proof for audit record."""
        for i, record in enumerate(self.audit_log):
            if record.id == audit_id:
                return MerkleProof(
                    leaf_hash=record.payload_hash,
                    tree_root=self.get_merkle_root(),
                    proof_path=[],  # Simplified for development
                    leaf_index=i,
                    tree_size=len(self.audit_log)
                )
        return None