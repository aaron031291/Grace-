"""Immutable log service - Merkle tree-based audit trail."""
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

from .schemas import MemoryStore, AuditRecord, MerkleProof


class ImmutableLogService:
    """Immutable audit log with Merkle proofs."""
    
    def __init__(self, memory_store: MemoryStore):
        self.store = memory_store
    
    def append(self, memory_id: str, action: str, payload_hash: str, actor: str = "system") -> str:
        """Append entry to immutable log."""
        audit_id = f"log_{len(self.store.audit_log)}_{memory_id}"
        
        audit_record = AuditRecord(
            id=audit_id,
            memory_id=memory_id,
            action=action,
            actor=actor,
            payload_hash=payload_hash
        )
        
        self.store.audit_log.append(audit_record)
        
        # Update proof for the new record (simplified)
        audit_record.merkle_proof = self.store.create_proof(audit_id)
        
        return audit_id
    
    def proof(self, audit_id: str) -> Optional[MerkleProof]:
        """Get Merkle proof for an audit record."""
        return self.store.create_proof(audit_id)
    
    def verify_proof(self, proof: MerkleProof, content: str) -> bool:
        """Verify a Merkle proof."""
        # Simplified verification for development
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return proof.leaf_hash == content_hash
    
    def get_audit_trail(self, memory_id: str) -> List[AuditRecord]:
        """Get complete audit trail for a memory entry."""
        return [record for record in self.store.audit_log if record.memory_id == memory_id]
    
    def get_record(self, audit_id: str) -> Optional[AuditRecord]:
        """Get specific audit record."""
        for record in self.store.audit_log:
            if record.id == audit_id:
                return record
        return None