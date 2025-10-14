"""Trust service (Trust Ledger) - manages trust attestations."""

import hashlib
from typing import Dict, List

from ..contracts.dto_common import TrustAttestation
from .schemas import MemoryStore, AuditRecord


class TrustService:
    """Trust ledger for managing attestations and trust scores."""

    def __init__(self, memory_store: MemoryStore):
        self.store = memory_store

    def init_trust(self, memory_id: str) -> str:
        """Initialize trust record for a new memory entry."""
        if memory_id not in self.store.trust_records:
            self.store.trust_records[memory_id] = []

        # Create audit record for trust initialization
        audit_id = f"trust_init_{memory_id}"
        audit_record = AuditRecord(
            id=audit_id,
            memory_id=memory_id,
            action="trust_init",
            payload_hash=hashlib.sha256(f"trust_init:{memory_id}".encode()).hexdigest(),
        )
        self.store.audit_log.append(audit_record)

        return audit_id

    def attest(self, memory_id: str, delta: Dict, attestor: str = "system") -> str:
        """Create a trust attestation."""
        attestation = TrustAttestation(
            memory_id=memory_id,
            delta=delta,
            attestor=attestor,
            confidence=delta.get("confidence", 0.8),
        )

        # Store attestation
        if memory_id not in self.store.trust_records:
            self.store.trust_records[memory_id] = []
        self.store.trust_records[memory_id].append(attestation)

        # Create audit record
        audit_record = AuditRecord(
            id=attestation.id,
            memory_id=memory_id,
            action="attest",
            payload_hash=hashlib.sha256(str(delta).encode()).hexdigest(),
        )
        self.store.audit_log.append(audit_record)

        return attestation.id

    def get_trust_score(self, memory_id: str) -> float:
        """Calculate aggregated trust score for a memory entry."""
        attestations = self.store.trust_records.get(memory_id, [])

        if not attestations:
            return 0.5  # Neutral trust

        # Simple weighted average
        total_score = 0.0
        total_weight = 0.0

        for attestation in attestations:
            weight = attestation.confidence
            score = attestation.delta.get("trust_score", 0.5)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def get_attestations(self, memory_id: str) -> List[TrustAttestation]:
        """Get all attestations for a memory entry."""
        return self.store.trust_records.get(memory_id, [])
