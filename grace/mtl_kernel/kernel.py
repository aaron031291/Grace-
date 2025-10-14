"""MTL Kernel - Main orchestration for Memory, Trust, Learning."""

from typing import Dict, List, Optional

from ..contracts.dto_common import MemoryEntry
from ..contracts.governed_decision import GovernedDecision
from .schemas import MemoryStore
from .memory_service import MemoryService
from .trust_service import TrustService
from .immutable_log_service import ImmutableLogService
from .trigger_ledger import TriggerLedger
from .w5h_indexer import W5HIndexer
from .librarian import Librarian
from .llm.kernel import LLMKernel


class MTLKernel:
    """Main MTL kernel orchestrating all memory, trust, and learning operations."""

    def __init__(self):
        # Initialize shared store
        self.store = MemoryStore()

        # Initialize services with shared store
        self.memory_service = MemoryService()
        # Share the store instance
        self.memory_service.store = self.store

        self.trust_service = TrustService(self.store)
        self.immutable_log = ImmutableLogService(self.store)
        self.trigger_ledger = TriggerLedger(self.store)
        self.w5h_indexer = W5HIndexer()
        self.llm_kernel = LLMKernel()
        self.librarian = Librarian(self.memory_service, self.llm_kernel)

    def write(self, entry: MemoryEntry) -> str:
        """Write entry with full MTL fan-out: store → trust.init → immutable.append → trigger.record."""
        # 1. Store in memory service
        memory_id = self.memory_service.store_entry(entry)

        # 2. Initialize trust record
        trust_id = self.trust_service.init_trust(memory_id)

        # 3. Append to immutable log
        log_id = self.immutable_log.append(
            memory_id=memory_id, action="write", payload_hash=entry.sha256 or "no_hash"
        )

        # 4. Record trigger event
        trigger_id = self.trigger_ledger.record(
            event_type="memory_write",
            source="mtl_kernel",
            target_id=memory_id,
            payload={"trust_id": trust_id, "log_id": log_id},
        )

        return memory_id

    def recall(
        self, query: str, *, filters: Optional[Dict] = None
    ) -> List[MemoryEntry]:
        """Recall memories using advanced search and ranking."""
        if filters:
            # Use filtered search
            results = self.memory_service.get_entries_by_filters(filters)

            # Apply text query to filtered results
            if query:
                filtered_results = []
                for entry in results:
                    if query.lower() in entry.content.lower():
                        filtered_results.append(entry)
                results = filtered_results
        else:
            # Use librarian's search and rank
            results = self.librarian.search_and_rank(query)

        return results

    def attest(self, memory_id: str, delta: Dict) -> str:
        """Create trust attestation with audit trail."""
        # Create attestation
        attestation_id = self.trust_service.attest(memory_id, delta)

        # Log attestation
        log_id = self.immutable_log.append(
            memory_id=memory_id,
            action="attest",
            payload_hash=delta.get("hash", "no_hash"),
        )

        # Record trigger
        trigger_id = self.trigger_ledger.record(
            event_type="trust_attest",
            source="mtl_kernel",
            target_id=memory_id,
            payload={"attestation_id": attestation_id, "log_id": log_id},
        )

        return attestation_id

    def feed_for_quorum(self, filters: Dict) -> List[str]:
        """Generate feed of memory IDs for quorum consensus."""
        entries = self.memory_service.get_entries_by_filters(filters)

        # Sort by trust score and recency
        scored_entries = []
        for entry in entries:
            trust_score = self.trust_service.get_trust_score(entry.id)
            scored_entries.append((trust_score, entry.created_at, entry.id))

        # Sort by trust score descending, then by recency
        scored_entries.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Return top memory IDs
        return [entry_id for _, _, entry_id in scored_entries[:10]]

    def store_decision(self, decision: GovernedDecision) -> str:
        """Store governance decision as memory entry."""
        decision_entry = MemoryEntry(
            content=f"Governance Decision: {decision.reasoning}",
            content_type="application/json",
            metadata={
                "decision_id": decision.id,
                "approved": decision.approved,
                "confidence": decision.confidence,
                "type": "governance_decision",
            },
        )

        # Store with full fan-out
        memory_id = self.write(decision_entry)

        # Add high-trust attestation for governance decisions
        self.attest(
            memory_id,
            {
                "trust_score": 0.9,
                "confidence": 1.0,
                "source": "governance_kernel",
                "hash": decision.id,
            },
        )

        return memory_id

    def get_audit_proof(self, audit_id: str):
        """Get Merkle proof for audit record."""
        return self.immutable_log.proof(audit_id)

    def get_stats(self) -> Dict:
        """Get comprehensive MTL kernel statistics."""
        return {
            "memory_entries": len(self.store.entries),
            "trust_records": sum(
                len(attestations) for attestations in self.store.trust_records.values()
            ),
            "audit_records": len(self.store.audit_log),
            "trigger_events": len(self.store.trigger_events),
            "merkle_root": self.store.get_merkle_root(),
            "services": {
                "memory": True,
                "trust": True,
                "immutable_log": True,
                "trigger_ledger": True,
                "librarian": True,
                "llm": True,
            },
        }
