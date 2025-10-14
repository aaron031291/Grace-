"""Memory service - storage and retrieval of memory entries."""

import hashlib
from typing import Dict, List, Optional

from ..contracts.dto_common import MemoryEntry
from .schemas import MemoryStore, AuditRecord


class MemoryService:
    """Core memory storage and retrieval service."""

    def __init__(self):
        self.store = MemoryStore()

    def store_entry(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        # Calculate content hash if not provided
        if not entry.sha256:
            entry.sha256 = hashlib.sha256(entry.content.encode()).hexdigest()

        # Store the entry
        self.store.entries[entry.id] = entry

        # Create audit record
        audit_record = AuditRecord(
            id=f"audit_{entry.id}",
            memory_id=entry.id,
            action="store",
            payload_hash=entry.sha256,
        )
        self.store.audit_log.append(audit_record)

        return entry.id

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        return self.store.entries.get(memory_id)

    def query(
        self, query_text: str, filters: Optional[Dict] = None, limit: int = 10
    ) -> List[MemoryEntry]:
        """Query memory entries (simple text matching for now)."""
        results = []

        for entry in self.store.entries.values():
            # Simple text search
            if query_text.lower() in entry.content.lower():
                results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_entries_by_filters(self, filters: Dict) -> List[MemoryEntry]:
        """Get entries matching filters."""
        results = []

        for entry in self.store.entries.values():
            match = True

            # Filter by content type
            if (
                "content_type" in filters
                and entry.content_type != filters["content_type"]
            ):
                match = False

            # Filter by tags (using w5h_index.what as tags)
            if "tags" in filters:
                entry_tags = entry.w5h_index.what
                if not any(tag in entry_tags for tag in filters["tags"]):
                    match = False

            if match:
                results.append(entry)

        return results
