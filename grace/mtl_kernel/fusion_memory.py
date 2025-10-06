"""Fusion Memory - Long-term storage with structured query support (<50ms operations)."""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionEntry:
    """Entry in Fusion long-term storage."""
    id: str
    table: str
    data: Dict[str, Any]
    created_at: datetime
    trust_score: float
    content_hash: str
    metadata: Dict[str, Any]


class FusionMemory:
    """
    Long-term storage with PostgreSQL-like semantics.
    
    Performance target: <50ms for queries
    Features:
    - Structured schema: patterns, interactions, decisions, precedents
    - Trust score storage
    - Constitutional validation metadata
    - Time-range queries
    - Full-text search support
    - Transaction support
    """
    
    # Define table schemas
    TABLES = {
        "learned_patterns": {
            "columns": ["pattern_type", "pattern_data", "confidence", "usage_count"]
        },
        "interactions": {
            "columns": ["interaction_type", "user_id", "content", "response", "sentiment"]
        },
        "governance_decisions": {
            "columns": ["decision_type", "context", "outcome", "reasoning", "approved"]
        },
        "constitutional_precedents": {
            "columns": ["article", "interpretation", "context", "binding"]
        },
        "trust_ledger": {
            "columns": ["entity_id", "trust_score", "event_type", "delta", "reason"]
        }
    }
    
    def __init__(self, storage_path: str = "/tmp/fusion_memory"):
        # In-memory storage (SQLite/PostgreSQL would be used in production)
        self._storage: Dict[str, List[FusionEntry]] = {
            table: [] for table in self.TABLES.keys()
        }
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_inserts": 0,
            "total_queries": 0,
            "total_updates": 0,
            "total_deletes": 0,
            "avg_query_time_ms": 0,
            "start_time": time.time()
        }
        
        logger.info(f"Fusion Memory initialized with tables: {list(self.TABLES.keys())}")
    
    async def insert(self, table: str, data: Dict[str, Any], trust_score: float = 0.5) -> str:
        """Insert data into table (<50ms target)."""
        start = time.time()
        
        if table not in self.TABLES:
            raise ValueError(f"Table '{table}' not found")
        
        async with self._lock:
            # Generate ID and hash
            entry_id = f"{table}_{len(self._storage[table])}_{int(time.time() * 1000)}"
            content_hash = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            
            entry = FusionEntry(
                id=entry_id,
                table=table,
                data=data,
                created_at=datetime.utcnow(),
                trust_score=trust_score,
                content_hash=content_hash,
                metadata={}
            )
            
            self._storage[table].append(entry)
            self._stats["total_inserts"] += 1
            
            elapsed = (time.time() - start) * 1000
            if elapsed > 50:
                logger.warning(f"Fusion insert exceeded 50ms: {elapsed:.2f}ms")
            
            return entry_id
    
    async def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """Query table with filters (<50ms target)."""
        start = time.time()
        
        if table not in self.TABLES:
            raise ValueError(f"Table '{table}' not found")
        
        async with self._lock:
            results = []
            
            for entry in self._storage[table]:
                # Apply time range filter
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= entry.created_at <= end_time):
                        continue
                
                # Apply data filters
                if filters:
                    match = all(
                        entry.data.get(key) == value
                        for key, value in filters.items()
                    )
                    if not match:
                        continue
                
                results.append({
                    "id": entry.id,
                    "data": entry.data,
                    "created_at": entry.created_at.isoformat(),
                    "trust_score": entry.trust_score,
                    "content_hash": entry.content_hash
                })
                
                if len(results) >= limit:
                    break
            
            self._stats["total_queries"] += 1
            
            elapsed = (time.time() - start) * 1000
            
            # Update average query time
            total_queries = self._stats["total_queries"]
            self._stats["avg_query_time_ms"] = (
                (self._stats["avg_query_time_ms"] * (total_queries - 1) + elapsed) 
                / total_queries
            )
            
            if elapsed > 50:
                logger.warning(f"Fusion query exceeded 50ms: {elapsed:.2f}ms")
            
            return results
    
    async def update(self, table: str, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update entry by ID."""
        if table not in self.TABLES:
            raise ValueError(f"Table '{table}' not found")
        
        async with self._lock:
            for entry in self._storage[table]:
                if entry.id == entry_id:
                    entry.data.update(updates)
                    entry.content_hash = hashlib.sha256(
                        json.dumps(entry.data, sort_keys=True).encode()
                    ).hexdigest()
                    self._stats["total_updates"] += 1
                    return True
            return False
    
    async def delete(self, table: str, entry_id: str) -> bool:
        """Delete entry by ID."""
        if table not in self.TABLES:
            raise ValueError(f"Table '{table}' not found")
        
        async with self._lock:
            for i, entry in enumerate(self._storage[table]):
                if entry.id == entry_id:
                    del self._storage[table][i]
                    self._stats["total_deletes"] += 1
                    return True
            return False
    
    async def transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute multiple operations in a transaction."""
        async with self._lock:
            # Store backup
            backup = {table: list(entries) for table, entries in self._storage.items()}
            
            try:
                for op in operations:
                    op_type = op.get("type")
                    
                    if op_type == "insert":
                        await self.insert(op["table"], op["data"], op.get("trust_score", 0.5))
                    elif op_type == "update":
                        await self.update(op["table"], op["entry_id"], op["updates"])
                    elif op_type == "delete":
                        await self.delete(op["table"], op["entry_id"])
                    else:
                        raise ValueError(f"Unknown operation type: {op_type}")
                
                return True
                
            except Exception as e:
                # Rollback on error
                logger.error(f"Transaction failed, rolling back: {e}")
                self._storage = backup
                return False
    
    async def search_precedents(self, context: str) -> List[Dict[str, Any]]:
        """Search constitutional precedents by context."""
        async with self._lock:
            results = []
            
            for entry in self._storage.get("constitutional_precedents", []):
                # Simple text search (would use full-text search in production)
                if context.lower() in str(entry.data).lower():
                    results.append({
                        "id": entry.id,
                        "data": entry.data,
                        "created_at": entry.created_at.isoformat(),
                        "trust_score": entry.trust_score
                    })
            
            return results
    
    async def get_trust_history(
        self,
        entity_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get trust history for entity."""
        return await self.query(
            "trust_ledger",
            filters={"entity_id": entity_id},
            limit=limit
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        async with self._lock:
            table_sizes = {
                table: len(entries) 
                for table, entries in self._storage.items()
            }
            
            total_entries = sum(table_sizes.values())
            uptime = time.time() - self._stats["start_time"]
            
            return {
                "total_entries": total_entries,
                "table_sizes": table_sizes,
                "total_inserts": self._stats["total_inserts"],
                "total_queries": self._stats["total_queries"],
                "total_updates": self._stats["total_updates"],
                "total_deletes": self._stats["total_deletes"],
                "avg_query_time_ms": round(self._stats["avg_query_time_ms"], 2),
                "uptime_seconds": round(uptime, 1)
            }
