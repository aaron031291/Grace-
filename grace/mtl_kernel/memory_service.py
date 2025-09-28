"""Memory service for storing and retrieving memory entries."""

from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
import json
from pathlib import Path

from .schemas import MemoryEntry


class MemoryService:
    """Service for managing memory storage and retrieval."""
    
    def __init__(self, db_path: str = "grace_memory.db"):
        """Initialize memory service with database."""
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT,
                    source TEXT,
                    w5h_index TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
                ON memories(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_source 
                ON memories(source)
            """)
    
    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, metadata, timestamp, source, w5h_index)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.content,
                json.dumps(entry.metadata),
                entry.timestamp.isoformat(),
                entry.source,
                json.dumps(entry.w5h_index)
            ))
        return entry.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?", 
                (memory_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
                
            return MemoryEntry(
                id=row['id'],
                content=row['content'],
                metadata=json.loads(row['metadata'] or '{}'),
                timestamp=datetime.fromisoformat(row['timestamp']),
                source=row['source'],
                w5h_index=json.loads(row['w5h_index'] or '{}')
            )
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """Search memory entries."""
        conditions = ["content LIKE ?"]
        params = [f"%{query}%"]
        
        if filters:
            if filters.get('source'):
                conditions.append("source = ?")
                params.append(filters['source'])
            
            if filters.get('since'):
                conditions.append("timestamp >= ?")
                params.append(filters['since'])
        
        sql = f"""
            SELECT * FROM memories 
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
        """
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(sql, params):
                results.append(MemoryEntry(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata'] or '{}'),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    source=row['source'],
                    w5h_index=json.loads(row['w5h_index'] or '{}')
                ))
        
        return results
    
    def list_all(self, limit: int = 100) -> List[MemoryEntry]:
        """List all memories with optional limit."""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", 
                (limit,)
            ):
                results.append(MemoryEntry(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata'] or '{}'),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    source=row['source'],
                    w5h_index=json.loads(row['w5h_index'] or '{}')
                ))
        
        return results