"""
Persistent Memory System for Grace

Grace's long-term memory that persists across sessions.
Stores:
- All task executions and outcomes
- Chat history
- Learned patterns
- Domain knowledge
- Successful strategies
- Code examples
- Multi-modal data (text, code, PDFs, web content, audio, video)

This is what makes Grace remember and learn over time.
"""

import asyncio
import logging
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry"""
    entry_id: str
    entry_type: str  # task_execution, chat_message, pattern, code, document
    content: Dict[str, Any]
    domain: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None  # For semantic search
    trust_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class PersistentMemory:
    """
    Grace's persistent memory system.
    
    Stores everything Grace learns and experiences.
    Enables:
    - Learning from past executions
    - Building autonomous capabilities
    - Reducing LLM dependence
    - Continuous knowledge accumulation
    """
    
    def __init__(self, db_path: str = "./data/grace_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
        logger.info(f"Persistent Memory initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize memory database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Memory entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_entries (
                entry_id TEXT PRIMARY KEY,
                entry_type TEXT NOT NULL,
                content TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                trust_score REAL DEFAULT 0.5,
                embedding TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Learned patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                pattern_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                last_used TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Domain knowledge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_knowledge (
                domain TEXT PRIMARY KEY,
                encounter_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                status TEXT DEFAULT 'new',
                last_updated TEXT
            )
        ''')
        
        # Ingested documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ingested_documents (
                document_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_url TEXT,
                title TEXT,
                content TEXT NOT NULL,
                metadata TEXT,
                ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER,
                indexed BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("  ✅ Memory database initialized")
    
    async def store_execution(
        self,
        task: Any,
        decision: Any,
        result: Dict[str, Any]
    ):
        """Store task execution in memory"""
        entry = MemoryEntry(
            entry_id=task.task_id,
            entry_type="task_execution",
            content={
                "task": task.description,
                "domain": task.domain,
                "intelligence_source": decision.intelligence_source.value,
                "result": str(result)[:1000],  # Truncate
                "success": result.get("success", False)
            },
            domain=task.domain,
            timestamp=datetime.utcnow(),
            metadata={
                "confidence": decision.confidence,
                "llm_used": decision.use_llm
            }
        )
        
        await self.store(entry)
        
        # Update domain knowledge
        await self._update_domain_knowledge(
            task.domain,
            result.get("success", False)
        )
    
    async def store(self, entry: MemoryEntry):
        """Store memory entry"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO memory_entries
            (entry_id, entry_type, content, domain, timestamp, metadata, trust_score, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.entry_id,
            entry.entry_type,
            json.dumps(entry.content),
            entry.domain,
            entry.timestamp.isoformat(),
            json.dumps(entry.metadata),
            entry.trust_score,
            json.dumps(entry.embedding) if entry.embedding else None
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored memory: {entry.entry_id} ({entry.entry_type})")
    
    async def search(
        self,
        query: str,
        domain: Optional[str] = None,
        entry_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memory"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        sql = "SELECT * FROM memory_entries WHERE 1=1"
        params = []
        
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        
        if entry_type:
            sql += " AND entry_type = ?"
            params.append(entry_type)
        
        if query:
            sql += " AND (content LIKE ? OR metadata LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to MemoryEntry objects
        entries = []
        for row in rows:
            entry = MemoryEntry(
                entry_id=row['entry_id'],
                entry_type=row['entry_type'],
                content=json.loads(row['content']),
                domain=row['domain'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                metadata=json.loads(row['metadata']),
                trust_score=row['trust_score']
            )
            entries.append(entry)
        
        return entries
    
    async def store_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store chat message for persistent history"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        message_id = hashlib.sha256(
            f"{session_id}{role}{content}{datetime.utcnow()}".encode()
        ).hexdigest()[:16]
        
        cursor.execute('''
            INSERT INTO chat_history
            (message_id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            message_id,
            session_id,
            role,
            content,
            datetime.utcnow().isoformat(),
            json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored chat message: {session_id} ({role})")
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            messages.append({
                "message_id": row['message_id'],
                "role": row['role'],
                "content": row['content'],
                "timestamp": row['timestamp'],
                "metadata": json.loads(row['metadata'])
            })
        
        return list(reversed(messages))  # Return in chronological order
    
    async def store_pattern(self, pattern: Dict[str, Any]):
        """Store learned pattern"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        pattern_id = hashlib.sha256(
            json.dumps(pattern, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        cursor.execute('''
            INSERT OR REPLACE INTO learned_patterns
            (pattern_id, domain, pattern_type, pattern_data, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            pattern.get("domain", "general"),
            pattern.get("type", "general"),
            json.dumps(pattern),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored pattern: {pattern_id}")
    
    async def get_patterns(
        self,
        domain: Optional[str] = None,
        min_success_rate: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve learned patterns"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        sql = "SELECT * FROM learned_patterns WHERE success_rate >= ?"
        params = [min_success_rate]
        
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        
        sql += " ORDER BY success_rate DESC"
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "pattern_id": row['pattern_id'],
                "domain": row['domain'],
                "pattern_type": row['pattern_type'],
                "data": json.loads(row['pattern_data']),
                "success_rate": row['success_rate']
            }
            for row in rows
        ]
    
    async def _update_domain_knowledge(self, domain: str, success: bool):
        """Update domain knowledge statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get current stats
        cursor.execute(
            "SELECT encounter_count, success_count FROM domain_knowledge WHERE domain = ?",
            (domain,)
        )
        row = cursor.fetchone()
        
        if row:
            encounter_count = row[0] + 1
            success_count = row[1] + (1 if success else 0)
        else:
            encounter_count = 1
            success_count = 1 if success else 0
        
        success_rate = success_count / encounter_count if encounter_count > 0 else 0.0
        
        # Determine status
        if encounter_count >= 100 and success_rate >= 0.90:
            status = "mastered"
        elif encounter_count >= 10 and success_rate >= 0.85:
            status = "established"
        elif encounter_count >= 1:
            status = "learning"
        else:
            status = "new"
        
        # Update
        cursor.execute('''
            INSERT OR REPLACE INTO domain_knowledge
            (domain, encounter_count, success_count, success_rate, status, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            domain,
            encounter_count,
            success_count,
            success_rate,
            status,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def ingest_document(
        self,
        source_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Ingest document into memory.
        
        Supports: PDF, web pages, code files, books, etc.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        doc_id = hashlib.sha256(
            f"{source_type}{content[:100]}{datetime.utcnow()}".encode()
        ).hexdigest()[:16]
        
        cursor.execute('''
            INSERT INTO ingested_documents
            (document_id, source_type, source_url, title, content, metadata, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            source_type,
            metadata.get("url"),
            metadata.get("title", "Untitled"),
            content,
            json.dumps(metadata),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Ingested document: {doc_id} ({source_type})")
        
        return doc_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Count entries by type
        cursor.execute("SELECT entry_type, COUNT(*) FROM memory_entries GROUP BY entry_type")
        entry_counts = dict(cursor.fetchall())
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM memory_entries")
        total_entries = cursor.fetchone()[0]
        
        # Domain stats
        cursor.execute("SELECT COUNT(*), AVG(success_rate) FROM domain_knowledge")
        domain_stats = cursor.fetchone()
        
        # Chat messages
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        total_messages = cursor.fetchone()[0]
        
        # Documents
        cursor.execute("SELECT COUNT(*), SUM(chunk_count) FROM ingested_documents")
        doc_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_memory_entries": total_entries,
            "entries_by_type": entry_counts,
            "total_domains": domain_stats[0] or 0,
            "avg_domain_success_rate": domain_stats[1] or 0.0,
            "total_chat_messages": total_messages,
            "total_documents_ingested": doc_stats[0] or 0,
            "total_document_chunks": doc_stats[1] or 0
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🧠 Persistent Memory Demo\n")
        
        memory = PersistentMemory()
        
        # Store some memories
        entry1 = MemoryEntry(
            entry_id="mem_001",
            entry_type="task_execution",
            content={"task": "Create API", "result": "Success"},
            domain="python_api",
            timestamp=datetime.utcnow(),
            metadata={"quality": 0.9}
        )
        
        await memory.store(entry1)
        print("✅ Stored task execution")
        
        # Store chat
        await memory.store_chat_message(
            session_id="session_001",
            role="user",
            content="How do I build an API?"
        )
        await memory.store_chat_message(
            session_id="session_001",
            role="assistant",
            content="I'll help you build an API using FastAPI..."
        )
        print("✅ Stored chat history")
        
        # Store pattern
        await memory.store_pattern({
            "domain": "python_api",
            "type": "fastapi_crud",
            "approach": "Use SQLAlchemy with async",
            "success": True
        })
        print("✅ Stored learned pattern")
        
        # Search memory
        results = await memory.search("API", domain="python_api")
        print(f"\n✅ Found {len(results)} relevant memories")
        
        # Get stats
        stats = memory.get_stats()
        print(f"\n📊 Memory Stats:")
        print(f"  Total entries: {stats['total_memory_entries']}")
        print(f"  Chat messages: {stats['total_chat_messages']}")
        print(f"  Domains tracked: {stats['total_domains']}")
    
    asyncio.run(demo())
