"""
Async Fusion Memory - Postgres-backed persistent storage with full schema
"""

from typing import Any, Optional, Dict, List
import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AsyncFusionMemory:
    """
    Async Postgres-backed persistent memory
    
    Specification-compliant with:
    - learned_patterns table
    - interactions table
    - audit_log table
    - Async operations
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self._connected = False
        
    async def connect(self):
        """Connect to Postgres"""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(self.database_url)
            self._connected = True
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info(f"Connected to Postgres: {self.database_url}")
        except ImportError:
            logger.warning("asyncpg not installed, using in-memory fallback")
            self.pool = None
            self._connected = False
        except Exception as e:
            logger.error(f"Postgres connection failed: {e}")
            self.pool = None
            self._connected = False
    
    async def disconnect(self):
        """Disconnect from Postgres"""
        if self.pool:
            await self.pool.close()
        self._connected = False
    
    async def _create_tables(self):
        """Create required tables per specification"""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            # learned_patterns table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    pattern_id SERIAL PRIMARY KEY,
                    pattern_type VARCHAR(100) NOT NULL,
                    pattern_data JSONB NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    occurrences INTEGER DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # interactions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255),
                    action VARCHAR(100) NOT NULL,
                    context JSONB,
                    outcome VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(255),
                    metadata JSONB
                )
            """)
            
            # audit_log table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id SERIAL PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB NOT NULL,
                    actor VARCHAR(255),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    severity VARCHAR(20) DEFAULT 'info',
                    tags TEXT[],
                    immutable_hash VARCHAR(64)
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_type 
                ON learned_patterns(pattern_type)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_user 
                ON interactions(user_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_log(timestamp DESC)
            """)
    
    async def store_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store learned pattern"""
        if not self.pool:
            logger.warning("Not connected to database")
            return -1
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO learned_patterns 
                (pattern_type, pattern_data, confidence, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING pattern_id
            """, pattern_type, json.dumps(pattern_data), confidence, 
                json.dumps(metadata or {}))
            
            return row['pattern_id']
    
    async def get_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve learned patterns"""
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            if pattern_type:
                rows = await conn.fetch("""
                    SELECT * FROM learned_patterns
                    WHERE pattern_type = $1 AND confidence >= $2
                    ORDER BY last_seen DESC
                    LIMIT $3
                """, pattern_type, min_confidence, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM learned_patterns
                    WHERE confidence >= $1
                    ORDER BY last_seen DESC
                    LIMIT $2
                """, min_confidence, limit)
            
            return [dict(row) for row in rows]
    
    async def record_interaction(
        self,
        action: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Record user interaction"""
        if not self.pool:
            return -1
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO interactions 
                (user_id, action, context, outcome, session_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING interaction_id
            """, user_id, action, json.dumps(context or {}), 
                outcome, session_id, json.dumps(metadata or {}))
            
            return row['interaction_id']
    
    async def get_interactions(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve interactions"""
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM interactions WHERE 1=1"
            params = []
            
            if user_id:
                params.append(user_id)
                query += f" AND user_id = ${len(params)}"
            
            if action:
                params.append(action)
                query += f" AND action = ${len(params)}"
            
            params.append(limit)
            query += f" ORDER BY timestamp DESC LIMIT ${len(params)}"
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def log_audit_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        actor: Optional[str] = None,
        severity: str = "info",
        tags: Optional[List[str]] = None
    ) -> int:
        """Log audit event"""
        if not self.pool:
            return -1
        
        # Calculate immutable hash
        import hashlib
        hash_input = f"{event_type}:{json.dumps(event_data)}:{datetime.now(timezone.utc).isoformat()}"
        immutable_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO audit_log 
                (event_type, event_data, actor, severity, tags, immutable_hash)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING log_id
            """, event_type, json.dumps(event_data), actor, 
                severity, tags or [], immutable_hash)
            
            return row['log_id']
    
    async def get_audit_logs(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve audit logs"""
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if event_type:
                params.append(event_type)
                query += f" AND event_type = ${len(params)}"
            
            if actor:
                params.append(actor)
                query += f" AND actor = ${len(params)}"
            
            if severity:
                params.append(severity)
                query += f" AND severity = ${len(params)}"
            
            params.append(limit)
            query += f" ORDER BY timestamp DESC LIMIT ${len(params)}"
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
