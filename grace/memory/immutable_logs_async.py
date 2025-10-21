"""
Async Immutable Logs - Durable append-only storage
"""

from typing import Dict, Any, List, Optional
import asyncio
import json
import hashlib
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AsyncImmutableLogs:
    """
    Async immutable audit logs with durable Postgres storage
    
    Features:
    - Cryptographic chaining
    - Durable persistence
    - Async operations
    - Batch writes
    """
    
    def __init__(self, database_url: str, batch_size: int = 100):
        self.database_url = database_url
        self.batch_size = batch_size
        self.pool = None
        self._buffer: List[Dict[str, Any]] = []
        self._last_hash: Optional[str] = None
        self._flush_task = None
        
    async def connect(self):
        """Connect to Postgres"""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(self.database_url)
            
            # Create immutable logs table
            await self._create_table()
            
            # Load last hash for chaining
            await self._load_last_hash()
            
            # Start auto-flush task
            self._flush_task = asyncio.create_task(self._auto_flush())
            
            logger.info("AsyncImmutableLogs connected and ready")
        except Exception as e:
            logger.error(f"Failed to connect AsyncImmutableLogs: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect and flush remaining logs"""
        if self._flush_task:
            self._flush_task.cancel()
        
        if self._buffer:
            await self._flush_batch()
        
        if self.pool:
            await self.pool.close()
    
    async def _create_table(self):
        """Create immutable logs table"""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS immutable_logs (
                    log_id SERIAL PRIMARY KEY,
                    operation_type VARCHAR(100) NOT NULL,
                    actor VARCHAR(255) NOT NULL,
                    action JSONB NOT NULL,
                    result JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    severity VARCHAR(20) DEFAULT 'info',
                    tags TEXT[],
                    chain_hash VARCHAR(64) NOT NULL,
                    previous_hash VARCHAR(64),
                    verified BOOLEAN DEFAULT TRUE,
                    metadata JSONB
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_immutable_timestamp 
                ON immutable_logs(timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_immutable_actor 
                ON immutable_logs(actor)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_immutable_operation 
                ON immutable_logs(operation_type)
            """)
    
    async def _load_last_hash(self):
        """Load the last chain hash"""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT chain_hash FROM immutable_logs
                ORDER BY log_id DESC
                LIMIT 1
            """)
            
            if row:
                self._last_hash = row['chain_hash']
                logger.info(f"Loaded last hash: {self._last_hash[:16]}...")
    
    async def log(
        self,
        operation_type: str,
        actor: str,
        action: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an immutable entry
        
        Returns the chain hash for verification
        """
        timestamp = datetime.now(timezone.utc)
        
        # Calculate chain hash
        hash_input = f"{operation_type}:{actor}:{json.dumps(action)}:{timestamp.isoformat()}:{self._last_hash or ''}"
        chain_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        entry = {
            "operation_type": operation_type,
            "actor": actor,
            "action": action,
            "result": result,
            "timestamp": timestamp,
            "severity": severity,
            "tags": tags or [],
            "chain_hash": chain_hash,
            "previous_hash": self._last_hash,
            "metadata": metadata
        }
        
        # Add to buffer
        self._buffer.append(entry)
        
        # Update last hash
        self._last_hash = chain_hash
        
        # Flush if buffer full
        if len(self._buffer) >= self.batch_size:
            await self._flush_batch()
        
        return chain_hash
    
    async def _flush_batch(self):
        """Flush buffered logs to Postgres"""
        if not self._buffer or not self.pool:
            return
        
        entries = self._buffer[:]
        self._buffer.clear()
        
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO immutable_logs 
                (operation_type, actor, action, result, timestamp, severity, 
                 tags, chain_hash, previous_hash, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, [
                (
                    entry["operation_type"],
                    entry["actor"],
                    json.dumps(entry["action"]),
                    json.dumps(entry["result"]) if entry["result"] else None,
                    entry["timestamp"],
                    entry["severity"],
                    entry["tags"],
                    entry["chain_hash"],
                    entry["previous_hash"],
                    json.dumps(entry["metadata"]) if entry["metadata"] else None
                )
                for entry in entries
            ])
        
        logger.info(f"Flushed {len(entries)} immutable log entries")
    
    async def _auto_flush(self):
        """Auto-flush buffer every 10 seconds"""
        while True:
            try:
                await asyncio.sleep(10)
                if self._buffer:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-flush error: {e}")
    
    async def verify_chain(
        self,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Verify integrity of the immutable log chain
        
        Returns verification results
        """
        if not self.pool:
            return {"verified": False, "error": "Not connected"}
        
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM immutable_logs ORDER BY log_id"
            params = []
            
            if start_id:
                query += f" WHERE log_id >= ${len(params) + 1}"
                params.append(start_id)
            
            if end_id:
                if start_id:
                    query += f" AND log_id <= ${len(params) + 1}"
                else:
                    query += f" WHERE log_id <= ${len(params) + 1}"
                params.append(end_id)
            
            rows = await conn.fetch(query, *params)
            
            if not rows:
                return {"verified": True, "entries_checked": 0}
            
            # Verify chain
            previous_hash = None
            errors = []
            
            for i, row in enumerate(rows):
                if row['previous_hash'] != previous_hash:
                    errors.append({
                        "log_id": row['log_id'],
                        "error": "Hash chain broken",
                        "expected": previous_hash,
                        "got": row['previous_hash']
                    })
                
                # Recalculate hash
                hash_input = f"{row['operation_type']}:{row['actor']}:{row['action']}:{row['timestamp'].isoformat()}:{previous_hash or ''}"
                expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()
                
                if row['chain_hash'] != expected_hash:
                    errors.append({
                        "log_id": row['log_id'],
                        "error": "Hash mismatch",
                        "expected": expected_hash,
                        "got": row['chain_hash']
                    })
                
                previous_hash = row['chain_hash']
            
            return {
                "verified": len(errors) == 0,
                "entries_checked": len(rows),
                "errors": errors
            }
    
    async def query(
        self,
        operation_type: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query immutable logs"""
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM immutable_logs WHERE 1=1"
            params = []
            
            if operation_type:
                params.append(operation_type)
                query += f" AND operation_type = ${len(params)}"
            
            if actor:
                params.append(actor)
                query += f" AND actor = ${len(params)}"
            
            if severity:
                params.append(severity)
                query += f" AND severity = ${len(params)}"
            
            if tags:
                params.append(tags)
                query += f" AND tags && ${len(params)}"
            
            if start_time:
                params.append(start_time)
                query += f" AND timestamp >= ${len(params)}"
            
            if end_time:
                params.append(end_time)
                query += f" AND timestamp <= ${len(params)}"
            
            params.append(limit)
            query += f" ORDER BY timestamp DESC LIMIT ${len(params)}"
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
