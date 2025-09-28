"""
Grace Dead Letter Queue (DLQ) - Handle failed message delivery and provide retry mechanisms.
"""
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import sqlite3
from pathlib import Path

from .envelope import GraceMessageEnvelope, Priority


logger = logging.getLogger(__name__)


class DLQReason(str, Enum):
    """Reasons for messages being sent to DLQ."""
    ROUTE_NOT_FOUND = "route_not_found"
    HANDLER_ERROR = "handler_error"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"
    VALIDATION_ERROR = "validation_error"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    POISON_MESSAGE = "poison_message"


class DLQEntry:
    """Dead letter queue entry."""
    
    def __init__(self,
                 envelope: GraceMessageEnvelope,
                 reason: DLQReason,
                 error_details: Optional[str] = None,
                 retry_count: int = 0,
                 max_retries: int = 3):
        self.id = f"dlq_{envelope.id}"
        self.envelope = envelope
        self.reason = reason
        self.error_details = error_details
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.created_at = datetime.utcnow()
        self.last_retry_at: Optional[datetime] = None
        self.next_retry_at = self._calculate_next_retry()
        
    def _calculate_next_retry(self) -> datetime:
        """Calculate next retry time using exponential backoff."""
        base_delay = 60  # 1 minute base
        delay_minutes = base_delay * (2 ** self.retry_count)
        return datetime.utcnow() + timedelta(minutes=min(delay_minutes, 60))  # Max 1 hour
    
    def can_retry(self) -> bool:
        """Check if entry can be retried."""
        return (self.retry_count < self.max_retries and 
                datetime.utcnow() >= self.next_retry_at)
    
    def increment_retry(self):
        """Increment retry count and update timestamps."""
        self.retry_count += 1
        self.last_retry_at = datetime.utcnow()
        self.next_retry_at = self._calculate_next_retry()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for storage."""
        return {
            'id': self.id,
            'envelope_json': self.envelope.model_dump_json(),
            'reason': self.reason.value,
            'error_details': self.error_details,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at.isoformat(),
            'last_retry_at': self.last_retry_at.isoformat() if self.last_retry_at else None,
            'next_retry_at': self.next_retry_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DLQEntry':
        """Create entry from dictionary."""
        envelope_data = json.loads(data['envelope_json'])
        envelope = GraceMessageEnvelope(**envelope_data)
        
        entry = cls(
            envelope=envelope,
            reason=DLQReason(data['reason']),
            error_details=data.get('error_details'),
            retry_count=data['retry_count'],
            max_retries=data['max_retries']
        )
        
        entry.id = data['id']
        entry.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_retry_at'):
            entry.last_retry_at = datetime.fromisoformat(data['last_retry_at'])
        entry.next_retry_at = datetime.fromisoformat(data['next_retry_at'])
        
        return entry


class DeadLetterQueue:
    """Grace Dead Letter Queue for failed message handling."""
    
    def __init__(self, db_path: str = "/tmp/grace_dlq.db", max_queue_size: int = 10000):
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.retry_queue: deque = deque()
        self.poison_messages: Dict[str, int] = {}
        self.dlq_stats = {
            'total_entries': 0,
            'retries_attempted': 0,
            'successful_retries': 0,
            'permanent_failures': 0
        }
        
        # Initialize database
        self._init_db()
        
        # Load existing entries
        self._load_entries()
    
    def _init_db(self):
        """Initialize SQLite database for DLQ persistence."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dlq_entries (
                    id TEXT PRIMARY KEY,
                    envelope_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    error_details TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    last_retry_at TEXT,
                    next_retry_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_next_retry 
                ON dlq_entries(next_retry_at, status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_reason 
                ON dlq_entries(reason)
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize DLQ database: {e}")
    
    def _load_entries(self):
        """Load pending entries from database into retry queue."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT * FROM dlq_entries 
                WHERE status = 'pending' 
                ORDER BY next_retry_at ASC
                LIMIT 1000
            """)
            
            for row in cursor.fetchall():
                entry = DLQEntry.from_dict(dict(row))
                if entry.can_retry():
                    self.retry_queue.append(entry)
            
            conn.close()
            logger.info(f"Loaded {len(self.retry_queue)} entries into DLQ retry queue")
            
        except Exception as e:
            logger.error(f"Failed to load DLQ entries: {e}")
    
    async def add_message(self,
                         envelope: GraceMessageEnvelope,
                         reason: DLQReason,
                         error_details: Optional[str] = None,
                         max_retries: int = 3) -> bool:
        """Add a failed message to the dead letter queue."""
        try:
            # Check for poison messages (too many failures)
            poison_count = self.poison_messages.get(envelope.id, 0)
            if poison_count >= 5:
                reason = DLQReason.POISON_MESSAGE
                max_retries = 0
            
            # Create DLQ entry
            entry = DLQEntry(
                envelope=envelope,
                reason=reason,
                error_details=error_details,
                max_retries=max_retries
            )
            
            # Store in database
            if await self._store_entry(entry):
                # Add to retry queue if retryable
                if entry.can_retry():
                    self.retry_queue.append(entry)
                
                self.dlq_stats['total_entries'] += 1
                self.poison_messages[envelope.id] = poison_count + 1
                
                logger.warning(f"Message {envelope.id} added to DLQ: {reason.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add message to DLQ: {e}")
            return False
    
    async def _store_entry(self, entry: DLQEntry) -> bool:
        """Store DLQ entry in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check queue size limit
            cursor = conn.execute("SELECT COUNT(*) FROM dlq_entries WHERE status = 'pending'")
            count = cursor.fetchone()[0]
            
            if count >= self.max_queue_size:
                # Remove oldest entries
                conn.execute("""
                    DELETE FROM dlq_entries 
                    WHERE status = 'pending' 
                    AND id IN (
                        SELECT id FROM dlq_entries 
                        WHERE status = 'pending' 
                        ORDER BY created_at ASC 
                        LIMIT 100
                    )
                """)
            
            # Insert new entry
            entry_data = entry.to_dict()
            conn.execute("""
                INSERT INTO dlq_entries (
                    id, envelope_json, reason, error_details, 
                    retry_count, max_retries, created_at, 
                    last_retry_at, next_retry_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                entry_data['id'],
                entry_data['envelope_json'],
                entry_data['reason'],
                entry_data['error_details'],
                entry_data['retry_count'],
                entry_data['max_retries'],
                entry_data['created_at'],
                entry_data['last_retry_at'],
                entry_data['next_retry_at']
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store DLQ entry: {e}")
            return False
    
    async def process_retries(self, retry_handler: Callable[[GraceMessageEnvelope], bool]) -> int:
        """Process retry queue with provided handler."""
        processed = 0
        current_time = datetime.utcnow()
        
        # Process eligible retries
        while self.retry_queue and processed < 100:  # Limit processing per batch
            entry = self.retry_queue.popleft()
            
            if current_time < entry.next_retry_at:
                # Not ready for retry, put back
                self.retry_queue.appendleft(entry)
                break
            
            try:
                # Attempt retry
                success = await retry_handler(entry.envelope)
                self.dlq_stats['retries_attempted'] += 1
                
                if success:
                    # Retry successful, mark as completed
                    await self._update_entry_status(entry.id, 'completed')
                    self.dlq_stats['successful_retries'] += 1
                    logger.info(f"DLQ retry successful for message {entry.envelope.id}")
                else:
                    # Retry failed
                    entry.increment_retry()
                    
                    if entry.retry_count < entry.max_retries:
                        # Can retry again
                        await self._update_entry(entry)
                        self.retry_queue.append(entry)
                    else:
                        # Max retries exceeded
                        await self._update_entry_status(entry.id, 'failed')
                        self.dlq_stats['permanent_failures'] += 1
                        logger.error(f"DLQ max retries exceeded for message {entry.envelope.id}")
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processing DLQ retry: {e}")
                entry.increment_retry()
                await self._update_entry(entry)
                self.retry_queue.append(entry)
                processed += 1
        
        return processed
    
    async def _update_entry(self, entry: DLQEntry):
        """Update DLQ entry in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            entry_data = entry.to_dict()
            conn.execute("""
                UPDATE dlq_entries SET
                    retry_count = ?,
                    last_retry_at = ?,
                    next_retry_at = ?
                WHERE id = ?
            """, (
                entry_data['retry_count'],
                entry_data['last_retry_at'],
                entry_data['next_retry_at'],
                entry.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update DLQ entry: {e}")
    
    async def _update_entry_status(self, entry_id: str, status: str):
        """Update entry status in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute("""
                UPDATE dlq_entries SET status = ? WHERE id = ?
            """, (status, entry_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update entry status: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get counts by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM dlq_entries 
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Get counts by reason
            cursor = conn.execute("""
                SELECT reason, COUNT(*) as count 
                FROM dlq_entries 
                WHERE status = 'pending'
                GROUP BY reason
            """)
            reason_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                **self.dlq_stats,
                'retry_queue_size': len(self.retry_queue),
                'poison_messages': len(self.poison_messages),
                'status_counts': status_counts,
                'reason_counts': reason_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get DLQ stats: {e}")
            return self.dlq_stats
    
    async def cleanup_old_entries(self, max_age_days: int = 7):
        """Clean up old DLQ entries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute("""
                DELETE FROM dlq_entries 
                WHERE created_at < ? AND status IN ('completed', 'failed')
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old DLQ entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup DLQ entries: {e}")
            return 0