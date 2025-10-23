"""
Grace AI Immutable Logger - Cryptographically secure audit trail
"""

import asyncio
import hashlib
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from ..config.environment import get_grace_config

logger = logging.getLogger(__name__)


class TransparencyLevel(IntEnum):
    """Transparency levels for log entries."""

    PUBLIC = 0  # Fully public
    DEMOCRATIC_OVERSIGHT = 1  # Democratic oversight only
    GOVERNANCE_INTERNAL = 2  # Internal governance team
    AUDIT_ONLY = 3  # Audit access only
    SECURITY_SENSITIVE = 4  # Security sensitive information


@dataclass
class LogEntry:
    """Immutable log entry structure."""

    log_id: str
    timestamp: datetime
    event_type: str
    component_id: str
    correlation_id: Optional[str]
    event_data: Dict[str, Any]
    transparency_level: TransparencyLevel
    hash_chain: Optional[str] = None
    retention_until: Optional[datetime] = None


class ImmutableLogger:
    """Immutable, cryptographically secure logging system."""

    def __init__(self, storage_dir: str = "logs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.entries: List[LogEntry] = []

        # Hash chain tracking
        self.last_hash = "0" * 64  # Genesis hash
        self.log_count = 0

        # In-memory buffer for recent logs
        self.log_buffer: List[LogEntry] = []
        self.buffer_limit = 10000

        # Batch processing
        self.batch_size = 100
        self.batch_interval = 30  # seconds

        # Retention configuration
        self.config = get_grace_config()
        self.retention_config = self.config["audit_config"]["transparency_levels"]

        logger.info("ImmutableLogger initialized")

    async def start(self):
        """Start the immutable logging system."""
        logger.info("Starting ImmutableLogger...")

        # Start background processing tasks
        asyncio.create_task(self._batch_processor())
        asyncio.create_task(self._retention_cleanup())

        # Log system start
        await self.log_event(
            event_type="system_start",
            component_id="immutable_logger",
            event_data={"system": "immutable_logger", "version": "1.0.0"},
            transparency_level=TransparencyLevel.AUDIT_ONLY,
        )

    async def stop(self):
        """Stop the immutable logging system."""
        # Log system stop
        await self.log_event(
            event_type="system_stop",
            component_id="immutable_logger",
            event_data={"system": "immutable_logger", "final_count": self.log_count},
            transparency_level=TransparencyLevel.AUDIT_ONLY,
        )

        # Flush remaining logs
        await self._flush_batch()

        logger.info("Stopped ImmutableLogger")

    async def log_event(
        self,
        event_type: str,
        component_id: str,
        event_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        transparency_level: TransparencyLevel = TransparencyLevel.GOVERNANCE_INTERNAL,
    ) -> str:
        """
        Log an immutable event with hash chaining.
        Returns the log entry ID.
        """
        # Generate unique log ID
        log_id = f"log_{int(time.time() * 1000000)}_{self.log_count:06d}"

        # Create log entry
        entry = LogEntry(
            log_id=log_id,
            timestamp=datetime.now(),
            event_type=event_type,
            component_id=component_id,
            correlation_id=correlation_id,
            event_data=event_data,
            transparency_level=transparency_level,
        )

        # Calculate retention date
        if transparency_level in self.retention_config:
            retention_days = self.retention_config[transparency_level]["retention_days"]
            entry.retention_until = entry.timestamp + timedelta(days=retention_days)

        # Calculate hash chain
        entry.hash_chain = self._calculate_hash(entry)
        self.last_hash = entry.hash_chain
        self.log_count += 1

        # Add to buffer
        self.log_buffer.append(entry)

        # Limit buffer size
        if len(self.log_buffer) > self.buffer_limit:
            self.log_buffer = self.log_buffer[-self.buffer_limit :]

        # Write to file
        log_file = self.storage_dir / f"immutable_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Logged event: {event_type} with correlation_id: {correlation_id}")
        return log_id

    def get_entries_by_correlation_id(self, correlation_id: str):
        """Retrieve all entries for a specific correlation_id."""
        return [e for e in self.entries if e["correlation_id"] == correlation_id]

    def get_logs_by_correlation(
        self, correlation_id: str, access_level: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all logs associated with a correlation ID (subject to access level)."""
        matching_logs = []

        for entry in self.log_buffer:
            if entry.correlation_id == correlation_id:
                # Check access level
                required_level = self.retention_config.get(
                    entry.transparency_level.name.lower(), {"access_level": 999}
                )["access_level"]

                if access_level >= required_level:
                    matching_logs.append(self._serialize_entry(entry))

        return matching_logs

    def get_recent_logs(
        self,
        component_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        access_level: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get recent logs with optional filtering."""
        filtered_logs = []

        for entry in reversed(self.log_buffer):  # Most recent first
            if len(filtered_logs) >= limit:
                break

            # Apply filters
            if component_id and entry.component_id != component_id:
                continue
            if event_type and entry.event_type != event_type:
                continue

            # Check access level
            required_level = self.retention_config.get(
                entry.transparency_level.name.lower(), {"access_level": 999}
            )["access_level"]

            if access_level >= required_level:
                filtered_logs.append(self._serialize_entry(entry))

        return filtered_logs

    def verify_hash_chain(self, start_index: int = 0) -> Dict[str, Any]:
        """Verify the integrity of the hash chain."""
        if not self.log_buffer:
            return {"valid": True, "message": "No logs to verify"}

        start_idx = max(0, min(start_index, len(self.log_buffer) - 1))
        previous_hash = (
            "0" * 64 if start_idx == 0 else self.log_buffer[start_idx - 1].hash_chain
        )

        for i in range(start_idx, len(self.log_buffer)):
            entry = self.log_buffer[i]
            expected_hash = self._calculate_hash(entry, previous_hash)

            if entry.hash_chain != expected_hash:
                return {
                    "valid": False,
                    "message": f"Hash chain broken at index {i}",
                    "entry_id": entry.log_id,
                    "expected_hash": expected_hash,
                    "actual_hash": entry.hash_chain,
                }

            previous_hash = entry.hash_chain

        return {
            "valid": True,
            "message": f"Hash chain verified for {len(self.log_buffer) - start_idx} entries",
            "verified_count": len(self.log_buffer) - start_idx,
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        transparency_counts = {}
        component_counts = {}

        for entry in self.log_buffer:
            # Count by transparency level
            level_name = entry.transparency_level.name.lower()
            transparency_counts[level_name] = transparency_counts.get(level_name, 0) + 1

            # Count by component
            component_counts[entry.component_id] = (
                component_counts.get(entry.component_id, 0) + 1
            )

        return {
            "total_logs": len(self.log_buffer),
            "total_logged": self.log_count,
            "transparency_distribution": transparency_counts,
            "component_distribution": component_counts,
            "last_hash": self.last_hash,
            "buffer_utilization": len(self.log_buffer) / self.buffer_limit,
        }

    def _calculate_hash(
        self, entry: LogEntry, previous_hash: Optional[str] = None
    ) -> str:
        """Calculate hash for a log entry including chain."""
        if previous_hash is None:
            previous_hash = self.last_hash

        # Create hash input
        hash_input = {
            "log_id": entry.log_id,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "component_id": entry.component_id,
            "correlation_id": entry.correlation_id,
            "event_data": entry.event_data,
            "transparency_level": entry.transparency_level.value,
            "previous_hash": previous_hash,
        }

        # Convert to JSON and hash
        hash_string = json.dumps(hash_input, sort_keys=True, default=str)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def _serialize_entry(self, entry: LogEntry) -> Dict[str, Any]:
        """Serialize a log entry for external use."""
        return {
            "log_id": entry.log_id,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "component_id": entry.component_id,
            "correlation_id": entry.correlation_id,
            "event_data": entry.event_data,
            "transparency_level": entry.transparency_level.name.lower(),
            "hash_chain": entry.hash_chain,
            "retention_until": entry.retention_until.isoformat()
            if entry.retention_until
            else None,
        }

    async def _batch_processor(self):
        """Process log batches for persistent storage."""
        while True:
            try:
                await asyncio.sleep(self.batch_interval)
                await self._flush_batch()

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _flush_batch(self):
        """Flush current batch to storage."""
        if not self.log_buffer:
            return

        # For now, just log the batch size (would integrate with actual storage)
        logger.debug(f"Would flush {len(self.log_buffer)} log entries to storage")

        # If we had a storage backend, we would write here:
        # if self.storage_backend:
        #     await self.storage_backend.write_batch(self.log_buffer)

    async def _retention_cleanup(self):
        """Clean up expired log entries based on retention policy."""
        while True:
            try:
                now = datetime.now()
                initial_count = len(self.log_buffer)

                # Remove expired entries
                self.log_buffer = [
                    entry
                    for entry in self.log_buffer
                    if entry.retention_until is None or entry.retention_until > now
                ]

                removed_count = initial_count - len(self.log_buffer)
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} expired log entries")

                # Run cleanup daily
                await asyncio.sleep(86400)  # 24 hours

            except Exception as e:
                logger.error(f"Error in retention cleanup: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
