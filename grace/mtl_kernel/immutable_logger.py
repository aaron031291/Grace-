"""Immutable Logger - Blockchain-style audit trail with cryptographic sealing."""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Immutable audit log entry."""
    audit_id: str
    timestamp: datetime
    event_type: str
    component_id: str
    payload: Dict[str, Any]
    trust_score: float
    constitutional_compliance: float
    previous_hash: str
    current_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "component_id": self.component_id,
            "payload": self.payload,
            "trust_score": self.trust_score,
            "constitutional_compliance": self.constitutional_compliance,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash
        }


class ImmutableLogger:
    """
    Blockchain-style audit trail with cryptographic sealing.
    
    Features:
    - Chain-based log structure (each entry references previous hash)
    - Cryptographic sealing (SHA-256)
    - Tamper detection
    - Transparency levels (public, internal, private)
    - Event type taxonomy
    - Constitutional compliance tagging
    - Query interface for audit trails
    """
    
    # Event type taxonomy
    EVENT_TYPES = [
        "MEMORY_STORED",
        "MEMORY_RETRIEVED",
        "DECISION_MADE",
        "TRUST_UPDATED",
        "CONSTITUTIONAL_CHECK",
        "POLICY_VIOLATION",
        "GOVERNANCE_ACTION",
        "USER_INTERACTION",
        "SYSTEM_EVENT"
    ]
    
    # Transparency levels
    TRANSPARENCY_LEVELS = ["public", "internal", "private"]
    
    def __init__(self):
        self._log_chain: List[AuditLog] = []
        self._lock = asyncio.Lock()
        
        # Index for fast lookups
        self._index: Dict[str, int] = {}  # audit_id -> position in chain
        
        # Statistics
        self._stats = {
            "total_logs": 0,
            "integrity_checks": 0,
            "tamper_detections": 0,
            "start_time": time.time()
        }
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info("Immutable Logger initialized with genesis block")
    
    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis_hash = hashlib.sha256(b"GRACE_AUDIT_GENESIS").hexdigest()
        
        genesis = AuditLog(
            audit_id="genesis",
            timestamp=datetime.utcnow(),
            event_type="SYSTEM_EVENT",
            component_id="immutable_logger",
            payload={"message": "Audit chain initialized"},
            trust_score=1.0,
            constitutional_compliance=1.0,
            previous_hash="0" * 64,
            current_hash=genesis_hash
        )
        
        self._log_chain.append(genesis)
        self._index["genesis"] = 0
    
    async def log(
        self,
        event_type: str,
        component_id: str,
        payload: Dict[str, Any],
        trust_score: float = 0.5,
        constitutional_compliance: float = 1.0,
        transparency: str = "internal"
    ) -> str:
        """
        Log an event to the immutable chain.
        
        Args:
            event_type: Type of event (from EVENT_TYPES)
            component_id: ID of component generating event
            payload: Event payload data
            trust_score: Trust score for the event
            constitutional_compliance: Compliance score (0.0-1.0)
            transparency: Transparency level
        
        Returns:
            Audit ID
        """
        if event_type not in self.EVENT_TYPES:
            logger.warning(f"Unknown event type: {event_type}")
        
        if transparency not in self.TRANSPARENCY_LEVELS:
            transparency = "internal"
        
        async with self._lock:
            # Get previous log
            previous_log = self._log_chain[-1] if self._log_chain else None
            previous_hash = previous_log.current_hash if previous_log else "0" * 64
            
            # Generate audit ID
            audit_id = f"audit_{len(self._log_chain)}_{int(time.time() * 1000)}"
            
            # Calculate current hash
            log_content = {
                "audit_id": audit_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "component_id": component_id,
                "payload": payload,
                "trust_score": trust_score,
                "constitutional_compliance": constitutional_compliance,
                "previous_hash": previous_hash,
                "transparency": transparency
            }
            
            current_hash = hashlib.sha256(
                json.dumps(log_content, sort_keys=True).encode()
            ).hexdigest()
            
            # Create log entry
            log_entry = AuditLog(
                audit_id=audit_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                component_id=component_id,
                payload=payload,
                trust_score=trust_score,
                constitutional_compliance=constitutional_compliance,
                previous_hash=previous_hash,
                current_hash=current_hash
            )
            
            # Append to chain
            self._log_chain.append(log_entry)
            self._index[audit_id] = len(self._log_chain) - 1
            
            self._stats["total_logs"] += 1
            
            logger.debug(f"Logged event: {event_type} from {component_id}")
            
            return audit_id
    
    async def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the entire chain.
        
        Returns:
            True if chain is intact, False if tampering detected
        """
        async with self._lock:
            self._stats["integrity_checks"] += 1
            
            for i in range(1, len(self._log_chain)):
                current_log = self._log_chain[i]
                previous_log = self._log_chain[i - 1]
                
                # Verify previous hash reference
                if current_log.previous_hash != previous_log.current_hash:
                    logger.error(f"Chain integrity broken at position {i}")
                    self._stats["tamper_detections"] += 1
                    return False
                
                # Verify current hash
                log_content = {
                    "audit_id": current_log.audit_id,
                    "timestamp": current_log.timestamp.isoformat(),
                    "event_type": current_log.event_type,
                    "component_id": current_log.component_id,
                    "payload": current_log.payload,
                    "trust_score": current_log.trust_score,
                    "constitutional_compliance": current_log.constitutional_compliance,
                    "previous_hash": current_log.previous_hash,
                    "transparency": "internal"  # Default
                }
                
                expected_hash = hashlib.sha256(
                    json.dumps(log_content, sort_keys=True).encode()
                ).hexdigest()
                
                if current_log.current_hash != expected_hash:
                    logger.error(f"Hash mismatch at position {i}")
                    self._stats["tamper_detections"] += 1
                    return False
            
            logger.info("Chain integrity verified successfully")
            return True
    
    async def get_log(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Get audit log by ID."""
        async with self._lock:
            if audit_id in self._index:
                position = self._index[audit_id]
                return self._log_chain[position].to_dict()
            return None
    
    async def query_logs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters.
        
        Args:
            filters: Dict with keys like event_type, component_id
            time_range: Tuple of (start_time, end_time)
            limit: Maximum results to return
        
        Returns:
            List of matching audit logs
        """
        async with self._lock:
            results = []
            
            for log_entry in self._log_chain:
                # Apply time range filter
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= log_entry.timestamp <= end_time):
                        continue
                
                # Apply filters
                if filters:
                    match = True
                    
                    if "event_type" in filters and log_entry.event_type != filters["event_type"]:
                        match = False
                    
                    if "component_id" in filters and log_entry.component_id != filters["component_id"]:
                        match = False
                    
                    if "min_trust_score" in filters and log_entry.trust_score < filters["min_trust_score"]:
                        match = False
                    
                    if not match:
                        continue
                
                results.append(log_entry.to_dict())
                
                if len(results) >= limit:
                    break
            
            return results
    
    async def get_chain_head(self) -> Dict[str, Any]:
        """Get the most recent log entry."""
        async with self._lock:
            if self._log_chain:
                return self._log_chain[-1].to_dict()
            return {}
    
    async def export_audit_trail(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Export audit trail for compliance reporting.
        
        Args:
            time_range: Optional time range filter
        
        Returns:
            Complete audit trail
        """
        return await self.query_logs(time_range=time_range, limit=100000)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        async with self._lock:
            uptime = time.time() - self._stats["start_time"]
            
            # Count by event type
            event_counts = {}
            for log_entry in self._log_chain:
                event_type = log_entry.event_type
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Calculate average compliance
            avg_compliance = 0.0
            if self._log_chain:
                avg_compliance = sum(
                    log.constitutional_compliance 
                    for log in self._log_chain
                ) / len(self._log_chain)
            
            return {
                "total_logs": self._stats["total_logs"],
                "chain_length": len(self._log_chain),
                "integrity_checks": self._stats["integrity_checks"],
                "tamper_detections": self._stats["tamper_detections"],
                "avg_constitutional_compliance": round(avg_compliance, 3),
                "event_type_distribution": event_counts,
                "chain_head_hash": self._log_chain[-1].current_hash if self._log_chain else None,
                "uptime_seconds": round(uptime, 1)
            }
