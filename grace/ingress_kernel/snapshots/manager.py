"""
Snapshot Manager - Manages versioned snapshots for Ingress state and rollback capabilities.
"""
import json
import logging
import hashlib
import asyncio
from datetime import datetime, timedelta
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid

from grace.contracts.ingress_contracts import IngressSnapshot


logger = logging.getLogger(__name__)


def generate_snapshot_id() -> str:
    """Generate a snapshot ID."""
    timestamp = utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')
    return f"ing_{timestamp}"


class SnapshotManager:
    """Manages versioned snapshots of Ingress Kernel state."""
    
    def __init__(self, storage_path: str = "/tmp/ingress_snapshots"):
        """
        Initialize snapshot manager.
        
        Args:
            storage_path: Base path for snapshot storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # In-memory snapshot cache
        self.snapshots: Dict[str, IngressSnapshot] = {}
        
        # Current state tracking
        self.current_state = {
            "active_sources": [],
            "registry_hash": "",
            "parser_versions": {"html": "1.3.0", "pdf": "2.1.4", "asr.en": "2.4.1"},
            "dedupe_threshold": 0.87,
            "pii_policy_defaults": "mask",
            "offsets": {},
            "watermarks": {},
            "gold_views_version": "1.2.0"
        }
        
        # Auto-snapshot configuration
        self.auto_snapshot_interval = timedelta(hours=6)  # Every 6 hours
        self.last_auto_snapshot = utc_now()
        self.max_snapshots = 100
        
        # Performance metrics for rollback decisions
        self.performance_history: List[Dict[str, Any]] = []
        self.performance_window = timedelta(hours=1)
    
    async def create_snapshot(self, state: Optional[Dict[str, Any]] = None, 
                            manual: bool = False) -> IngressSnapshot:
        """
        Create a snapshot of current Ingress state.
        
        Args:
            state: Optional state override
            manual: Whether this is a manual snapshot
            
        Returns:
            Created snapshot
        """
        try:
            if state:
                self.current_state.update(state)
            
            snapshot_id = generate_snapshot_id()
            if manual:
                snapshot_id = f"{snapshot_id}_manual"
            
            # Create snapshot object
            snapshot = IngressSnapshot(
                snapshot_id=snapshot_id,
                active_sources=self.current_state["active_sources"].copy(),
                registry_hash=self._compute_registry_hash(),
                parser_versions=self.current_state["parser_versions"].copy(),
                dedupe_threshold=self.current_state["dedupe_threshold"],
                pii_policy_defaults=self.current_state["pii_policy_defaults"],
                offsets=self.current_state["offsets"].copy(),
                watermarks=self.current_state["watermarks"].copy(),
                gold_views_version=self.current_state["gold_views_version"],
                hash=""  # Will be calculated
            )
            
            # Calculate and set hash
            snapshot.hash = self._calculate_snapshot_hash(snapshot)
            
            # Store in memory and disk
            self.snapshots[snapshot_id] = snapshot
            await self._persist_snapshot(snapshot)
            
            # Cleanup old snapshots
            await self._cleanup_old_snapshots()
            
            logger.info(f"Created snapshot: {snapshot_id}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise
    
    async def load_snapshot(self, snapshot_id: str) -> bool:
        """
        Load a specific snapshot and restore state.
        
        Args:
            snapshot_id: ID of snapshot to load
            
        Returns:
            Success status
        """
        try:
            # Get snapshot
            snapshot = self.snapshots.get(snapshot_id)
            if not snapshot:
                snapshot = await self._load_snapshot_from_disk(snapshot_id)
            
            if not snapshot:
                logger.error(f"Snapshot not found: {snapshot_id}")
                return False
            
            # Verify snapshot integrity
            if not await self._verify_snapshot(snapshot):
                logger.error(f"Snapshot verification failed: {snapshot_id}")
                return False
            
            # Restore state
            await self._restore_from_snapshot(snapshot)
            
            logger.info(f"Loaded snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return False
    
    async def get_latest_snapshot(self) -> Optional[IngressSnapshot]:
        """Get the most recent snapshot."""
        if not self.snapshots:
            await self._load_snapshots_from_disk()
        
        if not self.snapshots:
            return None
        
        # Sort by creation time
        latest = max(self.snapshots.values(), key=lambda s: s.created_at)
        return latest
    
    def list_snapshots(self, limit: int = 10) -> List[IngressSnapshot]:
        """
        List recent snapshots.
        
        Args:
            limit: Maximum number of snapshots to return
            
        Returns:
            List of snapshots sorted by creation time (newest first)
        """
        snapshots = list(self.snapshots.values())
        snapshots.sort(key=lambda s: s.created_at, reverse=True)
        return snapshots[:limit]
    
    async def rollback_to_snapshot(self, snapshot_id: str, 
                                 verify_hash: bool = True) -> bool:
        """
        Rollback to a specific snapshot with optional hash verification.
        
        Args:
            snapshot_id: ID of snapshot to rollback to
            verify_hash: Whether to verify snapshot hash
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Initiating rollback to snapshot: {snapshot_id}")
            
            # Load snapshot
            if not await self.load_snapshot(snapshot_id):
                return False
            
            # Create pre-rollback snapshot for safety
            pre_rollback = await self.create_snapshot(manual=True)
            logger.info(f"Created pre-rollback snapshot: {pre_rollback.snapshot_id}")
            
            # Perform rollback
            success = await self._perform_rollback(snapshot_id)
            
            if success:
                logger.info(f"Successfully rolled back to snapshot: {snapshot_id}")
            else:
                logger.error(f"Rollback failed for snapshot: {snapshot_id}")
                # Could attempt to restore pre-rollback snapshot here
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback to {snapshot_id} failed: {e}")
            return False
    
    async def should_rollback(self, current_metrics: Dict[str, Any]) -> tuple[bool, str]:
        """
        Determine if system should rollback based on performance metrics.
        
        Args:
            current_metrics: Current system performance metrics
            
        Returns:
            (should_rollback, reason) tuple
        """
        try:
            # Add current metrics to history
            self.performance_history.append({
                "timestamp": utc_now(),
                "metrics": current_metrics
            })
            
            # Clean old history
            cutoff = utc_now() - self.performance_window
            self.performance_history = [
                h for h in self.performance_history 
                if h["timestamp"] > cutoff
            ]
            
            if len(self.performance_history) < 10:  # Need enough data points
                return False, "Insufficient performance history"
            
            # Calculate recent performance trends
            recent_errors = sum(
                h["metrics"].get("error_rate", 0) 
                for h in self.performance_history[-5:]
            ) / 5
            
            recent_latency = sum(
                h["metrics"].get("avg_latency_ms", 0) 
                for h in self.performance_history[-5:]
            ) / 5
            
            historical_errors = sum(
                h["metrics"].get("error_rate", 0) 
                for h in self.performance_history[:-5]
            ) / max(1, len(self.performance_history) - 5)
            
            historical_latency = sum(
                h["metrics"].get("avg_latency_ms", 0) 
                for h in self.performance_history[:-5]
            ) / max(1, len(self.performance_history) - 5)
            
            # Rollback conditions
            error_threshold_multiplier = 3.0
            latency_threshold_multiplier = 2.0
            absolute_error_threshold = 0.20  # 20% error rate
            absolute_latency_threshold = 5000  # 5 seconds
            
            # Check conditions
            if recent_errors > absolute_error_threshold:
                return True, f"High error rate: {recent_errors:.2%}"
            
            if recent_latency > absolute_latency_threshold:
                return True, f"High latency: {recent_latency:.0f}ms"
            
            if recent_errors > historical_errors * error_threshold_multiplier:
                return True, f"Error rate spike: {recent_errors:.2%} vs {historical_errors:.2%}"
            
            if recent_latency > historical_latency * latency_threshold_multiplier:
                return True, f"Latency spike: {recent_latency:.0f}ms vs {historical_latency:.0f}ms"
            
            return False, "Performance within acceptable ranges"
            
        except Exception as e:
            logger.error(f"Rollback decision analysis failed: {e}")
            return False, f"Analysis error: {str(e)}"
    
    async def auto_snapshot_if_needed(self):
        """Create automatic snapshot if interval has passed."""
        if utc_now() - self.last_auto_snapshot > self.auto_snapshot_interval:
            await self.create_snapshot()
            self.last_auto_snapshot = utc_now()
    
    def update_current_state(self, updates: Dict[str, Any]):
        """Update current state tracking."""
        self.current_state.update(updates)
    
    def _compute_registry_hash(self) -> str:
        """Compute hash of current registry state."""
        # Mock implementation - would hash actual registry data
        registry_data = {
            "sources_count": len(self.current_state.get("active_sources", [])),
            "timestamp": iso_format()
        }
        registry_json = json.dumps(registry_data, sort_keys=True)
        return hashlib.sha256(registry_json.encode()).hexdigest()[:16]
    
    def _calculate_snapshot_hash(self, snapshot: IngressSnapshot) -> str:
        """Calculate hash for snapshot integrity."""
        snapshot_dict = snapshot.dict()
        snapshot_dict.pop('hash', None)  # Remove hash field for calculation
        snapshot_dict.pop('created_at', None)  # Exclude timestamp
        
        snapshot_json = json.dumps(snapshot_dict, sort_keys=True)
        return hashlib.sha256(snapshot_json.encode()).hexdigest()
    
    async def _persist_snapshot(self, snapshot: IngressSnapshot):
        """Persist snapshot to disk."""
        snapshot_file = self.storage_path / f"{snapshot.snapshot_id}.json"
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.dict(), f, indent=2, default=str)
        
        logger.debug(f"Persisted snapshot to {snapshot_file}")
    
    async def _load_snapshot_from_disk(self, snapshot_id: str) -> Optional[IngressSnapshot]:
        """Load snapshot from disk."""
        try:
            snapshot_file = self.storage_path / f"{snapshot_id}.json"
            
            if not snapshot_file.exists():
                return None
            
            with open(snapshot_file, 'r') as f:
                data = json.load(f)
            
            # Convert string timestamps back to datetime
            if isinstance(data.get('created_at'), str):
                data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            
            if isinstance(data.get('watermarks'), dict):
                for key, value in data['watermarks'].items():
                    if isinstance(value, str):
                        data['watermarks'][key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            
            snapshot = IngressSnapshot(**data)
            self.snapshots[snapshot_id] = snapshot
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id} from disk: {e}")
            return None
    
    async def _load_snapshots_from_disk(self):
        """Load all snapshots from disk."""
        try:
            for snapshot_file in self.storage_path.glob("*.json"):
                snapshot_id = snapshot_file.stem
                if snapshot_id not in self.snapshots:
                    await self._load_snapshot_from_disk(snapshot_id)
                    
        except Exception as e:
            logger.error(f"Failed to load snapshots from disk: {e}")
    
    async def _verify_snapshot(self, snapshot: IngressSnapshot) -> bool:
        """Verify snapshot integrity."""
        try:
            calculated_hash = self._calculate_snapshot_hash(snapshot)
            return calculated_hash == snapshot.hash
            
        except Exception as e:
            logger.error(f"Snapshot verification failed: {e}")
            return False
    
    async def _restore_from_snapshot(self, snapshot: IngressSnapshot):
        """Restore system state from snapshot."""
        # Update current state
        self.current_state.update({
            "active_sources": snapshot.active_sources,
            "registry_hash": snapshot.registry_hash,
            "parser_versions": snapshot.parser_versions,
            "dedupe_threshold": snapshot.dedupe_threshold,
            "pii_policy_defaults": snapshot.pii_policy_defaults,
            "offsets": snapshot.offsets,
            "watermarks": snapshot.watermarks,
            "gold_views_version": snapshot.gold_views_version
        })
        
        logger.info(f"Restored state from snapshot {snapshot.snapshot_id}")
    
    async def _perform_rollback(self, snapshot_id: str) -> bool:
        """Perform actual rollback operations."""
        try:
            # In a real implementation, this would:
            # 1. Stop active ingestion jobs
            # 2. Reset connector offsets
            # 3. Reload parser configurations
            # 4. Update policy settings
            # 5. Restart services with snapshot state
            
            logger.info(f"Mock rollback performed for snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback operations failed: {e}")
            return False
    
    async def _cleanup_old_snapshots(self):
        """Clean up old snapshots to maintain storage limits."""
        if len(self.snapshots) <= self.max_snapshots:
            return
        
        # Sort by creation time and keep most recent
        snapshots_by_time = sorted(
            self.snapshots.values(), 
            key=lambda s: s.created_at,
            reverse=True
        )
        
        # Remove excess snapshots
        for snapshot in snapshots_by_time[self.max_snapshots:]:
            snapshot_id = snapshot.snapshot_id
            
            # Don't delete manual snapshots or very recent ones
            if ("manual" in snapshot_id or 
                utc_now() - snapshot.created_at < timedelta(hours=24)):
                continue
            
            # Remove from memory and disk
            self.snapshots.pop(snapshot_id, None)
            
            snapshot_file = self.storage_path / f"{snapshot_id}.json"
            if snapshot_file.exists():
                snapshot_file.unlink()
            
            logger.debug(f"Cleaned up old snapshot: {snapshot_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get snapshot manager statistics."""
        return {
            "total_snapshots": len(self.snapshots),
            "storage_path": str(self.storage_path),
            "auto_snapshot_interval_hours": self.auto_snapshot_interval.total_seconds() / 3600,
            "last_auto_snapshot": self.last_auto_snapshot.isoformat(),
            "performance_history_size": len(self.performance_history),
            "current_state_keys": list(self.current_state.keys())
        }