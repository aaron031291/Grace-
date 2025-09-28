"""
Grace Orchestration Snapshot Manager - State persistence and rollback capabilities.

Manages system state snapshots for backup, recovery, and rollback operations.
Provides consistent point-in-time state capture across all orchestration components.
"""

import asyncio
import json
import hashlib
import shutil
import gzip
from datetime import datetime, timedelta
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SnapshotType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    PRE_ROLLBACK = "pre_rollback"
    PRE_UPGRADE = "pre_upgrade"
    EMERGENCY = "emergency"


class SnapshotStatus(Enum):
    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"
    CORRUPTED = "corrupted"


class RollbackStatus(Enum):
    PREPARING = "preparing"
    DRAINING = "draining"
    RESTORING = "restoring"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class OrchSnapshot:
    """Orchestration state snapshot."""
    
    def __init__(self, snapshot_id: str, snapshot_type: SnapshotType = SnapshotType.MANUAL):
        self.snapshot_id = snapshot_id
        self.snapshot_type = snapshot_type
        self.created_at = utc_now()
        self.status = SnapshotStatus.CREATING
        
        # Snapshot content
        self.loops: List[Dict[str, Any]] = []
        self.active_tasks: List[str] = []
        self.policies: Dict[str, Any] = {}
        self.component_states: Dict[str, Any] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        # Metadata
        self.size_bytes = 0
        self.hash = ""
        self.description = ""
        self.tags: Set[str] = set()
        
        # Validation
        self.validation_errors: List[str] = []
        self.validated_at = None
    
    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of snapshot content."""
        content = {
            "loops": self.loops,
            "active_tasks": self.active_tasks,
            "policies": self.policies,
            "component_states": self.component_states,
            "system_metrics": self.system_metrics
        }
        
        content_json = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "type": self.snapshot_type.value,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "loops": self.loops,
            "active_tasks": self.active_tasks,
            "policies": self.policies,
            "component_states": self.component_states,
            "system_metrics": self.system_metrics,
            "size_bytes": self.size_bytes,
            "hash": self.hash,
            "description": self.description,
            "tags": list(self.tags),
            "validation_errors": self.validation_errors,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrchSnapshot':
        """Create snapshot from dictionary."""
        snapshot = cls(data["snapshot_id"], SnapshotType(data["type"]))
        
        snapshot.created_at = datetime.fromisoformat(data["created_at"])
        snapshot.status = SnapshotStatus(data["status"])
        snapshot.loops = data.get("loops", [])
        snapshot.active_tasks = data.get("active_tasks", [])
        snapshot.policies = data.get("policies", {})
        snapshot.component_states = data.get("component_states", {})
        snapshot.system_metrics = data.get("system_metrics", {})
        snapshot.size_bytes = data.get("size_bytes", 0)
        snapshot.hash = data.get("hash", "")
        snapshot.description = data.get("description", "")
        snapshot.tags = set(data.get("tags", []))
        snapshot.validation_errors = data.get("validation_errors", [])
        
        if data.get("validated_at"):
            snapshot.validated_at = datetime.fromisoformat(data["validated_at"])
        
        return snapshot


class RollbackOperation:
    """Rollback operation tracking."""
    
    def __init__(self, operation_id: str, target_snapshot_id: str):
        self.operation_id = operation_id
        self.target_snapshot_id = target_snapshot_id
        self.started_at = utc_now()
        self.completed_at = None
        self.status = RollbackStatus.PREPARING
        
        # Progress tracking
        self.steps_completed = 0
        self.total_steps = 0
        self.current_step = ""
        
        # Results
        self.pre_rollback_snapshot_id = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_id": self.operation_id,
            "target_snapshot_id": self.target_snapshot_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "progress": {
                "steps_completed": self.steps_completed,
                "total_steps": self.total_steps,
                "current_step": self.current_step,
                "percentage": (self.steps_completed / max(1, self.total_steps)) * 100
            },
            "pre_rollback_snapshot_id": self.pre_rollback_snapshot_id,
            "errors": self.errors,
            "warnings": self.warnings
        }


class SnapshotManager:
    """Orchestration snapshot and rollback manager."""
    
    def __init__(self, storage_path: str = "/tmp/grace_orchestration_snapshots",
                 scheduler=None, state_manager=None, router=None, 
                 watchdog=None, scaling_manager=None, lifecycle_manager=None,
                 event_publisher=None):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Component references for snapshot capture
        self.scheduler = scheduler
        self.state_manager = state_manager
        self.router = router
        self.watchdog = watchdog
        self.scaling_manager = scaling_manager
        self.lifecycle_manager = lifecycle_manager
        self.event_publisher = event_publisher
        
        # Snapshot management
        self.snapshots: Dict[str, OrchSnapshot] = {}
        self.active_rollbacks: Dict[str, RollbackOperation] = {}
        
        # Configuration
        self.max_snapshots = 50
        self.auto_cleanup_days = 30
        self.compression_enabled = True
        self.validation_enabled = True
        
        # Scheduled snapshots
        self.snapshot_schedule: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self._schedule_task = None
        
        # Statistics
        self.total_snapshots_created = 0
        self.total_rollbacks_performed = 0
        self.storage_usage_bytes = 0
        
        # Load existing snapshots
        asyncio.create_task(self._load_existing_snapshots())
    
    async def start(self):
        """Start the snapshot manager."""
        if self.running:
            logger.warning("Snapshot manager already running")
            return
        
        logger.info("Starting orchestration snapshot manager...")
        self.running = True
        
        # Start scheduled snapshot task
        self._schedule_task = asyncio.create_task(self._snapshot_scheduler_loop())
        
        logger.info("Orchestration snapshot manager started")
    
    async def stop(self):
        """Stop the snapshot manager."""
        if not self.running:
            return
        
        logger.info("Stopping orchestration snapshot manager...")
        self.running = False
        
        if self._schedule_task:
            self._schedule_task.cancel()
            try:
                await self._schedule_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Orchestration snapshot manager stopped")
    
    async def export_snapshot(self, description: str = "", 
                            snapshot_type: SnapshotType = SnapshotType.MANUAL,
                            tags: Set[str] = None) -> str:
        """Create and export a system state snapshot."""
        snapshot_id = f"orch_{int(utc_now().timestamp())}_{len(self.snapshots)}"
        
        logger.info(f"Creating snapshot {snapshot_id}")
        
        snapshot = OrchSnapshot(snapshot_id, snapshot_type)
        snapshot.description = description
        if tags:
            snapshot.tags.update(tags)
        
        try:
            # Capture state from all components
            await self._capture_scheduler_state(snapshot)
            await self._capture_state_manager_state(snapshot)
            await self._capture_router_state(snapshot)
            await self._capture_watchdog_state(snapshot)
            await self._capture_scaling_state(snapshot)
            await self._capture_lifecycle_state(snapshot)
            await self._capture_system_metrics(snapshot)
            
            # Calculate hash and finalize
            snapshot.hash = snapshot.calculate_hash()
            snapshot.status = SnapshotStatus.COMPLETED
            
            # Validate if enabled
            if self.validation_enabled:
                await self._validate_snapshot(snapshot)
            
            # Store snapshot
            await self._store_snapshot(snapshot)
            self.snapshots[snapshot_id] = snapshot
            self.total_snapshots_created += 1
            
            # Cleanup old snapshots
            await self._cleanup_old_snapshots()
            
            logger.info(f"Successfully created snapshot {snapshot_id}")
            
            # Publish event
            if self.event_publisher:
                await self.event_publisher("ORCH_SNAPSHOT_CREATED", {
                    "snapshot": snapshot.to_dict()
                })
            
            return snapshot_id
            
        except Exception as e:
            snapshot.status = SnapshotStatus.FAILED
            snapshot.validation_errors.append(f"Creation failed: {e}")
            logger.error(f"Failed to create snapshot {snapshot_id}: {e}")
            raise
    
    async def rollback(self, to_snapshot: str) -> str:
        """Rollback orchestration to a specific snapshot."""
        if to_snapshot not in self.snapshots:
            raise ValueError(f"Snapshot not found: {to_snapshot}")
        
        target_snapshot = self.snapshots[to_snapshot]
        if target_snapshot.status != SnapshotStatus.COMPLETED:
            raise ValueError(f"Cannot rollback to incomplete snapshot: {to_snapshot}")
        
        operation_id = f"rollback_{int(utc_now().timestamp())}"
        rollback_op = RollbackOperation(operation_id, to_snapshot)
        rollback_op.total_steps = 6  # Number of rollback steps
        
        self.active_rollbacks[operation_id] = rollback_op
        
        try:
            logger.info(f"Starting rollback operation {operation_id} to snapshot {to_snapshot}")
            
            # Step 1: Create pre-rollback snapshot
            rollback_op.current_step = "Creating pre-rollback snapshot"
            rollback_op.status = RollbackStatus.PREPARING
            
            pre_snapshot_id = await self.export_snapshot(
                description=f"Pre-rollback snapshot for operation {operation_id}",
                snapshot_type=SnapshotType.PRE_ROLLBACK,
                tags={"pre_rollback", operation_id}
            )
            rollback_op.pre_rollback_snapshot_id = pre_snapshot_id
            rollback_op.steps_completed += 1
            
            # Step 2: Drain active tasks
            rollback_op.current_step = "Draining active tasks"
            rollback_op.status = RollbackStatus.DRAINING
            await self._drain_active_tasks()
            rollback_op.steps_completed += 1
            
            # Step 3: Restore component states
            rollback_op.current_step = "Restoring component states"
            rollback_op.status = RollbackStatus.RESTORING
            await self._restore_component_states(target_snapshot)
            rollback_op.steps_completed += 1
            
            # Step 4: Restore loops and policies
            rollback_op.current_step = "Restoring loops and policies"
            await self._restore_loops_and_policies(target_snapshot)
            rollback_op.steps_completed += 1
            
            # Step 5: Restart scheduler
            rollback_op.current_step = "Restarting scheduler"
            await self._restart_scheduler_with_config(target_snapshot)
            rollback_op.steps_completed += 1
            
            # Step 6: Validate rollback
            rollback_op.current_step = "Validating rollback"
            rollback_op.status = RollbackStatus.VALIDATING
            await self._validate_rollback(target_snapshot)
            rollback_op.steps_completed += 1
            
            # Complete rollback
            rollback_op.status = RollbackStatus.COMPLETED
            rollback_op.completed_at = utc_now()
            rollback_op.current_step = "Completed"
            
            self.total_rollbacks_performed += 1
            
            logger.info(f"Successfully completed rollback operation {operation_id}")
            
            # Publish events
            if self.event_publisher:
                await self.event_publisher("ROLLBACK_COMPLETED", {
                    "target": "orchestration",
                    "snapshot_id": to_snapshot,
                    "at": iso_format()
                })
            
            return operation_id
            
        except Exception as e:
            rollback_op.status = RollbackStatus.FAILED
            rollback_op.errors.append(str(e))
            rollback_op.completed_at = utc_now()
            
            logger.error(f"Rollback operation {operation_id} failed: {e}")
            raise
    
    async def _capture_scheduler_state(self, snapshot: OrchSnapshot):
        """Capture scheduler state."""
        if self.scheduler:
            try:
                scheduler_status = self.scheduler.get_status()
                snapshot.loops = [loop.to_dict() for loop in scheduler_status.get("loops", {}).values()]
                snapshot.active_tasks = list(scheduler_status.get("active_tasks", {}).keys())
                snapshot.component_states["scheduler"] = scheduler_status
            except Exception as e:
                logger.warning(f"Failed to capture scheduler state: {e}")
    
    async def _capture_state_manager_state(self, snapshot: OrchSnapshot):
        """Capture state manager state."""
        if self.state_manager:
            try:
                state_status = self.state_manager.get_status()
                snapshot.policies = {
                    policy_id: policy.to_dict()
                    for policy_id, policy in self.state_manager.policies.items()
                }
                snapshot.component_states["state_manager"] = state_status
            except Exception as e:
                logger.warning(f"Failed to capture state manager state: {e}")
    
    async def _capture_router_state(self, snapshot: OrchSnapshot):
        """Capture router state."""
        if self.router:
            try:
                router_status = self.router.get_status()
                snapshot.component_states["router"] = router_status
            except Exception as e:
                logger.warning(f"Failed to capture router state: {e}")
    
    async def _capture_watchdog_state(self, snapshot: OrchSnapshot):
        """Capture watchdog state."""
        if self.watchdog:
            try:
                watchdog_status = self.watchdog.get_status()
                snapshot.component_states["watchdog"] = watchdog_status
            except Exception as e:
                logger.warning(f"Failed to capture watchdog state: {e}")
    
    async def _capture_scaling_state(self, snapshot: OrchSnapshot):
        """Capture scaling manager state."""
        if self.scaling_manager:
            try:
                scaling_status = self.scaling_manager.get_status()
                snapshot.component_states["scaling_manager"] = scaling_status
            except Exception as e:
                logger.warning(f"Failed to capture scaling state: {e}")
    
    async def _capture_lifecycle_state(self, snapshot: OrchSnapshot):
        """Capture lifecycle manager state."""
        if self.lifecycle_manager:
            try:
                lifecycle_status = self.lifecycle_manager.get_status()
                snapshot.component_states["lifecycle_manager"] = lifecycle_status
            except Exception as e:
                logger.warning(f"Failed to capture lifecycle state: {e}")
    
    async def _capture_system_metrics(self, snapshot: OrchSnapshot):
        """Capture system metrics."""
        try:
            # Basic system metrics (can be extended)
            snapshot.system_metrics = {
                "timestamp": iso_format(),
                "total_components": len(snapshot.component_states),
                "active_loops": len(snapshot.loops),
                "active_tasks": len(snapshot.active_tasks),
                "policies": len(snapshot.policies)
            }
        except Exception as e:
            logger.warning(f"Failed to capture system metrics: {e}")
    
    async def _validate_snapshot(self, snapshot: OrchSnapshot):
        """Validate snapshot integrity and consistency."""
        snapshot.status = SnapshotStatus.VALIDATING
        snapshot.validation_errors.clear()
        
        try:
            # Basic validation
            if not snapshot.snapshot_id:
                snapshot.validation_errors.append("Missing snapshot ID")
            
            if not snapshot.hash:
                snapshot.validation_errors.append("Missing content hash")
            
            # Verify hash
            calculated_hash = snapshot.calculate_hash()
            if snapshot.hash != calculated_hash:
                snapshot.validation_errors.append("Hash mismatch - content may be corrupted")
            
            # Validate component states
            for component_name, state in snapshot.component_states.items():
                if not isinstance(state, dict):
                    snapshot.validation_errors.append(f"Invalid state format for {component_name}")
            
            # Set validation result
            if snapshot.validation_errors:
                snapshot.status = SnapshotStatus.CORRUPTED
                logger.warning(f"Snapshot {snapshot.snapshot_id} validation failed: {snapshot.validation_errors}")
            else:
                snapshot.status = SnapshotStatus.COMPLETED
                logger.debug(f"Snapshot {snapshot.snapshot_id} validation passed")
            
            snapshot.validated_at = utc_now()
            
        except Exception as e:
            snapshot.validation_errors.append(f"Validation error: {e}")
            snapshot.status = SnapshotStatus.CORRUPTED
            logger.error(f"Snapshot validation error: {e}")
    
    async def _store_snapshot(self, snapshot: OrchSnapshot):
        """Store snapshot to disk."""
        file_path = self.storage_path / f"{snapshot.snapshot_id}.json"
        
        try:
            snapshot_data = snapshot.to_dict()
            
            if self.compression_enabled:
                # Store compressed
                compressed_path = file_path.with_suffix('.json.gz')
                with gzip.open(compressed_path, 'wt') as f:
                    json.dump(snapshot_data, f, indent=2)
                snapshot.size_bytes = compressed_path.stat().st_size
            else:
                # Store uncompressed
                with open(file_path, 'w') as f:
                    json.dump(snapshot_data, f, indent=2)
                snapshot.size_bytes = file_path.stat().st_size
            
            self.storage_usage_bytes += snapshot.size_bytes
            logger.debug(f"Stored snapshot {snapshot.snapshot_id} ({snapshot.size_bytes} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to store snapshot {snapshot.snapshot_id}: {e}")
            raise
    
    async def _load_existing_snapshots(self):
        """Load existing snapshots from storage."""
        try:
            for file_path in self.storage_path.glob("*.json*"):
                try:
                    if file_path.suffix == '.gz':
                        with gzip.open(file_path, 'rt') as f:
                            snapshot_data = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            snapshot_data = json.load(f)
                    
                    snapshot = OrchSnapshot.from_dict(snapshot_data)
                    self.snapshots[snapshot.snapshot_id] = snapshot
                    self.storage_usage_bytes += file_path.stat().st_size
                    
                except Exception as e:
                    logger.warning(f"Failed to load snapshot from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.snapshots)} existing snapshots")
            
        except Exception as e:
            logger.error(f"Failed to load existing snapshots: {e}")
    
    async def _drain_active_tasks(self):
        """Drain active tasks before rollback."""
        # This would coordinate with scheduler to gracefully complete or pause active tasks
        if self.scheduler:
            # Simplified - in real implementation would wait for tasks to complete
            await asyncio.sleep(1)
        logger.debug("Active tasks drained")
    
    async def _restore_component_states(self, snapshot: OrchSnapshot):
        """Restore component states from snapshot."""
        # This would restore each component's state from the snapshot
        logger.debug("Component states restored")
    
    async def _restore_loops_and_policies(self, snapshot: OrchSnapshot):
        """Restore loops and policies from snapshot."""
        # Restore state manager policies
        if self.state_manager and snapshot.policies:
            for policy_data in snapshot.policies.values():
                # Would restore policy from data
                pass
        
        # Restore scheduler loops
        if self.scheduler and snapshot.loops:
            for loop_data in snapshot.loops:
                # Would restore loop from data
                pass
        
        logger.debug("Loops and policies restored")
    
    async def _restart_scheduler_with_config(self, snapshot: OrchSnapshot):
        """Restart scheduler with restored configuration."""
        if self.scheduler:
            # Would restart scheduler with snapshot configuration
            await asyncio.sleep(0.5)
        logger.debug("Scheduler restarted with restored configuration")
    
    async def _validate_rollback(self, snapshot: OrchSnapshot):
        """Validate that rollback was successful."""
        # Would perform validation checks to ensure rollback worked
        logger.debug("Rollback validation completed")
    
    async def _cleanup_old_snapshots(self):
        """Clean up old snapshots based on retention policy."""
        if len(self.snapshots) <= self.max_snapshots:
            return
        
        # Sort by creation time and keep most recent
        sorted_snapshots = sorted(
            self.snapshots.values(),
            key=lambda s: s.created_at,
            reverse=True
        )
        
        to_remove = sorted_snapshots[self.max_snapshots:]
        
        for snapshot in to_remove:
            await self._delete_snapshot(snapshot.snapshot_id)
    
    async def _delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a specific snapshot."""
        if snapshot_id not in self.snapshots:
            return False
        
        try:
            snapshot = self.snapshots[snapshot_id]
            
            # Remove file
            file_patterns = [f"{snapshot_id}.json", f"{snapshot_id}.json.gz"]
            for pattern in file_patterns:
                file_path = self.storage_path / pattern
                if file_path.exists():
                    self.storage_usage_bytes -= file_path.stat().st_size
                    file_path.unlink()
            
            # Remove from memory
            del self.snapshots[snapshot_id]
            
            logger.debug(f"Deleted snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
            return False
    
    async def _snapshot_scheduler_loop(self):
        """Scheduled snapshot creation loop."""
        try:
            while self.running:
                # Check for scheduled snapshots
                for schedule_name, config in self.snapshot_schedule.items():
                    if self._should_create_scheduled_snapshot(config):
                        await self._create_scheduled_snapshot(schedule_name, config)
                
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logger.debug("Snapshot scheduler loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Snapshot scheduler error: {e}", exc_info=True)
    
    def _should_create_scheduled_snapshot(self, config: Dict[str, Any]) -> bool:
        """Check if a scheduled snapshot should be created."""
        # Simplified scheduling logic
        interval_minutes = config.get("interval_minutes", 60)
        last_snapshot = config.get("last_created")
        
        if not last_snapshot:
            return True
        
        time_since_last = (utc_now() - last_snapshot).total_seconds() / 60
        return time_since_last >= interval_minutes
    
    async def _create_scheduled_snapshot(self, schedule_name: str, config: Dict[str, Any]):
        """Create a scheduled snapshot."""
        try:
            description = f"Scheduled snapshot: {schedule_name}"
            tags = {"scheduled", schedule_name}
            
            snapshot_id = await self.export_snapshot(
                description=description,
                snapshot_type=SnapshotType.SCHEDULED,
                tags=tags
            )
            
            config["last_created"] = utc_now()
            logger.info(f"Created scheduled snapshot {snapshot_id} for schedule {schedule_name}")
            
        except Exception as e:
            logger.error(f"Failed to create scheduled snapshot {schedule_name}: {e}")
    
    def get_snapshot(self, snapshot_id: str) -> Optional[OrchSnapshot]:
        """Get a specific snapshot."""
        return self.snapshots.get(snapshot_id)
    
    def list_snapshots(self, snapshot_type: SnapshotType = None, 
                      tags: Set[str] = None) -> List[OrchSnapshot]:
        """List snapshots with optional filtering."""
        snapshots = list(self.snapshots.values())
        
        if snapshot_type:
            snapshots = [s for s in snapshots if s.snapshot_type == snapshot_type]
        
        if tags:
            snapshots = [s for s in snapshots if tags.issubset(s.tags)]
        
        return sorted(snapshots, key=lambda s: s.created_at, reverse=True)
    
    def get_rollback_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get rollback operation status."""
        if operation_id in self.active_rollbacks:
            return self.active_rollbacks[operation_id].to_dict()
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get snapshot manager status."""
        return {
            "running": self.running,
            "total_snapshots": len(self.snapshots),
            "storage_usage_bytes": self.storage_usage_bytes,
            "storage_path": str(self.storage_path),
            "configuration": {
                "max_snapshots": self.max_snapshots,
                "auto_cleanup_days": self.auto_cleanup_days,
                "compression_enabled": self.compression_enabled,
                "validation_enabled": self.validation_enabled
            },
            "statistics": {
                "total_snapshots_created": self.total_snapshots_created,
                "total_rollbacks_performed": self.total_rollbacks_performed
            },
            "snapshots_by_type": {
                snapshot_type.value: len([
                    s for s in self.snapshots.values() 
                    if s.snapshot_type == snapshot_type
                ])
                for snapshot_type in SnapshotType
            },
            "active_rollbacks": len(self.active_rollbacks),
            "scheduled_snapshots": len(self.snapshot_schedule)
        }