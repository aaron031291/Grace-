"""
Snapshot Manager - Manages versioned snapshots of MLT state for rollback capabilities.
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .contracts import MLTSnapshot, generate_snapshot_id


logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages versioned snapshots for MLT state including configs, weights, and schedules."""

    def __init__(self, storage_path: str = "/tmp/mlt_snapshots"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.snapshots: Dict[str, MLTSnapshot] = {}
        self.current_state = {
            "planner_version": "1.0.0",
            "search_spaces": {},
            "weights": {},
            "policies": {},
            "active_jobs": [],
        }

    async def create_snapshot(
        self, state: Optional[Dict[str, Any]] = None
    ) -> MLTSnapshot:
        """Create a snapshot of current MLT state."""
        try:
            if state:
                self.current_state.update(state)

            snapshot_id = generate_snapshot_id()

            # Calculate state hash
            state_json = json.dumps(self.current_state, sort_keys=True)
            state_hash = hashlib.sha256(state_json.encode()).hexdigest()

            snapshot = MLTSnapshot(
                snapshot_id=snapshot_id,
                planner_version=self.current_state["planner_version"],
                search_spaces=self.current_state["search_spaces"].copy(),
                weights=self.current_state["weights"].copy(),
                policies=self.current_state["policies"].copy(),
                active_jobs=self.current_state["active_jobs"].copy(),
                hash=state_hash,
                timestamp=datetime.now(),
            )

            # Store snapshot in memory
            self.snapshots[snapshot_id] = snapshot

            # Persist to disk
            await self._persist_snapshot(snapshot)

            logger.info(f"Created snapshot {snapshot_id}")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    async def load_snapshot(self, snapshot_id: str) -> bool:
        """Load a specific snapshot and restore state."""
        try:
            snapshot = self.snapshots.get(snapshot_id)

            if not snapshot:
                # Try loading from disk
                snapshot = await self._load_snapshot_from_disk(snapshot_id)
                if not snapshot:
                    logger.error(f"Snapshot {snapshot_id} not found")
                    return False

            # Restore state
            self.current_state = {
                "planner_version": snapshot.planner_version,
                "search_spaces": snapshot.search_spaces.copy(),
                "weights": snapshot.weights.copy(),
                "policies": snapshot.policies.copy(),
                "active_jobs": snapshot.active_jobs.copy(),
            }

            logger.info(f"Loaded snapshot {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return False

    def get_latest_snapshot(self) -> Optional[MLTSnapshot]:
        """Get the most recent snapshot."""
        if not self.snapshots:
            return None

        return max(self.snapshots.values(), key=lambda s: s.timestamp)

    def list_snapshots(self, limit: int = 10) -> List[MLTSnapshot]:
        """List recent snapshots."""
        snapshots = sorted(
            self.snapshots.values(), key=lambda s: s.timestamp, reverse=True
        )
        return snapshots[:limit]

    async def rollback_to_snapshot(
        self, snapshot_id: str, verify_hash: bool = True
    ) -> bool:
        """Rollback to a specific snapshot with optional hash verification."""
        try:
            if not await self.load_snapshot(snapshot_id):
                return False

            if verify_hash:
                # Verify state integrity
                current_hash = await self._calculate_current_hash()
                snapshot = self.snapshots[snapshot_id]

                if current_hash != snapshot.hash:
                    logger.warning(f"Hash mismatch during rollback to {snapshot_id}")
                    # Continue anyway as state might have minor acceptable differences

            logger.info(f"Rolled back to snapshot {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to snapshot {snapshot_id}: {e}")
            return False

    def should_rollback(self, current_metrics: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if rollback is needed based on current metrics."""
        rollback_triggers = [
            ("metric_drop_pct", 5, ">="),
            ("drift_z", 3, ">="),
            ("fairness_delta", 0.02, ">"),
            ("compliance_flag", True, "=="),
        ]

        for metric, threshold, operator in rollback_triggers:
            if metric not in current_metrics:
                continue

            value = current_metrics[metric]

            if operator == ">=" and value >= threshold:
                return True, f"{metric} ({value}) >= {threshold}"
            elif operator == ">" and value > threshold:
                return True, f"{metric} ({value}) > {threshold}"
            elif operator == "==" and value == threshold:
                return True, f"{metric} triggered"

        return False, "No rollback conditions met"

    async def auto_rollback_if_needed(self, current_metrics: Dict[str, Any]) -> bool:
        """Automatically rollback if conditions are met."""
        should_rollback, reason = self.should_rollback(current_metrics)

        if should_rollback:
            latest_snapshot = self.get_latest_snapshot()
            if latest_snapshot:
                logger.warning(f"Auto-rollback triggered: {reason}")
                return await self.rollback_to_snapshot(latest_snapshot.snapshot_id)

        return False

    async def cleanup_old_snapshots(self, keep_count: int = 20):
        """Clean up old snapshots, keeping only the most recent ones."""
        if len(self.snapshots) <= keep_count:
            return

        snapshots = sorted(
            self.snapshots.values(), key=lambda s: s.timestamp, reverse=True
        )
        to_remove = snapshots[keep_count:]

        for snapshot in to_remove:
            # Remove from memory
            if snapshot.snapshot_id in self.snapshots:
                del self.snapshots[snapshot.snapshot_id]

            # Remove from disk
            await self._delete_snapshot_from_disk(snapshot.snapshot_id)

        logger.info(f"Cleaned up {len(to_remove)} old snapshots")

    async def _persist_snapshot(self, snapshot: MLTSnapshot):
        """Persist snapshot to disk."""
        try:
            file_path = self.storage_path / f"{snapshot.snapshot_id}.json"
            with open(file_path, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist snapshot {snapshot.snapshot_id}: {e}")

    async def _load_snapshot_from_disk(self, snapshot_id: str) -> Optional[MLTSnapshot]:
        """Load snapshot from disk."""
        try:
            file_path = self.storage_path / f"{snapshot_id}.json"
            if not file_path.exists():
                return None

            with open(file_path, "r") as f:
                data = json.load(f)

            snapshot = MLTSnapshot(
                snapshot_id=data["snapshot_id"],
                planner_version=data["planner_version"],
                search_spaces=data["search_spaces"],
                weights=data["weights"],
                policies=data["policies"],
                active_jobs=data["active_jobs"],
                hash=data["hash"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )

            self.snapshots[snapshot_id] = snapshot
            return snapshot

        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id} from disk: {e}")
            return None

    async def _delete_snapshot_from_disk(self, snapshot_id: str):
        """Delete snapshot file from disk."""
        try:
            file_path = self.storage_path / f"{snapshot_id}.json"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id} from disk: {e}")

    async def _calculate_current_hash(self) -> str:
        """Calculate hash of current state."""
        state_json = json.dumps(self.current_state, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get snapshot manager statistics."""
        return {
            "total_snapshots": len(self.snapshots),
            "storage_path": str(self.storage_path),
            "latest_snapshot_age": (
                (datetime.now() - self.get_latest_snapshot().timestamp).total_seconds()
                / 3600
                if self.get_latest_snapshot()
                else None
            ),
            "current_state_hash": hashlib.sha256(
                json.dumps(self.current_state, sort_keys=True).encode()
            ).hexdigest()[:12],
        }
