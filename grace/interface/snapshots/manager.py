"""Snapshot and rollback manager for Interface Kernel."""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class InterfaceSnapshot:
    """Represents a point-in-time snapshot of interface state."""

    def __init__(self, snapshot_data: Dict[str, Any]):
        self.snapshot_id = snapshot_data["snapshot_id"]
        self.timestamp = datetime.fromisoformat(snapshot_data["timestamp"])
        self.routes = snapshot_data.get("routes", {})
        self.rbac_version = snapshot_data.get("rbac_version", "1.0.0")
        self.theme = snapshot_data.get("theme", {})
        self.i18n = snapshot_data.get("i18n", {})
        self.a11y = snapshot_data.get("a11y", {})
        self.ws = snapshot_data.get("ws", {})
        self.consent_policies = snapshot_data.get("consent_policies", {})
        self.hash = snapshot_data.get("hash", "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "routes": self.routes,
            "rbac_version": self.rbac_version,
            "theme": self.theme,
            "i18n": self.i18n,
            "a11y": self.a11y,
            "ws": self.ws,
            "consent_policies": self.consent_policies,
            "hash": self.hash,
        }


class SnapshotManager:
    """Manages interface snapshots and rollback functionality."""

    def __init__(self, storage_path: str = "/tmp/interface_snapshots"):
        self.storage_path = storage_path
        self.snapshots: Dict[str, InterfaceSnapshot] = {}
        self.current_config = self._get_default_config()

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)

        # Load existing snapshots
        self._load_snapshots_from_disk()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default interface configuration."""
        return {
            "routes": {
                "enabled": ["/dashboard", "/intel", "/memory", "/governance", "/tasks"]
            },
            "rbac_version": "2.3.0",
            "theme": {"mode": "dark", "tokens_version": "1.2.0"},
            "i18n": {"locales": ["en-GB", "cy-GB"], "default": "en-GB"},
            "a11y": {"contrast_min": 7.0, "reduce_motion_default": True},
            "ws": {"enabled": True, "max_clients": 5000, "heartbeat_s": 20},
            "consent_policies": {"autonomy": {"required": True, "renew_days": 180}},
        }

    def export_snapshot(self) -> Dict[str, str]:
        """Create and export current interface state snapshot."""
        snapshot_id = f"ui_{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"

        snapshot_data = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.utcnow().isoformat(),
            **self.current_config,
        }

        # Calculate hash
        config_str = json.dumps(snapshot_data, sort_keys=True)
        snapshot_data["hash"] = (
            f"sha256:{hashlib.sha256(config_str.encode()).hexdigest()}"
        )

        # Create snapshot object
        snapshot = InterfaceSnapshot(snapshot_data)
        self.snapshots[snapshot_id] = snapshot

        # Persist to disk
        self._save_snapshot_to_disk(snapshot)

        logger.info(f"Created interface snapshot {snapshot_id}")

        return {
            "snapshot_id": snapshot_id,
            "uri": f"{self.storage_path}/{snapshot_id}.json",
        }

    async def rollback(self, to_snapshot: str) -> Dict[str, Any]:
        """Rollback interface to a specific snapshot."""
        if to_snapshot not in self.snapshots:
            raise ValueError(f"Snapshot {to_snapshot} not found")

        snapshot = self.snapshots[to_snapshot]

        logger.info(f"Starting rollback to snapshot {to_snapshot}")

        try:
            # Step 1: Quiesce WebSocket hubs (would integrate with actual WS manager)
            await self._quiesce_websockets()

            # Step 2: Load snapshot configuration
            self.current_config = {
                "routes": snapshot.routes,
                "rbac_version": snapshot.rbac_version,
                "theme": snapshot.theme,
                "i18n": snapshot.i18n,
                "a11y": snapshot.a11y,
                "ws": snapshot.ws,
                "consent_policies": snapshot.consent_policies,
            }

            # Step 3: Re-open hubs and broadcast notification
            await self._restore_websockets()

            # Step 4: Create rollback completion event
            rollback_event = {
                "target": "interface",
                "snapshot_id": to_snapshot,
                "at": datetime.utcnow().isoformat(),
            }

            logger.info(f"Successfully rolled back to snapshot {to_snapshot}")

            return rollback_event

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

    async def _quiesce_websockets(self):
        """Gracefully close WebSocket connections."""
        logger.info("Quiescing WebSocket connections for rollback")
        # Would integrate with actual WebSocket manager
        pass

    async def _restore_websockets(self):
        """Restore WebSocket connections after rollback."""
        logger.info("Restoring WebSocket connections after rollback")
        # Would integrate with actual WebSocket manager and send notification
        pass

    def list_snapshots(self, limit: int = 10) -> List[Dict]:
        """List available snapshots."""
        snapshots = list(self.snapshots.values())
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        return [s.to_dict() for s in snapshots[:limit]]

    def get_snapshot(self, snapshot_id: str) -> Optional[InterfaceSnapshot]:
        """Get specific snapshot."""
        return self.snapshots.get(snapshot_id)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        if snapshot_id not in self.snapshots:
            return False

        del self.snapshots[snapshot_id]

        # Remove from disk
        snapshot_file = f"{self.storage_path}/{snapshot_id}.json"
        if os.path.exists(snapshot_file):
            os.remove(snapshot_file)

        logger.info(f"Deleted snapshot {snapshot_id}")
        return True

    def _save_snapshot_to_disk(self, snapshot: InterfaceSnapshot):
        """Save snapshot to disk."""
        snapshot_file = f"{self.storage_path}/{snapshot.snapshot_id}.json"

        try:
            with open(snapshot_file, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save snapshot to disk: {e}")

    def _load_snapshots_from_disk(self):
        """Load existing snapshots from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    snapshot_file = os.path.join(self.storage_path, filename)

                    with open(snapshot_file, "r") as f:
                        snapshot_data = json.load(f)
                        snapshot = InterfaceSnapshot(snapshot_data)
                        self.snapshots[snapshot.snapshot_id] = snapshot

            logger.info(f"Loaded {len(self.snapshots)} snapshots from disk")

        except Exception as e:
            logger.error(f"Failed to load snapshots from disk: {e}")

    def update_config(self, config_updates: Dict[str, Any]):
        """Update current configuration."""

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    isinstance(value, dict)
                    and key in base_dict
                    and isinstance(base_dict[key], dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.current_config, config_updates)
        logger.info("Updated interface configuration")

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.current_config.copy()

    def cleanup_old_snapshots(self, keep_count: int = 20):
        """Clean up old snapshots, keeping only the most recent ones."""
        snapshots = list(self.snapshots.values())
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        if len(snapshots) <= keep_count:
            return 0

        old_snapshots = snapshots[keep_count:]
        deleted_count = 0

        for snapshot in old_snapshots:
            if self.delete_snapshot(snapshot.snapshot_id):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old snapshots")
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """Get snapshot manager statistics."""
        snapshots = list(self.snapshots.values())

        if snapshots:
            latest_snapshot = max(snapshots, key=lambda s: s.timestamp)
            oldest_snapshot = min(snapshots, key=lambda s: s.timestamp)

            return {
                "total_snapshots": len(snapshots),
                "latest_snapshot": latest_snapshot.snapshot_id,
                "oldest_snapshot": oldest_snapshot.snapshot_id,
                "storage_path": self.storage_path,
            }

        return {"total_snapshots": 0, "storage_path": self.storage_path}
