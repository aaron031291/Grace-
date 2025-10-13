"""
Snapshot Manager - handles system state snapshots and rollback functionality.
"""

import json
import logging
import hashlib
import sqlite3
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages system snapshots for rollback and state recovery."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None

        # Initialize database
        self._initialize_db()

        logger.info("Snapshot Manager initialized")

    def _initialize_db(self):
        """Initialize SQLite database for snapshot storage."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create snapshots table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT UNIQUE NOT NULL,
                snapshot_type TEXT NOT NULL,
                description TEXT,
                registry_index TEXT,  -- JSON string
                search_spaces TEXT,   -- JSON string
                calibration_config TEXT,  -- JSON string
                fairness_config TEXT,     -- JSON string
                deployment_policy TEXT,   -- JSON string
                state_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                size_bytes INTEGER,
                metadata TEXT  -- JSON string
            )
        """)

        # Create rollback history table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rollback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollback_id TEXT UNIQUE NOT NULL,
                from_snapshot_id TEXT,
                to_snapshot_id TEXT NOT NULL,
                reason TEXT,
                triggered_by TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                FOREIGN KEY (to_snapshot_id) REFERENCES snapshots (snapshot_id)
            )
        """)

        self.conn.commit()
        logger.info("Snapshot database initialized")

    async def create_snapshot(
        self, description: str = None, snapshot_type: str = "manual"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive system snapshot.

        Args:
            description: Optional description
            snapshot_type: Type of snapshot (manual, automatic, scheduled)

        Returns:
            Snapshot information
        """
        try:
            snapshot_id = f"mldl_{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}"

            # Collect current system state
            registry_index = await self._collect_registry_state()
            search_spaces = await self._collect_search_spaces()
            calibration_config = await self._collect_calibration_config()
            fairness_config = await self._collect_fairness_config()
            deployment_policy = await self._collect_deployment_policy()

            # Create snapshot payload
            snapshot_payload = {
                "snapshot_id": snapshot_id,
                "registry_index": registry_index,
                "search_spaces": search_spaces,
                "calibration": calibration_config,
                "fairness": fairness_config,
                "deployment_policy": deployment_policy,
                "created_at": datetime.now().isoformat(),
                "snapshot_type": snapshot_type,
            }

            # Calculate hash
            state_hash = self._calculate_state_hash(snapshot_payload)
            snapshot_payload["hash"] = f"sha256:{state_hash}"

            # Calculate size
            payload_json = json.dumps(snapshot_payload, indent=2)
            size_bytes = len(payload_json.encode("utf-8"))

            # Store in database
            self.conn.execute(
                """
                INSERT INTO snapshots (
                    snapshot_id, snapshot_type, description, registry_index,
                    search_spaces, calibration_config, fairness_config,
                    deployment_policy, state_hash, created_at, size_bytes, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    snapshot_id,
                    snapshot_type,
                    description or f"Snapshot created at {datetime.now().isoformat()}",
                    json.dumps(registry_index),
                    json.dumps(search_spaces),
                    json.dumps(calibration_config),
                    json.dumps(fairness_config),
                    json.dumps(deployment_policy),
                    state_hash,
                    datetime.now().isoformat(),
                    size_bytes,
                    json.dumps({"payload_size": size_bytes}),
                ),
            )

            self.conn.commit()

            logger.info(f"Snapshot {snapshot_id} created successfully")

            return {
                "snapshot_id": snapshot_id,
                "uri": f"mldl://snapshots/{snapshot_id}",
                "state_hash": f"sha256:{state_hash}",
                "size_bytes": size_bytes,
                "created_at": datetime.now().isoformat(),
                "description": description
                or f"Snapshot created at {datetime.now().isoformat()}",
            }

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    async def rollback(
        self, to_snapshot: str, reason: str = None, triggered_by: str = "manual"
    ) -> Dict[str, Any]:
        """
        Rollback system to a specific snapshot.

        Args:
            to_snapshot: Snapshot ID to rollback to
            reason: Reason for rollback
            triggered_by: Who/what triggered the rollback

        Returns:
            Rollback result
        """
        try:
            rollback_id = f"rollback_{uuid.uuid4().hex[:8]}"

            # Get snapshot to rollback to
            snapshot = await self.get_snapshot(to_snapshot)
            if not snapshot:
                raise ValueError(f"Snapshot {to_snapshot} not found")

            # Get current state snapshot as backup
            current_snapshot = await self.create_snapshot(
                f"Pre-rollback backup for {rollback_id}", "rollback_backup"
            )

            # Record rollback attempt
            self.conn.execute(
                """
                INSERT INTO rollback_history (
                    rollback_id, from_snapshot_id, to_snapshot_id, reason,
                    triggered_by, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rollback_id,
                    current_snapshot["snapshot_id"],
                    to_snapshot,
                    reason,
                    triggered_by,
                    "in_progress",
                    datetime.now().isoformat(),
                ),
            )

            self.conn.commit()

            # Perform rollback operations
            rollback_success = await self._execute_rollback(snapshot)

            # Update rollback status
            if rollback_success:
                self.conn.execute(
                    """
                    UPDATE rollback_history 
                    SET status = ?, completed_at = ?
                    WHERE rollback_id = ?
                """,
                    ("completed", datetime.now().isoformat(), rollback_id),
                )

                status = "completed"
                logger.info(f"Rollback {rollback_id} completed successfully")
            else:
                self.conn.execute(
                    """
                    UPDATE rollback_history 
                    SET status = ?, error_message = ?, completed_at = ?
                    WHERE rollback_id = ?
                """,
                    (
                        "failed",
                        "Rollback execution failed",
                        datetime.now().isoformat(),
                        rollback_id,
                    ),
                )

                status = "failed"
                logger.error(f"Rollback {rollback_id} failed")

            self.conn.commit()

            return {
                "rollback_id": rollback_id,
                "to_snapshot": to_snapshot,
                "from_snapshot": current_snapshot["snapshot_id"],
                "status": status,
                "reason": reason,
                "triggered_by": triggered_by,
                "completed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            # Update rollback status as failed
            try:
                self.conn.execute(
                    """
                    UPDATE rollback_history 
                    SET status = ?, error_message = ?, completed_at = ?
                    WHERE rollback_id = ?
                """,
                    ("failed", str(e), datetime.now().isoformat(), rollback_id),
                )
                self.conn.commit()
            except:
                pass
            raise

    async def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot by ID."""
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM snapshots WHERE snapshot_id = ?
            """,
                (snapshot_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            snapshot_data = dict(row)

            # Parse JSON fields
            json_fields = [
                "registry_index",
                "search_spaces",
                "calibration_config",
                "fairness_config",
                "deployment_policy",
                "metadata",
            ]

            for field in json_fields:
                if snapshot_data[field]:
                    try:
                        snapshot_data[field] = json.loads(snapshot_data[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON field {field}")
                        snapshot_data[field] = {}

            return snapshot_data

        except Exception as e:
            logger.error(f"Failed to get snapshot {snapshot_id}: {e}")
            return None

    async def list_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List available snapshots."""
        try:
            cursor = self.conn.execute(
                """
                SELECT snapshot_id, snapshot_type, description, state_hash,
                       created_at, size_bytes
                FROM snapshots 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []

    async def get_rollback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get rollback history."""
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM rollback_history
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []

    async def _collect_registry_state(self) -> Dict[str, Any]:
        """Collect current model registry state."""
        # Mock registry state - real implementation would query registry
        return {
            "tabular.classification.xgb": "1.3.2",
            "tabular.classification.rf": "2.0.1",
            "transformer.tabular": "1.2.0",
            "clustering.kmeans": "1.0.5",
        }

    async def _collect_search_spaces(self) -> Dict[str, Any]:
        """Collect current hyperparameter search spaces."""
        return {
            "xgb": {"max_depth": [3, 10], "eta": [0.01, 0.3], "subsample": [0.6, 1.0]},
            "svm": {"C": [0.1, 10], "kernel": ["rbf", "linear"]},
            "rf": {"n_estimators": [50, 500], "max_depth": [5, 20]},
        }

    async def _collect_calibration_config(self) -> Dict[str, Any]:
        """Collect calibration configuration."""
        return {"default": "isotonic", "fallback": "platt", "ece_threshold": 0.1}

    async def _collect_fairness_config(self) -> Dict[str, Any]:
        """Collect fairness configuration."""
        return {
            "delta_max": 0.02,
            "groups": ["gender", "region", "age_group"],
            "metrics": ["statistical_parity", "equal_opportunity"],
        }

    async def _collect_deployment_policy(self) -> Dict[str, Any]:
        """Collect deployment policy configuration."""
        return {
            "canary_steps": [5, 25, 50, 100],
            "promotion_window": 3600,
            "rollback_triggers": ["metric_drop", "drift_spike", "violation", "anomaly"],
            "slo_thresholds": {
                "p95_latency_ms": 500,
                "error_rate": 0.01,
                "accuracy_min": 0.8,
            },
        }

    def _calculate_state_hash(self, snapshot_payload: Dict[str, Any]) -> str:
        """Calculate hash of snapshot state."""
        # Create deterministic JSON representation
        payload_str = json.dumps(
            snapshot_payload, sort_keys=True, separators=(",", ":")
        )

        # Calculate SHA-256 hash
        hash_obj = hashlib.sha256(payload_str.encode("utf-8"))
        return hash_obj.hexdigest()

    async def _execute_rollback(self, snapshot: Dict[str, Any]) -> bool:
        """Execute the actual rollback operations."""
        try:
            # In a real implementation, this would:
            # 1. Pause all active deployments
            # 2. Update registry with snapshot registry_index
            # 3. Update HPO configurations with snapshot search_spaces
            # 4. Update calibration settings
            # 5. Update fairness settings
            # 6. Update deployment policies
            # 7. Invalidate pending deployments
            # 8. Resume operations

            logger.info("Executing rollback operations...")

            # Simulate rollback operations
            await asyncio.sleep(2)  # Simulate operation time

            # Mock success - real implementation would check each step
            return True

        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            return False

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        try:
            self.conn.execute(
                """
                DELETE FROM snapshots WHERE snapshot_id = ?
            """,
                (snapshot_id,),
            )

            self.conn.commit()

            if self.conn.rowcount > 0:
                logger.info(f"Snapshot {snapshot_id} deleted")
                return True
            else:
                logger.warning(f"Snapshot {snapshot_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
            return False

    async def get_snapshot_stats(self) -> Dict[str, Any]:
        """Get snapshot statistics."""
        try:
            # Total snapshots
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM snapshots")
            total_snapshots = cursor.fetchone()["count"]

            # Snapshots by type
            cursor = self.conn.execute("""
                SELECT snapshot_type, COUNT(*) as count 
                FROM snapshots 
                GROUP BY snapshot_type
            """)
            type_counts = {
                row["snapshot_type"]: row["count"] for row in cursor.fetchall()
            }

            # Total storage size
            cursor = self.conn.execute(
                "SELECT SUM(size_bytes) as total_size FROM snapshots"
            )
            total_size = cursor.fetchone()["total_size"] or 0

            # Recent rollbacks
            cursor = self.conn.execute("""
                SELECT COUNT(*) as count 
                FROM rollback_history 
                WHERE created_at > datetime('now', '-7 days')
            """)
            recent_rollbacks = cursor.fetchone()["count"]

            return {
                "total_snapshots": total_snapshots,
                "snapshot_types": type_counts,
                "total_size_bytes": total_size,
                "recent_rollbacks": recent_rollbacks,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get snapshot stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close snapshot manager."""
        if self.conn:
            self.conn.close()
            logger.info("Snapshot manager database connection closed")
