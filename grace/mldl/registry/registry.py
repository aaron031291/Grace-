"""
Model Registry - manages model artifacts, metadata, lineage, and versioning.
"""

import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry with lineage tracking and version management."""

    def __init__(self, event_bus=None, db_path: str = ":memory:"):
        self.event_bus = event_bus
        self.db_path = db_path
        self.conn = None

        # Initialize database
        self._initialize_db()

        logger.info(f"Model Registry initialized with database at {db_path}")

    def _initialize_db(self):
        """Initialize SQLite database for model registry."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                artifact_uri TEXT NOT NULL,
                metrics TEXT,  -- JSON string
                calibration TEXT,  -- JSON string
                fairness TEXT,  -- JSON string
                robustness TEXT,  -- JSON string
                data_schema TEXT,  -- JSON string
                lineage TEXT,  -- JSON string
                tags TEXT,  -- JSON string (array)
                validation_hash TEXT,
                status TEXT DEFAULT 'registered',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                UNIQUE(model_key, version)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                approval_type TEXT NOT NULL,  -- governance, security, fairness
                status TEXT NOT NULL,  -- pending, approved, rejected
                approver TEXT,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_key, version) REFERENCES models (model_key, version)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                deployment_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_key, version) REFERENCES models (model_key, version)
            )
        """)

        self.conn.commit()
        logger.info("Model registry database initialized")

    async def register(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a trained model bundle.

        Args:
            bundle: TrainedBundle dictionary

        Returns:
            Registration result
        """
        try:
            model_key = bundle["model_key"]
            version = bundle["version"]

            # Check if model version already exists
            existing = await self.get(model_key, version)
            if existing:
                raise ValueError(f"Model {model_key}@{version} already exists")

            # Insert into database
            self.conn.execute(
                """
                INSERT INTO models (
                    model_key, version, artifact_uri, metrics, calibration, fairness,
                    robustness, data_schema, lineage, tags, validation_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_key,
                    version,
                    bundle["artifact_uri"],
                    json.dumps(bundle.get("metrics", {})),
                    json.dumps(bundle.get("calibration", {})),
                    json.dumps(bundle.get("fairness", {})),
                    json.dumps(bundle.get("robustness", {})),
                    json.dumps(bundle.get("data_schema", {})),
                    json.dumps(bundle.get("lineage", {})),
                    json.dumps(bundle.get("tags", [])),
                    bundle.get("validation_hash", ""),
                    bundle.get("created_at", datetime.now().isoformat()),
                ),
            )

            self.conn.commit()

            # Publish registration event
            if self.event_bus:
                await self.event_bus.publish(
                    "MLDL_MODEL_REGISTERED",
                    {
                        "model_key": model_key,
                        "version": version,
                        "uri": bundle["artifact_uri"],
                    },
                )

            logger.info(f"Registered model {model_key}@{version}")

            return {
                "model_key": model_key,
                "version": version,
                "status": "registered",
                "registered_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    async def get(self, model_key: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered model bundle.

        Args:
            model_key: Model identifier
            version: Model version

        Returns:
            Model bundle or None if not found
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM models WHERE model_key = ? AND version = ?
            """,
                (model_key, version),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Convert row to dictionary and parse JSON fields
            model_data = dict(row)
            json_fields = [
                "metrics",
                "calibration",
                "fairness",
                "robustness",
                "data_schema",
                "lineage",
                "tags",
            ]

            for field in json_fields:
                if model_data[field]:
                    try:
                        model_data[field] = json.loads(model_data[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON field {field}")
                        model_data[field] = {}

            # Get approvals
            model_data["approvals"] = await self._get_model_approvals(
                model_key, version
            )

            # Get deployment history
            model_data["deployments"] = await self._get_model_deployments(
                model_key, version
            )

            return model_data

        except Exception as e:
            logger.error(f"Failed to get model {model_key}@{version}: {e}")
            raise

    async def query(
        self, task: str = None, constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query models by task and constraints.

        Args:
            task: Task type filter
            constraints: Constraint filters (latency_ms, cost_units, etc.)

        Returns:
            List of matching models
        """
        try:
            # Base query
            query = "SELECT * FROM models WHERE status = 'registered'"
            params = []

            # Add task filter (would need task field in schema for full implementation)
            # For now, filter by model_key pattern
            if task:
                query += " AND model_key LIKE ?"
                params.append(f"%{task}%")

            # Execute query
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()

            models = []
            for row in rows:
                model_data = dict(row)

                # Parse JSON fields
                json_fields = [
                    "metrics",
                    "calibration",
                    "fairness",
                    "robustness",
                    "data_schema",
                    "lineage",
                    "tags",
                ]

                for field in json_fields:
                    if model_data[field]:
                        try:
                            model_data[field] = json.loads(model_data[field])
                        except json.JSONDecodeError:
                            model_data[field] = {}

                # Apply constraints if specified
                if constraints and not self._check_constraints(model_data, constraints):
                    continue

                models.append(model_data)

            logger.info(f"Found {len(models)} models matching query")
            return models

        except Exception as e:
            logger.error(f"Model query failed: {e}")
            raise

    def _check_constraints(
        self, model_data: Dict[str, Any], constraints: Dict[str, Any]
    ) -> bool:
        """Check if model meets constraints."""
        try:
            metrics = model_data.get("metrics", {})

            # Latency constraint
            if "latency_ms" in constraints:
                model_latency = metrics.get("p95_latency_ms", float("inf"))
                if model_latency > constraints["latency_ms"]:
                    return False

            # Cost constraint
            if "cost_units" in constraints:
                model_cost = metrics.get("cost_units", float("inf"))
                if model_cost > constraints["cost_units"]:
                    return False

            # Fairness constraint
            if "fairness_delta_max" in constraints:
                fairness = model_data.get("fairness", {})
                model_delta = fairness.get("delta", float("inf"))
                if model_delta > constraints["fairness_delta_max"]:
                    return False

            # Calibration constraint
            if "min_calibration" in constraints:
                calibration = model_data.get("calibration", {})
                model_ece = calibration.get("ece", float("inf"))
                if model_ece > (
                    1 - constraints["min_calibration"]
                ):  # Lower ECE is better
                    return False

            return True

        except Exception as e:
            logger.warning(f"Constraint check failed: {e}")
            return True  # Be permissive on errors

    async def _get_model_approvals(
        self, model_key: str, version: str
    ) -> List[Dict[str, Any]]:
        """Get approval records for a model."""
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM model_approvals 
                WHERE model_key = ? AND version = ?
                ORDER BY created_at DESC
            """,
                (model_key, version),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.warning(f"Failed to get approvals: {e}")
            return []

    async def _get_model_deployments(
        self, model_key: str, version: str
    ) -> List[Dict[str, Any]]:
        """Get deployment records for a model."""
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM model_deployments 
                WHERE model_key = ? AND version = ?
                ORDER BY created_at DESC
            """,
                (model_key, version),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.warning(f"Failed to get deployments: {e}")
            return []

    async def add_approval(
        self,
        model_key: str,
        version: str,
        approval_type: str,
        status: str,
        approver: str = None,
        reason: str = None,
    ) -> Dict[str, Any]:
        """Add an approval record for a model."""
        try:
            self.conn.execute(
                """
                INSERT INTO model_approvals (
                    model_key, version, approval_type, status, approver, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_key,
                    version,
                    approval_type,
                    status,
                    approver,
                    reason,
                    datetime.now().isoformat(),
                ),
            )

            self.conn.commit()

            logger.info(
                f"Added {approval_type} approval for {model_key}@{version}: {status}"
            )

            return {
                "model_key": model_key,
                "version": version,
                "approval_type": approval_type,
                "status": status,
                "approver": approver,
                "reason": reason,
                "created_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to add approval: {e}")
            raise

    async def record_deployment(
        self,
        model_key: str,
        version: str,
        deployment_id: str,
        environment: str,
        status: str,
    ) -> Dict[str, Any]:
        """Record a model deployment."""
        try:
            self.conn.execute(
                """
                INSERT INTO model_deployments (
                    model_key, version, deployment_id, environment, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    model_key,
                    version,
                    deployment_id,
                    environment,
                    status,
                    datetime.now().isoformat(),
                ),
            )

            self.conn.commit()

            logger.info(
                f"Recorded deployment {deployment_id} for {model_key}@{version}"
            )

            return {
                "model_key": model_key,
                "version": version,
                "deployment_id": deployment_id,
                "environment": environment,
                "status": status,
                "created_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to record deployment: {e}")
            raise

    async def get_model_lineage(self, model_key: str, version: str) -> Dict[str, Any]:
        """Get full lineage information for a model."""
        try:
            model = await self.get(model_key, version)
            if not model:
                return {}

            lineage_info = {
                "model_key": model_key,
                "version": version,
                "direct_lineage": model.get("lineage", {}),
                "approvals": model.get("approvals", []),
                "deployments": model.get("deployments", []),
                "created_at": model.get("created_at"),
                "validation_hash": model.get("validation_hash"),
            }

            return lineage_info

        except Exception as e:
            logger.error(f"Failed to get lineage: {e}")
            raise

    async def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current registry state."""
        try:
            snapshot_id = f"registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get all registered models
            cursor = self.conn.execute(
                "SELECT * FROM models WHERE status = 'registered'"
            )
            models = [dict(row) for row in cursor.fetchall()]

            # Create snapshot
            snapshot = {
                "snapshot_id": snapshot_id,
                "created_at": datetime.now().isoformat(),
                "model_count": len(models),
                "models": models,
            }

            logger.info(
                f"Created registry snapshot {snapshot_id} with {len(models)} models"
            )
            return snapshot

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        try:
            # Total models
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM models")
            total_models = cursor.fetchone()["count"]

            # Models by status
            cursor = self.conn.execute("""
                SELECT status, COUNT(*) as count FROM models GROUP BY status
            """)
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Recent activity
            cursor = self.conn.execute("""
                SELECT COUNT(*) as count FROM models 
                WHERE created_at > datetime('now', '-7 days')
            """)
            recent_registrations = cursor.fetchone()["count"]

            return {
                "total_models": total_models,
                "status_breakdown": status_counts,
                "recent_registrations": recent_registrations,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Model registry database connection closed")
