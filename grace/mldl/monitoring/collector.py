"""
Monitoring Collector - collects live metrics, SLO guards, and performance data.
"""

import asyncio
import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringCollector:
    """Collects and manages live model metrics and SLO monitoring."""

    def __init__(self, event_bus=None, db_path: str = ":memory:"):
        self.event_bus = event_bus
        self.db_path = db_path
        self.conn = None

        # SLO thresholds
        self.default_slos = {
            "p95_latency_ms": 500,
            "p99_latency_ms": 1000,
            "error_rate": 0.01,
            "throughput_qps": 100,
            "accuracy_threshold": 0.8,
            "drift_threshold": 0.25,
        }

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None

        # Initialize database
        self._initialize_db()

        logger.info("Monitoring Collector initialized")

    def _initialize_db(self):
        """Initialize SQLite database for metrics storage."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                deployment_id TEXT,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                dimensions TEXT,  -- JSON string
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Create SLO violations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS slo_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                deployment_id TEXT,
                slo_name TEXT NOT NULL,
                threshold_value REAL,
                actual_value REAL,
                severity TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)

        # Create alerts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,  -- JSON string
                resolved BOOLEAN DEFAULT FALSE,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)

        self.conn.commit()
        logger.info("Monitoring database initialized")

    async def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring started")

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Monitoring stopped")

    async def record_metric(
        self,
        model_key: str,
        version: str,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        deployment_id: Optional[str] = None,
        dimensions: Optional[Dict[str, Any]] = None,
    ):
        """Record a metric for a model."""
        try:
            self.conn.execute(
                """
                INSERT INTO model_metrics (
                    model_key, version, deployment_id, metric_type, metric_name,
                    metric_value, dimensions, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_key,
                    version,
                    deployment_id,
                    metric_type,
                    metric_name,
                    metric_value,
                    json.dumps(dimensions or {}),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

            self.conn.commit()

            # Check SLOs
            await self._check_slos(
                model_key, version, metric_name, metric_value, deployment_id
            )

        except Exception as e:
            logger.error(f"Failed to record metric: {e}")

    async def get_metrics(
        self,
        model_key: Optional[str] = None,
        version: Optional[str] = None,
        since: Optional[str] = None,
        metric_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics with optional filters."""
        try:
            query = "SELECT * FROM model_metrics WHERE 1=1"
            params = []

            if model_key:
                query += " AND model_key = ?"
                params.append(model_key)

            if version:
                query += " AND version = ?"
                params.append(version)

            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)

            if since:
                query += " AND timestamp >= ?"
                params.append(since)

            query += " ORDER BY timestamp DESC LIMIT 1000"

            cursor = self.conn.execute(query, params)
            metrics = []

            for row in cursor.fetchall():
                metric_data = dict(row)
                if metric_data["dimensions"]:
                    metric_data["dimensions"] = json.loads(metric_data["dimensions"])
                metrics.append(metric_data)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
