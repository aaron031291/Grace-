"""Resilience state snapshot and rollback management."""

import json
import hashlib
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class SnapshotManager:
    """
    Manage resilience state snapshots for rollback and recovery.
    
    Captures current resilience configuration, policies, and state
    to enable rollback during incidents or configuration issues.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """Initialize snapshot manager."""
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
        
        logger.debug(f"Snapshot manager initialized with db: {db_path}")
    
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
                slo_policies TEXT,        -- JSON string
                resilience_policies TEXT, -- JSON string
                dependency_graphs TEXT,   -- JSON string
                circuit_states TEXT,      -- JSON string
                degradation_states TEXT,  -- JSON string
                chaos_config TEXT,        -- JSON string
                state_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                size_bytes INTEGER,
                metadata TEXT             -- JSON string
            )
        """)
        
        # Create rollback history table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rollback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollback_id TEXT UNIQUE NOT NULL,
                from_snapshot TEXT,
                to_snapshot TEXT NOT NULL,
                reason TEXT,
                triggered_by TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                affected_services TEXT    -- JSON array
            )
        """)
        
        self.conn.commit()
    
    async def create_snapshot(
        self, 
        description: str = None,
        snapshot_type: str = "manual",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a new resilience state snapshot.
        
        Args:
            description: Human-readable snapshot description
            snapshot_type: Type of snapshot (manual, scheduled, incident, etc.)
            metadata: Additional metadata
            
        Returns:
            Created snapshot information
        """
        try:
            snapshot_id = f"res_{datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')}"
            
            # Collect current state (this would integrate with actual services)
            state_data = await self._collect_current_state()
            
            # Calculate hash for integrity
            state_hash = self._calculate_hash(state_data)
            
            # Serialize data
            slo_policies = json.dumps(state_data.get("slo_policies", {}))
            resilience_policies = json.dumps(state_data.get("resilience_policies", {}))
            dependency_graphs = json.dumps(state_data.get("dependency_graphs", {}))
            circuit_states = json.dumps(state_data.get("circuit_states", {}))
            degradation_states = json.dumps(state_data.get("degradation_states", {}))
            chaos_config = json.dumps(state_data.get("chaos_config", {}))
            
            # Calculate size
            size_bytes = sum(len(s.encode('utf-8')) for s in [
                slo_policies, resilience_policies, dependency_graphs,
                circuit_states, degradation_states, chaos_config
            ])
            
            # Store snapshot
            self.conn.execute("""
                INSERT INTO snapshots (
                    snapshot_id, snapshot_type, description, slo_policies,
                    resilience_policies, dependency_graphs, circuit_states,
                    degradation_states, chaos_config, state_hash, created_at,
                    size_bytes, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id, snapshot_type, description, slo_policies,
                resilience_policies, dependency_graphs, circuit_states,
                degradation_states, chaos_config, state_hash,
                datetime.now().isoformat(), size_bytes,
                json.dumps(metadata or {})
            ))
            
            self.conn.commit()
            
            snapshot_info = {
                "snapshot_id": snapshot_id,
                "snapshot_type": snapshot_type,
                "description": description,
                "state_hash": state_hash,
                "size_bytes": size_bytes,
                "created_at": datetime.now().isoformat(),
                "uri": f"resilience://snapshots/{snapshot_id}"
            }
            
            logger.info(f"Created resilience snapshot: {snapshot_id}")
            return snapshot_info
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise
    
    async def rollback(
        self, 
        to_snapshot: str,
        reason: str = None,
        triggered_by: str = "manual"
    ) -> Dict[str, Any]:
        """
        Rollback to a previous snapshot.
        
        Args:
            to_snapshot: Target snapshot ID
            reason: Reason for rollback
            triggered_by: Who/what triggered the rollback
            
        Returns:
            Rollback operation result
        """
        try:
            import uuid
            rollback_id = str(uuid.uuid4())
            
            # Check if target snapshot exists
            snapshot = self.get_snapshot(to_snapshot)
            if not snapshot:
                raise ValueError(f"Snapshot not found: {to_snapshot}")
            
            # Record rollback start
            self.conn.execute("""
                INSERT INTO rollback_history (
                    rollback_id, to_snapshot, reason, triggered_by,
                    status, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rollback_id, to_snapshot, reason, triggered_by,
                "in_progress", datetime.now().isoformat()
            ))
            self.conn.commit()
            
            # Perform rollback
            affected_services = await self._apply_snapshot(snapshot)
            
            # Record rollback completion
            self.conn.execute("""
                UPDATE rollback_history 
                SET status = ?, completed_at = ?, affected_services = ?
                WHERE rollback_id = ?
            """, (
                "completed", datetime.now().isoformat(),
                json.dumps(affected_services), rollback_id
            ))
            self.conn.commit()
            
            result = {
                "rollback_id": rollback_id,
                "to_snapshot": to_snapshot,
                "status": "completed",
                "affected_services": affected_services,
                "completed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Completed rollback to snapshot {to_snapshot}")
            return result
            
        except Exception as e:
            # Record rollback failure
            try:
                self.conn.execute("""
                    UPDATE rollback_history 
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE rollback_id = ?
                """, (
                    "failed", datetime.now().isoformat(), str(e), rollback_id
                ))
                self.conn.commit()
            except:
                pass
            
            logger.error(f"Failed to rollback to snapshot {to_snapshot}: {e}")
            raise
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot by ID."""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM snapshots WHERE snapshot_id = ?
            """, (snapshot_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "snapshot_id": row["snapshot_id"],
                "snapshot_type": row["snapshot_type"],
                "description": row["description"],
                "slo_policies": json.loads(row["slo_policies"]) if row["slo_policies"] else {},
                "resilience_policies": json.loads(row["resilience_policies"]) if row["resilience_policies"] else {},
                "dependency_graphs": json.loads(row["dependency_graphs"]) if row["dependency_graphs"] else {},
                "circuit_states": json.loads(row["circuit_states"]) if row["circuit_states"] else {},
                "degradation_states": json.loads(row["degradation_states"]) if row["degradation_states"] else {},
                "chaos_config": json.loads(row["chaos_config"]) if row["chaos_config"] else {},
                "state_hash": row["state_hash"],
                "created_at": row["created_at"],
                "size_bytes": row["size_bytes"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get snapshot {snapshot_id}: {e}")
            return None
    
    def list_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List available snapshots."""
        try:
            cursor = self.conn.execute("""
                SELECT snapshot_id, snapshot_type, description, state_hash,
                       created_at, size_bytes
                FROM snapshots
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            snapshots = []
            for row in cursor.fetchall():
                snapshots.append(dict(row))
            
            return snapshots
            
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
    def get_rollback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get rollback history."""
        try:
            cursor = self.conn.execute("""
                SELECT rollback_id, to_snapshot, reason, triggered_by,
                       status, started_at, completed_at, error_message,
                       affected_services
                FROM rollback_history
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                record = dict(row)
                if record["affected_services"]:
                    record["affected_services"] = json.loads(record["affected_services"])
                history.append(record)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []
    
    async def _collect_current_state(self) -> Dict[str, Any]:
        """Collect current resilience state."""
        # This would integrate with actual resilience components
        # For now, return a mock state
        
        return {
            "slo_policies": {
                "service_a": {
                    "slos": [
                        {"sli": "latency_p95_ms", "objective": 800, "window": "30d"}
                    ],
                    "error_budget_days": 0.5
                }
            },
            "resilience_policies": {
                "service_a": {
                    "retries": {"max": 2, "backoff": "exp", "base_ms": 50},
                    "circuit_breaker": {"failure_rate_threshold_pct": 25}
                }
            },
            "dependency_graphs": {},
            "circuit_states": {
                "service_a/dependency_x": "closed"
            },
            "degradation_states": {
                "service_a": []
            },
            "chaos_config": {
                "enabled": True,
                "max_blast_radius_pct": 5
            }
        }
    
    async def _apply_snapshot(self, snapshot: Dict[str, Any]) -> List[str]:
        """Apply snapshot state to system."""
        affected_services = set()
        
        try:
            # Apply SLO policies
            slo_policies = snapshot.get("slo_policies", {})
            for service_id, policy in slo_policies.items():
                # Would apply to actual SLO system
                logger.info(f"Restoring SLO policy for {service_id}")
                affected_services.add(service_id)
            
            # Apply resilience policies
            resilience_policies = snapshot.get("resilience_policies", {})
            for service_id, policy in resilience_policies.items():
                # Would apply to actual resilience system
                logger.info(f"Restoring resilience policy for {service_id}")
                affected_services.add(service_id)
            
            # Apply circuit breaker states
            circuit_states = snapshot.get("circuit_states", {})
            for circuit_id, state in circuit_states.items():
                # Would apply to actual circuit breakers
                logger.info(f"Restoring circuit breaker {circuit_id} to {state}")
                service_id = circuit_id.split('/')[0]
                affected_services.add(service_id)
            
            # Apply degradation states
            degradation_states = snapshot.get("degradation_states", {})
            for service_id, modes in degradation_states.items():
                # Would apply to degradation manager
                logger.info(f"Restoring degradation state for {service_id}")
                affected_services.add(service_id)
            
            # Simulate application delay
            await asyncio.sleep(1)
            
            return list(affected_services)
            
        except Exception as e:
            logger.error(f"Failed to apply snapshot: {e}")
            raise
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash for data integrity."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def close(self):
        """Close snapshot manager."""
        if self.conn:
            self.conn.close()
            logger.info("Snapshot manager closed")
    
    async def start(self):
        """Start snapshot manager (placeholder for async initialization)."""
        logger.info("Snapshot manager started")
    
    async def stop(self):
        """Stop snapshot manager."""
        self.close()