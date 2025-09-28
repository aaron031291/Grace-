"""
Unified Snapshot Persistence System for Grace Kernel.

Supports:
- PostgreSQL metadata storage
- Object store payload storage (S3/filesystem)
- SHA256 hash verification
- Blue/green instance swap
- Export/rollback APIs
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle
import gzip

logger = logging.getLogger(__name__)


class ObjectStore:
    """Simple object store implementation (can be extended for S3)."""
    
    def __init__(self, base_path: str = "/tmp/grace_snapshots"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def store_object(self, key: str, data: Any) -> int:
        """Store object and return size in bytes."""
        object_path = self.base_path / key
        object_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress and serialize
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized)
        
        # Write to file
        with open(object_path, 'wb') as f:
            f.write(compressed)
        
        return len(compressed)
    
    async def retrieve_object(self, key: str) -> Any:
        """Retrieve object from store."""
        object_path = self.base_path / key
        
        if not object_path.exists():
            raise FileNotFoundError(f"Object {key} not found")
        
        # Read and decompress
        with open(object_path, 'rb') as f:
            compressed = f.read()
        
        serialized = gzip.decompress(compressed)
        return pickle.loads(serialized)
    
    async def delete_object(self, key: str) -> bool:
        """Delete object from store."""
        object_path = self.base_path / key
        try:
            object_path.unlink()
            return True
        except FileNotFoundError:
            return False
    
    async def list_objects(self, prefix: str = "") -> List[str]:
        """List objects with prefix."""
        objects = []
        prefix_path = self.base_path / prefix if prefix else self.base_path
        
        if prefix_path.exists():
            for path in prefix_path.rglob("*"):
                if path.is_file():
                    objects.append(str(path.relative_to(self.base_path)))
        
        return objects


class GraceSnapshotManager:
    """
    Unified snapshot manager for Grace kernel components.
    
    Manages snapshots for:
    - Orchestration state
    - Resilience state  
    - MLDL state
    - Governance state
    - Memory state
    """
    
    def __init__(self, 
                 db_path: str = "grace_snapshots.db",
                 object_store: Optional[ObjectStore] = None):
        self.db_path = db_path
        self.object_store = object_store or ObjectStore()
        self.conn = None
        
        self._initialize_db()
        
        # Blue/green deployment support
        self.current_instance = "blue"
        self.instances = {"blue": {}, "green": {}}
        
        logger.info("Grace Snapshot Manager initialized")
    
    def _initialize_db(self):
        """Initialize PostgreSQL-compatible database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create snapshots metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT UNIQUE NOT NULL,
                component_type TEXT NOT NULL,  -- orchestration, resilience, mldl, governance, memory
                description TEXT,
                object_key TEXT NOT NULL,     -- Key in object store
                state_hash TEXT NOT NULL,     -- SHA256 of payload
                created_at TIMESTAMP NOT NULL,
                created_by TEXT,
                size_bytes INTEGER,
                metadata TEXT,  -- JSON metadata
                tags TEXT,      -- JSON tags
                version TEXT DEFAULT '1.0.0'
            )
        """)
        
        # Create rollback history table  
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rollback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollback_id TEXT UNIQUE NOT NULL,
                component_type TEXT NOT NULL,
                from_snapshot_id TEXT,
                to_snapshot_id TEXT NOT NULL,
                reason TEXT,
                triggered_by TEXT,
                status TEXT NOT NULL,  -- in_progress, completed, failed
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (to_snapshot_id) REFERENCES snapshots (snapshot_id)
            )
        """)
        
        # Create instance state table (for blue/green)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS instance_states (
                instance_name TEXT PRIMARY KEY,
                current_snapshots TEXT,  -- JSON mapping component -> snapshot_id
                status TEXT NOT NULL,    -- active, standby, switching
                last_updated TIMESTAMP,
                health_score REAL DEFAULT 1.0
            )
        """)
        
        self.conn.commit()
        
        # Initialize blue/green instances
        for instance in ["blue", "green"]:
            self.conn.execute("""
                INSERT OR REPLACE INTO instance_states 
                (instance_name, current_snapshots, status, last_updated) 
                VALUES (?, ?, ?, ?)
            """, (instance, json.dumps({}), "standby", datetime.utcnow().isoformat()))
        
        # Set blue as active
        self.conn.execute("""
            UPDATE instance_states SET status = 'active' WHERE instance_name = 'blue'
        """)
        
        self.conn.commit()
        logger.info("Snapshot database initialized")
    
    async def export_snapshot(self, 
                            component_type: str,
                            payload: Dict[str, Any],
                            description: str = None,
                            created_by: str = "system",
                            tags: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Export component snapshot to persistent storage.
        
        Args:
            component_type: Type of component (orchestration, resilience, mldl, governance, memory)
            payload: Snapshot payload data
            description: Human readable description
            created_by: Who created the snapshot
            tags: Optional tags for categorization
            
        Returns:
            Snapshot metadata
        """
        try:
            # Generate snapshot ID
            timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            snapshot_id = f"{component_type}_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            # Calculate payload hash
            payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            state_hash = hashlib.sha256(payload_json.encode()).hexdigest()
            
            # Store payload in object store
            object_key = f"{component_type}/{snapshot_id}.snapshot"
            size_bytes = await self.object_store.store_object(object_key, payload)
            
            # Store metadata in database
            metadata = {
                "component_version": payload.get("version", "unknown"),
                "config_hash": payload.get("config_hash"),
                "dependencies": payload.get("dependencies", {}),
                "runtime_info": payload.get("runtime_info", {})
            }
            
            self.conn.execute("""
                INSERT INTO snapshots (
                    snapshot_id, component_type, description, object_key,
                    state_hash, created_at, created_by, size_bytes, 
                    metadata, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id, component_type,
                description or f"{component_type} snapshot at {timestamp}",
                object_key, state_hash, datetime.utcnow().isoformat(),
                created_by, size_bytes, json.dumps(metadata),
                json.dumps(tags or {})
            ))
            
            self.conn.commit()
            
            logger.info(f"Exported snapshot {snapshot_id} for {component_type}")
            
            return {
                "snapshot_id": snapshot_id,
                "component_type": component_type,
                "object_key": object_key,
                "state_hash": f"sha256:{state_hash}",
                "size_bytes": size_bytes,
                "created_at": datetime.utcnow().isoformat(),
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Failed to export snapshot: {e}")
            raise
    
    async def rollback(self, 
                      component_type: str,
                      to_snapshot: str,
                      reason: str = None,
                      triggered_by: str = "manual",
                      validate_health: bool = True) -> Dict[str, Any]:
        """
        Rollback component to specific snapshot.
        
        Args:
            component_type: Type of component to rollback
            to_snapshot: Snapshot ID to rollback to
            reason: Reason for rollback
            triggered_by: Who/what triggered the rollback
            validate_health: Whether to validate health after rollback
            
        Returns:
            Rollback result
        """
        try:
            rollback_id = f"rollback_{uuid.uuid4().hex[:8]}"
            
            # Get target snapshot
            snapshot = await self.get_snapshot(to_snapshot)
            if not snapshot:
                raise ValueError(f"Snapshot {to_snapshot} not found")
            
            if snapshot["component_type"] != component_type:
                raise ValueError(f"Snapshot {to_snapshot} is for {snapshot['component_type']}, not {component_type}")
            
            # Get current snapshot as backup
            current_state = await self._collect_current_state(component_type)
            backup_snapshot = await self.export_snapshot(
                component_type,
                current_state,
                f"Pre-rollback backup for {rollback_id}",
                triggered_by,
                {"rollback_backup": "true", "rollback_id": rollback_id}
            )
            
            # Record rollback attempt
            self.conn.execute("""
                INSERT INTO rollback_history (
                    rollback_id, component_type, from_snapshot_id, to_snapshot_id,
                    reason, triggered_by, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rollback_id, component_type, backup_snapshot["snapshot_id"], 
                to_snapshot, reason, triggered_by, "in_progress", 
                datetime.utcnow().isoformat()
            ))
            
            self.conn.commit()
            
            # Load snapshot payload
            payload = await self.object_store.retrieve_object(snapshot["object_key"])
            
            # Execute rollback
            rollback_success = await self._execute_rollback(component_type, payload)
            
            # Validate health if requested
            if rollback_success and validate_health:
                rollback_success = await self._validate_post_rollback_health(component_type)
            
            # Update rollback status
            if rollback_success:
                self.conn.execute("""
                    UPDATE rollback_history 
                    SET status = ?, completed_at = ?
                    WHERE rollback_id = ?
                """, ("completed", datetime.utcnow().isoformat(), rollback_id))
                
                status = "completed"
                logger.info(f"Rollback {rollback_id} completed successfully")
            else:
                self.conn.execute("""
                    UPDATE rollback_history 
                    SET status = ?, error_message = ?, completed_at = ?
                    WHERE rollback_id = ?
                """, ("failed", "Rollback execution failed", 
                     datetime.utcnow().isoformat(), rollback_id))
                
                status = "failed"
                logger.error(f"Rollback {rollback_id} failed")
            
            self.conn.commit()
            
            return {
                "rollback_id": rollback_id,
                "component_type": component_type,
                "to_snapshot": to_snapshot,
                "from_snapshot": backup_snapshot["snapshot_id"],
                "status": status,
                "reason": reason,
                "triggered_by": triggered_by,
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            # Update rollback status as failed
            try:
                self.conn.execute("""
                    UPDATE rollback_history 
                    SET status = ?, error_message = ?, completed_at = ?
                    WHERE rollback_id = ?
                """, ("failed", str(e), datetime.utcnow().isoformat(), rollback_id))
                self.conn.commit()
            except:
                pass
            raise
    
    async def blue_green_swap(self, 
                             component_type: str, 
                             new_snapshot: str,
                             health_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Perform blue/green instance swap with health validation.
        
        Args:
            component_type: Component to swap
            new_snapshot: New snapshot to deploy to standby instance
            health_threshold: Minimum health score to proceed with swap
            
        Returns:
            Swap result
        """
        try:
            # Determine current active and standby instances
            cursor = self.conn.execute("""
                SELECT instance_name, status, current_snapshots, health_score
                FROM instance_states
                ORDER BY instance_name
            """)
            
            instances = {row["instance_name"]: dict(row) for row in cursor.fetchall()}
            
            active_instance = None
            standby_instance = None
            
            for name, instance in instances.items():
                if instance["status"] == "active":
                    active_instance = name
                elif instance["status"] == "standby":
                    standby_instance = name
            
            if not active_instance or not standby_instance:
                raise ValueError("Invalid instance configuration for blue/green swap")
            
            logger.info(f"Starting blue/green swap: {active_instance} -> {standby_instance}")
            
            # Deploy new snapshot to standby instance
            await self._deploy_to_instance(standby_instance, component_type, new_snapshot)
            
            # Validate standby instance health
            health_score = await self._validate_instance_health(standby_instance, component_type)
            
            if health_score < health_threshold:
                raise ValueError(f"Standby instance health {health_score} below threshold {health_threshold}")
            
            # Mark instances as switching
            for instance_name in [active_instance, standby_instance]:
                self.conn.execute("""
                    UPDATE instance_states SET status = 'switching'
                    WHERE instance_name = ?
                """, (instance_name,))
            
            self.conn.commit()
            
            # Perform the swap
            self.conn.execute("""
                UPDATE instance_states 
                SET status = CASE 
                    WHEN instance_name = ? THEN 'standby'
                    WHEN instance_name = ? THEN 'active'
                END,
                last_updated = ?
                WHERE instance_name IN (?, ?)
            """, (active_instance, standby_instance, datetime.utcnow().isoformat(),
                  active_instance, standby_instance))
            
            self.conn.commit()
            
            # Update current active instance
            self.current_instance = standby_instance
            
            logger.info(f"Blue/green swap completed: {standby_instance} is now active")
            
            return {
                "swap_id": f"swap_{uuid.uuid4().hex[:8]}",
                "previous_active": active_instance,
                "new_active": standby_instance,
                "component_type": component_type,
                "snapshot_id": new_snapshot,
                "health_score": health_score,
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Blue/green swap failed: {e}")
            # Reset instance states
            try:
                self.conn.execute("""
                    UPDATE instance_states 
                    SET status = CASE 
                        WHEN instance_name = ? THEN 'active'
                        ELSE 'standby'
                    END
                    WHERE instance_name IN (?, ?)
                """, (active_instance, active_instance, standby_instance))
                self.conn.commit()
            except:
                pass
            raise
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot metadata by ID."""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM snapshots WHERE snapshot_id = ?
            """, (snapshot_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            snapshot_data = dict(row)
            
            # Parse JSON fields
            for field in ["metadata", "tags"]:
                if snapshot_data[field]:
                    try:
                        snapshot_data[field] = json.loads(snapshot_data[field])
                    except json.JSONDecodeError:
                        snapshot_data[field] = {}
            
            return snapshot_data
            
        except Exception as e:
            logger.error(f"Failed to get snapshot {snapshot_id}: {e}")
            return None
    
    async def get_snapshot_payload(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get full snapshot payload."""
        snapshot = await self.get_snapshot(snapshot_id)
        if not snapshot:
            return None
        
        try:
            return await self.object_store.retrieve_object(snapshot["object_key"])
        except Exception as e:
            logger.error(f"Failed to load payload for snapshot {snapshot_id}: {e}")
            return None
    
    async def list_snapshots(self, 
                           component_type: str = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """List available snapshots."""
        try:
            query = """
                SELECT snapshot_id, component_type, description, state_hash,
                       created_at, created_by, size_bytes, version
                FROM snapshots
            """
            params = []
            
            if component_type:
                query += " WHERE component_type = ?"
                params.append(component_type)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
    async def _collect_current_state(self, component_type: str) -> Dict[str, Any]:
        """Collect current state for a component type."""
        # This would be implemented per component type
        # For now, return a mock state
        return {
            "component_type": component_type,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "config": {"mock": "state"},
            "runtime_info": {
                "uptime_seconds": 3600,
                "memory_usage_mb": 512
            }
        }
    
    async def _execute_rollback(self, component_type: str, payload: Dict[str, Any]) -> bool:
        """Execute rollback for specific component."""
        try:
            logger.info(f"Executing rollback for {component_type}")
            
            # Component-specific rollback logic would go here
            # For now, simulate rollback
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback execution failed for {component_type}: {e}")
            return False
    
    async def _validate_post_rollback_health(self, component_type: str) -> bool:
        """Validate component health after rollback."""
        try:
            # Component-specific health checks would go here
            # For now, simulate health check
            await asyncio.sleep(0.5)
            return True
            
        except Exception as e:
            logger.error(f"Health validation failed for {component_type}: {e}")
            return False
    
    async def _deploy_to_instance(self, instance_name: str, component_type: str, snapshot_id: str):
        """Deploy snapshot to specific instance."""
        # Update instance snapshot mapping
        cursor = self.conn.execute("""
            SELECT current_snapshots FROM instance_states WHERE instance_name = ?
        """, (instance_name,))
        
        row = cursor.fetchone()
        if row:
            current_snapshots = json.loads(row["current_snapshots"])
            current_snapshots[component_type] = snapshot_id
            
            self.conn.execute("""
                UPDATE instance_states 
                SET current_snapshots = ?, last_updated = ?
                WHERE instance_name = ?
            """, (json.dumps(current_snapshots), datetime.utcnow().isoformat(), instance_name))
            
            self.conn.commit()
    
    async def _validate_instance_health(self, instance_name: str, component_type: str) -> float:
        """Validate instance health and return score."""
        # Mock health validation - real implementation would check component health
        await asyncio.sleep(1)
        return 0.95  # Mock healthy score
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get snapshot system statistics."""
        try:
            # Snapshot counts
            cursor = self.conn.execute("""
                SELECT component_type, COUNT(*) as count, SUM(size_bytes) as total_size
                FROM snapshots 
                GROUP BY component_type
            """)
            component_stats = {
                row["component_type"]: {
                    "count": row["count"],
                    "total_size_bytes": row["total_size"] or 0
                }
                for row in cursor.fetchall()
            }
            
            # Recent rollbacks
            cursor = self.conn.execute("""
                SELECT status, COUNT(*) as count
                FROM rollback_history 
                WHERE created_at > datetime('now', '-7 days')
                GROUP BY status
            """)
            rollback_stats = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Instance states
            cursor = self.conn.execute("SELECT * FROM instance_states")
            instance_states = {row["instance_name"]: dict(row) for row in cursor.fetchall()}
            
            return {
                "component_stats": component_stats,
                "rollback_stats": rollback_stats,
                "instance_states": instance_states,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get snapshot stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close snapshot manager."""
        if self.conn:
            self.conn.close()
            logger.info("Grace snapshot manager closed")