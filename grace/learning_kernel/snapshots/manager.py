"""Snapshot manager for Learning Kernel state and rollback capabilities."""

import json
import sqlite3
import hashlib
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Any
from pathlib import Path


class SnapshotManager:
    """Manages versioned snapshots of Learning Kernel state for rollback capabilities."""
    
    def __init__(self, db_path: str, storage_path: str = "/tmp/learning_snapshots"):
        self.db_path = db_path
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def create_snapshot(self, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a snapshot of current Learning Kernel state."""
        snapshot_id = f"learn_{utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}"
        
        # Collect current state from all components
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "description": description or "Automated snapshot",
            "created_at": iso_format(),
            "datasets": self._collect_datasets(),
            "versions": self._collect_dataset_versions(),
            "feature_views": self._collect_feature_views(),
            "label_policies_version": self._get_policies_version(),
            "active_query": self._collect_active_query_config(),
            "weak_labelers": self._collect_weak_labelers_config(),
            "augmentation": self._collect_augmentation_config(),
            "quality_thresholds": self._collect_quality_thresholds()
        }
        
        # Calculate state hash for integrity verification
        state_hash = self._calculate_snapshot_hash(snapshot_data)
        snapshot_data["hash"] = state_hash
        
        # Store snapshot in database and file system
        self._store_snapshot(snapshot_data)
        
        return {
            "snapshot_id": snapshot_id,
            "uri": f"learning://snapshots/{snapshot_id}.json",
            "hash": state_hash,
            "created_at": snapshot_data["created_at"]
        }
    
    def _collect_datasets(self) -> List[str]:
        """Collect list of all datasets."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT dataset_id FROM datasets")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def _collect_dataset_versions(self) -> Dict[str, str]:
        """Collect dataset to latest version mapping."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dataset_id, version, created_at,
                       ROW_NUMBER() OVER (PARTITION BY dataset_id ORDER BY created_at DESC) as rn
                FROM dataset_versions
            """)
            
            versions = {}
            for row in cursor.fetchall():
                if row["rn"] == 1:  # Latest version
                    versions[row["dataset_id"]] = row["version"]
            
            return versions
        finally:
            conn.close()
    
    def _collect_feature_views(self) -> Dict[str, str]:
        """Collect feature view mappings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dataset_id, version, view_uri
                FROM feature_views
                WHERE build_status = 'ready'
            """)
            
            feature_views = {}
            for row in cursor.fetchall():
                key = f"{row['dataset_id']}@{row['version']}"
                feature_views[key] = row["view_uri"]
            
            return feature_views
        finally:
            conn.close()
    
    def _get_policies_version(self) -> str:
        """Get current policies version hash."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT policy_id, rubric_json, gold_ratio, dual_label_ratio, min_agreement
                FROM label_policies
                ORDER BY policy_id
            """)
            
            policies_data = cursor.fetchall()
            policies_str = json.dumps(policies_data, sort_keys=True)
            return hashlib.sha256(policies_str.encode()).hexdigest()[:16]
        finally:
            conn.close()
    
    def _collect_active_query_config(self) -> Dict[str, Any]:
        """Collect active learning query configuration."""
        # Get most recent query configurations
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT strategy, AVG(batch_size) as avg_batch_size, COUNT(*) as query_count
                FROM active_queries
                WHERE created_at >= datetime('now', '-7 days')
                GROUP BY strategy
                ORDER BY query_count DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    "strategy": row["strategy"],
                    "batch_size": int(row["avg_batch_size"])
                }
            else:
                return {
                    "strategy": "hybrid", 
                    "batch_size": 128
                }
        finally:
            conn.close()
    
    def _collect_weak_labelers_config(self) -> Dict[str, Any]:
        """Collect weak supervision configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT labeler_id, name, labeler_type, threshold, active
                FROM weak_labelers
                WHERE active = 1
            """)
            
            labelers = {}
            for row in cursor.fetchall():
                labelers[f"{row['name']}@{row['labeler_id'][:8]}"] = {
                    "threshold": row["threshold"],
                    "type": row["labeler_type"]
                }
            
            return labelers
        finally:
            conn.close()
    
    def _collect_augmentation_config(self) -> Dict[str, Any]:
        """Collect data augmentation configuration."""
        # Mock augmentation config - in practice would collect from augmentation service
        return {
            "text": {
                "back_translate": ["en-de", "en-es"],
                "synonym_swap_prob": 0.1
            },
            "image": {
                "mixup": {"alpha": 0.4},
                "rand_crop_prob": 0.2
            }
        }
    
    def _collect_quality_thresholds(self) -> Dict[str, Any]:
        """Collect quality evaluation thresholds."""
        # Mock quality thresholds - in practice would be configurable
        return {
            "min_agreement": 0.8,
            "max_bias_score": 0.15,
            "max_drift_psi": 0.25,
            "min_coverage": 0.7
        }
    
    def _calculate_snapshot_hash(self, snapshot_data: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of snapshot data for integrity verification."""
        # Remove hash field if present and create deterministic representation
        data_copy = snapshot_data.copy()
        data_copy.pop("hash", None)
        data_copy.pop("created_at", None)  # Exclude timestamp from hash
        
        # Create deterministic JSON representation
        json_str = json.dumps(data_copy, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _store_snapshot(self, snapshot_data: Dict[str, Any]):
        """Store snapshot in database and file system."""
        snapshot_id = snapshot_data["snapshot_id"]
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO learning_snapshots (
                    snapshot_id, datasets_json, versions_json, feature_views_json,
                    policies_version, active_query_config_json, weak_labelers_json,
                    augmentation_config_json, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                json.dumps(snapshot_data["datasets"]),
                json.dumps(snapshot_data["versions"]),
                json.dumps(snapshot_data["feature_views"]),
                snapshot_data["label_policies_version"],
                json.dumps(snapshot_data["active_query"]),
                json.dumps(snapshot_data["weak_labelers"]),
                json.dumps(snapshot_data["augmentation"]),
                snapshot_data["hash"]
            ))
            conn.commit()
        finally:
            conn.close()
        
        # Store in file system
        snapshot_file = self.storage_path / f"{snapshot_id}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
    
    def rollback(self, to_snapshot: str) -> Dict[str, Any]:
        """Rollback Learning Kernel to a specific snapshot."""
        # Load snapshot data
        snapshot_data = self.load_snapshot(to_snapshot)
        if not snapshot_data:
            raise ValueError(f"Snapshot {to_snapshot} not found")
        
        # Verify hash integrity
        expected_hash = snapshot_data["hash"]
        actual_hash = self._calculate_snapshot_hash(snapshot_data)
        if expected_hash != actual_hash:
            raise ValueError(f"Snapshot {to_snapshot} integrity check failed")
        
        try:
            # 1. Freeze active operations (mock implementation)
            self._freeze_operations()
            
            # 2. Restore dataset version pointers
            self._restore_dataset_versions(snapshot_data["versions"])
            
            # 3. Restore feature views
            self._restore_feature_views(snapshot_data["feature_views"])
            
            # 4. Restore configurations
            self._restore_active_query_config(snapshot_data["active_query"])
            self._restore_weak_labelers_config(snapshot_data["weak_labelers"])
            self._restore_augmentation_config(snapshot_data["augmentation"])
            
            # 5. Resume operations
            self._resume_operations()
            
            rollback_result = {
                "snapshot_id": to_snapshot,
                "rollback_completed_at": iso_format(),
                "status": "success",
                "restored_components": [
                    "dataset_versions", "feature_views", "active_query_config", 
                    "weak_labelers", "augmentation_config"
                ]
            }
            
            return rollback_result
            
        except Exception as e:
            # Attempt to resume operations even on failure
            self._resume_operations()
            raise ValueError(f"Rollback failed: {e}")
    
    def _freeze_operations(self):
        """Freeze learning operations during rollback."""
        # Mock implementation - in practice would:
        # - Pause labeling queues
        # - Stop active learning queries  
        # - Checkpoint in-flight work
        pass
    
    def _resume_operations(self):
        """Resume learning operations after rollback."""
        # Mock implementation - in practice would:
        # - Resume labeling queues
        # - Restart active learning
        # - Process any checkpointed work
        pass
    
    def _restore_dataset_versions(self, versions: Dict[str, str]):
        """Restore dataset version pointers."""
        conn = sqlite3.connect(self.db_path)
        try:
            for dataset_id, version in versions.items():
                conn.execute("""
                    UPDATE datasets 
                    SET default_version = ?
                    WHERE dataset_id = ?
                """, (version, dataset_id))
            conn.commit()
        finally:
            conn.close()
    
    def _restore_feature_views(self, feature_views: Dict[str, str]):
        """Restore feature view configurations."""
        # Mock implementation - in practice would rebuild views if necessary
        pass
    
    def _restore_active_query_config(self, config: Dict[str, Any]):
        """Restore active learning query configuration."""
        # Mock implementation - in practice would update service configuration
        pass
    
    def _restore_weak_labelers_config(self, config: Dict[str, Any]):
        """Restore weak supervision configuration."""
        # Mock implementation - in practice would update labeler states
        pass
    
    def _restore_augmentation_config(self, config: Dict[str, Any]):
        """Restore augmentation configuration."""
        # Mock implementation - in practice would update augmentation pipelines
        pass
    
    def load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Load snapshot data from storage."""
        # Try file system first
        snapshot_file = self.storage_path / f"{snapshot_id}.json"
        if snapshot_file.exists():
            with open(snapshot_file, 'r') as f:
                return json.load(f)
        
        # Fall back to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM learning_snapshots WHERE snapshot_id = ?
            """, (snapshot_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "snapshot_id": row["snapshot_id"],
                "datasets": json.loads(row["datasets_json"]),
                "versions": json.loads(row["versions_json"]),
                "feature_views": json.loads(row["feature_views_json"]),
                "label_policies_version": row["policies_version"],
                "active_query": json.loads(row["active_query_config_json"]),
                "weak_labelers": json.loads(row["weak_labelers_json"]),
                "augmentation": json.loads(row["augmentation_config_json"]),
                "hash": row["hash"],
                "created_at": row["created_at"]
            }
        finally:
            conn.close()
    
    def list_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List available snapshots."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT snapshot_id, created_at
                FROM learning_snapshots
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            snapshots = []
            for row in cursor.fetchall():
                snapshots.append({
                    "snapshot_id": row["snapshot_id"],
                    "created_at": row["created_at"]
                })
            
            return snapshots
        finally:
            conn.close()
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        # Delete from file system
        snapshot_file = self.storage_path / f"{snapshot_id}.json"
        if snapshot_file.exists():
            snapshot_file.unlink()
        
        # Delete from database
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM learning_snapshots WHERE snapshot_id = ?", (snapshot_id,))
            return cursor.rowcount > 0
        finally:
            conn.close()