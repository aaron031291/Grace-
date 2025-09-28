"""Dataset registry and versioning management."""

import json
import hashlib
import sqlite3
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Any


class DatasetRegistry:
    """Manages dataset registration, versioning, and lineage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def register(self, manifest: Dict[str, Any]) -> str:
        """Register a new dataset manifest."""
        # Validate required fields
        required_fields = ["dataset_id", "task", "modality", "versions", "default_version"]
        for field in required_fields:
            if field not in manifest:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate dataset_id pattern
        dataset_id = manifest["dataset_id"]
        if not dataset_id.startswith("ds_") or len(dataset_id) < 7:
            raise ValueError("dataset_id must match pattern 'ds_[a-z0-9_-]{4,}'")
        
        # Validate enums
        valid_tasks = ["classification", "regression", "clustering", "dimred", "rl", "nlp", "vision", "timeseries"]
        valid_modalities = ["tabular", "text", "image", "audio", "video", "graph"]
        
        if manifest["task"] not in valid_tasks:
            raise ValueError(f"Invalid task. Must be one of: {valid_tasks}")
        
        if manifest["modality"] not in valid_modalities:
            raise ValueError(f"Invalid modality. Must be one of: {valid_modalities}")
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO datasets (
                    dataset_id, task, modality, schema_json, default_version
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                dataset_id,
                manifest["task"],
                manifest["modality"],
                json.dumps(manifest.get("schema", {})),
                manifest["default_version"]
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Dataset {dataset_id} already exists")
        finally:
            conn.close()
        
        return dataset_id
    
    def publish_version(self, dataset_id: str, version: str, refs: List[str]) -> Dict[str, Any]:
        """Publish a new dataset version."""
        # Validate dataset exists
        if not self.get_manifest(dataset_id):
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Generate lineage hash
        lineage_data = {
            "dataset_id": dataset_id,
            "version": version,
            "source_refs": sorted(refs),
            "timestamp": iso_format()
        }
        lineage_json = json.dumps(lineage_data, sort_keys=True)
        lineage_hash = hashlib.sha256(lineage_json.encode()).hexdigest()
        
        # Store version
        version_id = f"{dataset_id}@{version}"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO dataset_versions (
                    version_id, dataset_id, version, source_refs_json,
                    row_count, byte_size, lineage_hash, governance_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id, dataset_id, version, json.dumps(refs),
                0, 0, lineage_hash, "internal"  # defaults
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Version {version} already exists for dataset {dataset_id}")
        finally:
            conn.close()
        
        return {
            "dataset_id": dataset_id,
            "version": version,
            "lineage_hash": lineage_hash,
            "version_id": version_id
        }
    
    def get_manifest(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset manifest."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.*, GROUP_CONCAT(dv.version) as versions
                FROM datasets d
                LEFT JOIN dataset_versions dv ON d.dataset_id = dv.dataset_id
                WHERE d.dataset_id = ?
                GROUP BY d.dataset_id
            """, (dataset_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "dataset_id": row["dataset_id"],
                "task": row["task"],
                "modality": row["modality"],
                "schema": json.loads(row["schema_json"] or "{}"),
                "versions": row["versions"].split(",") if row["versions"] else [],
                "default_version": row["default_version"]
            }
        finally:
            conn.close()
    
    def get_version(self, dataset_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get specific dataset version details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM dataset_versions 
                WHERE dataset_id = ? AND version = ?
            """, (dataset_id, version))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "dataset_id": row["dataset_id"],
                "version": row["version"],
                "source_refs": json.loads(row["source_refs_json"]),
                "size": {
                    "rows": row["row_count"] or 0,
                    "bytes": row["byte_size"] or 0
                },
                "stats": json.loads(row["stats_json"] or "{}"),
                "splits": {
                    "train": row["train_split"],
                    "valid": row["valid_split"],
                    "test": row["test_split"]
                },
                "feature_view": row["feature_view"],
                "lineage_hash": row["lineage_hash"],
                "governance_label": row["governance_label"]
            }
        finally:
            conn.close()
    
    def list_datasets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all registered datasets."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.*, COUNT(dv.version) as version_count
                FROM datasets d
                LEFT JOIN dataset_versions dv ON d.dataset_id = dv.dataset_id
                GROUP BY d.dataset_id
                ORDER BY d.created_at DESC
                LIMIT ?
            """, (limit,))
            
            datasets = []
            for row in cursor.fetchall():
                datasets.append({
                    "dataset_id": row["dataset_id"],
                    "task": row["task"],
                    "modality": row["modality"],
                    "default_version": row["default_version"],
                    "version_count": row["version_count"],
                    "created_at": row["created_at"]
                })
            
            return datasets
        finally:
            conn.close()
    
    def list_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        """List all versions for a dataset."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, row_count, byte_size, governance_label, created_at
                FROM dataset_versions
                WHERE dataset_id = ?
                ORDER BY created_at DESC
            """, (dataset_id,))
            
            versions = []
            for row in cursor.fetchall():
                versions.append({
                    "version": row["version"],
                    "size": {
                        "rows": row["row_count"] or 0,
                        "bytes": row["byte_size"] or 0
                    },
                    "governance_label": row["governance_label"],
                    "created_at": row["created_at"]
                })
            
            return versions
        finally:
            conn.close()
    
    def update_version_stats(self, dataset_id: str, version: str, stats: Dict[str, Any]):
        """Update version statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE dataset_versions
                SET row_count = ?, byte_size = ?, stats_json = ?
                WHERE dataset_id = ? AND version = ?
            """, (
                stats.get("rows", 0),
                stats.get("bytes", 0),
                json.dumps(stats),
                dataset_id,
                version
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_lineage(self, dataset_id: str, version: str) -> Dict[str, Any]:
        """Get dataset version lineage information."""
        version_data = self.get_version(dataset_id, version)
        if not version_data:
            return {}
        
        return {
            "dataset_id": dataset_id,
            "version": version,
            "source_refs": version_data["source_refs"],
            "lineage_hash": version_data["lineage_hash"],
            "governance_label": version_data["governance_label"]
        }