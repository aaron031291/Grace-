"""Feature store for train/serve parity and feature view management."""

import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class FeatureStore:
    """Manages feature views for train/serve parity and feature engineering."""
    
    def __init__(self, db_path: str, storage_base: str = "/tmp/feature_store"):
        self.db_path = db_path
        self.storage_base = Path(storage_base)
        self.storage_base.mkdir(exist_ok=True)
    
    def build_view(self, dataset_id: str, version: str, format: str = "parquet") -> Dict[str, Any]:
        """Build feature view for a dataset version."""
        # Generate view ID and URI
        view_id = f"{dataset_id}@{version}"
        view_uri = f"fusion://features/{dataset_id}_{version}.{format}"
        
        # Check if view already exists
        existing_view = self.get_view(dataset_id, version)
        if existing_view and existing_view["build_status"] == "ready":
            return {
                "view_id": view_id,
                "view_uri": existing_view["view_uri"],
                "status": "already_exists",
                "built_at": existing_view["built_at"]
            }
        
        # Create or update feature view record
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO feature_views (
                    view_id, dataset_id, version, view_uri, format, build_status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                view_id, dataset_id, version, view_uri, format, "building"
            ))
            conn.commit()
        finally:
            conn.close()
        
        # Build the feature view (mock implementation)
        success = self._build_feature_view_process(dataset_id, version, view_uri, format)
        
        # Update status
        status = "ready" if success else "failed"
        built_at = datetime.now().isoformat() if success else None
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE feature_views 
                SET build_status = ?, built_at = ?
                WHERE view_id = ?
            """, (status, built_at, view_id))
            conn.commit()
        finally:
            conn.close()
        
        return {
            "view_id": view_id,
            "view_uri": view_uri if success else None,
            "status": status,
            "built_at": built_at
        }
    
    def _build_feature_view_process(self, dataset_id: str, version: str, view_uri: str, format: str) -> bool:
        """Execute the actual feature view building process."""
        try:
            # Mock feature engineering and view creation
            # In practice, this would:
            # 1. Load raw data from dataset version
            # 2. Apply feature transformations
            # 3. Ensure train/serve parity
            # 4. Export to specified format and location
            
            # Create mock feature view file
            local_path = self.storage_base / f"{dataset_id}_{version}.{format}"
            
            # Mock feature data
            mock_features = {
                "metadata": {
                    "dataset_id": dataset_id,
                    "version": version,
                    "feature_count": 50,
                    "sample_count": 10000,
                    "created_at": datetime.now().isoformat()
                },
                "schema": {
                    "features": [
                        {"name": "feature_1", "type": "numeric", "transform": "standard_scale"},
                        {"name": "feature_2", "type": "categorical", "transform": "one_hot"},
                        {"name": "text_embeddings", "type": "vector", "dim": 768},
                        {"name": "target", "type": "categorical", "classes": ["positive", "negative"]}
                    ]
                },
                "train_serve_parity": {
                    "transformation_pipeline": "sklearn_pipeline_v1.pkl",
                    "feature_names": ["feature_1_scaled", "feature_2_encoded", "text_embeddings_0_767"],
                    "validation_hash": "sha256:abc123..."
                }
            }
            
            # Write mock feature view
            with open(local_path, 'w') as f:
                json.dump(mock_features, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to build feature view: {e}")
            return False
    
    def get_view(self, dataset_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get feature view details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM feature_views
                WHERE dataset_id = ? AND version = ?
            """, (dataset_id, version))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "view_id": row["view_id"],
                "dataset_id": row["dataset_id"],
                "version": row["version"],
                "view_uri": row["view_uri"],
                "format": row["format"],
                "build_status": row["build_status"],
                "created_at": row["created_at"],
                "built_at": row["built_at"]
            }
        finally:
            conn.close()
    
    def list_views(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List feature views."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            if dataset_id:
                cursor.execute("""
                    SELECT view_id, dataset_id, version, format, build_status, built_at
                    FROM feature_views
                    WHERE dataset_id = ?
                    ORDER BY built_at DESC
                """, (dataset_id,))
            else:
                cursor.execute("""
                    SELECT view_id, dataset_id, version, format, build_status, built_at
                    FROM feature_views
                    ORDER BY built_at DESC
                """)
            
            views = []
            for row in cursor.fetchall():
                views.append({
                    "view_id": row["view_id"],
                    "dataset_id": row["dataset_id"],
                    "version": row["version"],
                    "format": row["format"],
                    "build_status": row["build_status"],
                    "built_at": row["built_at"]
                })
            
            return views
        finally:
            conn.close()
    
    def get_feature_schema(self, dataset_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get feature schema for a view."""
        view = self.get_view(dataset_id, version)
        if not view or view["build_status"] != "ready":
            return None
        
        # Mock schema retrieval - in practice would read from actual feature store
        return {
            "dataset_id": dataset_id,
            "version": version,
            "features": [
                {
                    "name": "numeric_feature_1",
                    "type": "float64",
                    "transform": "standard_scaler",
                    "nullable": False
                },
                {
                    "name": "categorical_feature_1", 
                    "type": "object",
                    "transform": "one_hot_encoder",
                    "categories": ["A", "B", "C"],
                    "nullable": False
                },
                {
                    "name": "text_embeddings",
                    "type": "float64",
                    "shape": [768],
                    "transform": "sentence_transformer",
                    "nullable": False
                },
                {
                    "name": "target",
                    "type": "int64", 
                    "transform": "label_encoder",
                    "classes": ["negative", "positive"],
                    "nullable": False
                }
            ],
            "train_serve_parity": {
                "pipeline_path": f"models/{dataset_id}_{version}_pipeline.pkl",
                "validation_data_hash": "sha256:def456...",
                "last_validated": datetime.now().isoformat()
            }
        }
    
    def validate_train_serve_parity(self, dataset_id: str, version: str) -> Dict[str, Any]:
        """Validate train/serve parity for a feature view."""
        view = self.get_view(dataset_id, version)
        if not view:
            return {"valid": False, "error": "Feature view not found"}
        
        if view["build_status"] != "ready":
            return {"valid": False, "error": f"Feature view status: {view['build_status']}"}
        
        # Mock parity validation - in practice would run comprehensive checks
        validation_results = {
            "feature_schema_match": True,
            "transformation_consistency": True,
            "data_type_consistency": True,
            "missing_value_handling": True,
            "feature_order_match": True
        }
        
        all_valid = all(validation_results.values())
        
        issues = []
        if not validation_results["feature_schema_match"]:
            issues.append("Feature schema mismatch between training and serving")
        if not validation_results["transformation_consistency"]:
            issues.append("Transformation pipeline inconsistency detected")
        if not validation_results["data_type_consistency"]:
            issues.append("Data type mismatches found")
        
        return {
            "valid": all_valid,
            "dataset_id": dataset_id,
            "version": version,
            "validation_details": validation_results,
            "issues": issues,
            "validated_at": datetime.now().isoformat()
        }
    
    def create_feature_view_snapshot(self, dataset_id: str, version: str) -> Dict[str, Any]:
        """Create a snapshot of a feature view for reproducibility."""
        view = self.get_view(dataset_id, version)
        if not view:
            raise ValueError("Feature view not found")
        
        snapshot_id = f"fv_snapshot_{dataset_id}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Mock snapshot creation
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "source_view": view,
            "feature_schema": self.get_feature_schema(dataset_id, version),
            "parity_validation": self.validate_train_serve_parity(dataset_id, version),
            "created_at": datetime.now().isoformat()
        }
        
        # In practice, would store snapshot to persistent storage
        snapshot_path = self.storage_base / f"{snapshot_id}.json"
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        return {
            "snapshot_id": snapshot_id,
            "snapshot_uri": f"fusion://snapshots/{snapshot_id}.json",
            "created_at": snapshot_data["created_at"]
        }
    
    def get_view_metrics(self, dataset_id: str, version: str) -> Dict[str, Any]:
        """Get performance and usage metrics for a feature view."""
        view = self.get_view(dataset_id, version)
        if not view:
            return {}
        
        # Mock metrics - in practice would collect from monitoring systems
        return {
            "dataset_id": dataset_id,
            "version": version,
            "usage_stats": {
                "total_requests": random.randint(100, 10000),
                "unique_consumers": random.randint(5, 50),
                "avg_response_time_ms": random.randint(10, 100),
                "error_rate": random.uniform(0.001, 0.01)
            },
            "data_stats": {
                "feature_count": 50,
                "sample_count": random.randint(1000, 100000),
                "last_updated": view["built_at"],
                "freshness_hours": random.randint(1, 24)
            },
            "quality_metrics": {
                "feature_drift_score": random.uniform(0.0, 0.3),
                "missing_value_rate": random.uniform(0.0, 0.05),
                "outlier_rate": random.uniform(0.01, 0.1)
            },
            "computed_at": datetime.now().isoformat()
        }
    
    def rebuild_view(self, dataset_id: str, version: str) -> Dict[str, Any]:
        """Rebuild an existing feature view."""
        # Mark as building
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE feature_views 
                SET build_status = 'building', built_at = NULL
                WHERE dataset_id = ? AND version = ?
            """, (dataset_id, version))
            conn.commit()
        finally:
            conn.close()
        
        # Rebuild
        return self.build_view(dataset_id, version)