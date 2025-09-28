"""Learning Kernel Service - FastAPI facade for data-centric learning."""

import os
import hashlib
import sqlite3
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .registry.datasets import DatasetRegistry
from .labeling.hitl.queue import HITLQueue
from .labeling.weak.rules import WeakSupervision
from .labeling.policies.service import PolicyService
from .active.strategies import ActiveLearningStrategies
from .semi_supervised.self_train import SemiSupervisedLearning
from .augmentation.pipelines import AugmentationPipelines
from .evaluation.quality import QualityEvaluator
from .feature_store.views import FeatureStore
from .snapshots.manager import SnapshotManager
from .bridges.mesh_bridge import MeshBridge
from .bridges.mlt_bridge import MLTBridge
from .bridges.gov_bridge import GovernanceBridge


class LearningService:
    """FastAPI service facade for Learning Kernel external interface."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.app = FastAPI(title="Learning Kernel API", version="1.0.0")
        
        # Database setup
        if db_path is None:
            db_path = os.environ.get("LEARNING_DB_PATH", "/tmp/learning.db")
        self.db_path = db_path
        self._init_db()
        
        # Initialize core components
        self.dataset_registry = DatasetRegistry(self.db_path)
        self.hitl_queue = HITLQueue(self.db_path)
        self.weak_supervision = WeakSupervision(self.db_path)
        self.policy_service = PolicyService(self.db_path)
        self.active_learning = ActiveLearningStrategies(self.db_path)
        self.semi_supervised = SemiSupervisedLearning(self.db_path)
        self.augmentation = AugmentationPipelines(self.db_path)
        self.quality_evaluator = QualityEvaluator(self.db_path)
        self.feature_store = FeatureStore(self.db_path)
        self.snapshot_manager = SnapshotManager(self.db_path)
        
        # Initialize bridges
        self.mesh_bridge = MeshBridge()
        self.mlt_bridge = MLTBridge()
        self.gov_bridge = GovernanceBridge()
        
        # Register routes
        self._register_routes()
    
    def _init_db(self):
        """Initialize the database with schema."""
        # Read and execute DDL
        ddl_path = Path(__file__).parent / "db" / "ddl" / "learning.sql"
        if ddl_path.exists():
            with open(ddl_path, 'r') as f:
                ddl_sql = f.read()
            
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(ddl_sql)
                conn.commit()
            finally:
                conn.close()
    
    def register_dataset(self, manifest: dict) -> str:
        """Register a new dataset."""
        return self.dataset_registry.register(manifest)
    
    def publish_version(self, ds_id: str, version: str, refs: list[str]) -> dict:
        """Publish a new dataset version."""
        return self.dataset_registry.publish_version(ds_id, version, refs)
    
    def create_label_task(self, task: dict) -> str:
        """Create a new labeling task."""
        return self.hitl_queue.create_task(task)
    
    def submit_label(self, label: dict) -> str:
        """Submit a label."""
        return self.hitl_queue.submit_label(label)
    
    def query_active_batch(self, cfg: dict) -> dict:
        """Query active learning batch."""
        return self.active_learning.select_batch(cfg)
    
    def apply_augmentation(self, ds_id: str, version: str, spec: dict) -> dict:
        """Apply data augmentation."""
        return self.augmentation.apply_spec(ds_id, version, spec)
    
    def build_feature_view(self, ds_id: str, version: str) -> dict:
        """Build feature view for train/serve parity."""
        return self.feature_store.build_view(ds_id, version)
    
    def export_snapshot(self) -> dict:
        """Export a learning kernel snapshot."""
        return self.snapshot_manager.create_snapshot()
    
    def rollback(self, to_snapshot: str) -> dict:
        """Rollback to a specific snapshot."""
        return self.snapshot_manager.rollback(to_snapshot)
    
    def _register_routes(self):
        """Register all FastAPI routes."""
        
        @self.app.get("/api/learning/v1/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": iso_format()
            }
        
        @self.app.post("/api/learning/v1/datasets")
        async def register_dataset_endpoint(manifest: dict):
            """Register a new dataset."""
            try:
                dataset_id = self.register_dataset(manifest)
                
                # Emit event
                await self.mesh_bridge.publish_event("LEARN_DATASET_REGISTERED", {
                    "manifest": manifest
                })
                
                return {"dataset_id": dataset_id}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/learning/v1/datasets/{dataset_id}")
        async def get_dataset(dataset_id: str):
            """Get dataset manifest."""
            manifest = self.dataset_registry.get_manifest(dataset_id)
            if not manifest:
                raise HTTPException(status_code=404, detail="Dataset not found")
            return manifest
        
        @self.app.post("/api/learning/v1/versions")
        async def publish_version_endpoint(version_data: dict):
            """Publish a new dataset version."""
            try:
                result = self.publish_version(
                    version_data["dataset_id"],
                    version_data["version"],
                    version_data["source_refs"]
                )
                
                # Emit event
                await self.mesh_bridge.publish_event("LEARN_VERSION_PUBLISHED", {
                    "dataset_id": version_data["dataset_id"],
                    "version": version_data["version"]
                })
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/learning/v1/versions/{dataset_id}/{version}")
        async def get_version(dataset_id: str, version: str):
            """Get dataset version details."""
            version_data = self.dataset_registry.get_version(dataset_id, version)
            if not version_data:
                raise HTTPException(status_code=404, detail="Version not found")
            return version_data
        
        @self.app.post("/api/learning/v1/label/tasks")
        async def create_label_task_endpoint(task: dict):
            """Create a labeling task."""
            try:
                task_id = self.create_label_task(task)
                
                # Emit event
                await self.mesh_bridge.publish_event("LEARN_LABEL_TASK_CREATED", {
                    "task": task
                })
                
                return {"task_id": task_id}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/learning/v1/label/submit")
        async def submit_label_endpoint(label: dict):
            """Submit a label."""
            try:
                label_id = self.submit_label(label)
                
                # Emit event
                await self.mesh_bridge.publish_event("LEARN_LABEL_ACCEPTED", {
                    "label": label
                })
                
                return {"label_id": label_id}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/learning/v1/active/query")
        async def query_active_batch_endpoint(config: dict):
            """Query active learning batch."""
            try:
                result = self.query_active_batch(config)
                
                # Emit event if batch ready
                if "items" in result:
                    await self.mesh_bridge.publish_event("LEARN_QUERY_BATCH_READY", {
                        "dataset_id": config.get("dataset_id"),
                        "version": config.get("version"),
                        "items": result["items"],
                        "strategy": config.get("strategy")
                    })
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/learning/v1/augment/apply")
        async def apply_augmentation_endpoint(request: dict):
            """Apply data augmentation."""
            try:
                result = self.apply_augmentation(
                    request["dataset_id"],
                    request["version"],
                    request["spec"]
                )
                
                # Emit event
                await self.mesh_bridge.publish_event("LEARN_AUGMENT_APPLIED", {
                    "dataset_id": request["dataset_id"],
                    "version": request["version"],
                    "spec_id": request["spec"]["spec_id"],
                    "delta_rows": result.get("delta_rows", 0)
                })
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/learning/v1/curriculum")
        async def update_curriculum_endpoint(curriculum: dict):
            """Update curriculum specification."""
            try:
                # Store curriculum spec (simple implementation)
                spec_id = curriculum.get("spec_id", f"curr_{format_for_filename()}")
                
                # Emit event
                await self.mesh_bridge.publish_event("LEARN_CURRICULUM_UPDATED", {
                    "curriculum": curriculum
                })
                
                return {"spec_id": spec_id}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/learning/v1/feature-view/build")
        async def build_feature_view_endpoint(request: dict):
            """Build feature view."""
            try:
                result = self.build_feature_view(
                    request["dataset_id"],
                    request["version"]
                )
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/learning/v1/snapshot/export")
        async def export_snapshot_endpoint():
            """Export learning kernel snapshot."""
            try:
                result = self.export_snapshot()
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/learning/v1/rollback")
        async def rollback_endpoint(request: dict):
            """Rollback to snapshot."""
            try:
                result = self.rollback(request["to_snapshot"])
                
                # Emit rollback completed event
                await self.mesh_bridge.publish_event("ROLLBACK_COMPLETED", {
                    "target": "learning",
                    "snapshot_id": request["to_snapshot"],
                    "at": iso_format()
                })
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/learning/v1/quality/report")
        async def quality_report_endpoint(dataset_id: str, version: Optional[str] = None):
            """Get data quality report."""
            try:
                report = self.quality_evaluator.generate_report(dataset_id, version)
                
                # Emit quality report event
                await self.mesh_bridge.publish_event("LEARN_DATA_QUALITY_REPORT", {
                    "dataset_id": dataset_id,
                    "version": version,
                    "metrics": report.get("metrics", {})
                })
                
                return report
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


# Create a default instance
learning_service = LearningService()
app = learning_service.get_app()