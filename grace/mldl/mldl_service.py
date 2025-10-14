"""
MLDL Service - FastAPI facade for the Model Lifecycle (train, evaluate, deploy).
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
import uuid

from .training.job import TrainingJobRunner
from .registry.registry import ModelRegistry
from .deployment.manager import DeploymentManager
from .monitoring.collector import MonitoringCollector
from .snapshots.manager import SnapshotManager

logger = logging.getLogger(__name__)


# Request/Response models
class ModelSpec(BaseModel):
    """Model specification for training."""

    model_key: str = Field(..., description="Model identifier")
    family: str = Field(..., description="Model family (lr, svm, knn, etc.)")
    task: str = Field(..., description="Task type (classification, regression, etc.)")
    adapter: str = Field(..., description="Adapter module path")
    hyperparams: Dict[str, Any] = Field(
        {}, description="Hyperparameters or search space"
    )
    feature_view: str = Field(..., description="Feature view reference")
    tags: List[str] = Field([], description="Model tags")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Model constraints")


class TrainingJobRequest(BaseModel):
    """Training job request."""

    dataset_id: str
    version: str
    spec: ModelSpec
    cv: Dict[str, Any] = Field({"folds": 5, "stratify": True, "time_aware": False})
    hpo: Dict[str, Any] = Field(
        {
            "strategy": "bayes",
            "max_trials": 50,
            "early_stop": True,
            "success_metric": "f1",
        }
    )


class EvaluationRequest(BaseModel):
    """Model evaluation request."""

    model_key: str
    version: str
    dataset_id: str
    metrics: List[str] = Field(["f1", "auroc", "calibration"])
    fairness_groups: Optional[List[str]] = Field(None)


class DeploymentRequest(BaseModel):
    """Model deployment request."""

    model_key: str
    version: str
    target_env: str = Field(..., description="staging or prod")
    canary_pct: int = Field(5, ge=0, le=100)
    shadow: bool = Field(False)
    guardrails: Dict[str, Any] = Field({})
    route: str = Field("single", description="single or ensemble")
    traffic_segment: Optional[Dict[str, Any]] = Field(None)


class CanaryPromoteRequest(BaseModel):
    """Canary promotion request."""

    deployment_id: str
    target_pct: Optional[int] = Field(None, ge=0, le=100)


class RollbackRequest(BaseModel):
    """Rollback request."""

    to_snapshot: str
    reason: Optional[str] = Field(None)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class JobStatusResponse(BaseModel):
    """Training job status response."""

    status: str
    best_score: Optional[float] = None
    trials: int = 0
    elapsed_time: Optional[float] = None
    metadata: Dict[str, Any] = {}


class MLDLService:
    """FastAPI service facade for MLDL kernel external interface."""

    def __init__(self, event_bus=None, governance_bridge=None, memory_bridge=None):
        self.app = FastAPI(
            title="MLDL Kernel API",
            description="Model Lifecycle Management API",
            version="1.0.0",
        )

        # Core components
        self.event_bus = event_bus
        self.governance_bridge = governance_bridge
        self.memory_bridge = memory_bridge

        # Initialize MLDL components
        self.training_runner = TrainingJobRunner(event_bus)
        self.registry = ModelRegistry(event_bus)
        self.deployment_manager = DeploymentManager(event_bus, governance_bridge)
        self.monitoring_collector = MonitoringCollector(event_bus)
        self.snapshot_manager = SnapshotManager()

        # State tracking
        self.active_jobs = {}
        self.running = False

        # Register routes
        self._register_routes()

        logger.info("MLDL Service initialized")

    def _register_routes(self):
        """Register all API routes."""

        @self.app.get("/api/mldl/v1/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="ok", version="1.0.0", timestamp=datetime.now().isoformat()
            )

        @self.app.post("/api/mldl/v1/train")
        async def train_model(
            request: TrainingJobRequest, background_tasks: BackgroundTasks
        ):
            """Start model training job."""
            job_id = f"job_{uuid.uuid4().hex[:8]}"

            try:
                # Submit training job
                background_tasks.add_task(
                    self._run_training_job, job_id, request.dict()
                )

                self.active_jobs[job_id] = {
                    "status": "queued",
                    "created_at": datetime.now().isoformat(),
                    "spec": request.spec.dict(),
                }

                logger.info(f"Training job {job_id} queued")
                return {"job_id": job_id}

            except Exception as e:
                logger.error(f"Failed to queue training job: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/api/mldl/v1/jobs/{job_id}/status", response_model=JobStatusResponse
        )
        async def get_job_status(job_id: str):
            """Get training job status."""
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job_info = self.active_jobs[job_id]
            return JobStatusResponse(**job_info)

        @self.app.post("/api/mldl/v1/evaluate")
        async def evaluate_model(request: EvaluationRequest):
            """Evaluate a registered model."""
            try:
                # Get model from registry
                model_bundle = await self.registry.get(
                    request.model_key, request.version
                )
                if not model_bundle:
                    raise HTTPException(status_code=404, detail="Model not found")

                # Run evaluation
                evaluation_report = await self._run_evaluation(request.dict())

                return evaluation_report

            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/mldl/v1/register")
        async def register_model(model_bundle: Dict[str, Any]):
            """Register a trained model."""
            try:
                result = await self.registry.register(model_bundle)
                return result

            except Exception as e:
                logger.error(f"Model registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/mldl/v1/registry/{model_key}/{version}")
        async def get_model(model_key: str, version: str):
            """Get registered model details."""
            model_bundle = await self.registry.get(model_key, version)
            if not model_bundle:
                raise HTTPException(status_code=404, detail="Model not found")

            return model_bundle

        @self.app.post("/api/mldl/v1/deploy")
        async def deploy_model(request: DeploymentRequest):
            """Deploy a model."""
            try:
                deployment = await self.deployment_manager.request(
                    request.model_key, request.version, request.dict()
                )

                return deployment

            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/mldl/v1/deployments/{deployment_id}")
        async def get_deployment(deployment_id: str):
            """Get deployment status."""
            deployment = await self.deployment_manager.get_deployment(deployment_id)
            if not deployment:
                raise HTTPException(status_code=404, detail="Deployment not found")

            return deployment

        @self.app.post("/api/mldl/v1/canary/promote")
        async def promote_canary(request: CanaryPromoteRequest):
            """Promote canary deployment."""
            try:
                result = await self.deployment_manager.promote(request.deployment_id)
                return result

            except Exception as e:
                logger.error(f"Canary promotion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/mldl/v1/snapshot/export")
        async def export_snapshot():
            """Export system snapshot."""
            try:
                snapshot = await self.snapshot_manager.create_snapshot()
                return snapshot

            except Exception as e:
                logger.error(f"Snapshot export failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/mldl/v1/rollback")
        async def rollback_system(request: RollbackRequest):
            """Rollback system to snapshot."""
            try:
                result = await self.snapshot_manager.rollback(
                    request.to_snapshot, request.reason
                )

                return result

            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/mldl/v1/metrics/live")
        async def get_live_metrics(
            model_key: Optional[str] = None,
            version: Optional[str] = None,
            since: Optional[str] = None,
        ):
            """Get live model metrics."""
            try:
                metrics = await self.monitoring_collector.get_metrics(
                    model_key=model_key, version=version, since=since
                )

                return {"timeseries": metrics}

            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _run_training_job(self, job_id: str, job_spec: Dict[str, Any]):
        """Run training job in background."""
        try:
            self.active_jobs[job_id]["status"] = "running"

            # Submit training job event
            if self.event_bus:
                await self.event_bus.publish(
                    "MLDL_TRAINING_STARTED", {"job_id": job_id, "job": job_spec}
                )

            # Run training
            result = await self.training_runner.run(job_spec)

            # Update job status
            self.active_jobs[job_id].update(
                {
                    "status": "completed",
                    "result": result,
                    "completed_at": datetime.now().isoformat(),
                }
            )

            # Submit completion event
            if self.event_bus:
                await self.event_bus.publish(
                    "MLDL_CANDIDATE_READY", {"job_id": job_id, "bundle": result}
                )

            logger.info(f"Training job {job_id} completed")

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            self.active_jobs[job_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                }
            )

    async def _run_evaluation(self, eval_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run model evaluation."""
        # This is a placeholder - real implementation would load model and test data
        evaluation_report = {
            "report_id": f"eval_{uuid.uuid4().hex[:8]}",
            "model_key": eval_spec["model_key"],
            "version": eval_spec["version"],
            "metrics": {"f1": 0.85, "auroc": 0.92, "precision": 0.88, "recall": 0.83},
            "calibration": {"ece": 0.05, "method": "isotonic"},
            "fairness": {"delta": 0.02, "groups": eval_spec.get("fairness_groups", [])},
            "drift_baseline": {"psi_threshold": 0.25, "kl_threshold": 0.1},
            "slices": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Submit evaluation event
        if self.event_bus:
            await self.event_bus.publish(
                "MLDL_EVALUATED", {"report": evaluation_report}
            )

        return evaluation_report

    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the MLDL service."""
        self.running = True
        logger.info(f"Starting MLDL Service on {host}:{port}")

        config = uvicorn.Config(app=self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        """Stop the MLDL service."""
        self.running = False
        logger.info("MLDL Service stopped")

    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance."""
        return self.app
