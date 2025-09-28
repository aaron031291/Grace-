"""
Intelligence Service - Main FastAPI service for the Intelligence Kernel.

Provides REST API endpoints for task routing, planning, inference,
and operational management including snapshots and rollbacks.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
import json
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

try:
    # Try relative imports first (when used as module)
    from .router.task_router import TaskRouter
    from .planner.plan_builder import PlanBuilder
    from .inference.engine import InferenceEngine
    from .ensembler.meta_learner import MetaEnsembler
    from .evaluation.metrics import MetricsCollector
    from .governance_bridge import GovernanceBridge
    from .mlt_bridge import MLTBridge
    from .memory_bridge import MemoryBridge
    from .snapshots.snapshot_manager import SnapshotManager
except ImportError:
    # Fall back to direct imports (when run as script)
    from router.task_router import TaskRouter
    from planner.plan_builder import PlanBuilder
    from inference.engine import InferenceEngine
    from ensembler.meta_learner import MetaEnsembler
    from evaluation.metrics import MetricsCollector
    from governance_bridge import GovernanceBridge
    from mlt_bridge import MLTBridge
    from memory_bridge import MemoryBridge
    from snapshots.snapshot_manager import SnapshotManager

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


class InferenceRequestResponse(BaseModel):
    req_id: str
    status: str = "accepted"


class IntelligenceService:
    """Main Intelligence Service orchestrating all Intelligence Kernel components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.version = "1.0.0"
        
        # Initialize core components
        self.router = TaskRouter()
        self.planner = PlanBuilder()
        self.inference_engine = InferenceEngine()
        self.meta_ensembler = MetaEnsembler()
        self.metrics_collector = MetricsCollector()
        self.snapshot_manager = SnapshotManager()
        
        # Initialize bridges
        self.governance_bridge = GovernanceBridge()
        self.mlt_bridge = MLTBridge()
        self.memory_bridge = MemoryBridge()
        
        # Request tracking
        self.active_requests: Dict[str, Dict] = {}
        self.completed_results: Dict[str, Dict] = {}
        
        logger.info("Intelligence Service initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Intelligence Service."""
        return {
            "policy": {
                "min_confidence": 0.75,
                "min_calibration": 0.96,
                "fairness_delta_max": 0.02,
                "require_explanations": False
            },
            "router": {
                "hybrid_weights": {"latency": 0.3, "quality": 0.7},
                "allow_shadow": True,
                "canary_pct_default": 10
            },
            "ensembler": {
                "type": "stack",
                "meta_model": "lr-meta@1.0.2"
            },
            "inference": {
                "timeout_ms": 800,
                "batch_max": 64
            }
        }
    
    def request(self, task_req: dict) -> str:
        """Process task request and return request ID."""
        try:
            # Generate request ID
            req_id = f"req_{format_for_filename()}_{hash(str(task_req)) % 10000:04d}"
            
            # Validate request
            if not self._validate_request(task_req):
                raise HTTPException(status_code=400, detail="Invalid request format")
            
            # Store request
            task_req["req_id"] = req_id
            task_req["timestamp"] = iso_format()
            self.active_requests[req_id] = task_req
            
            # Emit INTEL_REQUESTED event
            self._emit_event("INTEL_REQUESTED", {"request": task_req})
            
            # Start processing in background
            self._process_request_async(req_id, task_req)
            
            return req_id
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_result(self, req_id: str) -> dict:
        """Get result for completed request."""
        if req_id in self.completed_results:
            return self.completed_results[req_id]
        elif req_id in self.active_requests:
            return {"req_id": req_id, "status": "processing"}
        else:
            raise HTTPException(status_code=404, detail="Request not found")
    
    def plan_preview(self, task_req: dict) -> dict:
        """Preview execution plan for task request without executing."""
        try:
            # Route task
            route = self.router.route(task_req)
            
            # Build plan
            plan = self.planner.build_plan(task_req, route)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan preview: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def export_snapshot(self) -> dict:
        """Export current Intelligence Kernel state snapshot."""
        try:
            snapshot_data = {
                "snapshot_id": f"intel_{utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}",
                "router": self.router.get_state(),
                "ensembler": self.meta_ensembler.get_state(),
                "policies": self.config["policy"],
                "allowed_models": getattr(self.config, "allowed_models", []),
                "blocklist_models": getattr(self.config, "blocklist_models", []),
                "feature_views": self.memory_bridge.get_feature_views(),
                "version": self.version
            }
            
            # Calculate hash
            snapshot_hash = hashlib.sha256(
                json.dumps(snapshot_data, sort_keys=True).encode()
            ).hexdigest()
            snapshot_data["hash"] = f"sha256:{snapshot_hash}"
            
            # Save snapshot
            snapshot_id = self.snapshot_manager.save_snapshot(snapshot_data)
            
            return {"snapshot_id": snapshot_id, "uri": f"/snapshots/{snapshot_id}"}
            
        except Exception as e:
            logger.error(f"Error exporting snapshot: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def rollback(self, to_snapshot: str) -> dict:
        """Rollback to specified snapshot."""
        try:
            # Load snapshot
            success = self.snapshot_manager.load_snapshot(to_snapshot)
            if not success:
                raise HTTPException(status_code=404, detail="Snapshot not found")
            
            # Emit rollback completed event
            self._emit_event("ROLLBACK_COMPLETED", {
                "target": "intelligence",
                "snapshot_id": to_snapshot,
                "at": iso_format()
            })
            
            return {"status": "completed", "snapshot_id": to_snapshot}
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _validate_request(self, task_req: dict) -> bool:
        """Validate task request format."""
        required_fields = ["task", "input", "context"]
        return all(field in task_req for field in required_fields)
    
    def _process_request_async(self, req_id: str, task_req: dict):
        """Process request asynchronously."""
        try:
            # Step 1: Route and plan
            route = self.router.route(task_req)
            plan = self.planner.build_plan(task_req, route)
            plan["req_id"] = req_id
            
            # Emit INTEL_PLANNED event
            self._emit_event("INTEL_PLANNED", {"plan": plan})
            
            # Step 2: Governance check if required
            if self._requires_governance_check(plan):
                approval = self.governance_bridge.request_approval(plan)
                if not approval.get("approved", False):
                    self._complete_request(req_id, {
                        "req_id": req_id,
                        "status": "rejected", 
                        "reason": approval.get("reason", "Policy violation")
                    })
                    return
            
            # Step 3: Execute inference
            self._emit_event("INTEL_INFER_STARTED", {"plan_id": plan["plan_id"], "req_id": req_id})
            
            result = self.inference_engine.execute(plan, task_req)
            result["req_id"] = req_id
            
            # Step 4: Collect metrics and explanations
            result = self._enrich_result(result, plan, task_req)
            
            # Complete request
            self._emit_event("INTEL_INFER_COMPLETED", {"req_id": req_id, "result": result})
            self._complete_request(req_id, result)
            
        except Exception as e:
            logger.error(f"Error processing request {req_id}: {e}")
            self._complete_request(req_id, {
                "req_id": req_id,
                "status": "error",
                "error": str(e)
            })
    
    def _requires_governance_check(self, plan: dict) -> bool:
        """Check if plan requires governance approval."""
        # Check for high-risk conditions
        route = plan.get("route", {})
        return (
            route.get("canary_pct", 0) > 25 or  # High canary percentage
            len(route.get("models", [])) > 5 or  # Many models
            plan.get("policy", {}).get("min_confidence", 1.0) < 0.7  # Low confidence threshold
        )
    
    def _enrich_result(self, result: dict, plan: dict, task_req: dict) -> dict:
        """Enrich result with metrics, explanations, and lineage."""
        # Add runtime metrics
        result["metrics"] = self.metrics_collector.collect_runtime_metrics(result, plan)
        
        # Add explanations if requested
        if task_req.get("context", {}).get("explanation", False):
            result["explanations"] = self._generate_explanations(result, plan)
        
        # Add lineage information
        result["lineage"] = {
            "plan_id": plan["plan_id"],
            "models": plan["route"]["models"],
            "ensemble": plan["route"]["ensemble"],
            "feature_view": "default"
        }
        
        # Add governance info
        result["governance"] = {
            "approved": True,
            "policy_version": self.version,
            "redactions": []
        }
        
        return result
    
    def _generate_explanations(self, result: dict, plan: dict) -> dict:
        """Generate explanations for the result."""
        # Placeholder implementation
        return {
            "method": "auto",
            "feature_importance": {},
            "local_explanations": {}
        }
    
    def _complete_request(self, req_id: str, result: dict):
        """Complete request and move to completed results."""
        if req_id in self.active_requests:
            del self.active_requests[req_id]
        self.completed_results[req_id] = result
        
        # Cleanup old completed results (keep last 1000)
        if len(self.completed_results) > 1000:
            oldest_keys = sorted(self.completed_results.keys())[:100]
            for key in oldest_keys:
                del self.completed_results[key]
    
    def _emit_event(self, event_name: str, payload: dict):
        """Emit event to event mesh."""
        # TODO: Implement event emission
        logger.info(f"Event emitted: {event_name}")
    
    def get_health(self) -> HealthResponse:
        """Get service health status."""
        components = {
            "router": "healthy",
            "planner": "healthy", 
            "inference_engine": "healthy",
            "meta_ensembler": "healthy",
            "governance_bridge": "healthy",
            "mlt_bridge": "healthy",
            "memory_bridge": "healthy"
        }
        
        return HealthResponse(
            status="healthy",
            version=self.version,
            timestamp=iso_format(),
            components=components
        )


# FastAPI app setup
app = FastAPI(
    title="Grace Intelligence Kernel",
    description="ML/DL task routing, planning, and inference service",
    version="1.0.0"
)

# Global service instance
intelligence_service = IntelligenceService()


@app.get("/api/intel/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return intelligence_service.get_health()


@app.post("/api/intel/v1/request", response_model=InferenceRequestResponse)
async def request_inference(task_req: dict, background_tasks: BackgroundTasks):
    """Submit task request for processing."""
    req_id = intelligence_service.request(task_req)
    return InferenceRequestResponse(req_id=req_id)


@app.get("/api/intel/v1/result/{req_id}")
async def get_result(req_id: str):
    """Get inference result."""
    return intelligence_service.get_result(req_id)


@app.post("/api/intel/v1/plan/preview")
async def preview_plan(task_req: dict):
    """Preview execution plan."""
    return intelligence_service.plan_preview(task_req)


@app.post("/api/intel/v1/snapshot/export")
async def export_snapshot():
    """Export current state snapshot."""
    return intelligence_service.export_snapshot()


@app.post("/api/intel/v1/rollback")
async def rollback(rollback_req: dict):
    """Rollback to snapshot."""
    to_snapshot = rollback_req.get("to_snapshot")
    if not to_snapshot:
        raise HTTPException(status_code=400, detail="to_snapshot required")
    
    return intelligence_service.rollback(to_snapshot)


@app.get("/api/intel/v1/metrics")
async def get_metrics(since: Optional[str] = None, segment: Optional[str] = None):
    """Get metrics timeseries."""
    return intelligence_service.metrics_collector.get_timeseries(since, segment)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)