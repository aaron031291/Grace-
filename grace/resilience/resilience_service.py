"""
Grace Resilience Service - FastAPI facade for resilience kernel.

This service provides the main REST API for the resilience system,
orchestrating SLOs, policies, incident management, chaos engineering,
and self-healing capabilities.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ..resilience_kernel.kernel import ResilienceKernel
from .controllers.circuit import CircuitBreaker
from .controllers.degradation import enter_mode, exit_mode
from .chaos.runner import start_experiment
from .telemetry.budget import ErrorBudgetTracker
from .snapshots.manager import SnapshotManager
from .bridges.mesh_bridge import MeshBridge
from .bridges.gov_bridge import GovernanceBridge
from .bridges.orch_bridge import OrchestrationBridge

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(default="1.0.0", description="Service version")


class SLOPolicyRequest(BaseModel):
    service_id: str = Field(..., description="Service identifier")
    slos: List[Dict[str, Any]] = Field(..., description="SLO definitions")
    error_budget_days: float = Field(..., description="Error budget in days")


class ResiliencePolicyRequest(BaseModel):
    service_id: str = Field(..., description="Service identifier")
    retries: Optional[Dict[str, Any]] = Field(None, description="Retry configuration")
    circuit_breaker: Optional[Dict[str, Any]] = Field(
        None, description="Circuit breaker config"
    )
    rate_limit: Optional[Dict[str, Any]] = Field(
        None, description="Rate limiting config"
    )
    bulkhead: Optional[Dict[str, Any]] = Field(None, description="Bulkhead config")
    degradation_modes: Optional[List[Dict[str, Any]]] = Field(
        None, description="Degradation modes"
    )


class DependencyGraphRequest(BaseModel):
    service_id: str = Field(..., description="Service identifier")
    nodes: List[Dict[str, Any]] = Field(..., description="Dependency nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Dependency edges")


class IncidentRequest(BaseModel):
    incident_id: str = Field(..., description="Incident identifier")
    service_id: str = Field(..., description="Service identifier")
    severity: str = Field(..., description="Incident severity")
    signals: Dict[str, Any] = Field(..., description="Detector signals")


class ActionRequest(BaseModel):
    action: str = Field(..., description="Action to apply")
    params: Optional[Dict[str, Any]] = Field(None, description="Action parameters")


class CircuitStateRequest(BaseModel):
    state: str = Field(..., description="Circuit breaker state")


class DegradationRequest(BaseModel):
    mode_id: str = Field(..., description="Degradation mode ID")


class ChaosExperimentRequest(BaseModel):
    target: str = Field(..., description="Target service/component")
    blast_radius_pct: float = Field(
        ..., ge=0, le=100, description="Blast radius percentage"
    )
    duration_s: int = Field(..., ge=1, description="Experiment duration in seconds")


class SnapshotExportRequest(BaseModel):
    description: Optional[str] = Field(None, description="Snapshot description")
    tags: Optional[List[str]] = Field(None, description="Snapshot tags")


class RollbackRequest(BaseModel):
    to_snapshot: str = Field(..., description="Target snapshot ID")


class ResilienceService:
    """Main resilience service providing FastAPI interface."""

    def __init__(self):
        self.app = FastAPI(
            title="Grace Resilience Service",
            description="System resilience and self-healing capabilities",
            version="1.0.0",
        )

        # Core components
        self.kernel = ResilienceKernel()
        self.budget_tracker = ErrorBudgetTracker()
        self.snapshot_manager = SnapshotManager()

        # Bridges to other kernels
        self.mesh_bridge = MeshBridge()
        self.gov_bridge = GovernanceBridge()
        self.orch_bridge = OrchestrationBridge()

        # Storage for policies and state
        self.slo_policies: Dict[str, Dict] = {}
        self.resilience_policies: Dict[str, Dict] = {}
        self.dependency_graphs: Dict[str, Dict] = {}
        self.incidents: Dict[str, Dict] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.active_experiments: Dict[str, Dict] = {}

        # Register routes
        self._register_routes()

        logger.info("Resilience service initialized")

    def _register_routes(self):
        """Register FastAPI routes."""

        @self.app.get("/api/res/v1/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(status="healthy")

        @self.app.post("/api/res/v1/policy/slo")
        async def set_slo_policy(policy: SLOPolicyRequest):
            try:
                policy_dict = policy.dict()
                self.slo_policies[policy.service_id] = policy_dict

                # Initialize error budget tracking
                self.budget_tracker.set_policy(policy.service_id, policy_dict)

                return {"service_id": policy.service_id, "status": "created"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to set SLO policy: {str(e)}",
                )

        @self.app.post("/api/res/v1/policy/resilience")
        async def set_resilience_policy(policy: ResiliencePolicyRequest):
            try:
                policy_dict = policy.dict(exclude_none=True)
                self.resilience_policies[policy.service_id] = policy_dict

                # Initialize circuit breakers if configured
                if policy.circuit_breaker:
                    cb_key = f"{policy.service_id}/default"
                    self.circuit_breakers[cb_key] = CircuitBreaker(
                        failure_threshold_pct=policy.circuit_breaker.get(
                            "failure_rate_threshold_pct", 50
                        ),
                        volume_threshold=policy.circuit_breaker.get(
                            "request_volume_threshold", 20
                        ),
                        sleep_window_ms=policy.circuit_breaker.get(
                            "sleep_window_ms", 5000
                        ),
                        half_open_max_calls=policy.circuit_breaker.get(
                            "half_open_max_calls", 5
                        ),
                    )

                return {"service_id": policy.service_id, "status": "created"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to set resilience policy: {str(e)}",
                )

        @self.app.get("/api/res/v1/policy/{service_id}")
        async def get_policy(service_id: str):
            try:
                slo_policy = self.slo_policies.get(service_id)
                resilience_policy = self.resilience_policies.get(service_id)

                if not slo_policy and not resilience_policy:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No policies found for service: {service_id}",
                    )

                return {
                    "service_id": service_id,
                    "slo": slo_policy,
                    "resilience": resilience_policy,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get policy: {str(e)}",
                )

        @self.app.post("/api/res/v1/dependency/graph")
        async def submit_dependency_graph(graph: DependencyGraphRequest):
            try:
                graph_dict = graph.dict()
                self.dependency_graphs[graph.service_id] = graph_dict
                return {"service_id": graph.service_id, "status": "created"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to submit dependency graph: {str(e)}",
                )

        @self.app.post("/api/res/v1/incident/open")
        async def open_incident(incident: IncidentRequest):
            try:
                incident_dict = incident.dict()
                incident_dict["started_at"] = datetime.now().isoformat()
                incident_dict["status"] = "open"

                self.incidents[incident.incident_id] = incident_dict

                # Emit incident event via mesh bridge
                await self.mesh_bridge.publish_event(
                    "RES_INCIDENT_OPENED", {"incident": incident_dict}
                )

                return {"incident_id": incident.incident_id, "status": "created"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to open incident: {str(e)}",
                )

        @self.app.post("/api/res/v1/incident/{incident_id}/action")
        async def apply_action(incident_id: str, action_request: ActionRequest):
            try:
                if incident_id not in self.incidents:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Incident not found: {incident_id}",
                    )

                # Apply the healing action
                result = await self._execute_healing_action(
                    incident_id, action_request.action, action_request.params
                )

                # Emit action event
                await self.mesh_bridge.publish_event(
                    "RES_ACTION_APPLIED",
                    {
                        "incident_id": incident_id,
                        "action": action_request.action,
                        "result": result["status"],
                        "notes": result.get("notes"),
                    },
                )

                return {"accepted": True, "result": result}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to apply action: {str(e)}",
                )

        @self.app.post("/api/res/v1/breaker/{service_id}/{dep}/state")
        async def set_circuit_state(
            service_id: str, dep: str, state_request: CircuitStateRequest
        ):
            try:
                cb_key = f"{service_id}/{dep}"

                if cb_key not in self.circuit_breakers:
                    # Create circuit breaker with default settings
                    self.circuit_breakers[cb_key] = CircuitBreaker()

                breaker = self.circuit_breakers[cb_key]

                # Force state change (normally would be driven by metrics)
                if state_request.state == "open":
                    breaker.force_open()
                elif state_request.state == "closed":
                    breaker.force_closed()
                elif state_request.state == "half_open":
                    breaker.force_half_open()

                # Emit circuit state event
                await self.mesh_bridge.publish_event(
                    "RES_CIRCUIT_STATE",
                    {
                        "service_id": service_id,
                        "dependency": dep,
                        "state": breaker.state(),
                    },
                )

                return {"state": breaker.state()}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to set circuit state: {str(e)}",
                )

        @self.app.post("/api/res/v1/degradation/{service_id}/enter")
        async def enter_degradation(service_id: str, request: DegradationRequest):
            try:
                await enter_mode(service_id, request.mode_id)

                # Emit degradation event
                await self.mesh_bridge.publish_event(
                    "RES_DEGRADATION_ENTERED",
                    {"service_id": service_id, "mode_id": request.mode_id},
                )

                return {"mode_id": request.mode_id, "status": "entered"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to enter degradation mode: {str(e)}",
                )

        @self.app.post("/api/res/v1/degradation/{service_id}/exit")
        async def exit_degradation(service_id: str, request: DegradationRequest):
            try:
                await exit_mode(service_id, request.mode_id)

                # Emit degradation event
                await self.mesh_bridge.publish_event(
                    "RES_DEGRADATION_EXITED",
                    {"service_id": service_id, "mode_id": request.mode_id},
                )

                return {"mode_id": request.mode_id, "status": "exited"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to exit degradation mode: {str(e)}",
                )

        @self.app.post("/api/res/v1/chaos/start")
        async def start_chaos_experiment(experiment: ChaosExperimentRequest):
            try:
                experiment_id = await start_experiment(
                    experiment.target,
                    experiment.blast_radius_pct,
                    experiment.duration_s,
                )

                self.active_experiments[experiment_id] = experiment.dict()

                # Emit chaos started event
                await self.mesh_bridge.publish_event(
                    "RES_CHAOS_STARTED",
                    {
                        "experiment_id": experiment_id,
                        "target": experiment.target,
                        "blast_radius_pct": experiment.blast_radius_pct,
                    },
                )

                return {"experiment_id": experiment_id, "status": "started"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to start chaos experiment: {str(e)}",
                )

        @self.app.get("/api/res/v1/chaos/{experiment_id}/result")
        async def get_chaos_result(experiment_id: str):
            try:
                if experiment_id not in self.active_experiments:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Experiment not found: {experiment_id}",
                    )

                # Placeholder for actual result retrieval
                result = {
                    "experiment_id": experiment_id,
                    "outcome": "pass",  # Would be determined by experiment runner
                    "findings": {"status": "completed"},
                    "completed_at": datetime.now().isoformat(),
                }

                return result
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get chaos result: {str(e)}",
                )

        @self.app.post("/api/res/v1/snapshot/export")
        async def export_snapshot(request: SnapshotExportRequest):
            try:
                snapshot = await self.snapshot_manager.create_snapshot(
                    description=request.description or "Manual export",
                    snapshot_type="manual",
                )

                # Emit snapshot created event
                await self.mesh_bridge.publish_event(
                    "RES_SNAPSHOT_CREATED",
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "uri": snapshot.get("uri", ""),
                    },
                )

                return {
                    "snapshot_id": snapshot["snapshot_id"],
                    "uri": snapshot.get("uri", ""),
                    "created_at": snapshot["created_at"],
                }
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to export snapshot: {str(e)}",
                )

        @self.app.post("/api/res/v1/rollback")
        async def rollback_to_snapshot(request: RollbackRequest):
            try:
                result = await self.snapshot_manager.rollback(
                    to_snapshot=request.to_snapshot, reason="API rollback request"
                )

                # Emit rollback completed event
                await self.mesh_bridge.publish_event(
                    "ROLLBACK_COMPLETED",
                    {
                        "target": "resilience",
                        "snapshot_id": request.to_snapshot,
                        "at": datetime.now().isoformat(),
                    },
                )

                return result
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to rollback: {str(e)}",
                )

    async def _execute_healing_action(
        self, incident_id: str, action: str, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a healing action for an incident."""
        try:
            incident = self.incidents[incident_id]
            service_id = incident["service_id"]

            if action == "restart":
                # Request restart via orchestration bridge
                result = await self.orch_bridge.request_restart(service_id)
            elif action == "reprovision":
                # Request reprovision via orchestration bridge
                result = await self.orch_bridge.request_reprovision(service_id)
            elif action == "open_breaker":
                # Open circuit breaker
                cb_key = f"{service_id}/default"
                if cb_key in self.circuit_breakers:
                    self.circuit_breakers[cb_key].force_open()
                    result = {"status": "success", "notes": "Circuit breaker opened"}
                else:
                    result = {"status": "failed", "notes": "Circuit breaker not found"}
            elif action == "degrade_mode":
                # Enter degradation mode
                mode_id = (
                    params.get("mode_id", "cached_only") if params else "cached_only"
                )
                await enter_mode(service_id, mode_id)
                result = {
                    "status": "success",
                    "notes": f"Entered degradation mode: {mode_id}",
                }
            else:
                result = {"status": "failed", "notes": f"Unknown action: {action}"}

            return result
        except Exception as e:
            logger.error(f"Failed to execute healing action {action}: {e}")
            return {"status": "failed", "notes": str(e)}

    # Public interface methods (for integration with kernel)
    def set_slo(self, policy: dict) -> dict:
        """Set SLO policy."""
        service_id = policy["service_id"]
        self.slo_policies[service_id] = policy
        self.budget_tracker.set_policy(service_id, policy)
        return {"service_id": service_id}

    def set_resilience_policy(self, policy: dict) -> dict:
        """Set resilience policy."""
        service_id = policy["service_id"]
        self.resilience_policies[service_id] = policy
        return {"service_id": service_id}

    def submit_dependency_graph(self, graph: dict) -> dict:
        """Submit dependency graph."""
        service_id = graph["service_id"]
        self.dependency_graphs[service_id] = graph
        return {"service_id": service_id}

    def open_incident(self, incident: dict) -> str:
        """Open new incident."""
        incident_id = incident["incident_id"]
        incident["started_at"] = datetime.now().isoformat()
        incident["status"] = "open"
        self.incidents[incident_id] = incident
        return incident_id

    def apply_action(self, incident_id: str, action: str, params: dict = None) -> dict:
        """Apply healing action."""
        # This would be implemented asyncio.run in a sync context
        # For now, return success
        return {"status": "accepted"}

    def export_snapshot(self) -> dict:
        """Export current resilience state snapshot."""
        snapshot_id = f"res_{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}"
        snapshot = {
            "snapshot_id": snapshot_id,
            "services": {
                service_id: {
                    "slo": self.slo_policies.get(service_id, {}),
                    "policy": self.resilience_policies.get(service_id, {}),
                }
                for service_id in set(
                    list(self.slo_policies.keys())
                    + list(self.resilience_policies.keys())
                )
            },
            "dependency_graph_hash": "sha256:...",  # Would compute actual hash
            "chaos": {"enabled": True, "max_blast_radius_pct": 5},
            "hash": "sha256:...",  # Would compute actual hash
        }
        return snapshot

    def rollback(self, to_snapshot: str) -> dict:
        """Rollback to a previous snapshot."""
        return {"to_snapshot": to_snapshot, "status": "completed"}

    async def start(self):
        """Start the resilience service."""
        logger.info("Starting resilience service...")
        # Initialize components
        await self.snapshot_manager.start() if hasattr(
            self.snapshot_manager, "start"
        ) else None
        logger.info("Resilience service started")

    async def stop(self):
        """Stop the resilience service."""
        logger.info("Stopping resilience service...")
        # Cleanup components
        if hasattr(self.snapshot_manager, "stop"):
            await self.snapshot_manager.stop()
        logger.info("Resilience service stopped")
