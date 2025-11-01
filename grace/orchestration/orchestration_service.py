"""
Grace Orchestration Service - FastAPI service facade for orchestration kernel.

Provides RESTful API endpoints for managing orchestration loops, tasks,
snapshots, and system state. Acts as the main interface to the orchestration
kernel functionality.

Enhanced with Phase 4: Governance Gate - Constitutional validation integration.
"""

import asyncio
import time
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from .scheduler.scheduler import Scheduler
from .router.router import Router
from .state.state_manager import StateManager, OrchestrationState
from .recovery.watchdog import Watchdog
from .scaling.manager import ScalingManager
from .lifecycle.manager import LifecycleManager
from .snapshots.manager import SnapshotManager
from .bridges.mesh_bridge import MeshBridge
from .bridges.gov_bridge import GovernanceBridge
from .bridges.kernel_bridges import KernelBridges
from ..governance.constitutional_validator import ConstitutionalValidator

logger = logging.getLogger(__name__)


# Pydantic models for API
class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    version: str = Field(default="1.0.0", description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    active_loops: int = Field(..., description="Number of active loops")
    active_tasks: int = Field(..., description="Number of active tasks")


class LoopDefRequest(BaseModel):
    loop_id: str = Field(..., description="Unique loop identifier")
    name: str = Field(..., description="Loop name")
    priority: int = Field(..., ge=1, le=10, description="Priority level")
    interval_s: int = Field(..., ge=1, description="Execution interval in seconds")
    kernels: List[str] = Field(..., description="Required kernels")
    policies: Dict[str, Any] = Field(default_factory=dict, description="Loop policies")
    enabled: bool = Field(default=True, description="Whether loop is enabled")


class TaskDispatchRequest(BaseModel):
    loop_id: str = Field(..., description="Loop identifier")
    inputs: Dict[str, Any] = Field(..., description="Task inputs")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")


class LoopEnableRequest(BaseModel):
    enabled: bool = Field(..., description="Whether to enable the loop")


class RollbackRequest(BaseModel):
    to_snapshot: str = Field(..., description="Target snapshot ID for rollback")


class SnapshotExportRequest(BaseModel):
    description: str = Field(default="", description="Snapshot description")
    tags: List[str] = Field(default_factory=list, description="Snapshot tags")


class OrchestrationService:
    """Main orchestration service providing FastAPI interface with constitutional governance."""

    def __init__(self):
        self.app = FastAPI(
            title="Grace Orchestration Kernel API",
            description="Central conductor and scheduler for all Grace kernels with constitutional governance",
            version="1.0.0",
        )

        # Core components
        self.state_manager = StateManager()
        self.scheduler = Scheduler(
            state_manager=self.state_manager, event_publisher=self._publish_event
        )
        self.router = Router(
            event_publisher=self._publish_event,
            kernel_registry=None,  # Will be set by kernel bridges
        )
        self.watchdog = Watchdog(
            event_publisher=self._publish_event, kernel_registry=None
        )
        self.scaling_manager = ScalingManager(
            event_publisher=self._publish_event, scheduler=self.scheduler
        )
        self.lifecycle_manager = LifecycleManager(event_publisher=self._publish_event)
        self.snapshot_manager = SnapshotManager(
            scheduler=self.scheduler,
            state_manager=self.state_manager,
            router=self.router,
            watchdog=self.watchdog,
            scaling_manager=self.scaling_manager,
            lifecycle_manager=self.lifecycle_manager,
            event_publisher=self._publish_event,
        )

        # Integration bridges
        self.mesh_bridge = MeshBridge()
        self.governance_bridge = GovernanceBridge()
        self.kernel_bridges = KernelBridges(event_publisher=self._publish_event)

        # Phase 4: Governance Gate components
        self.constitutional_validator = ConstitutionalValidator(
            event_publisher=self._publish_event
        )
        self.verification_engine = None  # Will be initialized with proper dependencies

        # Service state
        self.started_at = None
        self.running = False

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/api/orch/v1/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            uptime = 0.0
            if self.started_at:
                uptime = time.time() - self.started_at

            scheduler_status = self.scheduler.get_status()

            return HealthResponse(
                status="healthy" if self.running else "starting",
                uptime_seconds=uptime,
                active_loops=len(scheduler_status.get("loops", {})),
                active_tasks=len(scheduler_status.get("active_tasks", {})),
            )

        @self.app.get("/api/orch/v1/loops")
        async def list_loops():
            """List all orchestration loops."""
            try:
                status = self.scheduler.get_status()
                loops = []

                for loop_data in status.get("loops", {}).values():
                    loops.append(loop_data)

                return loops

            except Exception as e:
                logger.error(f"Failed to list loops: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/orch/v1/loops")
        async def create_loop(loop_request: LoopDefRequest):
            """Create a new orchestration loop with constitutional validation."""
            try:
                # Phase 4: Validate against constitution first
                await self._validate_action_against_constitution(
                    {
                        "type": "create_loop",
                        "id": loop_request.loop_id,
                        "component_id": "orchestration_service",
                        "description": f"Creating orchestration loop: {loop_request.name}",
                        "rationale": f"Loop for {loop_request.name} with {len(loop_request.kernels)} kernels",
                        "risk_level": "medium" if loop_request.priority > 7 else "low",
                        "reversible": True,
                        "audit_trail": True,
                    }
                )

                # Validate with governance if available
                if self.governance_bridge:
                    is_valid = await self.governance_bridge.validate_loop_execution(
                        loop_request.loop_id, loop_request.dict()
                    )
                    if not is_valid:
                        raise HTTPException(
                            status_code=403,
                            detail="Loop creation denied by governance policies",
                        )

                loop_id = await self.scheduler.register_loop(loop_request.dict())

                return {"loop_id": loop_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create loop: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/orch/v1/loops/{loop_id}/enable")
        async def enable_loop(loop_id: str, request: LoopEnableRequest):
            """Enable or disable an orchestration loop."""
            try:
                success = await self.scheduler.enable_loop(loop_id, request.enabled)

                if not success:
                    raise HTTPException(status_code=404, detail="Loop not found")

                return {"status": "enabled" if request.enabled else "disabled"}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to enable/disable loop {loop_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/orch/v1/task/dispatch")
        async def dispatch_task(task_request: TaskDispatchRequest):
            """Dispatch a new orchestration task with constitutional validation."""
            try:
                # Phase 4: Validate against constitution
                await self._validate_action_against_constitution(
                    {
                        "type": "dispatch_task",
                        "id": f"task_{task_request.loop_id}",
                        "component_id": "orchestration_service",
                        "description": f"Dispatching task for loop {task_request.loop_id}",
                        "rationale": "Task dispatch requested via API",
                        "risk_level": "high" if task_request.priority > 8 else "medium",
                        "reversible": True,
                        "audit_trail": True,
                    }
                )

                # Validate with governance if available
                if self.governance_bridge:
                    is_valid = await self.governance_bridge.validate_task_dispatch(
                        "pending", task_request.loop_id, task_request.inputs
                    )
                    if not is_valid:
                        raise HTTPException(
                            status_code=403,
                            detail="Task dispatch denied by governance policies",
                        )

                task_id = await self.scheduler.dispatch_task(
                    task_request.loop_id, task_request.inputs, task_request.priority
                )

                return {"task_id": task_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to dispatch task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/orch/v1/task/{task_id}/status")
        async def get_task_status(task_id: str):
            """Get status of a specific task."""
            try:
                task_status = await self.scheduler.get_task_status(task_id)

                if task_status is None:
                    raise HTTPException(status_code=404, detail="Task not found")

                return task_status

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get task status {task_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/orch/v1/snapshot/export")
        async def export_snapshot(request: SnapshotExportRequest = None):
            """Create and export a system snapshot."""
            try:
                if request is None:
                    request = SnapshotExportRequest()

                snapshot_id = await self.snapshot_manager.export_snapshot(
                    description=request.description, tags=set(request.tags)
                )

                # Generate URI for snapshot download (simplified)
                uri = f"/api/orch/v1/snapshot/{snapshot_id}/download"

                return {"snapshot_id": snapshot_id, "uri": uri}

            except Exception as e:
                logger.error(f"Failed to export snapshot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/orch/v1/rollback")
        async def rollback_system(
            request: RollbackRequest, background_tasks: BackgroundTasks
        ):
            """Rollback orchestration to a previous snapshot."""
            try:
                # Validate rollback request with governance
                if self.governance_bridge:
                    is_valid = await self.governance_bridge.validate_rollback_request(
                        request.to_snapshot, "API rollback request"
                    )
                    if not is_valid:
                        raise HTTPException(
                            status_code=403,
                            detail="Rollback denied by governance policies",
                        )

                # Start rollback in background
                background_tasks.add_task(self._perform_rollback, request.to_snapshot)

                return {"to_snapshot": request.to_snapshot}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to initiate rollback: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/orch/v1/status")
        async def get_system_status():
            """Get comprehensive system status."""
            try:
                return {
                    "scheduler": self.scheduler.get_status(),
                    "router": self.router.get_status(),
                    "state_manager": self.state_manager.get_status(),
                    "watchdog": self.watchdog.get_status(),
                    "scaling_manager": self.scaling_manager.get_status(),
                    "lifecycle_manager": self.lifecycle_manager.get_status(),
                    "snapshot_manager": self.snapshot_manager.get_status(),
                    "mesh_bridge": self.mesh_bridge.get_status(),
                    "governance_bridge": self.governance_bridge.get_stats(),
                    "kernel_bridges": self.kernel_bridges.get_status(),
                    "service": {
                        "running": self.running,
                        "started_at": self.started_at.isoformat()
                        if self.started_at
                        else None,
                        "uptime_seconds": time.time() - self.started_at
                        if self.started_at
                        else 0,
                    },
                }

            except Exception as e:
                logger.error(f"Failed to get system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/orch/v1/kernels")
        async def list_kernels():
            """List registered kernels."""
            try:
                return self.kernel_bridges.get_registered_kernels()

            except Exception as e:
                logger.error(f"Failed to list kernels: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _perform_rollback(self, to_snapshot: str):
        """Perform rollback operation in background."""
        try:
            operation_id = await self.snapshot_manager.rollback(to_snapshot)
            logger.info(f"Rollback operation {operation_id} completed successfully")
        except Exception as e:
            logger.error(f"Rollback operation failed: {e}")

    async def _validate_action_against_constitution(self, action: Dict[str, Any]):
        """
        Phase 4: Constitutional validation for orchestrator decisions.
        Raises HTTPException if action violates constitutional principles.
        """
        try:
            # Perform constitutional validation
            validation_result = (
                await self.constitutional_validator.validate_against_constitution(
                    action
                )
            )

            if not validation_result.is_valid:
                # Log the constitutional violation
                logger.warning(
                    f"Constitutional violation detected for action {action.get('type', 'unknown')}: "
                    f"score {validation_result.compliance_score:.3f}"
                )

                # Prepare violation details for response
                violation_details = []
                for violation in validation_result.violations:
                    violation_details.append(
                        {
                            "principle": violation.principle,
                            "severity": violation.severity,
                            "description": violation.description,
                        }
                    )

                # Raise HTTP exception with constitutional violation details
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Constitutional violation",
                        "compliance_score": validation_result.compliance_score,
                        "violations": violation_details,
                        "validation_id": validation_result.validation_id,
                    },
                )

            logger.debug(
                f"Constitutional validation passed for {action.get('type', 'unknown')} "
                f"(score: {validation_result.compliance_score:.3f})"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during constitutional validation: {e}")
            # In case of validation system failure, log but don't block action
            # This ensures system remains operational even if validation fails
            logger.warning(
                "Constitutional validation failed due to system error - allowing action to proceed"
            )

    async def validate_against_constitution(
        self, action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Public method for constitutional validation (can be called by other components).
        Returns validation result instead of raising exception.
        """
        try:
            validation_result = (
                await self.constitutional_validator.validate_against_constitution(
                    action
                )
            )

            return {
                "is_valid": validation_result.is_valid,
                "compliance_score": validation_result.compliance_score,
                "violations": [
                    {
                        "principle": v.principle,
                        "severity": v.severity,
                        "description": v.description,
                        "recommendation": v.recommendation,
                    }
                    for v in validation_result.violations
                ],
                "validation_id": validation_result.validation_id,
            }

        except Exception as e:
            logger.error(f"Constitutional validation error: {e}")
            return {
                "is_valid": False,
                "compliance_score": 0.0,
                "violations": [
                    {
                        "principle": "system_error",
                        "severity": "critical",
                        "description": f"Validation system error: {str(e)}",
                        "recommendation": "Check validation system health",
                    }
                ],
                "validation_id": "error",
            }

    async def _publish_event(self, event_name: str, payload: Dict[str, Any]):
        """Publish events through the mesh bridge."""
        try:
            if self.mesh_bridge:
                await self.mesh_bridge.publish_event(event_name, payload)
        except Exception as e:
            logger.error(f"Failed to publish event {event_name}: {e}")

    async def start(self):
        """Start the orchestration service and all components."""
        if self.running:
            logger.warning("Orchestration service already running")
            return

        logger.info("Starting Grace Orchestration Service...")

        try:
            # Transition to starting state
            await self.state_manager.transition_state(
                OrchestrationState.ACTIVE,
                "service_startup",
                {"initiated_by": "orchestration_service"},
            )

            # Start core components
            await self.scheduler.start()
            await self.router.start()
            await self.watchdog.start()
            await self.scaling_manager.start()
            await self.snapshot_manager.start()

            # Start bridges
            await self.mesh_bridge.start()
            await self.governance_bridge.start()
            await self.kernel_bridges.start()

            # Register components with lifecycle manager
            from .lifecycle.manager import ManagedComponent

            # Register scheduler
            scheduler_component = ManagedComponent(
                "scheduler", self.scheduler, dependencies=["state_manager"]
            )
            self.lifecycle_manager.register_component(scheduler_component)

            # Register router
            router_component = ManagedComponent("router", self.router)
            self.lifecycle_manager.register_component(router_component)

            # Set startup order
            self.lifecycle_manager.set_startup_order(["scheduler", "router"])

            # Start lifecycle manager
            await self.lifecycle_manager.startup()

            self.started_at = time.time()
            self.running = True

            logger.info("Grace Orchestration Service started successfully")

        except Exception as e:
            logger.error(f"Failed to start orchestration service: {e}")
            await self.state_manager.transition_state(
                OrchestrationState.ERROR, "startup_failure", {"error": str(e)}
            )
            raise

    async def stop(self):
        """Stop the orchestration service and all components."""
        if not self.running:
            return

        logger.info("Stopping Grace Orchestration Service...")

        try:
            await self.state_manager.transition_state(
                OrchestrationState.SHUTDOWN,
                "service_shutdown",
                {"initiated_by": "orchestration_service"},
            )

            # Stop lifecycle manager
            await self.lifecycle_manager.shutdown()

            # Stop bridges
            await self.kernel_bridges.stop()
            await self.governance_bridge.stop()
            await self.mesh_bridge.stop()

            # Stop core components
            await self.snapshot_manager.stop()
            await self.scaling_manager.stop()
            await self.watchdog.stop()
            await self.router.stop()
            await self.scheduler.stop()

            self.running = False

            logger.info("Grace Orchestration Service stopped")

        except Exception as e:
            logger.error(f"Error during orchestration service shutdown: {e}")

    def dispatch(self, loop_id: str, task: Dict[str, Any]) -> str:
        """Legacy synchronous method for task dispatch."""
        # Convert to async and return task ID
        return asyncio.create_task(self.scheduler.dispatch_task(loop_id, task))

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Legacy synchronous method for getting task status."""
        # Convert to async
        return asyncio.create_task(self.scheduler.get_task_status(task_id))

    def export_snapshot(self) -> Dict[str, Any]:
        """Legacy synchronous method for snapshot export."""
        # Convert to async
        return asyncio.create_task(self.snapshot_manager.export_snapshot())

    def rollback(self, to_snapshot: str) -> Dict[str, Any]:
        """Legacy synchronous method for rollback."""
        # Convert to async
        return asyncio.create_task(self.snapshot_manager.rollback(to_snapshot))
