"""
Multi-OS Service - FastAPI facade for Multi-OS Kernel.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid

from orchestrator.scheduler import Scheduler
from inventory.registry import Registry
from telemetry.collector import TelemetryCollector
from snapshots.manager import SnapshotManager
from bridges.mesh_bridge import MeshBridge
from agents.linux import LinuxAdapter
from agents.windows import WindowsAdapter
from agents.macos import MacOSAdapter


logger = logging.getLogger(__name__)


# Request/Response models
class HostDescriptor(BaseModel):
    host_id: str
    os: str
    arch: str
    agent_version: str
    capabilities: List[str]
    labels: List[str]
    status: str
    endpoints: Optional[Dict[str, str]] = None


class RuntimeSpec(BaseModel):
    runtime: str
    version: str
    env: Optional[Dict[str, str]] = None
    packages: Optional[List[Dict[str, str]]] = None


class ExecTask(BaseModel):
    task_id: str
    command: str
    args: Optional[List[str]] = None
    cwd: Optional[str] = None
    runtime: RuntimeSpec
    io: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


class FSAction(BaseModel):
    action_id: str
    type: str
    path: str
    content_b64: Optional[str] = None
    recursive: Optional[bool] = False


class NetAction(BaseModel):
    action_id: str
    type: str
    url: Optional[str] = None
    body: Optional[str] = None
    timeout_s: Optional[int] = 30


class MultiOSService:
    """
    FastAPI service facade for Multi-OS Kernel external interface.
    Provides unified API for cross-platform operations with sandboxing, RBAC, and telemetry.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Initialize core components
        self.scheduler = Scheduler(self.config)
        self.registry = Registry()
        self.telemetry = TelemetryCollector(self.config)
        self.snapshot_manager = SnapshotManager()
        self.mesh_bridge = MeshBridge()
        
        # Initialize OS adapters
        self.adapters = {
            "linux": LinuxAdapter(self.config.get("linux", {})),
            "windows": WindowsAdapter(self.config.get("windows", {})),
            "macos": MacOSAdapter(self.config.get("macos", {}))
        }
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Multi-OS Kernel API",
            description="Unified execution layer across Linux/Windows/macOS with sandboxing, RBAC, and telemetry",
            version="1.0.0",
            docs_url="/api/mos/v1/docs",
            redoc_url="/api/mos/v1/redoc"
        )
        
        # Register routes
        self._register_routes()
        
        # Background tasks
        self.background_tasks = []
        
        logger.info("Multi-OS Service initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default service configuration."""
        return {
            "placement": {
                "weights": {"capability_fit": 0.4, "latency": 0.25, "success": 0.25, "gpu": 0.1}
            },
            "sandbox": {
                "default": "nsjail",
                "profiles": {"linux": "ns_v5", "windows": "appcontainer_low", "macos": "sandboxd_v3"}
            },
            "network": {
                "default_policy": "deny_all",
                "allowlist": ["api.company.local"]
            },
            "runtimes": {
                "prewarm": ["python@3.11", "node@18"]
            },
            "rollout": {
                "strategy": "blue_green",
                "rings": ["canary:5%", "ring1:25%", "ring2:50%", "ring3:100%"]
            },
            "timeouts": {
                "task_max_runtime_s": 1800
            }
        }
    
    def _register_routes(self):
        """Register FastAPI routes."""
        
        @self.app.get("/api/mos/v1/health")
        async def get_health():
            """Health check endpoint."""
            try:
                stats = {
                    "scheduler": self.scheduler.get_placement_stats(),
                    "registry": self.registry.get_inventory_stats(),
                    "telemetry": len(self.telemetry.metrics),
                    "snapshots": self.snapshot_manager.get_stats()
                }
                
                return {
                    "status": "ok",
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow().isoformat(),
                    "components": stats
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @self.app.get("/api/mos/v1/hosts", response_model=List[Dict])
        async def list_hosts(os: Optional[str] = None, status: Optional[str] = None):
            """Get list of registered hosts."""
            try:
                filters = {}
                if os:
                    filters["os"] = os
                if status:
                    filters["status"] = status
                
                hosts = self.registry.list_hosts(filters)
                return hosts
            except Exception as e:
                logger.error(f"Failed to list hosts: {e}")
                raise HTTPException(status_code=500, detail="Failed to list hosts")
        
        @self.app.post("/api/mos/v1/hosts/register", response_model=Dict[str, str])
        async def register_host(host_descriptor: HostDescriptor):
            """Register a new host."""
            try:
                host_id = self.registry.register_host(host_descriptor.dict())
                
                # Publish registration event
                await self.mesh_bridge.publish_host_registered(host_descriptor.dict())
                
                # Record telemetry
                self.telemetry.record_event(
                    "MOS_HOST_REGISTERED",
                    host_descriptor.dict(),
                    host_id=host_id
                )
                
                return {"host_id": host_id}
            except Exception as e:
                logger.error(f"Host registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/mos/v1/task/submit", response_model=Dict[str, str])
        async def submit_task(task: ExecTask, background_tasks: BackgroundTasks):
            """Submit a task for execution."""
            try:
                # Get available hosts
                hosts = self.registry.list_hosts({"status": "online"})
                
                if not hosts:
                    raise HTTPException(status_code=503, detail="No online hosts available")
                
                # Find optimal placement
                placement = await self.scheduler.place(task.dict(), hosts)
                
                if not placement["success"]:
                    raise HTTPException(status_code=503, detail=placement.get("reason", "Placement failed"))
                
                host_id = placement["host_id"]
                
                # Submit task to scheduler
                submission_id = await self.scheduler.submit(host_id, task.dict())
                
                # Publish events
                await self.mesh_bridge.publish_task_submitted(task.dict(), host_id)
                
                # Record telemetry
                self.telemetry.record_event("MOS_TASK_SUBMITTED", 
                                          {"task": task.dict(), "host_id": host_id}, 
                                          host_id=host_id)
                
                # Execute task in background
                background_tasks.add_task(self._execute_task_background, task.dict(), host_id)
                
                return {"task_id": task.task_id, "host_id": host_id}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Task submission failed: {e}")
                raise HTTPException(status_code=500, detail="Task submission failed")
        
        @self.app.get("/api/mos/v1/task/{task_id}/status", response_model=Dict[str, Any])
        async def get_task_status(task_id: str):
            """Get task execution status."""
            try:
                # In a real implementation, would query task database
                # For now, return mock status
                return {
                    "state": "completed",
                    "exit_code": 0,
                    "logs_uri": f"/api/mos/v1/logs/{task_id}"
                }
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get task status")
        
        @self.app.post("/api/mos/v1/fs", response_model=Dict[str, Any])
        async def execute_fs_action(action: FSAction):
            """Execute filesystem action."""
            try:
                # For demo, use Linux adapter
                adapter = self.adapters["linux"]
                result = await adapter.apply(action.dict())
                
                # Record telemetry
                self.telemetry.record_event("MOS_FS_APPLIED", 
                                          {"action_id": action.action_id, "result": result})
                
                return {"result": result}
            except Exception as e:
                logger.error(f"Filesystem action failed: {e}")
                raise HTTPException(status_code=500, detail="Filesystem action failed")
        
        @self.app.post("/api/mos/v1/net", response_model=Dict[str, Any])
        async def execute_net_action(action: NetAction):
            """Execute network action."""
            try:
                # For demo, use Linux adapter
                adapter = self.adapters["linux"]
                result = await adapter.apply(action.dict())
                
                # Record telemetry
                self.telemetry.record_event("MOS_NET_APPLIED",
                                          {"action_id": action.action_id, "result": result})
                
                return {"result": result}
            except Exception as e:
                logger.error(f"Network action failed: {e}")
                raise HTTPException(status_code=500, detail="Network action failed")
        
        @self.app.post("/api/mos/v1/runtime/ensure", response_model=Dict[str, str])
        async def ensure_runtime(runtime_spec: RuntimeSpec):
            """Ensure runtime environment is available."""
            try:
                job_id = str(uuid.uuid4())
                
                # For demo, use Linux adapter
                adapter = self.adapters["linux"]
                result = await adapter.ensure(runtime_spec.dict())
                
                return {"job_id": job_id}
            except Exception as e:
                logger.error(f"Runtime setup failed: {e}")
                raise HTTPException(status_code=500, detail="Runtime setup failed")
        
        @self.app.post("/api/mos/v1/agent/rollout", response_model=Dict[str, str])
        async def start_agent_rollout(rollout_spec: Dict[str, str]):
            """Start agent rollout."""
            try:
                mode = rollout_spec.get("mode", "blue")
                to_version = rollout_spec.get("to", "")
                
                # Publish rollout event
                await self.mesh_bridge.publish_agent_rollout("current", to_version, mode, 0)
                
                return {"mode": mode, "to": to_version}
            except Exception as e:
                logger.error(f"Agent rollout failed: {e}")
                raise HTTPException(status_code=500, detail="Agent rollout failed")
        
        @self.app.post("/api/mos/v1/snapshot/export", response_model=Dict[str, str])
        async def export_snapshot(snapshot_spec: Dict[str, str]):
            """Export system snapshot."""
            try:
                scope = snapshot_spec.get("scope", "agent")
                
                result = await self.snapshot_manager.create_snapshot(scope)
                
                # Publish snapshot event
                await self.mesh_bridge.publish_snapshot_created(
                    result["snapshot_id"], 
                    result["scope"], 
                    result["uri"]
                )
                
                return {
                    "snapshot_id": result["snapshot_id"],
                    "uri": result["uri"],
                    "scope": result["scope"]
                }
            except Exception as e:
                logger.error(f"Snapshot export failed: {e}")
                raise HTTPException(status_code=500, detail="Snapshot export failed")
        
        @self.app.post("/api/mos/v1/rollback", response_model=Dict[str, str])
        async def rollback_to_snapshot(rollback_spec: Dict[str, str]):
            """Rollback to snapshot."""
            try:
                to_snapshot = rollback_spec.get("to_snapshot", "")
                
                # Publish rollback request
                await self.mesh_bridge.publish_rollback_requested("multi_os", to_snapshot)
                
                # Execute rollback
                result = await self.snapshot_manager.rollback(to_snapshot)
                
                if result.get("success"):
                    # Publish completion
                    await self.mesh_bridge.publish_rollback_completed("multi_os", to_snapshot)
                
                return {"to_snapshot": to_snapshot}
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                raise HTTPException(status_code=500, detail="Rollback failed")
        
        # Additional monitoring endpoints
        @self.app.get("/api/mos/v1/metrics")
        async def get_metrics():
            """Get telemetry metrics."""
            return self.telemetry.get_metrics_summary()
        
        @self.app.get("/api/mos/v1/events")
        async def get_events(event_name: Optional[str] = None, limit: int = 100):
            """Get recent events."""
            return self.mesh_bridge.get_published_events(event_name, limit)
        
        @self.app.get("/api/mos/v1/placement/stats")
        async def get_placement_stats():
            """Get scheduler placement statistics."""
            return self.scheduler.get_placement_stats()
    
    async def _execute_task_background(self, task: Dict[str, Any], host_id: str):
        """Execute task in background and handle results."""
        try:
            task_id = task["task_id"]
            
            # Publish task started
            await self.mesh_bridge.publish_task_started(task_id, 12345, host_id)  # Mock PID
            
            # Determine which adapter to use based on host OS
            host = self.registry.get_host(host_id)
            if not host:
                logger.error(f"Host {host_id} not found")
                return
            
            os_type = host.get("os", "linux")
            adapter = self.adapters.get(os_type)
            
            if not adapter:
                logger.error(f"No adapter available for OS {os_type}")
                return
            
            # Execute task
            start_time = datetime.utcnow()
            result = await adapter.exec(task)
            end_time = datetime.utcnow()
            
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update host statistics
            success = result.get("success", False)
            self.registry.update_task_stats(host_id, success)
            
            # Publish completion event
            await self.mesh_bridge.publish_task_completed(
                task_id,
                "success" if success else "failed",
                result.get("exit_code", -1),
                result.get("outputs", []),
                f"/logs/{task_id}",
                duration_ms,
                host_id
            )
            
            # Record telemetry
            self.telemetry.record_event("MOS_TASK_COMPLETED", {
                "task_id": task_id,
                "status": "success" if success else "failed",
                "duration_ms": duration_ms,
                "host_id": host_id
            }, host_id=host_id)
            
            logger.info(f"Task {task_id} completed on {host_id} with success={success}")
            
        except Exception as e:
            logger.error(f"Background task execution failed: {e}")
    
    async def start_background_services(self):
        """Start background services and health monitoring."""
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        # Cleanup old telemetry data periodically
        asyncio.create_task(self._cleanup_telemetry())
        
        logger.info("Background services started")
    
    async def _health_monitor(self):
        """Background health monitoring of hosts."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check each registered host
                for host_id, host in self.registry.hosts.items():
                    # Mock health check - in real implementation would ping host
                    health_metrics = {
                        "cpu": 0.3,
                        "mem_used_mb": 2048,
                        "gpu_util": 0.0,
                        "disk_free_mb": 50000
                    }
                    
                    # Update host status
                    self.registry.update_host_status(host_id, "online", health_metrics)
                    
                    # Publish health event
                    await self.mesh_bridge.publish_host_health(host_id, "online", health_metrics)
                
                # Clean up stale hosts
                removed_count = self.registry.cleanup_stale_hosts()
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} stale hosts")
                    
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _cleanup_telemetry(self):
        """Background telemetry cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                stats = self.telemetry.cleanup_old_data()
                logger.info(f"Telemetry cleanup: {stats}")
            except Exception as e:
                logger.error(f"Telemetry cleanup error: {e}")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            "hosts": self.registry.get_inventory_stats(),
            "placement": self.scheduler.get_placement_stats(),
            "telemetry": self.telemetry.get_kpis(),
            "snapshots": self.snapshot_manager.get_stats(),
            "mesh": self.mesh_bridge.get_bridge_status(),
            "adapters": list(self.adapters.keys())
        }