"""
Resilience Kernel Service - FastAPI façade for Grace's self-healing layer.

Provides REST API endpoints for SLO management, policy configuration, incident handling,
circuit breakers, graceful degradation, chaos engineering, and system recovery.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
import json

from .policies.slo_manager import SLOManager
from .policies.resilience_policy_manager import ResiliencePolicyManager
from .detectors.slis import SLIEvaluator
from .detectors.anomaly import AnomalyDetector
from .detectors.dependency import DependencyHealthMonitor
from .controllers.circuit import CircuitBreaker
from .controllers.degradation import DegradationController
from .controllers.rate_limit import RateLimiter
from .repair.restart import RestartManager
from .repair.reprovision import ReprovisionManager
from .repair.snapshot import SnapshotRecovery
from .chaos.runner import ChaosRunner
from .telemetry.budget import ErrorBudgetManager
from .snapshots.manager import SnapshotManager
from .playbooks.executor import PlaybookExecutor
from .bridges.mesh_bridge import ResilienceMeshBridge
from .bridges.gov_bridge import GovernanceBridge
from .bridges.orch_bridge import OrchestrationBridge
from .bridges.immune_bridge import ImmuneBridge
from .bridges.multi_os_bridge import MultiOSBridge

logger = logging.getLogger(__name__)


class ResilienceService:
    """Main Resilience Service providing FastAPI interface and orchestration."""
    
    def __init__(self, event_bus=None, governance_engine=None, orchestration_engine=None):
        """Initialize the resilience service with dependencies."""
        self.event_bus = event_bus
        self.governance_engine = governance_engine
        self.orchestration_engine = orchestration_engine
        
        # Initialize managers and controllers
        self.slo_manager = SLOManager()
        self.resilience_policy_manager = ResiliencePolicyManager()
        self.sli_evaluator = SLIEvaluator()
        self.anomaly_detector = AnomalyDetector()
        self.dependency_monitor = DependencyHealthMonitor()
        self.degradation_controller = DegradationController()
        self.rate_limiter = RateLimiter()
        self.restart_manager = RestartManager()
        self.reprovision_manager = ReprovisionManager()
        self.snapshot_recovery = SnapshotRecovery()
        self.chaos_runner = ChaosRunner()
        self.error_budget_manager = ErrorBudgetManager()
        self.snapshot_manager = SnapshotManager()
        self.playbook_executor = PlaybookExecutor()
        
        # Initialize bridges
        self.mesh_bridge = ResilienceMeshBridge(self, event_bus)
        self.gov_bridge = GovernanceBridge(governance_engine, event_bus)
        self.orch_bridge = OrchestrationBridge(orchestration_engine, event_bus)
        self.immune_bridge = ImmuneBridge(event_bus)
        self.multi_os_bridge = MultiOSBridge(event_bus)
        
        # State management
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.incidents: Dict[str, Dict] = {}
        self.dependency_graphs: Dict[str, Dict] = {}
        self.running = False
        
        # Create FastAPI app
        self.app = self._create_app()
        
        logger.info("ResilienceService initialized")
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Grace Resilience Kernel",
            description="Self-healing layer for Grace AI system",
            version="1.0.0",
            openapi_url="/api/res/v1/openapi.json",
            docs_url="/api/res/v1/docs"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._register_routes(app)
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register all API routes."""
        
        @app.get("/api/res/v1/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy" if self.running else "stopped",
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "slo_manager": "healthy",
                    "circuit_breakers": len(self.circuit_breakers),
                    "active_incidents": len([i for i in self.incidents.values() if i["status"] == "open"]),
                    "chaos_experiments": self.chaos_runner.get_active_count()
                }
            }
        
        @app.post("/api/res/v1/policy/slo")
        async def set_slo_policy(policy: dict):
            """Set SLO policy for a service."""
            try:
                result = self.set_slo(policy)
                await self._emit_event("RES_SLO_POLICY_SET", {
                    "service_id": policy["service_id"],
                    "slos": policy["slos"]
                })
                return result
            except Exception as e:
                logger.error(f"Error setting SLO policy: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/policy/resilience")
        async def set_resilience_policy(policy: dict):
            """Set resilience policy for a service."""
            try:
                result = self.set_resilience_policy(policy)
                await self._emit_event("RES_POLICY_SET", {
                    "service_id": policy["service_id"],
                    "policy": policy
                })
                return result
            except Exception as e:
                logger.error(f"Error setting resilience policy: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/api/res/v1/policy/{service_id}")
        async def get_policy(service_id: str):
            """Get policies for a service."""
            try:
                slo = self.slo_manager.get_policy(service_id)
                resilience = self.resilience_policy_manager.get_policy(service_id)
                return {"slo": slo, "resilience": resilience}
            except Exception as e:
                logger.error(f"Error getting policy for {service_id}: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        @app.post("/api/res/v1/dependency/graph")
        async def set_dependency_graph(graph: dict):
            """Set dependency graph for a service."""
            try:
                result = self.submit_dependency_graph(graph)
                return result
            except Exception as e:
                logger.error(f"Error setting dependency graph: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/incident/open")
        async def open_incident(incident: dict):
            """Open a new incident."""
            try:
                incident_id = self.open_incident(incident)
                return {"incident_id": incident_id}
            except Exception as e:
                logger.error(f"Error opening incident: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/incident/{incident_id}/action")
        async def apply_action(incident_id: str, action_data: dict):
            """Apply action to an incident."""
            try:
                result = self.apply_action(incident_id, action_data["action"], action_data.get("params"))
                return result
            except Exception as e:
                logger.error(f"Error applying action to incident {incident_id}: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/breaker/{service_id}/{dep}/state")
        async def set_breaker_state(service_id: str, dep: str, state_data: dict):
            """Set circuit breaker state."""
            try:
                breaker_key = f"{service_id}:{dep}"
                if breaker_key not in self.circuit_breakers:
                    self.circuit_breakers[breaker_key] = CircuitBreaker(service_id, dep)
                
                breaker = self.circuit_breakers[breaker_key]
                state = state_data["state"]
                
                if state == "open":
                    breaker.force_open()
                elif state == "closed":
                    breaker.force_close()
                elif state == "half_open":
                    breaker.force_half_open()
                
                await self._emit_event("RES_CIRCUIT_STATE", {
                    "service_id": service_id,
                    "dependency": dep,
                    "state": breaker.state()
                })
                
                return {"state": breaker.state()}
            except Exception as e:
                logger.error(f"Error setting breaker state: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/degradation/{service_id}/enter")
        async def enter_degradation(service_id: str, mode_data: dict):
            """Enter degradation mode."""
            try:
                mode_id = mode_data["mode_id"]
                self.degradation_controller.enter_mode(service_id, mode_id)
                
                await self._emit_event("RES_DEGRADATION_ENTERED", {
                    "service_id": service_id,
                    "mode_id": mode_id
                })
                
                return {"accepted": True, "mode_id": mode_id}
            except Exception as e:
                logger.error(f"Error entering degradation mode: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/degradation/{service_id}/exit")
        async def exit_degradation(service_id: str, mode_data: dict):
            """Exit degradation mode."""
            try:
                mode_id = mode_data["mode_id"]
                self.degradation_controller.exit_mode(service_id, mode_id)
                
                await self._emit_event("RES_DEGRADATION_EXITED", {
                    "service_id": service_id,
                    "mode_id": mode_id
                })
                
                return {"accepted": True, "mode_id": mode_id}
            except Exception as e:
                logger.error(f"Error exiting degradation mode: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/chaos/start")
        async def start_chaos(experiment_data: dict):
            """Start chaos experiment."""
            try:
                experiment_id = await self.chaos_runner.start_experiment(
                    experiment_data["target"],
                    experiment_data.get("blast_radius_pct", 5.0),
                    experiment_data.get("duration_s", 300)
                )
                
                await self._emit_event("RES_CHAOS_STARTED", {
                    "experiment_id": experiment_id,
                    "target": experiment_data["target"],
                    "blast_radius_pct": experiment_data.get("blast_radius_pct", 5.0)
                })
                
                return {"accepted": True, "experiment_id": experiment_id}
            except Exception as e:
                logger.error(f"Error starting chaos experiment: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/api/res/v1/chaos/{experiment_id}/result")
        async def get_chaos_result(experiment_id: str):
            """Get chaos experiment result."""
            try:
                result = await self.chaos_runner.get_result(experiment_id)
                if result is None:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                return result
            except Exception as e:
                logger.error(f"Error getting chaos result: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/api/res/v1/snapshot/export")
        async def export_snapshot():
            """Export system snapshot."""
            try:
                result = self.export_snapshot()
                await self._emit_event("RES_SNAPSHOT_CREATED", {
                    "snapshot_id": result["snapshot_id"],
                    "uri": result["uri"]
                })
                return result
            except Exception as e:
                logger.error(f"Error exporting snapshot: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/res/v1/rollback")
        async def rollback(rollback_data: dict):
            """Rollback to snapshot."""
            try:
                await self._emit_event("ROLLBACK_REQUESTED", {
                    "target": "resilience",
                    "to_snapshot": rollback_data["to_snapshot"]
                })
                
                result = self.rollback(rollback_data["to_snapshot"])
                
                await self._emit_event("ROLLBACK_COMPLETED", {
                    "target": "resilience",
                    "snapshot_id": rollback_data["to_snapshot"],
                    "at": datetime.utcnow().isoformat()
                })
                
                return {"accepted": True, "to_snapshot": rollback_data["to_snapshot"]}
            except Exception as e:
                logger.error(f"Error during rollback: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # Core business logic methods
    
    def set_slo(self, policy: dict) -> dict:
        """Set SLO policy for a service."""
        service_id = policy["service_id"]
        self.slo_manager.set_policy(service_id, policy)
        self.error_budget_manager.initialize_budget(service_id, policy["error_budget_days"])
        return {"service_id": service_id}
    
    def set_resilience_policy(self, policy: dict) -> dict:
        """Set resilience policy for a service."""
        service_id = policy["service_id"]
        self.resilience_policy_manager.set_policy(service_id, policy)
        return {"service_id": service_id}
    
    def submit_dependency_graph(self, graph: dict) -> dict:
        """Submit dependency graph for a service."""
        service_id = graph["service_id"]
        self.dependency_graphs[service_id] = graph
        self.dependency_monitor.update_graph(service_id, graph)
        return {"service_id": service_id}
    
    def open_incident(self, incident: dict) -> str:
        """Open a new incident."""
        incident_id = incident.get("incident_id", str(uuid.uuid4()))
        incident["incident_id"] = incident_id
        incident["status"] = "open"
        incident["opened_at"] = datetime.utcnow().isoformat()
        
        self.incidents[incident_id] = incident
        
        # Start playbook execution
        asyncio.create_task(self._handle_incident_async(incident_id))
        
        return incident_id
    
    def apply_action(self, incident_id: str, action: str, params: dict = None) -> dict:
        """Apply action to an incident."""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        service_id = incident["service_id"]
        
        # Execute the action
        result = self._execute_action(service_id, action, params or {})
        
        # Record the action
        if "actions" not in incident:
            incident["actions"] = []
        incident["actions"].append({
            "action": action,
            "params": params,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"accepted": True, "result": result}
    
    def export_snapshot(self) -> dict:
        """Export current resilience state snapshot."""
        snapshot_id = f"res_{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "services": {},
            "dependency_graphs": self.dependency_graphs,
            "chaos": {
                "enabled": self.chaos_runner.is_enabled(),
                "max_blast_radius_pct": 5
            },
            "hash": "sha256:placeholder"
        }
        
        # Collect policies for all services
        for service_id in self.slo_manager.get_all_services():
            slo_policy = self.slo_manager.get_policy(service_id)
            resilience_policy = self.resilience_policy_manager.get_policy(service_id)
            
            snapshot["services"][service_id] = {
                "slo": slo_policy,
                "policy": resilience_policy
            }
        
        # Store snapshot
        uri = self.snapshot_manager.store_snapshot(snapshot)
        snapshot["uri"] = uri
        
        return snapshot
    
    def rollback(self, to_snapshot: str) -> dict:
        """Rollback to a specific snapshot."""
        # Load snapshot
        snapshot = self.snapshot_manager.load_snapshot(to_snapshot)
        if not snapshot:
            raise ValueError(f"Snapshot {to_snapshot} not found")
        
        # Restore policies
        for service_id, service_data in snapshot["services"].items():
            if "slo" in service_data:
                self.slo_manager.set_policy(service_id, service_data["slo"])
            if "policy" in service_data:
                self.resilience_policy_manager.set_policy(service_id, service_data["policy"])
        
        # Restore dependency graphs
        self.dependency_graphs = snapshot.get("dependency_graphs", {})
        
        # Reset circuit breakers to cautious state
        for breaker in self.circuit_breakers.values():
            breaker.force_close()
        
        return {
            "snapshot_id": to_snapshot,
            "restored_at": datetime.utcnow().isoformat()
        }
    
    async def _handle_incident_async(self, incident_id: str):
        """Handle incident asynchronously with playbook execution."""
        try:
            incident = self.incidents[incident_id]
            playbook_id = incident.get("playbook_id", "default_incident_response")
            
            # Execute playbook
            await self.playbook_executor.execute(playbook_id, incident)
            
            # Emit incident opened event
            await self._emit_event("RES_INCIDENT_OPENED", {"incident": incident})
            
        except Exception as e:
            logger.error(f"Error handling incident {incident_id}: {e}")
    
    def _execute_action(self, service_id: str, action: str, params: dict) -> str:
        """Execute a specific recovery action."""
        try:
            if action == "restart":
                return self.restart_manager.restart_service(service_id, params)
            elif action == "reprovision":
                return self.reprovision_manager.reprovision_service(service_id, params)
            elif action == "reload_snapshot":
                return self.snapshot_recovery.reload_snapshot(service_id, params)
            elif action == "warm_cache":
                return f"Cache warmed for {service_id}"
            elif action == "flip_shadow":
                return f"Shadow traffic flipped for {service_id}"
            elif action == "rollback":
                self.rollback(params.get("to_snapshot"))
                return f"Rollback completed for {service_id}"
            elif action == "shed_load":
                self.rate_limiter.enable_shedding(service_id, params.get("rate", 0.1))
                return f"Load shedding enabled for {service_id}"
            elif action == "degrade_mode":
                mode_id = params.get("mode_id", "cached_only")
                self.degradation_controller.enter_mode(service_id, mode_id)
                return f"Entered degradation mode {mode_id} for {service_id}"
            elif action == "open_breaker":
                dep = params.get("dependency")
                if dep:
                    breaker_key = f"{service_id}:{dep}"
                    if breaker_key not in self.circuit_breakers:
                        self.circuit_breakers[breaker_key] = CircuitBreaker(service_id, dep)
                    self.circuit_breakers[breaker_key].force_open()
                    return f"Circuit breaker opened for {service_id}:{dep}"
                return "No dependency specified for breaker"
            elif action == "close_breaker":
                dep = params.get("dependency")
                if dep:
                    breaker_key = f"{service_id}:{dep}"
                    if breaker_key in self.circuit_breakers:
                        self.circuit_breakers[breaker_key].force_close()
                        return f"Circuit breaker closed for {service_id}:{dep}"
                return f"No breaker found for {service_id}:{dep}"
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return f"Action failed: {str(e)}"
    
    async def _emit_event(self, event_type: str, payload: dict):
        """Emit event through mesh bridge."""
        try:
            await self.mesh_bridge.emit_event(event_type, payload)
        except Exception as e:
            logger.error(f"Failed to emit event {event_type}: {e}")
    
    async def start(self):
        """Start the resilience service."""
        if self.running:
            return
        
        self.running = True
        
        # Start bridges
        await self.mesh_bridge.start()
        await self.gov_bridge.start()
        await self.orch_bridge.start()
        await self.immune_bridge.start()
        await self.multi_os_bridge.start()
        
        # Start monitoring and detection loops
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("ResilienceService started")
    
    async def stop(self):
        """Stop the resilience service."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop bridges
        await self.mesh_bridge.stop()
        await self.gov_bridge.stop()
        await self.orch_bridge.stop()
        await self.immune_bridge.stop()
        await self.multi_os_bridge.stop()
        
        logger.info("ResilienceService stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for health checks and SLI evaluation."""
        while self.running:
            try:
                # Evaluate SLIs for all services
                for service_id in self.slo_manager.get_all_services():
                    slo_policy = self.slo_manager.get_policy(service_id)
                    if slo_policy:
                        # This would collect real metrics in production
                        mock_samples = {
                            "latency_p95_ms": [750, 820, 680, 920],
                            "availability_pct": [99.95],
                            "error_rate_pct": [0.8]
                        }
                        
                        evaluation = self.sli_evaluator.evaluate_sli(mock_samples, slo_policy)
                        
                        # Check for SLO violations
                        if evaluation["violations"]:
                            await self._handle_slo_violations(service_id, evaluation)
                
                # Monitor dependencies
                await self._monitor_dependencies()
                
                # Update error budgets
                self.error_budget_manager.update_all_budgets()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _handle_slo_violations(self, service_id: str, evaluation: dict):
        """Handle SLO violations by opening incidents or taking preventive actions."""
        for violation in evaluation["violations"]:
            # Check if we should open an incident
            severity = self._assess_violation_severity(violation)
            
            if severity in ["sev1", "sev2"]:
                incident = {
                    "service_id": service_id,
                    "severity": severity,
                    "signals": {"slo_violation": violation},
                    "hypothesis": f"SLO violation: {violation['sli']} is {violation['actual']} vs target {violation['target']}"
                }
                
                incident_id = self.open_incident(incident)
                logger.warning(f"Opened incident {incident_id} for SLO violation in {service_id}")
    
    async def _monitor_dependencies(self):
        """Monitor dependency health across all services."""
        for service_id, graph in self.dependency_graphs.items():
            health_status = await self.dependency_monitor.check_dependencies(service_id)
            
            # Handle unhealthy dependencies
            for dep_id, status in health_status.items():
                if status["status"] != "healthy":
                    await self._handle_dependency_issue(service_id, dep_id, status)
    
    async def _handle_dependency_issue(self, service_id: str, dep_id: str, status: dict):
        """Handle dependency health issues."""
        breaker_key = f"{service_id}:{dep_id}"
        
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = CircuitBreaker(service_id, dep_id)
        
        breaker = self.circuit_breakers[breaker_key]
        
        if status["status"] == "down":
            breaker.record_failure()
            
            # Check if we should enter degradation mode
            resilience_policy = self.resilience_policy_manager.get_policy(service_id)
            if resilience_policy and "degradation_modes" in resilience_policy:
                for mode in resilience_policy["degradation_modes"]:
                    if "dependency_down" in mode["triggers"]:
                        self.degradation_controller.enter_mode(service_id, mode["mode_id"])
                        break
    
    def _assess_violation_severity(self, violation: dict) -> str:
        """Assess the severity of an SLO violation."""
        sli = violation["sli"]
        actual = violation["actual"]
        target = violation["target"]
        
        if sli == "availability_pct":
            if actual < 99.0:
                return "sev1"
            elif actual < 99.5:
                return "sev2"
            else:
                return "sev3"
        elif sli == "latency_p95_ms":
            deviation = (actual - target) / target
            if deviation > 2.0:  # 200% over target
                return "sev1"
            elif deviation > 1.0:  # 100% over target
                return "sev2"
            else:
                return "sev3"
        elif sli == "error_rate_pct":
            if actual > 10.0:
                return "sev1"
            elif actual > 5.0:
                return "sev2"
            else:
                return "sev3"
        
        return "sev3"
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app
    
    def get_stats(self) -> dict:
        """Get comprehensive resilience service statistics."""
        return {
            "running": self.running,
            "services_monitored": len(self.slo_manager.get_all_services()),
            "circuit_breakers": len(self.circuit_breakers),
            "active_incidents": len([i for i in self.incidents.values() if i["status"] == "open"]),
            "dependency_graphs": len(self.dependency_graphs),
            "chaos_experiments": self.chaos_runner.get_stats(),
            "error_budgets": self.error_budget_manager.get_all_budgets_status()
        }


def create_resilience_app(event_bus=None, governance_engine=None, orchestration_engine=None) -> FastAPI:
    """Factory function to create ResilienceService FastAPI app."""
    service = ResilienceService(event_bus, governance_engine, orchestration_engine)
    return service.get_app()