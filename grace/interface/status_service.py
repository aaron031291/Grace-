"""
Grace Status Service - FastAPI service for live telemetry and system status.
Part of Phase 2: Core Spine Boot implementation.
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from ..core.event_bus import EventBus
from ..orchestration.orchestration_service import OrchestrationService
from ..config.environment import get_grace_config, validate_environment

logger = logging.getLogger(__name__)


# Pydantic models for API responses
class HealthStatus(BaseModel):
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Status timestamp")
    uptime_seconds: float = Field(..., description="System uptime")
    version: str = Field(..., description="Grace version")
    instance_id: str = Field(..., description="Instance identifier")


class SystemStatus(BaseModel):
    health: HealthStatus
    components: Dict[str, Any] = Field(..., description="Component status details")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    configuration: Dict[str, Any] = Field(..., description="Current configuration")


class TelemetryData(BaseModel):
    timestamp: str
    metrics: Dict[str, float]
    events: Dict[str, int]
    performance: Dict[str, float]


class GraceStatusService:
    """
    FastAPI service providing live telemetry and system status endpoints.
    Integrates with EventBus and OrchestrationService.
    """
    
    def __init__(self, event_bus: EventBus, orchestration_service: OrchestrationService):
        self.config = get_grace_config()
        self.event_bus = event_bus
        self.orchestration_service = orchestration_service
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Grace Governance Status API",
            description="Live telemetry and system status for Grace governance kernel",
            version=self.config["environment_config"]["version"],
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["*"],
        )
        
        # Service state
        self.started_at = None
        self.running = False
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Grace Status Service initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "service": "Grace Governance Status API",
                "version": self.config["environment_config"]["version"],
                "status": "online" if self.running else "offline",
                "documentation": "/docs"
            }
        
        @self.app.get("/status", response_model=HealthStatus)
        async def get_status():
            """Get basic system status."""
            if not self.running:
                raise HTTPException(status_code=503, detail="Service not running")
            
            uptime = 0.0
            if self.started_at:
                uptime = time.time() - self.started_at
            
            # Determine overall status
            overall_status = "healthy"
            try:
                # Check EventBus status
                eventbus_status = self.event_bus.get_system_status()
                if not eventbus_status.get("eventbus", {}).get("running", False):
                    overall_status = "degraded"
                
                # Check OrchestrationService status  
                if hasattr(self.orchestration_service, 'running') and not self.orchestration_service.running:
                    overall_status = "degraded"
                    
            except Exception as e:
                logger.error(f"Error checking system status: {e}")
                overall_status = "error"
            
            return HealthStatus(
                status=overall_status,
                timestamp=datetime.now().isoformat(),
                uptime_seconds=uptime,
                version=self.config["environment_config"]["version"],
                instance_id=self.config["environment_config"]["instance_id"]
            )
        
        @self.app.get("/status/detailed", response_model=SystemStatus)
        async def get_detailed_status():
            """Get comprehensive system status."""
            if not self.running:
                raise HTTPException(status_code=503, detail="Service not running")
            
            try:
                # Get basic health status
                health = await get_status()
                
                # Get component statuses
                components = {}
                
                # EventBus status
                if self.event_bus:
                    components["eventbus"] = self.event_bus.get_system_status()
                
                # OrchestrationService status
                if self.orchestration_service:
                    try:
                        components["orchestration"] = {
                            "running": getattr(self.orchestration_service, 'running', False),
                            "started_at": getattr(self.orchestration_service, 'started_at', None)
                        }
                        
                        # Get orchestration status if available
                        if hasattr(self.orchestration_service, 'app'):
                            # Try to get status from orchestration API
                            orchestration_status = await self._get_orchestration_status()
                            if orchestration_status:
                                components["orchestration"].update(orchestration_status)
                                
                    except Exception as e:
                        components["orchestration"] = {"error": str(e)}
                
                # System metrics
                metrics = {
                    "total_components": len(components),
                    "healthy_components": len([c for c in components.values() 
                                             if isinstance(c, dict) and c.get("running", False)]),
                    "configuration_valid": len(validate_environment()) == 0
                }
                
                # Configuration summary
                configuration = {
                    "instance_id": self.config["environment_config"]["instance_id"],
                    "version": self.config["environment_config"]["version"],
                    "debug_mode": self.config["environment_config"]["debug_mode"],
                    "database_type": "postgres" if self.config["database_config"]["use_postgres"] else "sqlite",
                    "redis_enabled": self.config["database_config"]["use_redis_cache"],
                    "monitoring_enabled": self.config["infrastructure_config"]["enable_telemetry"]
                }
                
                return SystemStatus(
                    health=health,
                    components=components,
                    metrics=metrics,
                    configuration=configuration
                )
                
            except Exception as e:
                logger.error(f"Error getting detailed status: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/telemetry", response_model=TelemetryData)
        async def get_telemetry():
            """Get real-time telemetry data."""
            if not self.running:
                raise HTTPException(status_code=503, detail="Service not running")
            
            try:
                timestamp = datetime.now().isoformat()
                
                # Collect metrics from EventBus
                eventbus_status = self.event_bus.get_system_status() if self.event_bus else {}
                
                # Extract metrics
                metrics = {}
                events = {}
                performance = {}
                
                if "eventbus" in eventbus_status:
                    eb_data = eventbus_status["eventbus"]
                    metrics["total_events"] = eb_data.get("total_events", 0)
                    metrics["failed_deliveries"] = eb_data.get("failed_deliveries", 0)
                    metrics["success_rate"] = eb_data.get("success_rate", 0.0)
                    metrics["uptime_seconds"] = eb_data.get("uptime_seconds", 0.0)
                    
                    # Event type distribution
                    subscribers = eb_data.get("subscribers", {})
                    for event_type, count in subscribers.items():
                        events[event_type] = count
                
                # KPI Monitor metrics
                if "kpi_monitor" in eventbus_status:
                    kpi_data = eventbus_status["kpi_monitor"]
                    metrics_summary = kpi_data.get("metrics_summary", {})
                    
                    for metric_name, metric_data in metrics_summary.items():
                        if isinstance(metric_data, dict) and "latest" in metric_data:
                            performance[f"kpi_{metric_name}"] = metric_data["latest"]
                
                # Immutable Logs metrics
                if "immutable_logs" in eventbus_status:
                    logs_data = eventbus_status["immutable_logs"]
                    metrics["total_logs"] = logs_data.get("total_logs", 0)
                    metrics["log_buffer_utilization"] = logs_data.get("buffer_utilization", 0.0)
                
                return TelemetryData(
                    timestamp=timestamp,
                    metrics=metrics,
                    events=events,
                    performance=performance
                )
                
            except Exception as e:
                logger.error(f"Error getting telemetry: {e}")
                raise HTTPException(status_code=500, detail=f"Telemetry error: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """Simple health check endpoint."""
            return {"status": "ok" if self.running else "not running"}
        
        @self.app.get("/config/validation")
        async def validate_config():
            """Validate current configuration."""
            try:
                missing_vars = validate_environment()
                
                return {
                    "valid": len(missing_vars) == 0,
                    "missing_variables": missing_vars,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error validating configuration: {e}")
                raise HTTPException(status_code=500, detail=f"Configuration validation error: {str(e)}")
    
    async def _get_orchestration_status(self) -> Optional[Dict[str, Any]]:
        """Get status from orchestration service if available."""
        try:
            if hasattr(self.orchestration_service, 'get_system_status'):
                # Call the orchestration service status method
                return await self.orchestration_service.get_system_status()
            return None
        except Exception as e:
            logger.error(f"Error getting orchestration status: {e}")
            return None
    
    async def start(self):
        """Start the status service."""
        if self.running:
            logger.warning("Status service already running")
            return
        
        self.started_at = time.time()
        self.running = True
        
        # Log service start
        if self.event_bus:
            await self.event_bus.immutable_logs.log_event(
                event_type="status_service_started",
                component_id="status_service",
                event_data={
                    "started_at": datetime.now().isoformat(),
                    "port": self.config["environment_config"]["api_port"]
                }
            )
        
        logger.info("Grace Status Service started")
    
    async def stop(self):
        """Stop the status service."""
        if not self.running:
            return
        
        # Log service stop
        if self.event_bus:
            await self.event_bus.immutable_logs.log_event(
                event_type="status_service_stopped", 
                component_id="status_service",
                event_data={
                    "stopped_at": datetime.now().isoformat(),
                    "uptime_seconds": time.time() - self.started_at if self.started_at else 0
                }
            )
        
        self.running = False
        logger.info("Grace Status Service stopped")


async def create_status_service(event_bus: EventBus, orchestration_service: OrchestrationService) -> GraceStatusService:
    """Factory function to create and initialize the status service."""
    service = GraceStatusService(event_bus, orchestration_service)
    await service.start()
    return service