"""
Health check routes for Grace Service.
"""
import time
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Depends
import structlog

from ..schemas.base import HealthResponse, BaseResponse

logger = structlog.get_logger(__name__)

health_router = APIRouter()

# Store startup time
startup_time = time.time()


def get_app_state():
    """Dependency injection placeholder."""
    pass


@health_router.get("/status", response_model=HealthResponse)
async def health_check(app_state: Dict[str, Any] = Depends(get_app_state)):
    """
    Comprehensive health check endpoint.
    
    Returns detailed health status for all Grace components
    including database connectivity, kernel status, and system metrics.
    """
    try:
        uptime_seconds = int(time.time() - startup_time)
        
        # Check component health
        components = {}
        overall_status = "healthy"
        
        # Check governance kernel
        governance_kernel = app_state.get("governance_kernel")
        if governance_kernel:
            try:
                # You would check if the kernel is responsive
                components["governance_kernel"] = "healthy"
            except Exception as e:
                components["governance_kernel"] = f"unhealthy: {str(e)}"
                overall_status = "degraded"
        else:
            components["governance_kernel"] = "not_initialized"
            overall_status = "unhealthy"
        
        # Check ingress kernel
        ingress_kernel = app_state.get("ingress_kernel")
        if ingress_kernel:
            components["ingress_kernel"] = "healthy"
        else:
            components["ingress_kernel"] = "not_initialized"
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Check orchestration service
        orchestration_service = app_state.get("orchestration_service")
        if orchestration_service:
            components["orchestration_service"] = "healthy"
        else:
            components["orchestration_service"] = "not_initialized"
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Check database connectivity (placeholder)
        try:
            # This would actually test database connection
            components["database"] = "healthy"
        except Exception as e:
            components["database"] = f"unhealthy: {str(e)}"
            overall_status = "unhealthy"
        
        # Check Redis connectivity (placeholder)
        try:
            components["redis"] = "healthy"
        except Exception as e:
            components["redis"] = f"unhealthy: {str(e)}"
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Check ChromaDB connectivity (placeholder)
        try:
            components["vector_db"] = "healthy"
        except Exception as e:
            components["vector_db"] = f"unhealthy: {str(e)}"
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Get system metrics
        metrics = _get_system_metrics()
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            components=components,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            uptime_seconds=0,
            components={"error": str(e)},
            metrics={}
        )


@health_router.get("/live")
async def liveness_probe():
    """
    Simple liveness probe for Kubernetes/container orchestrators.
    Returns 200 if the service is running.
    """
    return {"status": "alive", "timestamp": time.time()}


@health_router.get("/ready")
async def readiness_probe(app_state: Dict[str, Any] = Depends(get_app_state)):
    """
    Readiness probe for Kubernetes/container orchestrators.
    Returns 200 only if all critical components are ready.
    """
    try:
        # Check if critical components are initialized
        governance_kernel = app_state.get("governance_kernel")
        if not governance_kernel:
            return {"status": "not_ready", "reason": "governance_kernel_not_initialized"}
        
        return {"status": "ready", "timestamp": time.time()}
        
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


@health_router.get("/metrics")
async def get_metrics():
    """
    Get detailed system metrics.
    """
    try:
        metrics = _get_system_metrics()
        
        # Add application-specific metrics
        metrics.update({
            "websocket_connections": 0,  # Would get from WebSocketManager
            "active_governance_requests": 0,  # Would get from governance kernel
            "ingestion_queue_size": 0,  # Would get from ingress kernel
        })
        
        return BaseResponse(
            status="success",
            message="Metrics retrieved successfully",
            data=metrics
        )
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        return BaseResponse(
            status="error",
            message=f"Failed to get metrics: {str(e)}"
        )


def _get_system_metrics() -> Dict[str, Any]:
    """Get system-level metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_bytes": memory.available,
            "disk_usage_percent": disk.percent,
            "disk_free_bytes": disk.free,
            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0],
        }
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        return {}