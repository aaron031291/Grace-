"""
Grace Service - FastAPI application wrapper for Grace Governance Kernel
Provides REST API and WebSocket endpoints for production deployment.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import structlog

from .routes.governance import governance_router
from .routes.health import health_router
from .routes.ingest import ingest_router
from .routes.events import events_router
from .schemas.base import BaseResponse
from .websocket_manager import WebSocketManager

# Import Grace kernels
try:
    from grace.governance.grace_governance_kernel import GraceGovernanceKernel
    from grace.config.environment import get_grace_config
    # Optional imports - gracefully handle missing components
    try:
        from grace.ingress_kernel.ingress_kernel import IngressKernel
    except ImportError:
        IngressKernel = None
    try:
        from grace.orchestration.orchestration_service import OrchestrationService
    except ImportError:
        OrchestrationService = None
except ImportError as e:
    logging.error(f"Failed to import Grace components: {e}")
    logging.warning("Some components may not be available")
    GraceGovernanceKernel = None
    IngressKernel = None 
    OrchestrationService = None

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global state
app_state = {
    "governance_kernel": None,
    "ingress_kernel": None,
    "orchestration_service": None,
    "websocket_manager": None,
    "config": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Grace Service...")
    
    try:
        # Load configuration
        app_state["config"] = get_grace_config()
        
        # Initialize WebSocket manager
        app_state["websocket_manager"] = WebSocketManager()
        
        # Initialize Grace kernels with error handling
        if GraceGovernanceKernel:
            try:
                logger.info("Initializing Grace Governance Kernel...")
                app_state["governance_kernel"] = GraceGovernanceKernel(config=app_state["config"])
                await app_state["governance_kernel"].initialize()
                await app_state["governance_kernel"].start()
            except Exception as e:
                logger.error(f"Failed to initialize governance kernel: {e}")
                app_state["governance_kernel"] = None
        
        if IngressKernel:
            try:
                logger.info("Initializing Grace Ingress Kernel...")
                app_state["ingress_kernel"] = IngressKernel(
                    storage_path=app_state["config"].get("storage_path", "/tmp/grace_ingress")
                )
                await app_state["ingress_kernel"].start()
            except Exception as e:
                logger.error(f"Failed to initialize ingress kernel: {e}")
                app_state["ingress_kernel"] = None
        
        if OrchestrationService:
            try:
                logger.info("Initializing Orchestration Service...")
                app_state["orchestration_service"] = OrchestrationService()
                await app_state["orchestration_service"].start()
            except Exception as e:
                logger.error(f"Failed to initialize orchestration service: {e}")
                app_state["orchestration_service"] = None
        
        logger.info("Grace Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start Grace Service", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Grace Service...")
        
        if app_state["governance_kernel"]:
            await app_state["governance_kernel"].shutdown()
        if app_state["ingress_kernel"]:
            await app_state["ingress_kernel"].stop()
        if app_state["orchestration_service"]:
            await app_state["orchestration_service"].stop()
        
        logger.info("Grace Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Grace Governance Service",
    description="Production API for Grace Governance Kernel",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(governance_router, prefix="/api/v1/governance", tags=["governance"])
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(ingest_router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(events_router, prefix="/api/v1/events", tags=["events"])


@app.get("/", response_model=BaseResponse)
async def root():
    """Root endpoint."""
    return BaseResponse(
        status="success",
        message="Grace Governance Service is running",
        data={
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health/status",
            "metrics": "/metrics"
        }
    )


@app.websocket("/ws/events")
async def websocket_events_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    websocket_manager = app_state["websocket_manager"]
    if not websocket_manager:
        await websocket.close(code=1000, reason="Service not initialized")
        return
    
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            logger.info("Received WebSocket message", data=data)
            
            # Echo back for now (could be enhanced for bidirectional communication)
            await websocket_manager.send_personal_message(
                {"type": "ack", "message": "Message received"}, websocket
            )
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        websocket_manager.disconnect(websocket)


def get_app_state():
    """Dependency to get application state."""
    return app_state


def get_governance_kernel():
    """Dependency to get governance kernel."""
    kernel = app_state.get("governance_kernel")
    if not kernel:
        raise HTTPException(status_code=503, detail="Governance kernel not initialized")
    return kernel


def get_ingress_kernel():
    """Dependency to get ingress kernel."""
    kernel = app_state.get("ingress_kernel")
    if not kernel:
        raise HTTPException(status_code=503, detail="Ingress kernel not initialized")
    return kernel


def get_orchestration_service():
    """Dependency to get orchestration service."""
    service = app_state.get("orchestration_service")
    if not service:
        raise HTTPException(status_code=503, detail="Orchestration service not initialized")
    return service


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    log_level = os.getenv("GRACE_LOG_LEVEL", "INFO").upper()
    
    uvicorn.run(
        "grace_service.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        reload=os.getenv("HOT_RELOAD", "false").lower() == "true",
        log_level=log_level.lower()
    )