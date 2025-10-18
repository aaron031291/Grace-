"""
Grace API - FastAPI application setup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from grace.database import init_db
from grace.middleware.logging import LoggingMiddleware, setup_logging
from grace.middleware.rate_limit import RateLimitMiddleware
from grace.middleware.metrics import MetricsMiddleware, get_metrics_response

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Setup structured logging
    setup_logging(log_level="INFO", json_logs=True)
    
    app = FastAPI(
        title="Grace AI System API",
        description="Advanced Multi-Agent AI System with Authentication, Documents, Vector Search, Governance, and Real-time WebSocket",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware (order matters - last added is executed first)
    
    # 1. Metrics middleware (outermost - measures everything)
    app.add_middleware(MetricsMiddleware)
    
    # 2. Logging middleware
    app.add_middleware(
        LoggingMiddleware,
        exclude_paths=["/health", "/metrics"],
        log_request_body=False,
        log_response_body=False
    )
    
    # 3. Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=100,
        window_seconds=60,
        redis_client=None,  # Use Redis in production
        exclude_paths=["/health", "/metrics"]
    )
    
    # Include routers
    from grace.api.v1.auth import router as auth_router
    from grace.api.v1.documents import router as documents_router
    from grace.api.v1.policies import router as policies_router
    from grace.api.v1.sessions import router as sessions_router
    from grace.api.v1.tasks import router as tasks_router
    from grace.api.v1.websocket import router as websocket_router
    from grace.api.v1.logs import router as logs_router
    from grace.api.v1.avn import router as avn_router
    
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(policies_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/api/v1")
    app.include_router(logs_router, prefix="/api/v1")
    app.include_router(avn_router, prefix="/api/v1")
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Grace AI System API")
        init_db()
        logger.info("Database initialized")
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "Grace AI System",
            "version": "1.0.0",
            "features": ["auth", "documents", "vector_search", "policies", "sessions", "tasks", "websocket"]
        }
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint"""
        return get_metrics_response()
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Grace AI System API",
            "docs": "/api/docs",
            "version": "1.0.0",
            "endpoints": {
                "auth": "/api/v1/auth",
                "documents": "/api/v1/documents",
                "search": "/api/v1/documents/search",
                "policies": "/api/v1/policies",
                "sessions": "/api/v1/sessions",
                "tasks": "/api/v1/tasks",
                "logs": "/api/v1/logs",
                "avn": "/api/v1/avn",
                "websocket": "ws://localhost:8000/api/v1/ws/connect?token=<jwt>",
                "metrics": "/metrics",
                "health": "/health"
            }
        }
    
    return app


app = create_app()

__all__ = ['app', 'create_app']
