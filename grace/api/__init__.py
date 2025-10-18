"""
Grace API - FastAPI application setup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from grace.config import get_settings
from grace.database import init_db
from grace.middleware.logging import LoggingMiddleware, setup_logging
from grace.middleware.rate_limit import RateLimitMiddleware
from grace.middleware.metrics import MetricsMiddleware, get_metrics_response

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Load configuration
    settings = get_settings()
    
    # Setup structured logging
    setup_logging(
        log_level=settings.observability.log_level,
        json_output=settings.observability.json_logs,
        log_file=settings.observability.log_file
    )
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        description="Constitutional AI System with Multi-Agent Coordination",
        version=settings.api_version,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        debug=settings.debug
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    if settings.observability.metrics_enabled:
        app.add_middleware(MetricsMiddleware)
    
    app.add_middleware(
        LoggingMiddleware,
        exclude_paths=["/health", "/metrics"],
        log_request_body=False,
        log_response_body=False
    )
    
    if settings.rate_limit.enabled:
        redis_client = None
        if settings.rate_limit.redis_url:
            try:
                from redis import Redis
                redis_client = Redis.from_url(settings.rate_limit.redis_url)
                logger.info("Redis rate limiting enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory: {e}")
        
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=settings.rate_limit.default_limit,
            window_seconds=settings.rate_limit.window_seconds,
            redis_client=redis_client,
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
    from grace.api.public import router as public_router
    
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(documents_router, prefix=settings.api_prefix)
    app.include_router(policies_router, prefix=settings.api_prefix)
    app.include_router(sessions_router, prefix=settings.api_prefix)
    app.include_router(tasks_router, prefix=settings.api_prefix)
    app.include_router(websocket_router, prefix=settings.api_prefix)
    app.include_router(logs_router, prefix=settings.api_prefix)
    app.include_router(public_router, prefix=settings.api_prefix)
    
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {settings.api_title} v{settings.api_version}")
        logger.info(f"Environment: {settings.environment}")
        
        if settings.environment == "production":
            issues = settings.validate_production_config()
            if any("CRITICAL" in issue or "ERROR" in issue for issue in issues):
                logger.error("❌ Critical configuration issues found:")
                for issue in issues:
                    logger.error(f"  - {issue}")
                raise RuntimeError("Invalid production configuration")
        
        init_db()
        logger.info("Database initialized")
        
        deployment_info = settings.get_deployment_info()
        logger.info(f"Deployment info: {deployment_info}")
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": settings.api_title,
            "version": settings.api_version,
            "environment": settings.environment,
            "features": settings.get_deployment_info()["features"]
        }
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint"""
        if not settings.observability.metrics_enabled:
            return {"error": "Metrics disabled"}
        return get_metrics_response()
    
    @app.get("/")
    async def root():
        return {
            "message": settings.api_title,
            "version": settings.api_version,
            "environment": settings.environment,
            "docs": f"{settings.api_prefix}/docs",
            "health": "/health",
            "metrics": "/metrics" if settings.observability.metrics_enabled else None,
            "status": "operational" if settings.environment == "production" else "development"
        }
    
    return app


app = create_app()

__all__ = ['app', 'create_app']
