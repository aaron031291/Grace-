"""
Grace API module
"""

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)


def create_app(config: Optional[dict] = None) -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        config: Optional configuration dict
        
    Returns:
        FastAPI application instance
    """
    from grace.config import get_settings
    
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add middleware (lazy import to avoid circular deps)
    from grace.middleware.logging import LoggingMiddleware, setup_logging
    from grace.middleware.rate_limit import RateLimitMiddleware
    from grace.middleware.metrics import MetricsMiddleware
    
    # Setup logging
    setup_logging(
        log_level=settings.observability.log_level,
        json_output=settings.observability.json_logs,
        log_file=settings.observability.log_file
    )
    
    # Add middleware
    app.add_middleware(LoggingMiddleware)
    
    if settings.rate_limit.enabled:
        app.add_middleware(RateLimitMiddleware)
    
    if settings.observability.metrics_enabled:
        app.add_middleware(MetricsMiddleware)
    
    # Include routers (lazy import)
    from grace.api.v1.auth import router as auth_router
    from grace.api.v1.documents import router as documents_router
    
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(documents_router, prefix=settings.api_prefix)
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.api_version,
            "environment": settings.environment
        }
    
    logger.info(f"Grace API initialized: {settings.api_title} v{settings.api_version}")
    
    return app


__all__ = ['create_app']
