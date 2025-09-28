"""
Grace API Service - Main HTTP API server that handles REST and WebSocket endpoints.

This service reads from DATABASE_URL, REDIS_URL, S3_ENDPOINT, VECTOR_URL.
"""
import os
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import redis.asyncio as redis

from ..core.config import get_settings
from ..core.observability import setup_logging, request_id_middleware, metrics_middleware

# Metrics
REQUEST_COUNT = Counter('grace_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('grace_api_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('grace_api_active_connections', 'Active connections')

logger = logging.getLogger(__name__)

# Global instances
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    global redis_client
    
    settings = get_settings()
    
    try:
        # Initialize Redis
        if settings.redis_url:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis connection established")
        
        logger.info("Grace API service startup complete")
        yield
        
    finally:
        # Cleanup
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
        
        logger.info("Grace API service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Grace AI Governance API",
        description="API for Grace AI Governance System with constitutional decision-making",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add observability middleware
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(metrics_middleware)
    
    # Mount metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        checks = {
            "status": "healthy",
            "redis": False,
            "database": False
        }
        
        # Check Redis
        if redis_client:
            try:
                await redis_client.ping()
                checks["redis"] = True
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
        
        # Overall health
        all_healthy = all([
            checks["redis"],
            checks["database"]
        ])
        
        if not all_healthy:
            checks["status"] = "degraded"
            
        return checks
    
    # Basic endpoints
    @app.get("/api/v1/sessions")
    async def list_sessions():
        """List user sessions."""
        return {"sessions": [], "message": "Sessions endpoint - to be implemented"}
    
    @app.get("/api/v1/search")
    async def search_knowledge(q: str = "", filters: str = ""):
        """Search knowledge base with vector and keyword filters."""
        return {
            "query": q,
            "filters": filters,
            "results": [],
            "message": "Search endpoint - to be implemented"
        }
    
    @app.post("/api/v1/memory/ingest")
    async def ingest_memory(file_data: dict):
        """Ingest file into memory system."""
        return {"status": "queued", "message": "Memory ingestion - to be implemented"}
    
    return app


async def main():
    """Main entry point for the API service."""
    setup_logging()
    settings = get_settings()
    
    logger.info("Starting Grace API Service...")
    logger.info(f"Database URL: {settings.database_url}")
    logger.info(f"Redis URL: {settings.redis_url}")
    logger.info(f"S3 Endpoint: {settings.s3_endpoint}")
    logger.info(f"Vector URL: {settings.vector_url}")
    
    app = create_app()
    
    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=True,
        loop="uvloop" if os.name != "nt" else "asyncio"
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())