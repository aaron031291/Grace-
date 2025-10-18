"""
Grace API - FastAPI application setup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from grace.database import init_db

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Grace AI System API",
        description="Advanced Multi-Agent AI System with Authentication, Documents, and Vector Search",
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
    
    # Include routers
    from grace.api.v1.auth import router as auth_router
    from grace.api.v1.documents import router as documents_router
    
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    
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
            "features": ["auth", "documents", "vector_search"]
        }
    
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
                "search": "/api/v1/documents/search"
            }
        }
    
    return app


app = create_app()

__all__ = ['app', 'create_app']
