"""
Main FastAPI application for Grace system.
Integrates all API routes with the new repository-based architecture.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from grace.core.database import init_database, close_database
from grace.api.auth import router as auth_router
from grace.api.memories import router as memory_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await init_database()
    yield
    # Shutdown  
    await close_database()

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Grace AI Governance System",
        description="Advanced AI system with repository-based data layer",
        version="0.3.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(auth_router)
    app.include_router(memory_router)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Grace AI Governance System",
            "version": "0.3.0",
            "status": "running",
            "features": [
                "SQLAlchemy-based data layer",
                "JWT authentication with RBAC", 
                "Repository pattern implementation",
                "Async database operations",
                "Memory management APIs"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": "2024-09-28T19:20:00Z"}
    
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "grace.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )