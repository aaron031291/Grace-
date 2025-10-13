"""
Grace API Service - Main HTTP API server that handles REST and WebSocket endpoints.

This service reads from DATABASE_URL, REDIS_URL, S3_ENDPOINT, VECTOR_URL.
"""

import os
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import redis.asyncio as redis

from ..core.config import get_settings
from ..core.observability import (
    setup_logging,
    request_id_middleware,
    metrics_middleware,
)
from ..auth.jwt_auth import get_current_user

# Metrics
REQUEST_COUNT = Counter(
    "grace_api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram("grace_api_request_duration_seconds", "Request duration")
ACTIVE_CONNECTIONS = Gauge("grace_api_active_connections", "Active connections")

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
        lifespan=lifespan,
    )

    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Configure appropriately for production
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

    # Add authentication middleware (optional - can be disabled for development)
    if settings.jwt_secret_key and not settings.debug:
        from ..auth.jwt_auth import JWTManager, AuthMiddleware

        jwt_manager = JWTManager(settings.jwt_secret_key)
        app.add_middleware(AuthMiddleware, jwt_manager=jwt_manager)

    # Mount metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        checks = {"status": "healthy", "redis": False, "database": False}

        # Check Redis
        if redis_client:
            try:
                await redis_client.ping()
                checks["redis"] = True
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")

        # Overall health
        all_healthy = all([checks["redis"], checks["database"]])

        if not all_healthy:
            checks["status"] = "degraded"

        return checks

    # Authentication endpoints
    @app.post("/api/v1/auth/token")
    async def create_token(request_data: dict):
        """Create JWT token for user authentication."""
        from ..auth.jwt_auth import get_jwt_manager, GraceScopes

        user_id = request_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # In production, validate credentials here
        # For now, accept any user_id for demo purposes

        # Grant scopes based on user type (simplified)
        scopes = []
        roles = request_data.get("roles", ["user"])

        if "admin" in roles:
            scopes = [
                GraceScopes.READ_CHAT,
                GraceScopes.WRITE_MEMORY,
                GraceScopes.GOVERN_TASKS,
                GraceScopes.SANDBOX_BUILD,
                GraceScopes.NETWORK_ACCESS,
                GraceScopes.ADMIN,
            ]
        elif "developer" in roles:
            scopes = [
                GraceScopes.READ_CHAT,
                GraceScopes.WRITE_MEMORY,
                GraceScopes.SANDBOX_BUILD,
            ]
        else:
            scopes = [GraceScopes.READ_CHAT]

        jwt_manager = get_jwt_manager()
        token = jwt_manager.create_token(user_id, scopes, roles)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_id,
            "scopes": scopes,
            "roles": roles,
        }

    @app.get("/api/v1/auth/me")
    async def get_current_user_info(current_user: dict = Depends(get_current_user)):
        """Get information about the current authenticated user."""
        return {
            "user_id": current_user["user_id"],
            "scopes": current_user["scopes"],
            "roles": current_user["roles"],
            "token_info": {
                "issued_at": current_user["token_data"].get("iat"),
                "expires_at": current_user["token_data"].get("exp"),
                "issuer": current_user["token_data"].get("iss"),
            },
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint with authentication."""
        from ..auth.websocket_auth import authenticate_websocket, GraceScopes

        try:
            # Authenticate WebSocket connection
            auth_ws = await authenticate_websocket(
                websocket, required_scopes=[GraceScopes.READ_CHAT]
            )

            logger.info(f"WebSocket connected: {auth_ws.user_id}")

            try:
                while True:
                    # Receive message from client
                    data = await auth_ws.receive_json()

                    # Echo back with user context
                    response = {
                        "type": "echo",
                        "user_id": auth_ws.user_id,
                        "message": data.get("message", ""),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    await auth_ws.send_json(response)

            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        except Exception as e:
            logger.error(f"WebSocket authentication failed: {e}")
            await websocket.close(code=1008, reason="Authentication failed")

    # Basic endpoints
    @app.get("/api/v1/sessions")
    async def list_sessions():
        """List user sessions."""
        return {"sessions": [], "message": "Sessions endpoint - to be implemented"}

    @app.get("/api/v1/search")
    async def search_knowledge(
        q: str = "",
        filters: str = "",
        trust_threshold: float = 0.5,
        limit: int = 10,
        current_user: dict = Depends(get_current_user),
    ):
        """Search knowledge base with vector and keyword filters."""
        from ..auth.jwt_auth import GraceScopes
        from ..audit.golden_path_auditor import append_audit

        # Check if user has read permissions
        user_scopes = current_user.get("scopes", [])
        if GraceScopes.READ_CHAT not in user_scopes:
            raise HTTPException(
                status_code=403, detail="Insufficient permissions for search"
            )

        if not q.strip():
            return {"results": [], "message": "Empty query"}

        # Audit the search operation
        audit_data = {
            "query": q,
            "filters": filters,
            "trust_threshold": trust_threshold,
            "limit": limit,
            "user_scopes": user_scopes,
        }

        try:
            # Log memory read operation
            audit_id = await append_audit(
                operation_type="memory_read",
                operation_data=audit_data,
                user_id=current_user["user_id"],
                transparency_level="democratic_oversight",
            )

            from ..memory_ingestion.pipeline import get_memory_ingestion_pipeline

            pipeline = get_memory_ingestion_pipeline(settings.vector_url)

            results = await pipeline.search_memory(
                query=q,
                user_id=current_user["user_id"],
                trust_threshold=trust_threshold,
                limit=limit,
            )

            response_data = {
                "query": q,
                "filters": filters,
                "trust_threshold": trust_threshold,
                "results": results,
                "count": len(results),
                "user_id": current_user["user_id"],
                "audit_id": audit_id,
            }

            # Log the API response
            await append_audit(
                operation_type="api_response",
                operation_data={
                    "endpoint": "/api/v1/search",
                    "response_type": "search_results",
                    "result_count": len(results),
                    "audit_id": audit_id,
                },
                user_id=current_user["user_id"],
            )

            return response_data

        except Exception as e:
            logger.error(f"Search failed: {e}")

            # Log the error
            await append_audit(
                operation_type="api_error",
                operation_data={
                    "endpoint": "/api/v1/search",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": q,
                },
                user_id=current_user["user_id"],
                transparency_level="governance_internal",
            )

            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    @app.post("/api/v1/memory/ingest")
    async def ingest_memory(
        request_data: dict, current_user: dict = Depends(get_current_user)
    ):
        """Ingest file into memory system."""
        from ..auth.jwt_auth import GraceScopes
        from ..audit.golden_path_auditor import append_audit

        # Check if user has write permissions
        user_scopes = current_user.get("scopes", [])
        if GraceScopes.WRITE_MEMORY not in user_scopes:
            raise HTTPException(
                status_code=403, detail="Insufficient permissions for memory ingestion"
            )

        from ..memory_ingestion.pipeline import get_memory_ingestion_pipeline

        pipeline = get_memory_ingestion_pipeline(settings.vector_url)

        try:
            # Add user context to request
            request_data["user_id"] = current_user["user_id"]

            # Log memory write operation
            audit_id = await append_audit(
                operation_type="memory_write",
                operation_data={
                    "ingestion_type": "file" if "file_path" in request_data else "text",
                    "file_path": request_data.get("file_path"),
                    "has_text": "text" in request_data,
                    "tags": request_data.get("tags"),
                    "trust_score": request_data.get("trust_score", 0.7),
                },
                user_id=current_user["user_id"],
                transparency_level="governance_internal",  # Write operations are more restricted
            )

            # Handle different ingestion types
            if "file_path" in request_data:
                # File ingestion
                result = await pipeline.ingest_file(
                    file_path=request_data["file_path"],
                    session_id=request_data.get("session_id"),
                    user_id=current_user["user_id"],
                    tags=request_data.get("tags"),
                    trust_score=request_data.get("trust_score", 0.7),
                )
            elif "text" in request_data:
                # Direct text ingestion
                result = await pipeline.ingest_text_content(
                    text=request_data["text"],
                    title=request_data.get("title", "Text Content"),
                    session_id=request_data.get("session_id"),
                    user_id=current_user["user_id"],
                    tags=request_data.get("tags"),
                    trust_score=request_data.get("trust_score", 0.7),
                )
            else:
                raise HTTPException(
                    status_code=400, detail="Either 'file_path' or 'text' is required"
                )

            # Add audit info to result
            result["audit_id"] = audit_id

            # Log successful ingestion
            await append_audit(
                operation_type="api_response",
                operation_data={
                    "endpoint": "/api/v1/memory/ingest",
                    "response_type": "ingestion_success",
                    "ingestion_result": {
                        k: v for k, v in result.items() if k != "audit_id"
                    },
                    "audit_id": audit_id,
                },
                user_id=current_user["user_id"],
            )

            return result

        except Exception as e:
            logger.error(f"Memory ingestion failed: {e}")

            # Log the error
            await append_audit(
                operation_type="api_error",
                operation_data={
                    "endpoint": "/api/v1/memory/ingest",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request_data": {
                        k: v
                        for k, v in request_data.items()
                        if k not in ["text", "content"]
                    },  # Don't log sensitive content
                },
                user_id=current_user["user_id"],
                transparency_level="governance_internal",
            )

            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

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
        loop="uvloop" if os.name != "nt" else "asyncio",
    )

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
