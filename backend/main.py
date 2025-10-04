#!/usr/bin/env python3
"""
Grace Backend - Main FastAPI Application (Minimal Implementation)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.datastructures import Headers
import hashlib

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from collections import defaultdict, deque

from .config import get_settings



class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        msg = super().format(record)
        # Redact secrets
        try:
            settings = get_settings()
            secrets = [settings.jwt_secret_key, settings.secret_key, settings.storage_secret_key, settings.storage_access_key]
            for secret in secrets:
                if secret:
                    msg = msg.replace(secret, "[REDACTED]")
        except Exception:
            pass
        return msg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
for handler in logging.getLogger().handlers:
    handler.setFormatter(RedactingFormatter(handler.formatter._fmt))

settings = get_settings()

# --- Idempotency Key Middleware ---
class IdempotencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.cache = {}

    async def dispatch(self, request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            key = request.headers.get("Idempotency-Key")
            if not key:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": "IDEMPOTENCY_KEY_REQUIRED",
                            "message": "Idempotency-Key header is required for mutating requests."
                        }
                    }
                )
            hash_key = hashlib.sha256(key.encode()).hexdigest()
            if hash_key in self.cache:
                return self.cache[hash_key]
            response = await call_next(request)
            # Only cache successful responses
            if response.status_code < 400:
                self.cache[hash_key] = response
            return response
        return await call_next(request)

# --- Rate Limiting Middleware ---
class TokenBucket:
    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.time()

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.timestamp
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.timestamp = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: int = 5, capacity: int = 10):
        super().__init__(app)
        self.rate = rate
        self.capacity = capacity
        self.buckets = defaultdict(lambda: TokenBucket(rate, capacity))

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        bucket = self.buckets[ip]
        if not bucket.consume():
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later."
                    }
                }
            )
        response = await call_next(request)
        return response

# --- Pydantic Schemas ---
class HealthCheckResponse(BaseModel):
    status: str
    version: str
    service: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("🚀 Starting Grace Backend...")
    logger.info("✅ Grace Backend startup complete")
    
    yield
    
    logger.info("🔄 Shutting down Grace Backend...")
    logger.info("✅ Grace Backend shutdown complete")


def create_app() -> FastAPI:
    from starlette.middleware.base import BaseHTTPMiddleware
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none';"
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
            response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
            return response

    app.add_middleware(SecurityHeadersMiddleware)
    from .auth import create_access_token, create_refresh_token, verify_token
    from pydantic import BaseModel

    class TokenRequest(BaseModel):
        username: str
        password: str

    class TokenResponse(BaseModel):
        access_token: str
        refresh_token: str
        token_type: str = "bearer"

    class RefreshRequest(BaseModel):
        refresh_token: str

    @app.post("/api/auth/token", response_model=TokenResponse)
    async def login(request: TokenRequest):
        # TODO: Replace with real user validation
        if request.username == "admin" and request.password == "admin":
            access_token = create_access_token({"sub": request.username, "role": "admin"})
            refresh_token = create_refresh_token({"sub": request.username, "role": "admin"})
            return TokenResponse(access_token=access_token, refresh_token=refresh_token)
        return JSONResponse(status_code=401, content={"error": {"code": "INVALID_CREDENTIALS", "message": "Invalid username or password."}})

    @app.post("/api/auth/refresh", response_model=TokenResponse)
    async def refresh_token(request: RefreshRequest):
        try:
            payload = verify_token(request.refresh_token)
            if payload.get("type") != "refresh":
                raise Exception("Not a refresh token")
            access_token = create_access_token({"sub": payload["sub"], "role": payload["role"]})
            refresh_token = create_refresh_token({"sub": payload["sub"], "role": payload["role"]})
            return TokenResponse(access_token=access_token, refresh_token=refresh_token)
        except Exception:
            return JSONResponse(status_code=401, content={"error": {"code": "INVALID_REFRESH_TOKEN", "message": "Invalid or expired refresh token."}})
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Grace Backend",
        description="Comprehensive AI governance system backend",
        version="1.0.0",
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

        # Add global rate-limit middleware (per-IP, token bucket)
        app.add_middleware(RateLimitMiddleware, rate=5, capacity=10)

        # Add idempotency key middleware for mutating endpoints
        app.add_middleware(IdempotencyMiddleware)
    
    # Basic health endpoint
        @app.get("/api/health", response_model=HealthCheckResponse)
        async def health_check() -> HealthCheckResponse:
            return HealthCheckResponse(
                status="healthy",
                version="1.0.0",
                service="grace-backend"
            )
    
    # Global exception handler with standardized error envelope
    import uuid
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        trace_id = str(uuid.uuid4())
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "trace_id": trace_id,
                    "detail": str(exc) if settings.debug else None
                }
            }
        )
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )