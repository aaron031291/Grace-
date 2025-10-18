"""
Structured logging middleware using structlog
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

# Try to import verify_token for JWT decoding (optional)
try:
    from grace.auth.security import verify_token
except Exception:
    verify_token = None  # type: ignore

from grace.config import get_settings

# Configure a minimal structlog config if not configured elsewhere
def setup_structlog(json_output: bool = True, log_level: str = "INFO"):
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(processors=processors)

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request and response metadata with structured logging.

    Logged fields:
      - request_id
      - method, path, query_string
      - client_ip, user_agent
      - user_id (from request.state.user or decoded JWT)
      - status_code, duration_ms
    """

    def __init__(self, app: ASGIApp, exclude_paths: Optional[list[str]] = None, json_logs: bool = True):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        setup_structlog(json_output=json_logs)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log metadata"""

        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        start = time.time()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        method = request.method
        path = request.url.path
        query = dict(request.query_params)

        # Try to obtain user id: prefer request.state.user (set by dependency), else decode Bearer token
        user_id = None
        username = None
        try:
            if hasattr(request.state, "user") and request.state.user is not None:
                user = request.state.user
                user_id = getattr(user, "id", None)
                username = getattr(user, "username", None)
            else:
                # decode Authorization header if token available and verify_token imported
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer ") and verify_token:
                    token = auth.split(" ", 1)[1]
                    payload = verify_token(token, token_type="access")
                    if payload:
                        user_id = payload.get("user_id")
                        username = payload.get("username")
        except Exception:
            # Never raise from logging middleware
            logger.exception("failed_to_extract_user", path=path)

        # Bind context for this request
        bound = logger.bind(
            request_id=request_id,
            method=method,
            path=path,
            query=query,
            client_ip=client_ip,
            user_agent=user_agent,
            user_id=user_id,
            username=username,
        )
        bound.info("request_start")

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = int((time.time() - start) * 1000)
            bound.error("request_exception", error=str(exc), duration_ms=duration_ms)
            raise
        duration_ms = int((time.time() - start) * 1000)

        # Response metadata
        status_code = getattr(response, "status_code", 500)
        content_length = response.headers.get("content-length")

        bound.info(
            "request_end",
            status_code=status_code,
            duration_ms=duration_ms,
            response_size=content_length,
        )

        # Warn on slow requests
        if duration_ms > 1000:
            bound.warning("slow_request", duration_ms=duration_ms)

        return response


def get_request_logger(request: Request) -> structlog.BoundLogger:
    """
    Get a logger bound with request context
    
    Usage in endpoints:
        logger = get_request_logger(request)
        logger.info("processing_data", item_count=10)
    """
    return structlog.get_logger().bind(
        request_id=request.headers.get("X-Request-ID", "unknown"),
        path=request.url.path,
        method=request.method
    )
