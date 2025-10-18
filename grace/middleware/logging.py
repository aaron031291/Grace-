"""
Structured logging middleware using structlog
"""

import time
import uuid
from typing import Callable, Optional
from datetime import datetime, timezone

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

# Configure structlog
def setup_logging(log_level: str = "INFO", json_logs: bool = True):
    """
    Configure structured logging with structlog
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output JSON formatted logs
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        TimeStamper(fmt="iso", utc=True),
    ]
    
    if json_logs:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, log_level.upper(), structlog.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Get logger
logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request and response metadata with structured logging
    
    Logs:
    - Request ID (generated or from header)
    - HTTP method and path
    - Query parameters
    - User ID (if authenticated)
    - Client IP address
    - User agent
    - Response status code
    - Response time (duration)
    - Request/response size
    - Errors and exceptions
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list[str]] = None,
        log_request_body: bool = False,
        log_response_body: bool = False
    ):
        """
        Initialize logging middleware
        
        Args:
            app: ASGI application
            exclude_paths: List of paths to exclude from logging (e.g., /health)
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log metadata"""
        
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Extract client information
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Extract user ID from request state (set by auth dependencies)
        user_id = None
        username = None
        if hasattr(request.state, "user"):
            user = request.state.user
            user_id = getattr(user, "id", None)
            username = getattr(user, "username", None)
        
        # Start timing
        start_time = time.time()
        
        # Build context
        log_context = structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=client_host,
            user_agent=user_agent,
            user_id=user_id,
            username=username,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Log request
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=client_host,
            user_id=user_id,
            username=username
        )
        
        # Log request body if enabled
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    logger.debug(
                        "request_body",
                        body_size=len(body),
                        content_type=request.headers.get("Content-Type")
                    )
            except Exception as e:
                logger.warning("failed_to_read_request_body", error=str(e))
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
            
        except Exception as e:
            error = e
            logger.error(
                "request_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True
            )
            raise
        
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            if response:
                logger.info(
                    "request_completed",
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                    response_size=response.headers.get("Content-Length"),
                )
                
                # Log slow requests
                if duration_ms > 1000:  # > 1 second
                    logger.warning(
                        "slow_request",
                        duration_ms=round(duration_ms, 2),
                        threshold_ms=1000
                    )
                
                # Log errors
                if response.status_code >= 400:
                    log_level = "error" if response.status_code >= 500 else "warning"
                    getattr(logger, log_level)(
                        "request_error",
                        status_code=response.status_code,
                        duration_ms=round(duration_ms, 2)
                    )
            
            # Clear context
            structlog.contextvars.clear_contextvars()
        
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
