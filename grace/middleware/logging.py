"""
Structured logging middleware using structlog
"""

from typing import Callable, Set, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import uuid

logger = logging.getLogger(__name__)

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


def setup_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None
) -> None:
    """Setup global logging configuration"""
    import structlog
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.root.addHandler(file_handler)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    def __init__(
        self,
        app,
        exclude_paths: Optional[Set[str]] = None,
        log_request_body: bool = False,
        log_response_body: bool = False
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or set()
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log metadata"""
        
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        start_time = time.time()
        
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        logger.info(
            f"Request completed: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": duration
            }
        )
        
        response.headers["X-Request-ID"] = request_id
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
