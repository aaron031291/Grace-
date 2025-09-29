"""
Grace HTTP Middleware - Request ID propagation and logging integration.
"""
import logging
from contextvars import ContextVar
from typing import Callable, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .utils import generate_request_id

# Context variable for request ID propagation
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to handle X-Request-ID propagation and logging context."""
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = generate_request_id()
        
        # Set in context for logging
        request_id_ctx.set(request_id)
        
        # Add to request state for endpoint access
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            f"Request started - {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query) if request.url.query else None,
                "user_agent": request.headers.get("user-agent"),
                "remote_addr": request.client.host if request.client else None,
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers[self.header_name] = request_id
            
            # Log successful response
            logger.info(
                f"Request completed - {request.method} {request.url.path} -> {response.status_code}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_size": response.headers.get("content-length"),
                }
            )
            
        except Exception as e:
            # Log error
            logger.error(
                f"Request failed - {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True
            )
            raise
        
        return response


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_ctx.get('')


def set_request_id(request_id: str) -> None:
    """Set request ID in context (for manual setting)."""
    request_id_ctx.set(request_id)


class LoggingContextFilter(logging.Filter):
    """Logging filter to add request ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add request ID to log record if available
        record.request_id = get_request_id()
        return True


class GraceHTTPExceptionHandler:
    """
    Centralized exception handler for consistent error responses.
    """
    
    @staticmethod
    def create_error_envelope(code: str, message: str, detail: str = None, 
                            status_code: int = 500) -> Dict[str, Any]:
        """Create standardized error response."""
        error = {
            "error": {
                "code": code,
                "message": message,
                "request_id": get_request_id(),
                "timestamp": logger.handlers[0].formatter.formatTime(logging.LogRecord(
                    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
                )) if logger.handlers else None
            }
        }
        
        if detail:
            error["error"]["detail"] = detail
            
        return error
    
    @staticmethod
    async def handle_validation_error(request: Request, exc: Exception) -> Response:
        """Handle validation errors consistently."""
        from starlette.responses import JSONResponse
        
        error_response = GraceHTTPExceptionHandler.create_error_envelope(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            detail=str(exc),
            status_code=422
        )
        
        logger.warning(
            f"Validation error in {request.method} {request.url.path}: {exc}",
            extra={"request_id": get_request_id()}
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    @staticmethod
    async def handle_http_exception(request: Request, exc: Exception) -> Response:
        """Handle HTTP exceptions consistently."""
        from starlette.responses import JSONResponse
        from fastapi import HTTPException
        
        if isinstance(exc, HTTPException):
            status_code = exc.status_code
            detail = exc.detail
        else:
            status_code = 500
            detail = str(exc)
        
        # Map status codes to error codes
        error_code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED", 
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            409: "CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            422: "VALIDATION_ERROR",
            429: "TOO_MANY_REQUESTS",
            500: "INTERNAL_SERVER_ERROR",
            502: "BAD_GATEWAY",
            503: "SERVICE_UNAVAILABLE"
        }
        
        error_code = error_code_map.get(status_code, "HTTP_ERROR")
        
        error_response = GraceHTTPExceptionHandler.create_error_envelope(
            code=error_code,
            message=f"HTTP {status_code} error",
            detail=detail,
            status_code=status_code
        )
        
        # Log based on severity
        if status_code >= 500:
            logger.error(
                f"Server error in {request.method} {request.url.path}: {detail}",
                extra={"request_id": get_request_id()},
                exc_info=True
            )
        elif status_code >= 400:
            logger.warning(
                f"Client error in {request.method} {request.url.path}: {detail}",
                extra={"request_id": get_request_id()}
            )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )


def setup_logging_with_request_id():
    """Configure logging to include request ID in all log messages."""
    
    # Add context filter to root logger
    logging_filter = LoggingContextFilter()
    
    # Update existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(logging_filter)
        
        # Update formatter to include request ID if not already present
        if handler.formatter and 'request_id' not in handler.formatter._fmt:
            current_format = handler.formatter._fmt
            new_format = current_format.replace(
                '%(message)s', 
                '[%(request_id)s] %(message)s'
            )
            handler.setFormatter(logging.Formatter(new_format))
    
    logger.info("Logging configured with request ID context")