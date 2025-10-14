"""
Grace Observability Components

Provides request tracking, metrics, and logging middleware.
"""

import time
import uuid
import logging
from typing import Callable
import json

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter(
    "grace_requests_total", "Total requests", ["method", "endpoint", "status_code"]
)
REQUEST_DURATION = Histogram(
    "grace_request_duration_seconds", "Request duration", ["method", "endpoint"]
)
ACTIVE_REQUESTS = Gauge("grace_active_requests", "Active requests")

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up structured JSON logging for Grace services."""

    class JSONFormatter(logging.Formatter):
        """JSON formatter for structured logging."""

        def format(self, record):
            log_data = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)

            # Add extra fields if present
            if hasattr(record, "request_id"):
                log_data["request_id"] = record.request_id
            if hasattr(record, "session_id"):
                log_data["session_id"] = record.session_id
            if hasattr(record, "user_id"):
                log_data["user_id"] = record.user_id
            if hasattr(record, "route"):
                log_data["route"] = record.route

            return json.dumps(log_data)

    # Set up JSON logging
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to add request IDs and structured logging context."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to request state
    request.state.request_id = request_id

    # Add to logging context
    log_extra = {
        "request_id": request_id,
        "route": str(request.url.path),
        "method": request.method,
        "user_agent": request.headers.get("user-agent", ""),
        "remote_addr": request.client.host if request.client else "",
    }

    # Extract session/user context if available
    # This would be populated by auth middleware
    if hasattr(request.state, "session_id"):
        log_extra["session_id"] = request.state.session_id
    if hasattr(request.state, "user_id"):
        log_extra["user_id"] = request.state.user_id

    logger.info("Request started", extra=log_extra)

    # Process request
    try:
        ACTIVE_REQUESTS.inc()
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        log_extra.update(
            {"status_code": response.status_code, "duration_seconds": duration}
        )

        logger.info("Request completed", extra=log_extra)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        duration = time.time() - start_time
        log_extra.update(
            {"status_code": 500, "duration_seconds": duration, "error": str(e)}
        )

        logger.error("Request failed", extra=log_extra, exc_info=True)
        raise

    finally:
        ACTIVE_REQUESTS.dec()


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to collect Prometheus metrics."""
    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        # Record metrics
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
            time.time() - start_time
        )

        return response

    except Exception:
        # Record error metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code="500").inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
            time.time() - start_time
        )
        raise


class ObservabilityContext:
    """Context manager for adding observability data to logging."""

    def __init__(self, **context):
        self.context = context
        self.original_factory = None

    def __enter__(self):
        # Store original log record factory
        self.original_factory = logging.getLogRecordFactory()

        # Create new factory that adds our context
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original factory
        logging.setLogRecordFactory(self.original_factory)
