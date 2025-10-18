"""
Prometheus metrics middleware
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import re

# Define metrics
HTTP_REQUESTS = Counter(
    "grace_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)
HTTP_REQUEST_DURATION = Histogram(
    "grace_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)
HTTP_REQUEST_ACTIVE = Gauge(
    "grace_http_requests_active",
    "Number of active HTTP requests",
    ["method", "path"]
)

WEBSOCKET_CONNECTIONS = Gauge(
    "grace_websocket_connections_active",
    "Active websocket connections"
)


def _normalize_path(path: str) -> str:
    # Replace UUIDs and numbers with placeholder to reduce cardinality
    path = re.sub(r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "/{id}", path)
    path = re.sub(r"/\d+", "/{id}", path)
    return path


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection middleware"""

    def __init__(self, app):
        super().__init__(app)

        self.request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )

        self.request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request"""

        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # Record metrics
        self.request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        self.request_duration.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response


def get_metrics_response() -> Response:
    """Generate Prometheus metrics response"""
    metrics = generate_latest()
    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST,
    )
