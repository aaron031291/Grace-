"""
Prometheus metrics middleware
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Gauge
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
    """
    Collect request counts and durations for Prometheus.
    """
    def __init__(self, app: ASGIApp, exclude_paths=None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/metrics", "/health"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path in self.exclude_paths:
            return await call_next(request)

        method = request.method
        norm_path = _normalize_path(path)
        HTTP_REQUEST_ACTIVE.labels(method=method, path=norm_path).inc()
        start = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start
            HTTP_REQUEST_DURATION.labels(method=method, path=norm_path).observe(duration)
            HTTP_REQUESTS.labels(method=method, path=norm_path, status=str(status_code)).inc()
            HTTP_REQUEST_ACTIVE.labels(method=method, path=norm_path).dec()
        return response


def get_metrics_response() -> Response:
    output = generate_latest()
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)
