"""
Prometheus metrics middleware
"""

from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Lazy import prometheus_client to avoid initialization at import time
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Only create metrics if prometheus is available
if HAS_PROMETHEUS:
    HTTP_REQUESTS_TOTAL = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    HTTP_REQUEST_DURATION = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration',
        ['method', 'endpoint']
    )
    
    HTTP_REQUEST_ACTIVE = Gauge(
        'http_requests_active',
        'Active HTTP requests'
    )
else:
    HTTP_REQUESTS_TOTAL = None
    HTTP_REQUEST_DURATION = None
    HTTP_REQUEST_ACTIVE = None


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.has_prometheus = HAS_PROMETHEUS
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request"""
        
        if not self.has_prometheus:
            return await call_next(request)
        
        # Track active requests
        HTTP_REQUEST_ACTIVE.inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            duration = time.time() - start_time
            
            # Record metrics
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
        
        finally:
            HTTP_REQUEST_ACTIVE.dec()


def get_metrics_response() -> Response:
    """Generate Prometheus metrics response"""
    if not HAS_PROMETHEUS:
        return Response(
            content="Prometheus metrics not available (prometheus_client not installed)",
            media_type="text/plain"
        )
    
    metrics = generate_latest()
    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST
    )
