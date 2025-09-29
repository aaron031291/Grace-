"""Metrics middleware."""

from starlette.middleware.base import BaseHTTPMiddleware


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection middleware."""
    
    async def dispatch(self, request, call_next):
        # TODO: Implement metrics collection
        response = await call_next(request)
        return response