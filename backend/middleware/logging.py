"""Logging middleware."""

import logging
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""

    async def dispatch(self, request, call_next):
        # TODO: Implement structured logging
        response = await call_next(request)
        return response
