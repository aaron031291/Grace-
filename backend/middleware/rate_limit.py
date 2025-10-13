"""Rate limiting middleware."""

from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    async def dispatch(self, request, call_next):
        # TODO: Implement rate limiting
        response = await call_next(request)
        return response
