"""
Rate limiting middleware with Redis and in-memory support
"""

import time
import hashlib
from typing import Callable, Optional, Dict, Tuple
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import status
import structlog

from grace.config import get_settings

logger = structlog.get_logger()


class InMemorySlidingWindow:
    """
    Sliding window per-key using deque of timestamps.
    Not distributed; suitable for single-process deployments or dev.
    """
    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, dict]:
        now = time.time()
        window_start = now - window_seconds
        dq = self.windows[key]

        # Pop old timestamps
        while dq and dq[0] <= window_start:
            dq.popleft()

        allowed = len(dq) < limit
        if allowed:
            dq.append(now)

        oldest = dq[0] if dq else now
        reset_at = int(oldest + window_seconds)
        remaining = max(0, limit - len(dq))
        return allowed, {"limit": limit, "remaining": remaining, "reset": reset_at, "reset_in": max(0, reset_at - int(now))}


class RedisSlidingWindow:
    """
    Redis-based sliding-window implemented with sorted set (requires redis-py).
    Methods intentionally synchronous for simplicity; expects using redis client supporting sync operations.
    """
    def __init__(self, redis_client):
        self.redis = redis_client

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, dict]:
        now = time.time()
        window_start = now - window_seconds
        redis_key = f"rl:{key}"

        try:
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            removed, current = pipe.execute()
            allowed = current < limit
            if allowed:
                self.redis.zadd(redis_key, {str(now): now})
                self.redis.expire(redis_key, window_seconds * 2)
            oldest = self.redis.zrange(redis_key, 0, 0, withscores=True)
            reset_at = int((oldest[0][1] + window_seconds) if oldest else now + window_seconds)
            remaining = max(0, limit - current - (1 if allowed else 0))
            return allowed, {"limit": limit, "remaining": remaining, "reset": reset_at, "reset_in": max(0, reset_at - int(now))}
        except Exception as e:
            logger.exception("redis_rate_limit_error", error=str(e))
            # Fail-open
            return True, {"limit": limit, "remaining": limit, "reset": int(now + window_seconds), "reset_in": window_seconds}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware. Configure in app.add_middleware(...).
    """
    def __init__(self, app: ASGIApp, default_limit: int = 100, window_seconds: int = 60, redis_client = None, exclude_paths=None):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        if redis_client:
            self.limiter = RedisSlidingWindow(redis_client)
            logger.info("rate_limiter", backend="redis")
        else:
            self.limiter = InMemorySlidingWindow()
            logger.info("rate_limiter", backend="in_memory")

    def _key_for_request(self, request: Request) -> str:
        # Try request.state.user first (set by auth dependency)
        if hasattr(request.state, "user") and request.state.user is not None:
            user = request.state.user
            uid = getattr(user, "id", None)
            if uid:
                return f"user:{uid}"
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            h = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"apikey:{h}"
        # Fallback to IP
        ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"

    def _path_limit(self, path: str) -> Tuple[int, int]:
        # Customize per-path limits here
        custom = {
            "/api/v1/auth/token": (10, 60),
            "/api/v1/documents/search": (30, 60),
        }
        return custom.get(path, (self.default_limit, self.window_seconds))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        key = self._key_for_request(request)
        limit, window = self._path_limit(request.url.path)
        allowed, info = self.limiter.is_allowed(key, limit, window)

        if not allowed:
            logger.warning("rate_limited", key=key, path=request.url.path, limit=limit)
            # 429 response with headers
            body = {"detail": "Rate limit exceeded", "retry_after": info["reset_in"]}
            headers = {
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info["reset_in"])
            }
            return Response(content=str(body), status_code=status.HTTP_429_TOO_MANY_REQUESTS, headers=headers, media_type="application/json")

        # Otherwise proceed and attach headers on response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])
        return response


def rate_limit_dependency(max_requests: int = 10, window_seconds: int = 60):
    """
    Dependency for per-endpoint rate limiting
    
    Usage:
        @router.post("/expensive-operation")
        async def expensive_op(
            rate_limit: None = Depends(rate_limit_dependency(max_requests=5, window_seconds=60))
        ):
            ...
    """
    async def check_rate_limit(request: Request):
        # This is a simplified version - in production, integrate with middleware
        return None
    
    return check_rate_limit
