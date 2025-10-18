"""
Rate limiting middleware with Redis and in-memory support
"""

import time
import hashlib
from typing import Callable, Optional, Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

logger = structlog.get_logger()


class InMemoryRateLimiter:
    """In-memory rate limiter using token bucket algorithm"""
    
    def __init__(self):
        # Store: key -> (tokens, last_refill_time)
        self.buckets: Dict[str, tuple[float, float]] = {}
        self.request_counts: Dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, dict]:
        """
        Check if request is allowed using sliding window
        
        Returns:
            (is_allowed, rate_limit_info)
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Get request history for this key
        requests = self.request_counts[key]
        
        # Remove old requests outside window
        requests[:] = [req_time for req_time in requests if req_time > window_start]
        
        # Check if limit exceeded
        current_count = len(requests)
        is_allowed = current_count < max_requests
        
        if is_allowed:
            requests.append(now)
        
        # Calculate rate limit info
        oldest_request = requests[0] if requests else now
        reset_time = oldest_request + window_seconds
        remaining = max(0, max_requests - current_count - (1 if is_allowed else 0))
        
        return is_allowed, {
            "limit": max_requests,
            "remaining": remaining,
            "reset": int(reset_time),
            "reset_in": int(reset_time - now)
        }
    
    def cleanup(self, max_age_seconds: int = 3600):
        """Remove old entries to prevent memory leak"""
        now = time.time()
        cutoff = now - max_age_seconds
        
        # Clean up request counts
        to_remove = []
        for key, requests in self.request_counts.items():
            requests[:] = [req_time for req_time in requests if req_time > cutoff]
            if not requests:
                to_remove.append(key)
        
        for key in to_remove:
            del self.request_counts[key]


class RedisRateLimiter:
    """Redis-based rate limiter (production-ready)"""
    
    def __init__(self, redis_client):
        """
        Initialize Redis rate limiter
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
    
    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, dict]:
        """
        Check if request is allowed using Redis
        
        Returns:
            (is_allowed, rate_limit_info)
        """
        now = time.time()
        window_start = now - window_seconds
        redis_key = f"rate_limit:{key}"
        
        try:
            # Use Redis sorted set with timestamps as scores
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            
            # Count current requests
            pipe.zcard(redis_key)
            
            # Execute pipeline
            _, current_count = pipe.execute()
            
            is_allowed = current_count < max_requests
            
            if is_allowed:
                # Add current request
                self.redis.zadd(redis_key, {str(now): now})
                
                # Set expiry
                self.redis.expire(redis_key, window_seconds * 2)
            
            # Calculate rate limit info
            oldest = self.redis.zrange(redis_key, 0, 0, withscores=True)
            reset_time = (oldest[0][1] + window_seconds) if oldest else now + window_seconds
            remaining = max(0, max_requests - current_count - (1 if is_allowed else 0))
            
            return is_allowed, {
                "limit": max_requests,
                "remaining": remaining,
                "reset": int(reset_time),
                "reset_in": int(reset_time - now)
            }
        
        except Exception as e:
            logger.error("redis_rate_limit_error", error=str(e))
            # Fail open - allow request if Redis fails
            return True, {
                "limit": max_requests,
                "remaining": max_requests,
                "reset": int(now + window_seconds),
                "reset_in": window_seconds
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with per-user and per-IP throttling
    
    Supports:
    - Per-user rate limiting (for authenticated requests)
    - Per-IP rate limiting (for anonymous requests)
    - Custom rate limits per endpoint
    - Redis or in-memory backend
    """
    
    def __init__(
        self,
        app: ASGIApp,
        default_limit: int = 100,
        window_seconds: int = 60,
        redis_client = None,
        exclude_paths: Optional[list[str]] = None
    ):
        """
        Initialize rate limiting middleware
        
        Args:
            app: ASGI application
            default_limit: Default max requests per window
            window_seconds: Time window in seconds
            redis_client: Redis client (optional, uses in-memory if None)
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        
        # Initialize rate limiter backend
        if redis_client:
            self.limiter = RedisRateLimiter(redis_client)
            logger.info("rate_limiter_initialized", backend="redis")
        else:
            self.limiter = InMemoryRateLimiter()
            logger.info("rate_limiter_initialized", backend="in_memory")
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """
        Generate rate limit key based on user or IP
        
        Priority:
        1. User ID (if authenticated)
        2. API key (if present)
        3. Client IP address
        """
        # Check for authenticated user
        if hasattr(request.state, "user"):
            user = request.state.user
            user_id = getattr(user, "id", None)
            if user_id:
                return f"user:{user_id}"
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"apikey:{key_hash}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _get_rate_limit_for_path(self, path: str) -> tuple[int, int]:
        """
        Get rate limit configuration for specific path
        
        Can be customized per endpoint
        """
        # Custom limits for specific endpoints
        custom_limits = {
            "/api/v1/auth/token": (10, 60),  # 10 requests per minute
            "/api/v1/documents/search": (30, 60),  # 30 searches per minute
        }
        
        return custom_limits.get(path, (self.default_limit, self.window_seconds))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get rate limit key and limits
        rate_limit_key = self._get_rate_limit_key(request)
        max_requests, window_seconds = self._get_rate_limit_for_path(request.url.path)
        
        # Check rate limit
        is_allowed, rate_info = self.limiter.is_allowed(
            rate_limit_key,
            max_requests,
            window_seconds
        )
        
        # Add rate limit headers to response
        if is_allowed:
            response = await call_next(request)
            
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
            
            return response
        else:
            # Rate limit exceeded
            logger.warning(
                "rate_limit_exceeded",
                key=rate_limit_key,
                path=request.url.path,
                limit=max_requests,
                window=window_seconds
            )
            
            # Return 429 Too Many Requests
            return Response(
                content='{"detail":"Rate limit exceeded","retry_after":' + str(rate_info["reset_in"]) + '}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["reset_in"])
                }
            )


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
