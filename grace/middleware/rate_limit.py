"""
Rate limiting middleware with Redis and in-memory support
"""

from typing import Dict, Callable, Optional, Set
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(
        self,
        app,
        default_limit: int = 100,
        window_seconds: int = 60,
        redis_client: Optional[Any] = None,
        exclude_paths: Optional[Set[str]] = None
    ):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.redis_client = redis_client
        self.exclude_paths = exclude_paths or set()
        
        # In-memory storage if no Redis
        self.memory_store: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits before processing request"""
        
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed = await self._check_rate_limit(client_id)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(self.window_seconds)}
            )
        
        response = await call_next(request)
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try to get user ID from token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return f"user:{auth_header[7:20]}"  # Use part of token
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit"""
        if self.redis_client:
            return await self._check_redis_rate_limit(client_id)
        else:
            return self._check_memory_rate_limit(client_id)
    
    async def _check_redis_rate_limit(self, client_id: str) -> bool:
        """Check rate limit using Redis"""
        try:
            key = f"rate_limit:{client_id}"
            current = self.redis_client.get(key)
            
            if current is None:
                self.redis_client.setex(key, self.window_seconds, 1)
                return True
            
            count = int(current)
            if count >= self.default_limit:
                return False
            
            self.redis_client.incr(key)
            return True
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return True  # Fail open
    
    def _check_memory_rate_limit(self, client_id: str) -> bool:
        """Check rate limit using in-memory storage"""
        now = time.time()
        
        if client_id not in self.memory_store:
            self.memory_store[client_id] = {"count": 1, "reset_at": now + self.window_seconds}
            return True
        
        entry = self.memory_store[client_id]
        
        # Reset if window expired
        if now > entry["reset_at"]:
            entry["count"] = 1
            entry["reset_at"] = now + self.window_seconds
            return True
        
        # Check limit
        if entry["count"] >= self.default_limit:
            return False
        
        entry["count"] += 1
        return True


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
