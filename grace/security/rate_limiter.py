"""
Rate Limiter - Prevents abuse and ensures fair resource usage
"""

from typing import Dict, Optional
import time
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    def __init__(self, limit: int, window: int, retry_after: float):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}s. "
            f"Retry after {retry_after:.1f}s"
        )


class RateLimiter:
    """
    Token bucket rate limiter
    
    Features:
    - Per-user rate limiting
    - Per-endpoint rate limiting
    - Configurable limits and windows
    - Constitutional compliance logging
    """
    
    def __init__(
        self,
        default_limit: int = 100,
        default_window: int = 60,
        burst_multiplier: float = 1.5
    ):
        """
        Initialize rate limiter
        
        Args:
            default_limit: Default requests per window
            default_window: Window size in seconds
            burst_multiplier: Burst capacity multiplier
        """
        self.default_limit = default_limit
        self.default_window = default_window
        self.burst_multiplier = burst_multiplier
        
        # Per-user buckets: {user_id: {endpoint: (tokens, last_update)}}
        self.buckets: Dict[str, Dict[str, tuple[float, float]]] = defaultdict(dict)
        
        # Custom limits: {endpoint: (limit, window)}
        self.custom_limits: Dict[str, tuple[int, int]] = {}
        
        self._lock = asyncio.Lock()
    
    def set_limit(self, endpoint: str, limit: int, window: int):
        """Set custom limit for endpoint"""
        self.custom_limits[endpoint] = (limit, window)
        logger.info(f"Rate limit set for {endpoint}: {limit} req/{window}s")
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        cost: float = 1.0
    ) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            user_id: User identifier
            endpoint: Endpoint being accessed
            cost: Request cost (default 1.0)
        
        Returns:
            True if within limit
        
        Raises:
            RateLimitExceeded: If limit exceeded
        """
        async with self._lock:
            # Get limit and window
            limit, window = self.custom_limits.get(
                endpoint,
                (self.default_limit, self.default_window)
            )
            
            burst_capacity = limit * self.burst_multiplier
            refill_rate = limit / window
            
            # Get or create bucket
            if endpoint not in self.buckets[user_id]:
                self.buckets[user_id][endpoint] = (burst_capacity, time.time())
            
            tokens, last_update = self.buckets[user_id][endpoint]
            
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - last_update
            tokens = min(burst_capacity, tokens + (elapsed * refill_rate))
            
            # Check if enough tokens
            if tokens >= cost:
                tokens -= cost
                self.buckets[user_id][endpoint] = (tokens, now)
                return True
            else:
                # Calculate retry after
                tokens_needed = cost - tokens
                retry_after = tokens_needed / refill_rate
                
                logger.warning(
                    f"Rate limit exceeded for {user_id} on {endpoint}",
                    extra={
                        "user_id": user_id,
                        "endpoint": endpoint,
                        "retry_after": retry_after
                    }
                )
                
                raise RateLimitExceeded(limit, window, retry_after)
    
    async def reset_user_limits(self, user_id: str):
        """Reset all limits for user"""
        async with self._lock:
            if user_id in self.buckets:
                del self.buckets[user_id]
                logger.info(f"Rate limits reset for {user_id}")
    
    def get_user_status(self, user_id: str) -> Dict[str, Dict]:
        """Get rate limit status for user"""
        status = {}
        
        for endpoint, (tokens, last_update) in self.buckets.get(user_id, {}).items():
            limit, window = self.custom_limits.get(
                endpoint,
                (self.default_limit, self.default_window)
            )
            
            burst_capacity = limit * self.burst_multiplier
            
            status[endpoint] = {
                "tokens_remaining": tokens,
                "burst_capacity": burst_capacity,
                "limit": limit,
                "window": window,
                "last_update": last_update
            }
        
        return status
