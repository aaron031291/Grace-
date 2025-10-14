"""Rate limiting controller for resilience."""

import time
from collections import defaultdict, deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter implementation.

    Allows burst traffic up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(self, rate_per_second: int, burst_capacity: int):
        """
        Initialize token bucket.

        Args:
            rate_per_second: Token refill rate per second
            burst_capacity: Maximum tokens in bucket (burst capacity)
        """
        self.rate_per_second = rate_per_second
        self.burst_capacity = burst_capacity

        self._tokens = burst_capacity
        self._last_refill = time.time()

        logger.debug(
            f"Token bucket initialized: {rate_per_second} rps, burst {burst_capacity}"
        )

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limited
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def peek(self) -> int:
        """Get current token count without consuming."""
        self._refill()
        return self._tokens

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "rate_per_second": self.rate_per_second,
            "burst_capacity": self.burst_capacity,
            "current_tokens": self.peek(),
            "utilization_pct": (1 - self.peek() / self.burst_capacity) * 100,
        }

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill

        if elapsed > 0:
            tokens_to_add = elapsed * self.rate_per_second
            self._tokens = min(self.burst_capacity, self._tokens + tokens_to_add)
            self._last_refill = now


class RateLimiter:
    """
    Multi-service rate limiter with token bucket algorithm.

    Provides per-service rate limiting with configurable rates and burst capacity.
    """

    def __init__(self):
        """Initialize rate limiter."""
        self._buckets: Dict[str, TokenBucket] = {}
        self._configs: Dict[str, Dict] = {}

        logger.debug("Rate limiter initialized")

    def configure(self, service_id: str, rps: int, burst: int):
        """
        Configure rate limiting for a service.

        Args:
            service_id: Service identifier
            rps: Requests per second limit
            burst: Burst capacity
        """
        self._configs[service_id] = {"rps": rps, "burst": burst}
        self._buckets[service_id] = TokenBucket(rps, burst)
        logger.info(
            f"Rate limiter configured for {service_id}: {rps} rps, burst {burst}"
        )

    def is_allowed(self, service_id: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed under rate limit.

        Args:
            service_id: Service identifier
            tokens: Number of tokens to consume (default 1)

        Returns:
            True if request is allowed, False if rate limited
        """
        if service_id not in self._buckets:
            # No rate limit configured, allow by default
            return True

        bucket = self._buckets[service_id]
        allowed = bucket.consume(tokens)

        if not allowed:
            logger.warning(f"Rate limit exceeded for service {service_id}")

        return allowed

    def get_stats(self, service_id: Optional[str] = None) -> Dict:
        """
        Get rate limiter statistics.

        Args:
            service_id: Specific service ID, or None for all services

        Returns:
            Statistics dictionary
        """
        if service_id:
            if service_id in self._buckets:
                return {
                    "service_id": service_id,
                    **self._buckets[service_id].get_stats(),
                }
            return {"service_id": service_id, "configured": False}

        return {
            service_id: bucket.get_stats()
            for service_id, bucket in self._buckets.items()
        }

    def reset(self, service_id: str):
        """Reset rate limiter for a service."""
        if service_id in self._configs:
            config = self._configs[service_id]
            self._buckets[service_id] = TokenBucket(config["rps"], config["burst"])
            logger.info(f"Rate limiter reset for service {service_id}")

    def remove(self, service_id: str):
        """Remove rate limiting for a service."""
        if service_id in self._buckets:
            del self._buckets[service_id]
            del self._configs[service_id]
            logger.info(f"Rate limiter removed for service {service_id}")


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for more precise rate limiting.

    Uses a sliding window to track request counts over time,
    providing more accurate rate limiting than token bucket.
    """

    def __init__(self, window_size_ms: int = 60000):
        """
        Initialize sliding window rate limiter.

        Args:
            window_size_ms: Sliding window size in milliseconds
        """
        self.window_size_ms = window_size_ms
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._limits: Dict[str, int] = {}

        logger.debug(
            f"Sliding window rate limiter initialized: {window_size_ms}ms window"
        )

    def configure(self, service_id: str, requests_per_window: int):
        """
        Configure rate limit for a service.

        Args:
            service_id: Service identifier
            requests_per_window: Max requests allowed in window
        """
        self._limits[service_id] = requests_per_window
        logger.info(
            f"Sliding window configured for {service_id}: {requests_per_window} requests per window"
        )

    def is_allowed(self, service_id: str) -> bool:
        """
        Check if request is allowed under rate limit.

        Args:
            service_id: Service identifier

        Returns:
            True if request is allowed, False if rate limited
        """
        if service_id not in self._limits:
            # No limit configured, allow by default
            return True

        now = time.time() * 1000  # Convert to milliseconds
        window = self._windows[service_id]
        limit = self._limits[service_id]

        # Remove old entries outside the window
        cutoff = now - self.window_size_ms
        while window and window[0] < cutoff:
            window.popleft()

        # Check if we're under the limit
        if len(window) < limit:
            window.append(now)
            return True

        logger.warning(f"Sliding window rate limit exceeded for service {service_id}")
        return False

    def get_stats(self, service_id: Optional[str] = None) -> Dict:
        """Get rate limiter statistics."""
        if service_id:
            if service_id in self._limits:
                now = time.time() * 1000
                window = self._windows[service_id]

                # Clean up old entries for accurate count
                cutoff = now - self.window_size_ms
                while window and window[0] < cutoff:
                    window.popleft()

                return {
                    "service_id": service_id,
                    "limit": self._limits[service_id],
                    "current_count": len(window),
                    "utilization_pct": (len(window) / self._limits[service_id]) * 100,
                }
            return {"service_id": service_id, "configured": False}

        # Return stats for all services
        stats = {}
        for service_id in self._limits:
            stats[service_id] = self.get_stats(service_id)
        return stats
