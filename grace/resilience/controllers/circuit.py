"""Circuit breaker implementation for resilience control."""

import time
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.

    Provides automatic failure detection and recovery, preventing
    cascading failures by temporarily blocking calls to failing services.
    """

    def __init__(
        self,
        failure_threshold_pct: float = 50.0,
        volume_threshold: int = 20,
        sleep_window_ms: int = 5000,
        half_open_max_calls: int = 5,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold_pct: Failure rate threshold to open circuit
            volume_threshold: Minimum requests before circuit can open
            sleep_window_ms: Time circuit stays open before half-open
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold_pct = failure_threshold_pct
        self.volume_threshold = volume_threshold
        self.sleep_window_ms = sleep_window_ms
        self.half_open_max_calls = half_open_max_calls

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._total_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

        logger.debug(
            f"Circuit breaker initialized: threshold={failure_threshold_pct}%, volume={volume_threshold}"
        )

    def on_success(self):
        """Record successful execution."""
        with self._get_lock():
            self._success_count += 1
            self._total_count += 1
            self._half_open_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    # Successful recovery, close circuit
                    self._state = CircuitState.CLOSED
                    self._reset_counters()
                    logger.info("Circuit breaker closed after successful recovery")

    def on_failure(self):
        """Record failed execution."""
        with self._get_lock():
            self._failure_count += 1
            self._total_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failure during half-open, go back to open
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(
                    "Circuit breaker reopened after failure in half-open state"
                )
            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker opened: {self._get_failure_rate():.1f}% failure rate"
                    )

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._get_lock():
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker moved to half-open state")
                    return True
                return False
            else:  # HALF_OPEN
                return self._half_open_calls < self.half_open_max_calls

    def state(self) -> str:
        """Get current circuit state."""
        return self._state.value

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._get_lock():
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_count": self._total_count,
                "failure_rate_pct": self._get_failure_rate(),
                "last_failure_time": self._last_failure_time,
                "half_open_calls": self._half_open_calls
                if self._state == CircuitState.HALF_OPEN
                else 0,
            }

    def force_open(self):
        """Force circuit to open state (for testing/manual control)."""
        with self._get_lock():
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            logger.info("Circuit breaker forced to open state")

    def force_closed(self):
        """Force circuit to closed state (for testing/manual control)."""
        with self._get_lock():
            self._state = CircuitState.CLOSED
            self._reset_counters()
            logger.info("Circuit breaker forced to closed state")

    def force_half_open(self):
        """Force circuit to half-open state (for testing/manual control)."""
        with self._get_lock():
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.info("Circuit breaker forced to half-open state")

    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on current metrics."""
        if self._total_count < self.volume_threshold:
            return False

        failure_rate = self._get_failure_rate()
        return failure_rate >= self.failure_threshold_pct

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open to half-open."""
        if not self._last_failure_time:
            return True

        elapsed_ms = (time.time() - self._last_failure_time) * 1000
        return elapsed_ms >= self.sleep_window_ms

    def _get_failure_rate(self) -> float:
        """Calculate current failure rate percentage."""
        if self._total_count == 0:
            return 0.0
        return (self._failure_count / self._total_count) * 100.0

    def _reset_counters(self):
        """Reset failure/success counters."""
        self._failure_count = 0
        self._success_count = 0
        self._total_count = 0
        self._half_open_calls = 0

    def _get_lock(self):
        """Get thread lock for state management."""
        # In a real implementation, would use threading.Lock()
        # For simplicity, using a context manager that does nothing
        return _NoOpLock()


class _NoOpLock:
    """No-op lock for single-threaded environments."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Backwards-compatible alias expected by older tests
CircuitBreakerController = CircuitBreaker
