"""Circuit Breaker Controller - Implements circuit breaker pattern for fault tolerance."""
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting services from cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests are blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(self, service_id: str, dependency: str, config: Optional[Dict] = None):
        """
        Initialize circuit breaker.
        
        Args:
            service_id: ID of the service using the circuit breaker
            dependency: ID of the dependency being protected
            config: Circuit breaker configuration
        """
        self.service_id = service_id
        self.dependency = dependency
        
        # Default configuration
        self.config = {
            "failure_rate_threshold_pct": 50,  # Open when failure rate > 50%
            "request_volume_threshold": 20,    # Minimum requests before considering failure rate
            "sleep_window_ms": 60000,          # How long to stay open (1 minute)
            "half_open_max_calls": 10,         # Max requests in half-open state
            "timeout_ms": 30000                # Request timeout (30 seconds)
        }
        
        if config:
            self.config.update(config)
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._state_changed_at = datetime.utcnow()
        self._half_open_calls = 0
        
        # Request tracking (sliding window)
        self._request_window = []
        self._window_size_ms = 60000  # 1 minute window
        
        logger.info(f"Circuit breaker initialized for {service_id}:{dependency}")
    
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state.value
    
    def can_execute(self) -> bool:
        """Check if request can be executed through circuit breaker."""
        self._cleanup_old_requests()
        
        if self._state == CircuitState.CLOSED:
            return True
        
        elif self._state == CircuitState.OPEN:
            # Check if sleep window has elapsed
            if self._should_attempt_reset():
                self._transition_to_half_open()
                return True
            return False
        
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self._half_open_calls < self.config["half_open_max_calls"]
        
        return False
    
    def on_success(self):
        """Record successful request."""
        current_time = datetime.utcnow()
        self._record_request(current_time, success=True)
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1
            
            # If we've had enough successful calls, close the circuit
            if self._success_count >= 3:  # Require 3 successes to close
                self._transition_to_closed()
        
        elif self._state == CircuitState.CLOSED:
            self._success_count += 1
            # Reset failure count on success
            if self._failure_count > 0:
                self._failure_count = max(0, self._failure_count - 1)
    
    def on_failure(self):
        """Record failed request."""
        current_time = datetime.utcnow()
        self._record_request(current_time, success=False)
        self._last_failure_time = current_time
        
        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens the circuit
            self._transition_to_open()
        
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
    
    def on_timeout(self):
        """Record timeout as failure."""
        logger.warning(f"Circuit breaker timeout for {self.service_id}:{self.dependency}")
        self.on_failure()
    
    def force_open(self):
        """Force circuit breaker to open state."""
        logger.info(f"Forcing circuit breaker open for {self.service_id}:{self.dependency}")
        self._transition_to_open()
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        logger.info(f"Forcing circuit breaker closed for {self.service_id}:{self.dependency}")
        self._transition_to_closed()
    
    def force_half_open(self):
        """Force circuit breaker to half-open state."""
        logger.info(f"Forcing circuit breaker half-open for {self.service_id}:{self.dependency}")
        self._transition_to_half_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure rate."""
        total_requests = len(self._request_window)
        
        # Need minimum volume before considering failure rate
        if total_requests < self.config["request_volume_threshold"]:
            return False
        
        failures = sum(1 for req in self._request_window if not req["success"])
        failure_rate = (failures / total_requests) * 100
        
        return failure_rate > self.config["failure_rate_threshold_pct"]
    
    def _should_attempt_reset(self) -> bool:
        """Check if sleep window has elapsed and we should try half-open."""
        if self._last_failure_time is None:
            return True
        
        elapsed_ms = (datetime.utcnow() - self._state_changed_at).total_seconds() * 1000
        return elapsed_ms >= self.config["sleep_window_ms"]
    
    def _transition_to_open(self):
        """Transition to open state."""
        if self._state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker opening for {self.service_id}:{self.dependency}")
            self._state = CircuitState.OPEN
            self._state_changed_at = datetime.utcnow()
            self._half_open_calls = 0
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        if self._state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker closing for {self.service_id}:{self.dependency}")
            self._state = CircuitState.CLOSED
            self._state_changed_at = datetime.utcnow()
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        if self._state != CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker half-opening for {self.service_id}:{self.dependency}")
            self._state = CircuitState.HALF_OPEN
            self._state_changed_at = datetime.utcnow()
            self._success_count = 0
            self._failure_count = 0
            self._half_open_calls = 0
    
    def _record_request(self, timestamp: datetime, success: bool):
        """Record request in sliding window."""
        self._request_window.append({
            "timestamp": timestamp,
            "success": success
        })
        
        # Keep window manageable
        if len(self._request_window) > 1000:
            self._request_window = self._request_window[-500:]
    
    def _cleanup_old_requests(self):
        """Remove old requests from sliding window."""
        cutoff_time = datetime.utcnow() - timedelta(milliseconds=self._window_size_ms)
        self._request_window = [
            req for req in self._request_window 
            if req["timestamp"] > cutoff_time
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        self._cleanup_old_requests()
        
        total_requests = len(self._request_window)
        failures = sum(1 for req in self._request_window if not req["success"])
        
        failure_rate = (failures / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "service_id": self.service_id,
            "dependency": self.dependency,
            "state": self._state.value,
            "state_changed_at": self._state_changed_at.isoformat(),
            "total_requests": total_requests,
            "failure_count": failures,
            "success_count": total_requests - failures,
            "failure_rate_pct": failure_rate,
            "half_open_calls": self._half_open_calls,
            "config": self.config
        }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        logger.info(f"Resetting circuit breaker for {self.service_id}:{self.dependency}")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._state_changed_at = datetime.utcnow()
        self._half_open_calls = 0
        self._request_window = []


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(self, service_id: str, dependency: str, config: Optional[Dict] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        key = f"{service_id}:{dependency}"
        
        if key not in self.breakers:
            self.breakers[key] = CircuitBreaker(service_id, dependency, config)
        
        return self.breakers[key]
    
    def get_breaker(self, service_id: str, dependency: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        key = f"{service_id}:{dependency}"
        return self.breakers.get(key)
    
    def remove_breaker(self, service_id: str, dependency: str):
        """Remove circuit breaker."""
        key = f"{service_id}:{dependency}"
        if key in self.breakers:
            del self.breakers[key]
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all circuit breakers."""
        return {
            key: breaker.get_metrics() 
            for key, breaker in self.breakers.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_breakers = len(self.breakers)
        states = {}
        
        for breaker in self.breakers.values():
            state = breaker.state()
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_breakers": total_breakers,
            "state_distribution": states,
            "breaker_keys": list(self.breakers.keys())
        }


# Module-level registry instance
_registry = CircuitBreakerRegistry()

def get_circuit_breaker(service_id: str, dependency: str, config: Optional[Dict] = None) -> CircuitBreaker:
    """Get or create circuit breaker for service-dependency pair."""
    return _registry.get_or_create(service_id, dependency, config)

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all circuit breakers."""
    return _registry.breakers

def get_circuit_breaker_stats() -> Dict[str, Any]:
    """Get circuit breaker statistics."""
    return _registry.get_stats()