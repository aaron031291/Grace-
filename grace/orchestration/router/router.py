"""
Grace Orchestration Router - Event and signal routing across kernels.

Provides reliable event routing across the Event Mesh with backpressure, retry,
and circuit breaker patterns. Routes signals between governance, memory, mldl,
learning, ingress, intelligence, interface, multi-OS, and immune kernels.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RoutingPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RouteConfig:
    """Configuration for a routing rule."""
    
    def __init__(self, source: str, target: str, event_pattern: str, 
                 priority: RoutingPriority = RoutingPriority.NORMAL,
                 max_retries: int = 3, timeout_s: int = 30,
                 backpressure_limit: int = 100):
        self.source = source
        self.target = target
        self.event_pattern = event_pattern
        self.priority = priority
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.backpressure_limit = backpressure_limit
        
        # Metrics
        self.messages_routed = 0
        self.messages_failed = 0
        self.total_latency_ms = 0.0
        self.last_used = None
    
    def get_success_rate(self) -> float:
        """Get routing success rate (0-1)."""
        total = self.messages_routed + self.messages_failed
        if total == 0:
            return 1.0
        return self.messages_routed / total
    
    def get_average_latency_ms(self) -> float:
        """Get average routing latency in milliseconds."""
        if self.messages_routed == 0:
            return 0.0
        return self.total_latency_ms / self.messages_routed


class CircuitBreaker:
    """Circuit breaker for route protection."""
    
    def __init__(self, failure_threshold: int = 5, timeout_s: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_s = timeout_s
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        
        # Half-open state allows single attempt
        return True
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        if not self.last_failure_time:
            return True
        return time.time() - self.last_failure_time > self.timeout_s


class RoutingMessage:
    """Message to be routed through the system."""
    
    def __init__(self, event_type: str, payload: Dict[str, Any],
                 source: str, targets: List[str] = None,
                 priority: RoutingPriority = RoutingPriority.NORMAL,
                 correlation_id: str = None):
        self.event_type = event_type
        self.payload = payload
        self.source = source
        self.targets = targets or []
        self.priority = priority
        self.correlation_id = correlation_id or f"msg_{int(time.time() * 1000)}"
        self.created_at = datetime.now()
        self.attempts = 0
        self.max_retries = 3
        self.status = "pending"
        self.error = None
    
    def __lt__(self, other):
        """Priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.attempts < self.max_retries
    
    def increment_attempt(self):
        """Increment attempt counter."""
        self.attempts += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "targets": self.targets,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
            "attempts": self.attempts,
            "status": self.status,
            "error": self.error
        }


class Router:
    """Event and signal router for orchestration kernel."""
    
    def __init__(self, event_publisher=None, kernel_registry=None):
        self.event_publisher = event_publisher
        self.kernel_registry = kernel_registry
        
        # Routing configuration
        self.routes: Dict[str, List[RouteConfig]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.target_handlers: Dict[str, Callable] = {}
        
        # Message queues by priority
        self.message_queues: Dict[RoutingPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in RoutingPriority
        }
        
        # Backpressure tracking
        self.target_queues: Dict[str, int] = {}
        self.max_queue_size = 1000
        
        # Statistics
        self.total_messages = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.average_latency_ms = 0.0
        
        # State
        self.running = False
        self._routing_tasks: List[asyncio.Task] = []
        
        # Initialize default routes
        self._initialize_default_routes()
    
    def _initialize_default_routes(self):
        """Initialize default routing rules for Grace kernels."""
        default_routes = [
            # Governance kernel routes
            ("orchestration", "governance", "ORCH_*", RoutingPriority.HIGH),
            ("governance", "orchestration", "GOV_VIOLATION", RoutingPriority.CRITICAL),
            ("governance", "orchestration", "GOV_POLICY_UPDATE", RoutingPriority.HIGH),
            
            # Memory/MLT kernel routes
            ("orchestration", "memory", "ORCH_EXPERIENCE", RoutingPriority.NORMAL),
            ("memory", "orchestration", "MEMORY_INSIGHT", RoutingPriority.NORMAL),
            
            # Learning kernel routes
            ("orchestration", "learning", "ORCH_EXPERIENCE", RoutingPriority.NORMAL),
            ("learning", "orchestration", "LEARNING_ADAPTATION", RoutingPriority.HIGH),
            
            # Intelligence kernel routes
            ("orchestration", "intelligence", "ORCH_TASK_*", RoutingPriority.NORMAL),
            ("intelligence", "orchestration", "INTELLIGENCE_RESULT", RoutingPriority.NORMAL),
            
            # Interface kernel routes
            ("orchestration", "interface", "ORCH_STATUS_*", RoutingPriority.LOW),
            ("interface", "orchestration", "INTERFACE_REQUEST", RoutingPriority.NORMAL),
            
            # Ingress kernel routes
            ("ingress", "orchestration", "INGRESS_EVENT", RoutingPriority.NORMAL),
            ("orchestration", "ingress", "ORCH_CONFIG_UPDATE", RoutingPriority.HIGH),
            
            # Event mesh routes
            ("orchestration", "event_mesh", "*", RoutingPriority.NORMAL),
            ("event_mesh", "orchestration", "*", RoutingPriority.NORMAL),
        ]
        
        for source, target, pattern, priority in default_routes:
            self.add_route(RouteConfig(source, target, pattern, priority))
    
    async def start(self):
        """Start the router."""
        if self.running:
            logger.warning("Router already running")
            return
        
        logger.info("Starting orchestration router...")
        self.running = True
        
        # Start routing workers for each priority
        for priority in RoutingPriority:
            task = asyncio.create_task(self._routing_worker(priority))
            self._routing_tasks.append(task)
        
        logger.info("Orchestration router started")
    
    async def stop(self):
        """Stop the router."""
        if not self.running:
            return
        
        logger.info("Stopping orchestration router...")
        self.running = False
        
        # Cancel routing tasks
        for task in self._routing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._routing_tasks, return_exceptions=True)
        self._routing_tasks.clear()
        
        logger.info("Orchestration router stopped")
    
    def add_route(self, route: RouteConfig):
        """Add a routing rule."""
        source_key = f"{route.source}->{route.target}"
        
        if route.source not in self.routes:
            self.routes[route.source] = []
        
        self.routes[route.source].append(route)
        
        # Initialize circuit breaker
        if source_key not in self.circuit_breakers:
            self.circuit_breakers[source_key] = CircuitBreaker()
        
        logger.debug(f"Added route: {route.source} -> {route.target} ({route.event_pattern})")
    
    def register_target_handler(self, target: str, handler: Callable):
        """Register a handler for a specific target."""
        self.target_handlers[target] = handler
        logger.debug(f"Registered handler for target: {target}")
    
    async def route(self, event: Dict[str, Any]) -> bool:
        """Route an event through the system."""
        try:
            # Extract event information
            event_type = event.get("event_type", "UNKNOWN")
            payload = event.get("payload", {})
            source = event.get("source", "unknown")
            
            # Find matching routes
            matching_routes = self._find_matching_routes(source, event_type)
            
            if not matching_routes:
                logger.debug(f"No routes found for {source} -> {event_type}")
                return True  # Not an error if no routes
            
            # Create routing message
            targets = [route.target for route in matching_routes]
            priority = max(route.priority for route in matching_routes)
            
            message = RoutingMessage(
                event_type=event_type,
                payload=payload,
                source=source,
                targets=targets,
                priority=priority,
                correlation_id=event.get("correlation_id")
            )
            
            # Queue message for processing
            await self.message_queues[priority].put(message)
            self.total_messages += 1
            
            logger.debug(f"Queued message for routing: {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to route event: {e}", exc_info=True)
            return False
    
    def _find_matching_routes(self, source: str, event_type: str) -> List[RouteConfig]:
        """Find routes that match the source and event type."""
        if source not in self.routes:
            return []
        
        matching = []
        for route in self.routes[source]:
            if self._event_matches_pattern(event_type, route.event_pattern):
                matching.append(route)
        
        return matching
    
    def _event_matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches routing pattern."""
        if pattern == "*":
            return True
        
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return event_type.startswith(prefix)
        
        return event_type == pattern
    
    async def _routing_worker(self, priority: RoutingPriority):
        """Worker to process messages of a specific priority."""
        queue = self.message_queues[priority]
        
        try:
            while self.running:
                # Get next message (with timeout to allow graceful shutdown)
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                await self._process_message(message)
                
        except asyncio.CancelledError:
            logger.debug(f"Routing worker for {priority} cancelled")
            raise
        except Exception as e:
            logger.error(f"Routing worker error for {priority}: {e}", exc_info=True)
    
    async def _process_message(self, message: RoutingMessage):
        """Process a routing message."""
        start_time = time.time()
        
        try:
            message.increment_attempt()
            
            # Route to each target
            for target in message.targets:
                await self._route_to_target(message, target)
            
            # Record success
            message.status = "completed"
            latency_ms = (time.time() - start_time) * 1000
            
            self.successful_routes += 1
            self._update_average_latency(latency_ms)
            
            # Update route metrics
            self._update_route_metrics(message, latency_ms, success=True)
            
            logger.debug(f"Successfully routed {message.event_type} to {len(message.targets)} targets")
            
        except Exception as e:
            message.status = "failed"
            message.error = {"message": str(e), "type": type(e).__name__}
            
            self.failed_routes += 1
            self._update_route_metrics(message, 0, success=False)
            
            # Retry if possible
            if message.can_retry():
                logger.warning(f"Retrying message {message.correlation_id} (attempt {message.attempts})")
                await asyncio.sleep(min(2 ** message.attempts, 10))  # Exponential backoff
                await self.message_queues[message.priority].put(message)
            else:
                logger.error(f"Failed to route message {message.correlation_id}: {e}")
    
    async def _route_to_target(self, message: RoutingMessage, target: str):
        """Route message to specific target."""
        circuit_key = f"{message.source}->{target}"
        circuit_breaker = self.circuit_breakers.get(circuit_key)
        
        # Check circuit breaker
        if circuit_breaker and not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker open for {circuit_key}")
        
        # Check backpressure
        if self._check_backpressure(target):
            raise Exception(f"Backpressure limit exceeded for {target}")
        
        try:
            # Route via registered handler
            if target in self.target_handlers:
                await self.target_handlers[target](message.to_dict())
            else:
                # Fallback to event publisher
                if self.event_publisher:
                    await self.event_publisher(f"ROUTE_TO_{target.upper()}", message.to_dict())
                else:
                    logger.warning(f"No handler or publisher for target: {target}")
            
            # Record success
            if circuit_breaker:
                circuit_breaker.record_success()
            
            self._update_target_queue(target, -1)  # Decrease queue size
            
        except Exception as e:
            # Record failure
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            raise e
    
    def _check_backpressure(self, target: str) -> bool:
        """Check if target is under backpressure."""
        queue_size = self.target_queues.get(target, 0)
        return queue_size > self.max_queue_size
    
    def _update_target_queue(self, target: str, delta: int):
        """Update target queue size tracking."""
        self.target_queues[target] = max(0, self.target_queues.get(target, 0) + delta)
    
    def _update_route_metrics(self, message: RoutingMessage, latency_ms: float, success: bool):
        """Update metrics for routes used."""
        for target in message.targets:
            routes = self.routes.get(message.source, [])
            for route in routes:
                if route.target == target:
                    if success:
                        route.messages_routed += 1
                        route.total_latency_ms += latency_ms
                    else:
                        route.messages_failed += 1
                    route.last_used = datetime.now()
    
    def _update_average_latency(self, latency_ms: float):
        """Update average latency with exponential moving average."""
        alpha = 0.1
        self.average_latency_ms = alpha * latency_ms + (1 - alpha) * self.average_latency_ms
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status and metrics."""
        route_stats = {}
        for source, routes in self.routes.items():
            route_stats[source] = [
                {
                    "target": route.target,
                    "pattern": route.event_pattern,
                    "priority": route.priority.name,
                    "messages_routed": route.messages_routed,
                    "messages_failed": route.messages_failed,
                    "success_rate": route.get_success_rate(),
                    "avg_latency_ms": route.get_average_latency_ms()
                }
                for route in routes
            ]
        
        circuit_stats = {
            key: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time
            }
            for key, breaker in self.circuit_breakers.items()
        }
        
        return {
            "running": self.running,
            "total_messages": self.total_messages,
            "successful_routes": self.successful_routes,
            "failed_routes": self.failed_routes,
            "success_rate": self.successful_routes / max(1, self.total_messages),
            "average_latency_ms": self.average_latency_ms,
            "routes": route_stats,
            "circuits": circuit_stats,
            "queue_sizes": {
                priority.name: queue.qsize() 
                for priority, queue in self.message_queues.items()
            },
            "target_queues": self.target_queues
        }