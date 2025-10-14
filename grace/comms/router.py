"""
Grace Message Router - Intelligent routing based on message patterns and system state.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

from .envelope import GraceMessageEnvelope


logger = logging.getLogger(__name__)


class RouteStrategy(str, Enum):
    """Routing strategies."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    TOPIC_BASED = "topic_based"


class RouteHealth(str, Enum):
    """Route health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class Route:
    """Individual message route configuration."""

    def __init__(
        self,
        route_id: str,
        pattern: str,
        handler: Callable,
        strategy: RouteStrategy = RouteStrategy.ROUND_ROBIN,
        priority: int = 100,
        max_retries: int = 3,
    ):
        self.route_id = route_id
        self.pattern = pattern
        self.handler = handler
        self.strategy = strategy
        self.priority = priority
        self.max_retries = max_retries
        self.health = RouteHealth.HEALTHY
        self.failure_count = 0
        self.last_success = datetime.utcnow()
        self.last_failure: Optional[datetime] = None

    def matches(self, envelope: GraceMessageEnvelope) -> bool:
        """Check if message matches route pattern."""
        try:
            # Simple pattern matching - can be extended
            if self.pattern == "*":
                return True
            if self.pattern.startswith("topic:"):
                topic_pattern = self.pattern[6:]
                return envelope.routing.topic.startswith(topic_pattern)
            if self.pattern.startswith("kind:"):
                kind_pattern = self.pattern[5:]
                return envelope.kind == kind_pattern
            if self.pattern.startswith("priority:"):
                priority_pattern = self.pattern[9:]
                return envelope.priority == priority_pattern

            return envelope.routing.topic == self.pattern
        except Exception as e:
            logger.error(f"Route pattern matching error: {e}")
            return False

    def update_health(self, success: bool):
        """Update route health based on success/failure."""
        if success:
            self.last_success = datetime.utcnow()
            self.failure_count = max(0, self.failure_count - 1)
        else:
            self.last_failure = datetime.utcnow()
            self.failure_count += 1

        # Update health status
        if self.failure_count == 0:
            self.health = RouteHealth.HEALTHY
        elif self.failure_count < 5:
            self.health = RouteHealth.DEGRADED
        else:
            self.health = RouteHealth.UNHEALTHY


class MessageRouter:
    """Grace message router with intelligent routing and health monitoring."""

    def __init__(self, enable_circuit_breaker: bool = True):
        self.routes: Dict[str, Route] = {}
        self.topic_routes: Dict[str, List[str]] = defaultdict(list)
        self.enable_circuit_breaker = enable_circuit_breaker
        self.message_stats = defaultdict(int)
        self.route_metrics = defaultdict(dict)
        self.circuit_breakers: Dict[str, bool] = {}

    def add_route(self, route: Route) -> bool:
        """Add a new route to the router."""
        try:
            self.routes[route.route_id] = route

            # Index by topic for faster lookup
            if route.pattern.startswith("topic:"):
                topic = route.pattern[6:]
                self.topic_routes[topic].append(route.route_id)
            elif not route.pattern.startswith(("kind:", "priority:")):
                self.topic_routes[route.pattern].append(route.route_id)

            logger.info(f"Added route {route.route_id} with pattern {route.pattern}")
            return True

        except Exception as e:
            logger.error(f"Failed to add route {route.route_id}: {e}")
            return False

    def remove_route(self, route_id: str) -> bool:
        """Remove a route from the router."""
        try:
            if route_id not in self.routes:
                return False

            route = self.routes[route_id]
            del self.routes[route_id]

            # Remove from topic index
            for topic, route_ids in self.topic_routes.items():
                if route_id in route_ids:
                    route_ids.remove(route_id)

            logger.info(f"Removed route {route_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove route {route_id}: {e}")
            return False

    async def route_message(self, envelope: GraceMessageEnvelope) -> bool:
        """Route a message to appropriate handlers."""
        try:
            self.message_stats["total"] += 1

            # Find matching routes
            matching_routes = []
            for route in self.routes.values():
                if route.matches(envelope) and route.health != RouteHealth.UNHEALTHY:
                    matching_routes.append(route)

            if not matching_routes:
                logger.warning(f"No healthy routes found for message {envelope.id}")
                self.message_stats["unroutable"] += 1
                return False

            # Sort by priority
            matching_routes.sort(key=lambda r: r.priority, reverse=True)

            # Attempt delivery
            for route in matching_routes:
                if self.enable_circuit_breaker and self.circuit_breakers.get(
                    route.route_id, False
                ):
                    continue

                try:
                    success = await self._deliver_message(route, envelope)
                    if success:
                        self.message_stats["delivered"] += 1
                        route.update_health(True)
                        self._update_route_metrics(route.route_id, True)
                        return True
                    else:
                        route.update_health(False)
                        self._update_route_metrics(route.route_id, False)

                except Exception as e:
                    logger.error(f"Delivery failed for route {route.route_id}: {e}")
                    route.update_health(False)
                    self._update_route_metrics(route.route_id, False)

                    # Activate circuit breaker if too many failures
                    if route.failure_count > 10:
                        self.circuit_breakers[route.route_id] = True
                        logger.warning(
                            f"Circuit breaker activated for route {route.route_id}"
                        )

            self.message_stats["failed"] += 1
            return False

        except Exception as e:
            logger.error(f"Message routing failed: {e}")
            self.message_stats["errors"] += 1
            return False

    async def _deliver_message(
        self, route: Route, envelope: GraceMessageEnvelope
    ) -> bool:
        """Deliver message to route handler."""
        try:
            if asyncio.iscoroutinefunction(route.handler):
                await route.handler(envelope)
            else:
                route.handler(envelope)
            return True
        except Exception as e:
            logger.error(f"Handler execution failed: {e}")
            return False

    def _update_route_metrics(self, route_id: str, success: bool):
        """Update route performance metrics."""
        if route_id not in self.route_metrics:
            self.route_metrics[route_id] = {
                "delivered": 0,
                "failed": 0,
                "last_update": datetime.utcnow(),
            }

        if success:
            self.route_metrics[route_id]["delivered"] += 1
        else:
            self.route_metrics[route_id]["failed"] += 1

        self.route_metrics[route_id]["last_update"] = datetime.utcnow()

    def get_route_health(self, route_id: str) -> Dict[str, Any]:
        """Get health information for a route."""
        if route_id not in self.routes:
            return {}

        route = self.routes[route_id]
        metrics = self.route_metrics.get(route_id, {})

        return {
            "route_id": route_id,
            "health": route.health.value,
            "failure_count": route.failure_count,
            "last_success": route.last_success.isoformat()
            if route.last_success
            else None,
            "last_failure": route.last_failure.isoformat()
            if route.last_failure
            else None,
            "circuit_breaker_active": self.circuit_breakers.get(route_id, False),
            "delivered": metrics.get("delivered", 0),
            "failed": metrics.get("failed", 0),
        }

    def get_router_stats(self) -> Dict[str, Any]:
        """Get overall router statistics."""
        return {
            "total_routes": len(self.routes),
            "healthy_routes": sum(
                1 for r in self.routes.values() if r.health == RouteHealth.HEALTHY
            ),
            "degraded_routes": sum(
                1 for r in self.routes.values() if r.health == RouteHealth.DEGRADED
            ),
            "unhealthy_routes": sum(
                1 for r in self.routes.values() if r.health == RouteHealth.UNHEALTHY
            ),
            "active_circuit_breakers": len(
                [cb for cb in self.circuit_breakers.values() if cb]
            ),
            "message_stats": dict(self.message_stats),
        }

    async def health_check(self):
        """Perform health check and reset circuit breakers if needed."""
        current_time = datetime.utcnow()

        for route_id, route in self.routes.items():
            # Reset circuit breaker if route has been failing for too long
            if (
                self.circuit_breakers.get(route_id, False)
                and route.last_failure
                and current_time - route.last_failure > timedelta(minutes=5)
            ):
                self.circuit_breakers[route_id] = False
                route.failure_count = 0
                route.health = RouteHealth.DEGRADED
                logger.info(f"Circuit breaker reset for route {route_id}")
