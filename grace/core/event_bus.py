"""
EventBus - Central event routing system for Grace governance kernel.
Enhanced with KPITrustMonitor and ImmutableLogs integration.
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
import logging

from .contracts import generate_correlation_id
from .kpi_trust_monitor import KPITrustMonitor
from .immutable_logs import ImmutableLogs, TransparencyLevel


logger = logging.getLogger(__name__)


class EventBus:
    """
    Event-driven messaging system for governance components.
    Handles event routing, subscription management, and message delivery.
    Integrated with KPITrustMonitor and ImmutableLogs for comprehensive monitoring.
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.correlation_tracking: Dict[str, List[str]] = {}

        # Enhanced monitoring and logging
        self.kpi_monitor = KPITrustMonitor(event_publisher=self._publish_internal_event)
        self.immutable_logs = ImmutableLogs()

        # Performance metrics
        self.event_count = 0
        self.failed_deliveries = 0
        self.started_at: Optional[datetime] = None
        self.running = False

    async def start(self):
        """Start the EventBus and integrated components."""
        if self.running:
            logger.warning("EventBus already running")
            return

        logger.info("Starting EventBus with monitoring integration...")

        # Start integrated components
        await self.kpi_monitor.start()
        await self.immutable_logs.start()

        self.started_at = datetime.now()
        self.running = True

        # Log system start
        await self.immutable_logs.log_event(
            event_type="eventbus_started",
            component_id="eventbus",
            event_data={"started_at": self.started_at.isoformat(), "version": "2.0.0"},
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        logger.info("EventBus started successfully")

    async def stop(self):
        """Stop the EventBus and integrated components."""
        if not self.running:
            return

        logger.info("Stopping EventBus...")

        # Log system stop
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="eventbus_stopped",
                component_id="eventbus",
                event_data={
                    "stopped_at": datetime.now().isoformat(),
                    "total_events_processed": self.event_count,
                    "failed_deliveries": self.failed_deliveries,
                },
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
            )

        # Stop integrated components
        await self.immutable_logs.stop()
        await self.kpi_monitor.stop()

        self.running = False
        logger.info("EventBus stopped")

    async def subscribe(self, event_type: str, handler: Callable):
        """Subscribe a handler to a specific event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

        # Log subscription
        await self.immutable_logs.log_event(
            event_type="event_subscription",
            component_id="eventbus",
            event_data={
                "event_type": event_type,
                "handler_name": getattr(handler, "__name__", str(handler)),
                "total_subscribers": len(self.subscribers[event_type]),
            },
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        logger.info(f"Subscribed handler to {event_type}")

    async def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe a handler from an event type."""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)

                # Log unsubscription
                await self.immutable_logs.log_event(
                    event_type="event_unsubscription",
                    component_id="eventbus",
                    event_data={
                        "event_type": event_type,
                        "handler_name": getattr(handler, "__name__", str(handler)),
                        "remaining_subscribers": len(self.subscribers[event_type]),
                    },
                    transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
                )

                logger.info(f"Unsubscribed handler from {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type}")

    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: str = "normal",
    ) -> str:
        """
        Publish an event to all subscribers with enhanced monitoring.
        Returns the correlation ID for tracking.
        """
        start_time = asyncio.get_event_loop().time()

        if correlation_id is None:
            correlation_id = generate_correlation_id()

        event = {
            "type": event_type,
            "payload": payload,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "id": f"evt_{len(self.message_history):06d}",
            "priority": priority,
        }

        # Store in history
        self.message_history.append(event)

        # Track correlation
        if correlation_id not in self.correlation_tracking:
            self.correlation_tracking[correlation_id] = []
        self.correlation_tracking[correlation_id].append(event["id"])

        # Log event publication
        await self.immutable_logs.log_event(
            event_type="event_published",
            component_id="eventbus",
            event_data={
                "published_event_type": event_type,
                "event_id": event["id"],
                "priority": priority,
                "subscriber_count": len(self.subscribers.get(event_type, [])),
            },
            correlation_id=correlation_id,
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        # Deliver to subscribers
        delivery_results = []
        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                tasks.append(self._safe_deliver(handler, event))

            if tasks:
                delivery_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate performance metrics
        end_time = asyncio.get_event_loop().time()
        delivery_time = end_time - start_time

        # Count failed deliveries
        failed_count = sum(
            1 for result in delivery_results if isinstance(result, Exception)
        )
        self.failed_deliveries += failed_count

        # Record performance metrics
        await self.kpi_monitor.record_metric(
            name="event_delivery_time",
            value=delivery_time * 1000,  # Convert to milliseconds
            component_id="eventbus",
            threshold_warning=100.0,  # 100ms warning
            threshold_critical=500.0,  # 500ms critical
            tags={"event_type": event_type, "priority": priority},
        )

        await self.kpi_monitor.record_metric(
            name="event_delivery_failures",
            value=failed_count,
            component_id="eventbus",
            threshold_warning=1.0,
            threshold_critical=3.0,
            tags={"event_type": event_type},
        )

        # Update trust score based on delivery success
        success_rate = (len(delivery_results) - failed_count) / max(
            1, len(delivery_results)
        )
        await self.kpi_monitor.update_trust_score("eventbus", success_rate)

        self.event_count += 1

        logger.info(
            f"Published {event_type} event with correlation_id {correlation_id} "
            f"(delivery_time: {delivery_time * 1000:.1f}ms, failures: {failed_count})"
        )

        return correlation_id

    async def _safe_deliver(self, handler: Callable, event: Dict[str, Any]):
        """Safely deliver event to handler with error handling and monitoring."""
        handler_name = getattr(handler, "__name__", str(handler))
        start_time = asyncio.get_event_loop().time()

        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

            # Record successful delivery
            delivery_time = (asyncio.get_event_loop().time() - start_time) * 1000
            await self.kpi_monitor.record_metric(
                name="handler_execution_time",
                value=delivery_time,
                component_id=handler_name,
                tags={"event_type": event["type"]},
            )

            return True

        except Exception as e:
            error_details = {
                "handler_name": handler_name,
                "event_type": event["type"],
                "event_id": event["id"],
                "error": str(e),
                "error_type": type(e).__name__,
            }

            # Log delivery failure
            await self.immutable_logs.log_event(
                event_type="event_delivery_failed",
                component_id="eventbus",
                event_data=error_details,
                correlation_id=event["correlation_id"],
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
            )

            logger.error(f"Error delivering event to handler {handler_name}: {e}")
            return e

    async def _publish_internal_event(self, event_name: str, payload: Dict[str, Any]):
        """Handle internal events from integrated components."""
        try:
            # Don't create infinite loops - just log internal events
            if not event_name.startswith("internal_"):
                await self.immutable_logs.log_event(
                    event_type=f"internal_{event_name}",
                    component_id="eventbus",
                    event_data=payload,
                    transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
                )
        except Exception as e:
            logger.error(f"Error handling internal event {event_name}: {e}")

    def get_events_by_correlation(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all events associated with a correlation ID."""
        if correlation_id not in self.correlation_tracking:
            return []

        event_ids = self.correlation_tracking[correlation_id]
        return [event for event in self.message_history if event["id"] in event_ids]

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent events."""
        return self.message_history[-limit:]

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = (
            (datetime.now() - self.started_at).total_seconds() if self.started_at else 0
        )

        status = {
            "eventbus": {
                "running": self.running,
                "uptime_seconds": uptime,
                "total_events": self.event_count,
                "failed_deliveries": self.failed_deliveries,
                "success_rate": (self.event_count - self.failed_deliveries)
                / max(1, self.event_count),
                "subscribers": {
                    event_type: len(handlers)
                    for event_type, handlers in self.subscribers.items()
                },
                "message_history_size": len(self.message_history),
            }
        }

        # Add KPI monitor status
        if self.kpi_monitor:
            status["kpi_monitor"] = self.kpi_monitor.get_system_health()

        # Add immutable logs status
        if self.immutable_logs:
            status["immutable_logs"] = self.immutable_logs.get_system_stats()

        return status

    async def clear_history(self, keep_recent: int = 1000):
        """Clear old events, keeping only the most recent ones."""
        if len(self.message_history) > keep_recent:
            # Log the cleanup operation
            await self.immutable_logs.log_event(
                event_type="message_history_cleanup",
                component_id="eventbus",
                event_data={
                    "previous_size": len(self.message_history),
                    "new_size": keep_recent,
                    "cleaned_count": len(self.message_history) - keep_recent,
                },
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
            )

            self.message_history = self.message_history[-keep_recent:]
            logger.info(f"Cleared event history, kept {keep_recent} recent events")
