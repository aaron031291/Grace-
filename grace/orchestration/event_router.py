"""
Event Router for TriggerMesh Orchestration Layer.

Routes events from the EventBus to matching workflows in the WorkflowRegistry.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogger
from grace.core.kpi_trust_monitor import KPITrustMonitor
from grace.orchestration.workflow_engine import WorkflowEngine
from grace.orchestration.workflow_registry import WorkflowRegistry

logger = logging.getLogger(__name__)


class EventFilter:
    """A filter to determine if an event should be processed by a workflow."""

    def __init__(self, criteria: Dict[str, Any]):
        self.criteria = criteria

    def match(self, event_data: Dict[str, Any]) -> bool:
        """Checks if the event data matches the filter criteria."""
        for key, value in self.criteria.items():
            if event_data.get(key) != value:
                return False
        return True


class EventRouter:
    """Routes events to registered workflows based on trigger patterns."""

    def __init__(
        self,
        workflow_registry: WorkflowRegistry,
        workflow_engine: WorkflowEngine,
        event_bus: EventBus,
        immutable_logger: ImmutableLogger,
        kpi_monitor: Optional[KPITrustMonitor] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.registry = workflow_registry
        self.engine = workflow_engine
        self.event_bus = event_bus
        self.logger = immutable_logger
        self.kpi_monitor = kpi_monitor
        self.config = config or {}

        self._stats = {
            "events_received": 0,
            "workflows_triggered": 0,
            "events_filtered": 0,
            "events_rate_limited": 0,
            "last_event_timestamp": None,
        }
        self._recent_events: Dict[str, List[datetime]] = {}

    async def setup_subscriptions(self):
        """Subscribe to all event types that can trigger workflows."""
        trigger_event_types = self.registry.get_all_trigger_event_types()
        for event_type in trigger_event_types:
            await self.event_bus.subscribe(event_type, self.handle_event)
            logger.info(f"Subscribed to event type: {event_type}")

    async def handle_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Observe: Handle an incoming event, assign a correlation ID, and route it.
        This is the start of the OODA loop.
        """
        self._stats["events_received"] += 1
        self._stats["last_event_timestamp"] = datetime.utcnow()

        # OBSERVE: Ensure every event has a cryptographic, trackable correlation ID.
        if "correlation_id" not in payload:
            payload["correlation_id"] = uuid.uuid4().hex
        correlation_id = payload["correlation_id"]

        logger.debug(
            f"Event received: {event_type} [correlation_id={correlation_id}]",
            extra={"correlation_id": correlation_id},
        )

        matching_workflows = self.registry.get_workflows_for_event(event_type)

        if not matching_workflows:
            logger.debug(f"No workflows found for event type: {event_type}", extra={"correlation_id": correlation_id})
            return

        # Create a context object for this event
        event_context = {
            "event_type": event_type,
            "payload": payload,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow(),
        }

        for workflow in matching_workflows:
            if not workflow.get("enabled", True):
                continue

            if self._is_rate_limited(workflow, event_type, correlation_id):
                self._stats["events_rate_limited"] += 1
                logger.warning(
                    f"Workflow '{workflow['name']}' rate-limited for event '{event_type}'.",
                    extra={"correlation_id": correlation_id},
                )
                continue

            if self._passes_filters(workflow, payload):
                # ORIENT & DECIDE: The router has oriented the event and decided on a workflow.
                # ACT: Trigger the workflow execution in the engine.
                await self.engine.execute_workflow(workflow, event_context)
                self._stats["workflows_triggered"] += 1
            else:
                self._stats["events_filtered"] += 1
                logger.debug(
                    f"Event payload did not pass filters for workflow '{workflow['name']}'.",
                    extra={"correlation_id": correlation_id},
                )

    def _is_rate_limited(self, workflow: Dict[str, Any], event_type: str, correlation_id: str) -> bool:
        """Check if the workflow is currently rate-limited for this event."""
        rate_limit_config = workflow.get("rate_limit")
        if not rate_limit_config:
            return False

        limit = rate_limit_config.get("limit", 1)
        period_seconds = rate_limit_config.get("period_seconds", 60)
        now = datetime.utcnow()
        event_key = f"{workflow['name']}:{event_type}"

        if event_key not in self._recent_events:
            self._recent_events[event_key] = []

        # Prune old timestamps
        self._recent_events[event_key] = [
            ts for ts in self._recent_events[event_key] if now - ts < timedelta(seconds=period_seconds)
        ]

        if len(self._recent_events[event_key]) >= limit:
            return True

        self._recent_events[event_key].append(now)
        return False

    def _passes_filters(self, workflow: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        """Check if the event payload passes the workflow's filters."""
        filters = workflow.get("filters", [])
        if not filters:
            return True

        for f in filters:
            field = f.get("field")
            operator = f.get("operator")
            value = f.get("value")

            payload_value = payload.get(field)
            if payload_value is None:
                return False  # Field must exist in payload to be filtered

            if operator == "==" and not (payload_value == value):
                return False
            if operator == "!=" and not (payload_value != value):
                return False
            if operator == ">" and not (payload_value > value):
                return False
            if operator == "<" and not (payload_value < value):
                return False
            if operator == ">=" and not (payload_value >= value):
                return False
            if operator == "<=" and not (payload_value <= value):
                return False
            if operator == "in" and not (payload_value in value):
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return the router's operational statistics."""
        return self._stats
