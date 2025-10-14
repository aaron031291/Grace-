"""
Event Router - Core routing engine for TriggerMesh orchestration.

Routes events to appropriate workflows based on trigger patterns and filters.
Integrates with EventBus, ImmutableLogs, and KPITrustMonitor.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import re
import json

from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogs, TransparencyLevel
from grace.core.kpi_trust_monitor import KPITrustMonitor

logger = logging.getLogger(__name__)


class EventFilter:
    """Filters events based on conditions, thresholds, and rate limits."""

    def __init__(self):
        self.event_counts: defaultdict[str, List[datetime]] = defaultdict(list)
        self.seen_events: Dict[str, datetime] = {}  # For deduplication

    def matches_filters(
        self, event: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """Check if event matches filter conditions."""
        if not filters:
            return True

        payload = event.get("payload", {})

        # Check exact matches (e.g., severity: "CRITICAL")
        for key, expected in filters.items():
            if key in [
                "min_delta_percent",
                "max_score",
                "min_degradation_rate",
                "min_gap",
            ]:
                # Numeric comparisons handled separately
                continue

            value = payload.get(key)

            # Handle list of acceptable values
            if isinstance(expected, list):
                if value not in expected:
                    return False
            # Handle exact match
            else:
                if value != expected:
                    return False

        # Numeric threshold filters
        if "min_delta_percent" in filters:
            delta = payload.get("delta", 0)
            threshold = payload.get("threshold", 1)
            if threshold > 0:
                delta_percent = abs(delta / threshold) * 100
                if delta_percent < filters["min_delta_percent"]:
                    return False

        if "max_score" in filters:
            score = payload.get("current_score", 1.0)
            if score > filters["max_score"]:
                return False

        if "min_degradation_rate" in filters:
            rate = payload.get("degradation_rate", 0.0)
            if rate < filters["min_degradation_rate"]:
                return False

        if "min_gap" in filters:
            gap = payload.get("gap", 0.0)
            if gap < filters["min_gap"]:
                return False

        return True

    def check_rate_limit(
        self, event_type: str, max_per_minute: int = 100, burst_size: int = 20
    ) -> bool:
        """Check if event is within rate limits."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Clean old timestamps
        self.event_counts[event_type] = [
            ts for ts in self.event_counts[event_type] if ts > cutoff
        ]

        # Check burst (last 5 seconds)
        burst_cutoff = now - timedelta(seconds=5)
        recent_count = sum(
            1 for ts in self.event_counts[event_type] if ts > burst_cutoff
        )
        if recent_count >= burst_size:
            return False

        # Check overall rate
        if len(self.event_counts[event_type]) >= max_per_minute:
            return False

        # Add current event
        self.event_counts[event_type].append(now)
        return True

    def is_duplicate(
        self, event: Dict[str, Any], window_seconds: int = 60
    ) -> bool:
        """Check if event is a duplicate within the time window."""
        # Create unique key from event type + critical payload fields
        key_parts = [event.get("type", "")]
        payload = event.get("payload", {})

        # Add identifying fields to key
        for field in ["component_id", "metric_name", "workflow_name"]:
            if field in payload:
                key_parts.append(str(payload[field]))

        event_key = "|".join(key_parts)

        now = datetime.now()

        # Check if we've seen this recently
        if event_key in self.seen_events:
            last_seen = self.seen_events[event_key]
            if (now - last_seen).total_seconds() < window_seconds:
                return True

        # Record this event
        self.seen_events[event_key] = now

        # Cleanup old entries
        cutoff = now - timedelta(seconds=window_seconds * 2)
        self.seen_events = {
            k: v for k, v in self.seen_events.items() if v > cutoff
        }

        return False


class EventRouter:
    """
    Routes events to workflows based on trigger patterns.
    Main orchestration engine for TriggerMesh.
    """

    def __init__(
        self,
        workflow_registry,
        event_bus: EventBus,
        immutable_logs: ImmutableLogs,
        kpi_monitor: Optional[KPITrustMonitor] = None,
    ):
        self.workflow_registry = workflow_registry
        self.event_bus = event_bus
        self.immutable_logs = immutable_logs
        self.kpi_monitor = kpi_monitor

        self.event_filter = EventFilter()
        self.running = False

        # Statistics
        self.stats = {
            "events_received": 0,
            "events_filtered": 0,
            "events_rate_limited": 0,
            "events_duplicated": 0,
            "workflows_triggered": 0,
            "workflows_failed": 0,
        }

        # Configuration
        self.config = {
            "rate_limiting": {"enabled": True, "max_per_minute": 100, "burst_size": 20},
            "deduplication": {"enabled": True, "window_seconds": 60},
            "logging": {
                "log_all_events": False,
                "log_workflow_execution": True,
            },
        }

    async def start(self):
        """Start the event router and subscribe to all workflow trigger events."""
        if self.running:
            logger.warning("EventRouter already running")
            return

        logger.info("Starting EventRouter...")

        # Get all unique event types from workflow triggers
        event_types = self.workflow_registry.get_trigger_event_types()

        # Subscribe to each event type
        for event_type in event_types:
            await self.event_bus.subscribe(event_type, self.handle_event)
            logger.debug(f"Subscribed to event type: {event_type}")

        self.running = True

        # Log startup
        await self.immutable_logs.log_event(
            event_type="event_router_started",
            component_id="trigger_mesh",
            event_data={
                "subscribed_events": list(event_types),
                "workflows_loaded": len(self.workflow_registry.workflows),
            },
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        logger.info(
            f"EventRouter started - monitoring {len(event_types)} event types"
        )

    async def stop(self):
        """Stop the event router."""
        if not self.running:
            return

        logger.info("Stopping EventRouter...")
        self.running = False

        # Log shutdown with stats
        await self.immutable_logs.log_event(
            event_type="event_router_stopped",
            component_id="trigger_mesh",
            event_data=self.stats,
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        logger.info("EventRouter stopped")

    async def handle_event(self, event: Dict[str, Any]):
        """Main event handler - routes events to matching workflows."""
        self.stats["events_received"] += 1

        event_type = event.get("type", "")

        # Optional: Log all events
        if self.config["logging"]["log_all_events"]:
            await self.immutable_logs.log_event(
                event_type="event_received",
                component_id="trigger_mesh",
                event_data={
                    "received_event_type": event_type,
                    "correlation_id": event.get("correlation_id"),
                },
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
            )

        # Check rate limiting
        if self.config["rate_limiting"]["enabled"]:
            if not self.event_filter.check_rate_limit(
                event_type,
                self.config["rate_limiting"]["max_per_minute"],
                self.config["rate_limiting"]["burst_size"],
            ):
                self.stats["events_rate_limited"] += 1
                logger.warning(f"Rate limit exceeded for {event_type}")
                return

        # Check deduplication
        if self.config["deduplication"]["enabled"]:
            if self.event_filter.is_duplicate(
                event, self.config["deduplication"]["window_seconds"]
            ):
                self.stats["events_duplicated"] += 1
                logger.debug(f"Duplicate event filtered: {event_type}")
                return

        # Find matching workflows
        matching_workflows = self.workflow_registry.find_matching_workflows(event)

        if not matching_workflows:
            logger.debug(f"No workflows match event: {event_type}")
            return

        # Execute matching workflows
        for workflow in matching_workflows:
            # Check workflow-specific filters
            if not self.event_filter.matches_filters(event, workflow.trigger.filters):
                self.stats["events_filtered"] += 1
                logger.debug(
                    f"Event filtered by workflow {workflow.name}: {event_type}"
                )
                continue

            # Trigger workflow
            try:
                await self._trigger_workflow(workflow, event)
                self.stats["workflows_triggered"] += 1
            except Exception as e:
                self.stats["workflows_failed"] += 1
                logger.error(f"Failed to trigger workflow {workflow.name}: {e}")

                # Log failure
                await self.immutable_logs.log_event(
                    event_type="workflow_trigger_failed",
                    component_id="trigger_mesh",
                    event_data={
                        "workflow_name": workflow.name,
                        "event_type": event_type,
                        "error": str(e),
                    },
                    correlation_id=event.get("correlation_id"),
                    transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
                )

    async def _trigger_workflow(self, workflow, event: Dict[str, Any]):
        """Trigger a specific workflow with the event."""
        logger.info(f"Triggering workflow: {workflow.name}")

        # Log workflow start
        if self.config["logging"]["log_workflow_execution"]:
            await self.immutable_logs.log_event(
                event_type="workflow.started",
                component_id="trigger_mesh",
                event_data={
                    "workflow_name": workflow.name,
                    "trigger_event": event.get("type"),
                    "workflow_version": workflow.version,
                },
                correlation_id=event.get("correlation_id"),
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
            )

        # Execute workflow (imported to avoid circular dependency)
        from grace.orchestration.workflow_engine import WorkflowEngine

        engine = WorkflowEngine(self.event_bus, self.immutable_logs, self.kpi_monitor)
        await engine.execute_workflow(workflow, event)

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            **self.stats,
            "running": self.running,
            "workflows_loaded": len(self.workflow_registry.workflows),
            "subscribed_events": len(
                self.workflow_registry.get_trigger_event_types()
            ),
        }

    def update_config(self, config: Dict[str, Any]):
        """Update router configuration."""
        if "rate_limiting" in config:
            self.config["rate_limiting"].update(config["rate_limiting"])

        if "deduplication" in config:
            self.config["deduplication"].update(config["deduplication"])

        if "logging" in config:
            self.config["logging"].update(config["logging"])

        logger.info(f"EventRouter configuration updated: {config}")
