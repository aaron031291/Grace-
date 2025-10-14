"""
Grace Orchestration Event Mesh Bridge - Integration with the Grace event mesh.

Provides seamless integration between the orchestration kernel and the
Grace event mesh for reliable event publishing and consumption.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Callable
import logging

logger = logging.getLogger(__name__)


class MeshBridge:
    """Bridge for integrating Orchestration Kernel with the Grace event mesh."""

    def __init__(self, event_bus=None, trigger_mesh=None):
        self.event_bus = event_bus
        self.trigger_mesh = trigger_mesh

        # Event routing configuration
        self.routing_rules = {
            "ORCH_LOOP_STARTED": ["monitoring", "audit"],
            "ORCH_TASK_DISPATCHED": ["audit", "monitoring"],
            "ORCH_TASK_COMPLETED": ["audit", "monitoring", "mlt"],
            "ORCH_ERROR": ["governance", "alert_manager", "monitoring"],
            "ORCH_INSTANCE_SPAWNED": ["monitoring", "audit"],
            "ORCH_INSTANCE_RETIRED": ["monitoring", "audit"],
            "ORCH_SNAPSHOT_CREATED": ["audit", "backup_systems"],
            "ROLLBACK_REQUESTED": ["governance", "audit", "alert_manager"],
            "ROLLBACK_COMPLETED": ["audit", "monitoring"],
            "ORCH_EXPERIENCE": ["mlt_kernel", "learning_kernel"],
        }

        self.running = False
        self.published_events: List[Dict[str, Any]] = []
        self.event_handlers: Dict[str, List[Callable]] = {}

        # Statistics
        self.events_published = 0
        self.events_consumed = 0
        self.routing_failures = 0

    async def start(self):
        """Start the mesh bridge."""
        if self.running:
            return

        logger.info("Starting Orchestration-Event Mesh Bridge...")
        self.running = True

        # Start event processing
        if self.event_bus:
            asyncio.create_task(self._process_events())

        logger.info("Orchestration-Event Mesh Bridge started")

    async def stop(self):
        """Stop the mesh bridge."""
        logger.info("Stopping Orchestration-Event Mesh Bridge...")
        self.running = False

    async def publish_event(self, event_name: str, payload: Dict[str, Any]):
        """Publish an event to the mesh."""
        event = {
            "event_name": event_name,
            "payload": payload,
            "source": "orchestration_kernel",
            "timestamp": datetime.now().isoformat(),
            "event_id": f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        }

        # Store for debugging
        self.published_events.append(event)
        if len(self.published_events) > 1000:  # Keep last 1000 events
            self.published_events = self.published_events[-1000:]

        try:
            # Route to configured targets
            if event_name in self.routing_rules:
                for target in self.routing_rules[event_name]:
                    await self._route_to_target(target, event)

            # Publish to event bus if available
            if self.event_bus:
                await self._publish_to_event_bus(event)

            # Route through trigger mesh if available
            if self.trigger_mesh:
                await self._route_through_trigger_mesh(event)

            self.events_published += 1
            logger.debug(f"[ORCHESTRATION] Published event: {event_name}")

            # Trigger any registered handlers (for testing)
            if event_name in self.event_handlers:
                for handler in self.event_handlers[event_name]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Event handler error for {event_name}: {e}")

        except Exception as e:
            self.routing_failures += 1
            logger.error(f"Failed to publish event {event_name}: {e}")

    async def _route_to_target(self, target: str, event: Dict[str, Any]):
        """Route event to specific target."""
        # In a real implementation, this would route to actual target systems
        logger.debug(f"Routing event {event['event_name']} to {target}")

    async def _publish_to_event_bus(self, event: Dict[str, Any]):
        """Publish event to the event bus."""
        try:
            if hasattr(self.event_bus, "publish"):
                await self.event_bus.publish(event["event_name"], event["payload"])
            else:
                # Fallback for mock event bus
                logger.debug(f"Mock event bus publish: {event['event_name']}")
        except Exception as e:
            logger.error(f"Event bus publish failed: {e}")

    async def _route_through_trigger_mesh(self, event: Dict[str, Any]):
        """Route event through trigger mesh."""
        try:
            if hasattr(self.trigger_mesh, "route_event"):
                # Construct event dict as expected by TriggerMesh
                event_dict = {
                    "type": event.get("event_name"),
                    "payload": event.get("payload"),
                    "event_id": event.get("event_id"),
                    "source": event.get("source"),
                    "correlation_id": event.get("correlation_id"),
                }
                await self.trigger_mesh.route_event(event_dict)
            else:
                logger.debug(f"Mock trigger mesh route: {event['event_name']}")
        except Exception as e:
            logger.error(f"Trigger mesh routing failed: {e}")

    async def _process_events(self):
        """Process events from the event bus."""
        while self.running:
            try:
                if hasattr(self.event_bus, "subscribe"):
                    async for event in self.event_bus.subscribe("orchestration.*"):
                        await self._handle_external_event(event)
                        self.events_consumed += 1
                else:
                    # Fallback for mock event bus
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(5)

    async def _handle_external_event(self, event: Dict[str, Any]):
        """Handle external events that affect orchestration operations."""
        event_type = event.get("event_type", "UNKNOWN")

        # Handle specific event types
        if event_type == "GOVERNANCE_VIOLATION":
            await self._handle_governance_violation(event)
        elif event_type == "KERNEL_FAILURE":
            await self._handle_kernel_failure(event)
        elif event_type == "SCALING_TRIGGER":
            await self._handle_scaling_trigger(event)
        elif event_type == "ROLLBACK_REQUEST":
            await self._handle_rollback_request(event)
        else:
            logger.debug(f"Received external event: {event_type}")

    async def _handle_governance_violation(self, event: Dict[str, Any]):
        """Handle governance violation events."""
        logger.warning(f"Governance violation received: {event.get('payload', {})}")

        # Could trigger emergency actions like:
        # - Pause affected loops
        # - Create emergency snapshot
        # - Escalate to human operators

    async def _handle_kernel_failure(self, event: Dict[str, Any]):
        """Handle kernel failure events."""
        payload = event.get("payload", {})
        kernel = payload.get("kernel", "unknown")

        logger.error(f"Kernel failure reported for: {kernel}")

        # Could trigger recovery actions like:
        # - Isolate failed kernel
        # - Redistribute tasks
        # - Initiate kernel restart

    async def _handle_scaling_trigger(self, event: Dict[str, Any]):
        """Handle scaling trigger events."""
        payload = event.get("payload", {})
        action = payload.get("action", "unknown")

        logger.info(f"Scaling trigger received: {action}")

        # Could trigger scaling actions like:
        # - Scale up/down instances
        # - Adjust resource allocation
        # - Update scaling policies

    async def _handle_rollback_request(self, event: Dict[str, Any]):
        """Handle rollback requests from governance or other systems."""
        payload = event.get("payload", {})

        if payload.get("target") == "orchestration":
            snapshot_id = payload.get("to_snapshot")
            if snapshot_id:
                logger.info(f"External rollback request to snapshot: {snapshot_id}")
                # Would trigger actual rollback via snapshot manager

    def subscribe_to_event(self, event_name: str, handler: Callable):
        """Subscribe to an event type."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
        logger.debug(f"Subscribed handler to event: {event_name}")

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently published events."""
        return self.published_events[-limit:]

    def clear_event_history(self):
        """Clear event history (for testing)."""
        self.published_events.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status and statistics."""
        return {
            "running": self.running,
            "routing_rules": len(self.routing_rules),
            "event_handlers": {
                event_type: len(handlers)
                for event_type, handlers in self.event_handlers.items()
            },
            "statistics": {
                "events_published": self.events_published,
                "events_consumed": self.events_consumed,
                "routing_failures": self.routing_failures,
            },
            "recent_events_count": len(self.published_events),
            "connected_event_bus": self.event_bus is not None,
            "connected_trigger_mesh": self.trigger_mesh is not None,
        }
