"""Event mesh bridge for Learning Kernel event publishing and consumption."""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Callable


class MeshBridge:
    """Bridge for integrating Learning Kernel with the Grace event mesh."""

    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.published_events: List[Dict[str, Any]] = []  # For testing/debugging

    async def publish_event(self, event_name: str, payload: Dict[str, Any]):
        """Publish an event to the mesh."""
        event = {
            "event_name": event_name,
            "payload": payload,
            "source": "learning_kernel",
            "timestamp": datetime.now().isoformat(),
            "event_id": f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        }

        # Store for debugging
        self.published_events.append(event)

        # In a real implementation, would publish to actual event mesh
        print(f"[LEARNING] Published event: {event_name}")

        # Trigger any registered handlers (for testing)
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error in event handler for {event_name}: {e}")

    def subscribe_to_event(self, event_name: str, handler: Callable):
        """Subscribe to an event type."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    async def handle_rollback_request(self, event_data: Dict[str, Any]):
        """Handle rollback requests from governance or other systems."""
        payload = event_data.get("payload", {})
        target = payload.get("target")
        to_snapshot = payload.get("to_snapshot")

        if target == "learning":
            print(f"[LEARNING] Received rollback request to snapshot: {to_snapshot}")

            # In practice, would delegate to snapshot manager
            # For now, just acknowledge the request
            await self.publish_event(
                "ROLLBACK_COMPLETED",
                {
                    "target": "learning",
                    "snapshot_id": to_snapshot,
                    "at": datetime.now().isoformat(),
                },
            )

    async def handle_mlt_adaptation_plan(self, event_data: Dict[str, Any]):
        """Handle adaptation plans from MLT kernel."""
        payload = event_data.get("payload", {})
        plan = payload.get("plan", {})

        # Process learning-specific adaptations
        learning_actions = [
            action
            for action in plan.get("actions", [])
            if action.get("type") in ["policy_delta", "hpo"]
            and "learning" in action.get("path", "")
        ]

        for action in learning_actions:
            await self._apply_adaptation_action(action)

    async def _apply_adaptation_action(self, action: Dict[str, Any]):
        """Apply a specific adaptation action."""
        action_type = action.get("type")
        path = action.get("path", "")

        if action_type == "policy_delta":
            if "learning.labeling.qa.min_agreement" in path:
                new_value = action.get("to")
                print(f"[LEARNING] Adapting min_agreement to: {new_value}")
                # Would update policy service

            elif "learning.weak.label_threshold" in path:
                new_value = action.get("to")
                print(f"[LEARNING] Adapting weak labeler threshold to: {new_value}")
                # Would update weak supervision service

        elif action_type == "hpo":
            if "active.strategy.hybrid" in action.get("target", ""):
                budget = action.get("budget", {})
                print(
                    f"[LEARNING] Starting HPO for active learning with budget: {budget}"
                )
                # Would trigger hyperparameter optimization

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently published events."""
        return self.published_events[-limit:]

    def clear_event_history(self):
        """Clear event history (for testing)."""
        self.published_events.clear()


# Default instance
mesh_bridge = MeshBridge()
