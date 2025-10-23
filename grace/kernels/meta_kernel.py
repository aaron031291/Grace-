"""
MetaKernel: The central decision-making hub for Grace's OODA loop.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Dict

logger = logging.getLogger(__name__)

# Type alias for the event publisher function
EventPublisher = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, None]]


class MetaKernel:
    """
    Decides on the appropriate course of action based on incoming events.
    This is the "Decide" part of the OODA loop.
    """

    def __init__(self, event_publisher: EventPublisher):
        self.publish = event_publisher

    async def decide_and_route(
        self,
        event_type: str,
        payload: Dict[str, Any],
        correlation_id: str,
    ):
        """
        Analyzes an event and routes it to the correct action workflow.
        """
        logger.info(
            f"MetaKernel deciding for event: {event_type}",
            extra={"correlation_id": correlation_id},
        )

        # Example decision logic
        is_critical = "critical" in payload.get("reason", "").lower() or payload.get("severity") == "CRITICAL"
        is_recurring = payload.get("is_recurring_pattern", False)

        # 1. Prioritize critical self-healing
        if is_critical:
            await self.publish(
                "healing.request.critical",
                {
                    "reason": f"Critical event: {event_type}",
                    "details": payload,
                    "component_id": payload.get("component_id"),
                    "correlation_id": correlation_id,
                },
            )
        # 2. Check for self-improvement opportunities
        elif is_recurring:
            await self.publish(
                "transcendence.request.improve",
                {
                    "reason": f"Recurring pattern detected for event: {event_type}",
                    "details": payload,
                    "component_id": payload.get("component_id"),
                    "correlation_id": correlation_id,
                },
            )
        # 3. Log for learning if it's a minor issue
        else:
            await self.publish(
                "learning.request.log_opportunity",
                {
                    "reason": f"Minor event observed: {event_type}",
                    "details": payload,
                    "component_id": payload.get("component_id"),
                    "correlation_id": correlation_id,
                },
            )
