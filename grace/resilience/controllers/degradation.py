"""Degradation mode controller for graceful service degradation."""

import asyncio
import logging
from typing import Dict, Set, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DegradationMode(Enum):
    """Predefined degradation modes."""

    LITE_EXPLANATIONS = "lite_explanations"
    CACHED_ONLY = "cached_only"
    REDUCE_BATCH = "reduce_batch"
    DISABLE_EXPLAIN = "disable_explain"
    SHADOW_TRAFFIC = "shadow_traffic"
    ESSENTIAL_ONLY = "essential_only"


class DegradationManager:
    """
    Manages graceful degradation modes for services.

    Provides controlled reduction of service capabilities during
    high load, errors, or resource constraints to maintain core functionality.
    """

    def __init__(self):
        """Initialize degradation manager."""
        self._active_modes: Dict[
            str, Set[str]
        ] = {}  # service_id -> set of active mode_ids
        self._mode_configs: Dict[str, Dict] = {}  # service_id -> degradation config
        self._mode_history: List[Dict] = []  # History of mode changes
        self._event_handlers: Dict[str, List] = {}  # Event handlers for mode changes

        logger.debug("Degradation manager initialized")

    def configure_modes(self, service_id: str, degradation_modes: List[Dict]):
        """
        Configure degradation modes for a service.

        Args:
            service_id: Service identifier
            degradation_modes: List of degradation mode configurations
        """
        self._mode_configs[service_id] = {
            mode["mode_id"]: mode for mode in degradation_modes
        }

        if service_id not in self._active_modes:
            self._active_modes[service_id] = set()

        logger.info(
            f"Configured {len(degradation_modes)} degradation modes for {service_id}"
        )

    async def enter_mode(
        self, service_id: str, mode_id: str, reason: str = "manual"
    ) -> bool:
        """
        Enter a degradation mode.

        Args:
            service_id: Service identifier
            mode_id: Degradation mode identifier
            reason: Reason for entering mode

        Returns:
            True if mode was entered successfully
        """
        try:
            if service_id not in self._mode_configs:
                logger.warning(f"No degradation config found for service {service_id}")
                return False

            if mode_id not in self._mode_configs[service_id]:
                logger.warning(
                    f"Unknown degradation mode {mode_id} for service {service_id}"
                )
                return False

            # Check if already in this mode
            if mode_id in self._active_modes.get(service_id, set()):
                logger.info(
                    f"Service {service_id} already in degradation mode {mode_id}"
                )
                return True

            # Get mode configuration
            mode_config = self._mode_configs[service_id][mode_id]

            # Execute mode entry actions
            success = await self._execute_actions(
                service_id, mode_config.get("actions", []), "enter"
            )

            if success:
                if service_id not in self._active_modes:
                    self._active_modes[service_id] = set()

                self._active_modes[service_id].add(mode_id)

                # Record mode change
                self._record_mode_change(service_id, mode_id, "entered", reason)

                # Notify event handlers
                await self._notify_handlers("mode_entered", service_id, mode_id)

                logger.info(
                    f"Service {service_id} entered degradation mode {mode_id}: {reason}"
                )
                return True
            else:
                logger.error(
                    f"Failed to enter degradation mode {mode_id} for service {service_id}"
                )
                return False

        except Exception as e:
            logger.error(
                f"Error entering degradation mode {mode_id} for service {service_id}: {e}"
            )
            return False

    async def exit_mode(
        self, service_id: str, mode_id: str, reason: str = "manual"
    ) -> bool:
        """
        Exit a degradation mode.

        Args:
            service_id: Service identifier
            mode_id: Degradation mode identifier
            reason: Reason for exiting mode

        Returns:
            True if mode was exited successfully
        """
        try:
            if (
                service_id not in self._active_modes
                or mode_id not in self._active_modes[service_id]
            ):
                logger.info(f"Service {service_id} not in degradation mode {mode_id}")
                return True

            # Get mode configuration
            if (
                service_id in self._mode_configs
                and mode_id in self._mode_configs[service_id]
            ):
                mode_config = self._mode_configs[service_id][mode_id]

                # Execute mode exit actions (reverse of entry actions)
                success = await self._execute_actions(
                    service_id, mode_config.get("actions", []), "exit"
                )
            else:
                success = True  # No config to reverse

            if success:
                self._active_modes[service_id].discard(mode_id)

                # Record mode change
                self._record_mode_change(service_id, mode_id, "exited", reason)

                # Notify event handlers
                await self._notify_handlers("mode_exited", service_id, mode_id)

                logger.info(
                    f"Service {service_id} exited degradation mode {mode_id}: {reason}"
                )
                return True
            else:
                logger.error(
                    f"Failed to exit degradation mode {mode_id} for service {service_id}"
                )
                return False

        except Exception as e:
            logger.error(
                f"Error exiting degradation mode {mode_id} for service {service_id}: {e}"
            )
            return False

    def get_active_modes(self, service_id: str) -> Set[str]:
        """Get active degradation modes for a service."""
        return self._active_modes.get(service_id, set()).copy()

    def is_in_mode(self, service_id: str, mode_id: str) -> bool:
        """Check if service is in a specific degradation mode."""
        return mode_id in self._active_modes.get(service_id, set())

    def get_stats(self, service_id: Optional[str] = None) -> Dict:
        """Get degradation statistics."""
        if service_id:
            return {
                "service_id": service_id,
                "active_modes": list(self._active_modes.get(service_id, set())),
                "configured_modes": list(self._mode_configs.get(service_id, {}).keys()),
                "total_mode_changes": len(
                    [h for h in self._mode_history if h["service_id"] == service_id]
                ),
            }

        return {
            "total_services": len(self._mode_configs),
            "services_with_active_modes": len(
                [s for s, modes in self._active_modes.items() if modes]
            ),
            "total_mode_changes": len(self._mode_history),
            "services": {
                service_id: self.get_stats(service_id)
                for service_id in self._mode_configs
            },
        }

    def get_mode_history(
        self, service_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Get degradation mode change history."""
        history = self._mode_history

        if service_id:
            history = [h for h in history if h["service_id"] == service_id]

        return history[-limit:] if limit else history

    def register_handler(self, event_type: str, handler):
        """Register an event handler for mode changes."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def evaluate_triggers(
        self, service_id: str, signals: Dict[str, float]
    ) -> List[str]:
        """
        Evaluate if any degradation modes should be triggered based on signals.

        Args:
            service_id: Service identifier
            signals: Current signal values (e.g., {"latency": 1200, "error_rate": 5.5})

        Returns:
            List of mode IDs that should be entered
        """
        if service_id not in self._mode_configs:
            return []

        triggered_modes = []

        for mode_id, mode_config in self._mode_configs[service_id].items():
            if self.is_in_mode(service_id, mode_id):
                continue  # Already in this mode

            triggers = mode_config.get("triggers", [])
            should_trigger = False

            for trigger in triggers:
                if trigger == "high_latency" and signals.get("latency_p95_ms", 0) > 800:
                    should_trigger = True
                elif trigger == "high_error" and signals.get("error_rate_pct", 0) > 5.0:
                    should_trigger = True
                elif trigger == "drift" and signals.get("drift_psi", 0) > 0.1:
                    should_trigger = True
                elif (
                    trigger == "dependency_down"
                    and signals.get("dependency_health", 1.0) < 0.5
                ):
                    should_trigger = True
                elif (
                    trigger == "low_budget"
                    and signals.get("error_budget_remaining_pct", 100) < 10
                ):
                    should_trigger = True

            if should_trigger:
                triggered_modes.append(mode_id)

        return triggered_modes

    async def _execute_actions(
        self, service_id: str, actions: List[str], direction: str
    ) -> bool:
        """Execute degradation mode actions."""
        try:
            for action in actions:
                if direction == "enter":
                    await self._execute_enter_action(service_id, action)
                else:
                    await self._execute_exit_action(service_id, action)
            return True
        except Exception as e:
            logger.error(f"Failed to execute {direction} actions for {service_id}: {e}")
            return False

    async def _execute_enter_action(self, service_id: str, action: str):
        """Execute a degradation entry action."""
        if action == "disable_explain":
            logger.info(f"Disabling explanations for {service_id}")
            # Would integrate with ML/DL kernel to disable explanations
        elif action == "use_cache":
            logger.info(f"Switching to cached responses for {service_id}")
            # Would integrate with caching layer
        elif action == "reduce_batch":
            logger.info(f"Reducing batch size for {service_id}")
            # Would integrate with processing kernel
        elif action == "shed_load":
            logger.info(f"Shedding load for {service_id}")
            # Would integrate with load balancer
        else:
            logger.warning(f"Unknown degradation action: {action}")

        # Simulate action execution delay
        await asyncio.sleep(0.1)

    async def _execute_exit_action(self, service_id: str, action: str):
        """Execute a degradation exit action (reverse of entry)."""
        if action == "disable_explain":
            logger.info(f"Re-enabling explanations for {service_id}")
        elif action == "use_cache":
            logger.info(f"Switching back to live responses for {service_id}")
        elif action == "reduce_batch":
            logger.info(f"Restoring normal batch size for {service_id}")
        elif action == "shed_load":
            logger.info(f"Restoring normal load for {service_id}")

        # Simulate action execution delay
        await asyncio.sleep(0.1)

    def _record_mode_change(
        self, service_id: str, mode_id: str, action: str, reason: str
    ):
        """Record a mode change in history."""
        self._mode_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "service_id": service_id,
                "mode_id": mode_id,
                "action": action,
                "reason": reason,
            }
        )

        # Keep history limited to prevent memory growth
        if len(self._mode_history) > 1000:
            self._mode_history = self._mode_history[-800:]  # Keep last 800 entries

    async def _notify_handlers(self, event_type: str, service_id: str, mode_id: str):
        """Notify registered event handlers."""
        handlers = self._event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(service_id, mode_id)
                else:
                    handler(service_id, mode_id)
            except Exception as e:
                logger.error(f"Error in degradation event handler: {e}")


# Global degradation manager instance
_degradation_manager = DegradationManager()


async def enter_mode(
    service_id: str, mode_id: str, reason: str = "api_request"
) -> None:
    """Global function to enter degradation mode."""
    await _degradation_manager.enter_mode(service_id, mode_id, reason)


async def exit_mode(service_id: str, mode_id: str, reason: str = "api_request") -> None:
    """Global function to exit degradation mode."""
    await _degradation_manager.exit_mode(service_id, mode_id, reason)


def get_degradation_manager() -> DegradationManager:
    """Get the global degradation manager instance."""
    return _degradation_manager


# Backwards-compatible alias expected by older tests
DegradationController = DegradationManager
