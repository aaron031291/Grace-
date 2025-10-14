"""
Workflow Engine - Executes TriggerMesh workflows by routing actions to kernels.

Handles parameter substitution, parallel execution, error handling, and logging.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import json

from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogs, TransparencyLevel
from grace.core.kpi_trust_monitor import KPITrustMonitor

logger = logging.getLogger(__name__)


class ParameterSubstitutor:
    """Substitutes template variables in workflow parameters."""

    def substitute(
        self, template: Any, context: Dict[str, Any]
    ) -> Any:
        """
        Recursively substitute template variables.

        Supports: {{ payload.field }}, {{ event.field }}
        """
        if isinstance(template, str):
            return self._substitute_string(template, context)
        elif isinstance(template, dict):
            return {k: self.substitute(v, context) for k, v in template.items()}
        elif isinstance(template, list):
            return [self.substitute(item, context) for item in template]
        else:
            return template

    def _substitute_string(self, template: str, context: Dict[str, Any]) -> Any:
        """Substitute variables in a single string."""
        # Find all {{ variable }} patterns
        pattern = r"\{\{\s*([^}]+)\s*\}\}"

        def replacer(match):
            path = match.group(1).strip()
            value = self._resolve_path(path, context)

            # If entire string is just the template, return raw value
            if match.group(0) == template:
                return value

            # Otherwise, convert to string for interpolation
            return str(value) if value is not None else ""

        result = re.sub(pattern, replacer, template)

        # If entire string was a template, return the resolved value type
        if template.startswith("{{") and template.endswith("}}"):
            path = template[2:-2].strip()
            return self._resolve_path(path, context)

        return result

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """Resolve a dotted path in the context."""
        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value


class WorkflowEngine:
    """
    Executes workflows by routing actions to appropriate kernels.
    Handles parallel execution, error handling, and result logging.
    """

    def __init__(
        self,
        event_bus: EventBus,
        immutable_logs: ImmutableLogs,
        kpi_monitor: Optional[KPITrustMonitor] = None,
    ):
        self.event_bus = event_bus
        self.immutable_logs = immutable_logs
        self.kpi_monitor = kpi_monitor
        self.parameter_substitutor = ParameterSubstitutor()

        # Kernel registry (maps kernel names to handlers)
        self.kernel_handlers: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            "workflows_executed": 0,
            "actions_executed": 0,
            "actions_succeeded": 0,
            "actions_failed": 0,
        }

    def register_kernel(self, kernel_name: str, handler: Any):
        """Register a kernel handler for workflow actions."""
        self.kernel_handlers[kernel_name] = handler
        logger.info(f"Registered kernel: {kernel_name}")

    async def execute_workflow(self, workflow, event: Dict[str, Any]):
        """Execute a complete workflow."""
        start_time = datetime.now()
        workflow_name = workflow.name
        correlation_id = event.get("correlation_id")

        logger.info(f"Executing workflow: {workflow_name}")

        # Build execution context
        context = {
            "event": event,
            "payload": event.get("payload", {}),
            "workflow": {"name": workflow_name, "version": workflow.version},
        }

        # Execute all actions
        results = []
        for action in workflow.actions:
            try:
                result = await self._execute_action(action, context)
                results.append(result)

                if result["status"] == "SUCCESS":
                    self.stats["actions_succeeded"] += 1
                else:
                    self.stats["actions_failed"] += 1

            except Exception as e:
                logger.error(f"Action {action.name} failed: {e}")
                results.append(
                    {
                        "action": action.name,
                        "status": "FAILED",
                        "error": str(e),
                        "latency_ms": 0,
                    }
                )
                self.stats["actions_failed"] += 1

        # Calculate workflow results
        total_actions = len(results)
        successful_actions = sum(1 for r in results if r["status"] == "SUCCESS")
        end_time = datetime.now()
        total_latency_ms = (end_time - start_time).total_seconds() * 1000

        # Determine overall status
        if successful_actions == total_actions:
            status = "SUCCESS"
        elif successful_actions > 0:
            status = "PARTIAL_SUCCESS"
        else:
            status = "FAILED"

        # Log workflow completion
        await self.immutable_logs.log_event(
            event_type="workflow.completed",
            component_id="trigger_mesh",
            event_data={
                "workflow_name": workflow_name,
                "status": status,
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "total_latency_ms": total_latency_ms,
                "action_results": results,
            },
            correlation_id=correlation_id,
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        # Record metrics
        if self.kpi_monitor:
            await self.kpi_monitor.record_metric(
                name="workflow_latency",
                value=total_latency_ms,
                component_id="trigger_mesh",
                tags={"workflow": workflow_name, "status": status},
            )

        self.stats["workflows_executed"] += 1

        logger.info(
            f"Workflow {workflow_name} completed: {status} "
            f"({successful_actions}/{total_actions} actions succeeded, "
            f"{total_latency_ms:.1f}ms)"
        )

        return {"status": status, "results": results}

    async def _execute_action(
        self, action, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow action."""
        start_time = datetime.now()
        action_name = action.name
        target_kernel = action.target_kernel

        logger.debug(f"Executing action: {action_name} → {target_kernel}")

        # Substitute parameters
        parameters = self.parameter_substitutor.substitute(
            action.parameters, context
        )

        # Get kernel handler
        if target_kernel not in self.kernel_handlers:
            # If kernel not registered, publish event to event bus
            logger.warning(
                f"Kernel {target_kernel} not registered, publishing event instead"
            )
            result = await self._publish_kernel_event(
                target_kernel, action.action, parameters
            )
        else:
            # Call kernel handler directly
            handler = self.kernel_handlers[target_kernel]
            result = await self._call_kernel_handler(
                handler, action.action, parameters, action.timeout_ms
            )

        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # Build result
        action_result = {
            "action": action_name,
            "target_kernel": target_kernel,
            "status": result.get("status", "SUCCESS"),
            "latency_ms": latency_ms,
        }

        if "error" in result:
            action_result["error"] = result["error"]

        # Log action execution
        await self.immutable_logs.log_event(
            event_type="workflow.action_executed",
            component_id="trigger_mesh",
            event_data={
                "workflow_name": context["workflow"]["name"],
                "action_name": action_name,
                "target_kernel": target_kernel,
                "result": action_result["status"],
                "latency_ms": latency_ms,
            },
            correlation_id=context["event"].get("correlation_id"),
            transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
        )

        # Handle on_success / on_failure events
        if action_result["status"] == "SUCCESS" and hasattr(action, "on_success"):
            await self._handle_action_events(action.on_success, context, action_result)
        elif action_result["status"] != "SUCCESS" and hasattr(action, "on_failure"):
            await self._handle_action_events(action.on_failure, context, action_result)

        self.stats["actions_executed"] += 1

        return action_result

    async def _call_kernel_handler(
        self, handler: Any, action: str, parameters: Dict[str, Any], timeout_ms: int
    ) -> Dict[str, Any]:
        """Call a registered kernel handler."""
        try:
            # Get the action method from handler
            if hasattr(handler, action):
                method = getattr(handler, action)

                # Call with timeout
                if asyncio.iscoroutinefunction(method):
                    result = await asyncio.wait_for(
                        method(**parameters), timeout=timeout_ms / 1000
                    )
                else:
                    result = method(**parameters)

                return {"status": "SUCCESS", "result": result}
            else:
                return {
                    "status": "FAILED",
                    "error": f"Action {action} not found on kernel",
                }

        except asyncio.TimeoutError:
            return {"status": "TIMEOUT", "error": f"Action timed out after {timeout_ms}ms"}
        except Exception as e:
            logger.error(f"Kernel handler error: {e}")
            return {"status": "FAILED", "error": str(e)}

    async def _publish_kernel_event(
        self, kernel: str, action: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish event to event bus for unregistered kernel."""
        try:
            event_type = f"kernel.{kernel}.{action}"
            await self.event_bus.publish(event_type, parameters)

            return {"status": "SUCCESS", "result": "event_published"}

        except Exception as e:
            logger.error(f"Failed to publish kernel event: {e}")
            return {"status": "FAILED", "error": str(e)}

    async def _handle_action_events(
        self, event_configs: List[Dict], context: Dict[str, Any], action_result: Dict
    ):
        """Handle on_success / on_failure event publishing."""
        for event_config in event_configs:
            try:
                event_type = event_config["event"]

                # Build event context with action result
                event_context = {
                    **context,
                    "action": action_result,
                }

                # Substitute payload parameters
                payload = self.parameter_substitutor.substitute(
                    event_config["payload"], event_context
                )

                # Publish event
                await self.event_bus.publish(
                    event_type, payload, correlation_id=context["event"].get("correlation_id")
                )

                logger.debug(f"Published action event: {event_type}")

            except Exception as e:
                logger.error(f"Failed to publish action event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "registered_kernels": list(self.kernel_handlers.keys()),
        }
