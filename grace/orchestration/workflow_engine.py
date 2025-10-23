 """
Workflow Engine for TriggerMesh Orchestration Layer.

Executes the actions defined in a workflow.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogger, TransparencyLevel
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
    """Executes workflow actions by calling registered kernel handlers."""

    def __init__(
        self,
        event_bus: EventBus,
        immutable_logger: ImmutableLogger,
        kpi_monitor: Optional[KPITrustMonitor] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.event_bus = event_bus
        self.logger = immutable_logger
        self.kpi_monitor = kpi_monitor
        self.config = config or {}
        self.kernel_handlers: Dict[str, Callable] = {}
        self.parameter_substitutor = ParameterSubstitutor()

        # Statistics
        self.stats = {
            "workflows_executed": 0,
            "actions_executed": 0,
            "actions_succeeded": 0,
            "actions_failed": 0,
        }

    def register_kernel_handler(self, name: str, handler: Callable):
        """Register a function from a kernel that can be called as a workflow action."""
        if name in self.kernel_handlers:
            logger.warning(f"Overwriting kernel handler for action: {name}")
        self.kernel_handlers[name] = handler
        logger.info(f"Registered kernel handler for action: {name}")

    async def execute_workflow(self, workflow: Dict[str, Any], trigger_payload: Dict[str, Any]):
        """Execute all actions for a given workflow."""
        start_time = datetime.now()
        workflow_name = workflow["name"]
        correlation_id = trigger_payload.get("correlation_id")

        logger.info(f"Executing workflow: {workflow_name}")
        await self.event_bus.publish(
            "workflow.started", {"workflow_name": workflow_name, "trigger_payload": trigger_payload}
        )

        actions = workflow.get("actions", [])
        results = []
        for action in actions:
            try:
                result = await self._execute_action(workflow_name, action, trigger_payload)
                results.append(result)

                if result["status"] == "SUCCESS":
                    self.stats["actions_succeeded"] += 1
                else:
                    self.stats["actions_failed"] += 1

            except Exception as e:
                logger.error(f"Action {action['name']} failed: {e}")
                results.append(
                    {
                        "action": action["name"],
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
        await self.logger.log_event(
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
        self, workflow_name: str, action: Dict[str, Any], trigger_payload: Dict[str, Any]
    ):
        """Execute a single action from a workflow."""
        start_time = datetime.now()
        action_name = action.get("name")
        target = action.get("target")
        params = self.parameter_substitutor.substitute(action.get("params", {}), trigger_payload)
        timeout = action.get("timeout_seconds", 10)

        handler = self.kernel_handlers.get(target)
        if not handler:
            logger.error(f"No kernel handler found for target: {target} in workflow: {workflow_name}")
            return

        logger.debug(f"Executing action '{action_name}' with target '{target}' and params: {params}")

        try:
            await asyncio.wait_for(handler(**params), timeout=timeout)
            await self.event_bus.publish(
                "workflow.action_executed",
                {"workflow_name": workflow_name, "action_name": action_name, "status": "SUCCESS"},
            )
        except asyncio.TimeoutError:
            logger.error(f"Action '{action_name}' in workflow '{workflow_name}' timed out.")
            await self.event_bus.publish(
                "workflow.action_executed",
                {"workflow_name": workflow_name, "action_name": action_name, "status": "TIMEOUT"},
            )
        except Exception as e:
            logger.exception(f"Action '{action_name}' in workflow '{workflow_name}' failed.")
            await self.event_bus.publish(
                "workflow.action_executed",
                {"workflow_name": workflow_name, "action_name": action_name, "status": "FAILURE", "error": str(e)},
            )

    def _resolve_params(self, params: Dict[str, Any], trigger_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template variables in parameters from the trigger payload."""
        resolved_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Simple template substitution, e.g., {{ payload.component_id }}
                parts = value[2:-2].strip().split(".")
                if len(parts) == 2 and parts[0] == "payload":
                    resolved_params[key] = trigger_payload.get(parts[1])
                else:
                    resolved_params[key] = None
            else:
                resolved_params[key] = value
        return resolved_params

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "registered_kernels": list(self.kernel_handlers.keys()),
        }
