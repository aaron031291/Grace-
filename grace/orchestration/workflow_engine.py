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
import inspect

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

        Supports: {{ payload.field }}, {{ event_type }}, {{ correlation_id }}
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
        pattern = r"\{\{\s*([^}]+)\s*\}\}"

        # Handle the case where the entire string is a variable, to preserve type
        match = re.fullmatch(pattern, template.strip())
        if match:
            path = match.group(1).strip()
            return self._resolve_path(path, context)

        # For string interpolation
        def replacer(m):
            path = m.group(1).strip()
            value = self._resolve_path(path, context)
            return str(value) if value is not None else ""

        return re.sub(pattern, replacer, template)


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
    Executes the actions defined in a workflow.
    """

    def __init__(self, service_registry):
        """
        Initialize the WorkflowEngine with access to the service registry.
        """
        self.service_registry = service_registry
        logger.info("Workflow Engine initialized with service registry.")

        # Statistics
        self.stats = {
            "workflows_executed": 0,
            "actions_executed": 0,
            "actions_succeeded": 0,
            "actions_failed": 0,
            "ooda_loops_completed": 0,
        }

    def register_kernel_handler(self, name: str, handler: Callable):
        """Register a function from a kernel that can be called as a workflow action."""
        if name in self.kernel_handlers:
            logger.warning(f"Overwriting kernel handler for action: {name}")
        self.kernel_handlers[name] = handler
        logger.info(f"Registered kernel handler for action: {name}")

    async def execute_workflow(self, workflow, event_context: Dict[str, Any]):
        """
        Execute a workflow with full immutable logging phases:
        HANDLER_EXECUTED → HANDLER_COMMITTED (or FAILED)
        """
        # Get immutable logger
        immutable_logger = self.service_registry.get('immutable_logger') if self.service_registry else None
        
        # Normalize interface
        if isinstance(workflow, dict):
            workflow_name = workflow.get("name", "UNKNOWN")
            
            # Check if this is a YAML workflow with actions
            if "actions" in workflow:
                return await self._execute_yaml_workflow(workflow, event_context, immutable_logger)
            
            # Otherwise it should have an execute function
            execute_fn = workflow.get("execute")
        else:
            workflow_name = getattr(workflow, "name", workflow.__class__.__name__)
            execute_fn = getattr(workflow, "execute", None)

        if execute_fn is None or not callable(execute_fn):
            logger.error(f"Workflow {workflow_name} missing callable 'execute' and has no 'actions'; skipping")
            if immutable_logger:
                immutable_logger.append_phase(
                    event=event_context,
                    phase="FAILED",
                    status="error",
                    metadata={"workflow": workflow_name, "error": "missing callable execute"}
                )
            raise TypeError(f"Workflow {workflow_name} missing callable 'execute'")

        event_id = event_context.get("id", "unknown")
        
        # Phase: HANDLER_EXECUTED
        logger.info(f"Executing workflow={workflow_name} event_id={event_id}")
        if immutable_logger:
            immutable_logger.append_phase(
                event=event_context,
                phase="HANDLER_EXECUTED",
                status="ok",
                metadata={"workflow": workflow_name}
            )
        
        try:
            result = await execute_fn(event_context)
            
            # Phase: HANDLER_COMMITTED
            logger.info(f"Workflow {workflow_name} done event_id={event_id}")
            if immutable_logger:
                immutable_logger.append_phase(
                    event=event_context,
                    phase="HANDLER_COMMITTED",
                    status="ok",
                    metadata={"workflow": workflow_name, "result": str(result)[:200]}
                )
            
            return result
        except Exception as e:
            # Phase: FAILED
            logger.exception(f"Workflow {workflow_name} failed event_id={event_id}: {e}")
            if immutable_logger:
                immutable_logger.append_phase(
                    event=event_context,
                    phase="FAILED",
                    status="error",
                    metadata={"workflow": workflow_name, "error": str(e)}
                )
            raise

    async def _execute_yaml_workflow(self, workflow: Dict[str, Any], event_context: Dict[str, Any], immutable_logger):
        """
        Execute a YAML-based workflow with actions and full phase logging.
        """
        workflow_name = workflow.get("name", "UNKNOWN")
        event_id = event_context.get("id", "unknown")
        actions = workflow.get("actions", [])
        
        logger.info(f"Executing YAML workflow={workflow_name} with {len(actions)} actions, event_id={event_id}")
        
        # Phase: HANDLER_EXECUTED
        if immutable_logger:
            immutable_logger.append_phase(
                event=event_context,
                phase="HANDLER_EXECUTED",
                status="ok",
                metadata={"workflow": workflow_name, "action_count": len(actions)}
            )
        
        results = []
        try:
            for action in actions:
                action_type = action.get("action_type", "unknown")
                target = action.get("target", "unknown")
                
                logger.info(f"  Action: {action_type} -> {target}")
                
                if action_type == "call_kernel" and self.service_registry:
                    kernel = self.service_registry.get(target)
                    if kernel and hasattr(kernel, 'execute'):
                        try:
                            task = {**action.get("params", {}), "triggering_event": event_context}
                            result = await kernel.execute(task)
                            results.append({"action": target, "status": "success", "result": result})
                        except Exception as e:
                            logger.error(f"    Kernel {target} failed: {e}")
                            results.append({"action": target, "status": "failed", "error": str(e)})
                    else:
                        logger.warning(f"    Kernel {target} not found or not executable")
                        results.append({"action": target, "status": "skipped", "reason": "kernel not found"})
            
            logger.info(f"YAML workflow {workflow_name} completed with {len(results)} action results")
            
            # Phase: HANDLER_COMMITTED
            if immutable_logger:
                immutable_logger.append_phase(
                    event=event_context,
                    phase="HANDLER_COMMITTED",
                    status="ok",
                    metadata={"workflow": workflow_name, "actions_completed": len(results)}
                )
            
            return {"status": "completed", "actions": results}
            
        except Exception as e:
            # Phase: FAILED
            logger.exception(f"YAML workflow {workflow_name} failed: {e}")
            if immutable_logger:
                immutable_logger.append_phase(
                    event=event_context,
                    phase="FAILED",
                    status="error",
                    metadata={"workflow": workflow_name, "error": str(e)}
                )
            raise

        # ORIENT & DECIDE: The selection of this workflow by the router
        # based on the event is the orientation and decision step.
        logger.info(
            f"Executing workflow: {workflow_name}",
            extra={"correlation_id": correlation_id},
        )
        await self.event_bus.publish(
            "workflow.started",
            {
                "workflow_name": workflow_name,
                "trigger_event": event_context["event_type"],
                "correlation_id": correlation_id,
            },
        )

        actions = workflow.get("actions", [])
        results = []
        for action in actions:
            try:
                # ACT: Execute the individual action.
                result = await self._execute_action(workflow_name, action, event_context)
                results.append(result)

                if result["status"] == "SUCCESS":
                    self.stats["actions_succeeded"] += 1
                else:
                    self.stats["actions_failed"] += 1

            except Exception as e:
                logger.error(
                    f"Action {action['name']} failed: {e}",
                    extra={"correlation_id": correlation_id},
                )
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

        # Log workflow completion, ensuring the cryptographic key is central.
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
                correlation_id=correlation_id,
            )

        self.stats["workflows_executed"] += 1
        self.stats["ooda_loops_completed"] += 1

        logger.info(
            f"Workflow {workflow_name} completed: {status} "
            f"({successful_actions}/{total_actions} actions succeeded, "
            f"{total_latency_ms:.1f}ms)",
            extra={"correlation_id": correlation_id},
        )

        return {"status": status, "results": results}

    async def _execute_action(
        self, workflow_name: str, action: Dict[str, Any], event_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single action from a workflow."""
        start_time = datetime.now()
        action_name = action.get("name")
        target = action.get("target")
        correlation_id = event_context.get("correlation_id")

        # Substitute parameters from the full event context
        params = self.parameter_substitutor.substitute(action.get("params", {}), event_context)
        timeout = action.get("timeout_seconds", 10)

        handler = self.kernel_handlers.get(target)
        if not handler:
            error_msg = f"No kernel handler found for target: {target} in workflow: {workflow_name}"
            logger.error(error_msg, extra={"correlation_id": correlation_id})
            return {"action": action_name, "status": "FAILED", "error": error_msg, "latency_ms": 0}

        logger.debug(
            f"Executing action '{action_name}' with target '{target}' and params: {params}",
            extra={"correlation_id": correlation_id},
        )

        status = "UNKNOWN"
        error = None
        try:
            # Pass correlation_id to kernel handlers that accept it
            sig = inspect.signature(handler)
            if "correlation_id" in sig.parameters:
                params["correlation_id"] = correlation_id

            await asyncio.wait_for(handler(**params), timeout=timeout)
            status = "SUCCESS"
        except asyncio.TimeoutError as e:
            status = "TIMEOUT"
            error = str(e)
            logger.error(
                f"Action '{action_name}' in workflow '{workflow_name}' timed out.",
                extra={"correlation_id": correlation_id},
            )
        except Exception as e:
            status = "FAILURE"
            error = str(e)
            logger.exception(
                f"Action '{action_name}' in workflow '{workflow_name}' failed.",
                extra={"correlation_id": correlation_id},
            )

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.stats["actions_executed"] += 1

        await self.event_bus.publish(
            "workflow.action_executed",
            {
                "workflow_name": workflow_name,
                "action_name": action_name,
                "status": status,
                "error": error,
                "correlation_id": correlation_id,
            },
        )

        return {
            "action": action_name,
            "status": status,
            "error": error,
            "latency_ms": latency_ms,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "registered_kernels": list(self.kernel_handlers.keys()),
        }
