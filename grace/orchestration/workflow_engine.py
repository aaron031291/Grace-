# -*- coding: utf-8 -*-
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
    Tolerant engine that correctly initializes and executes workflows.
    - __init__(registry, *args, **kwargs)
    - Resolves modules via WorkflowRegistry.get_module(name)
    - Runs async handlers directly; sync via default executor
    """
    def __init__(self, registry, *args, **kwargs):
        if registry is None:
            raise TypeError("WorkflowEngine requires a 'registry' instance.")
        self.registry = registry
        self.workflow_registry = registry.get('workflow_registry')

    async def execute_workflow(self, workflow_name: str, event: dict):
        """
        Executes a single workflow handler.
        """
        module = self.workflow_registry.get_module(workflow_name)
        if not module:
            raise RuntimeError(f"Workflow '{workflow_name}' not registered or loaded")
        
        handler = getattr(module, "handle", None)
        if not callable(handler):
            raise RuntimeError(f"Workflow '{workflow_name}' has no callable 'handle' function")
        
        # Execute handler, supporting both sync and async functions
        if asyncio.iscoroutinefunction(handler):
            return await handler(event, self.registry)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, handler, event, self.registry)
