"""
Grace AI - Workflow Registry for TriggerMesh Orchestration Layer
"""
import importlib.util
import logging
import os
import inspect
import asyncio
from types import ModuleType
from typing import List, Any, Dict, Callable
import yaml

logger = logging.getLogger(__name__)


class DictWorkflowAdapter:
    """
    Wrap a dict-like workflow spec to a uniform interface:
      .name        -> str
      .EVENTS      -> list[str]
      .execute(e)  -> async
    Supported dict keys (any capitalization variant is tolerated):
      - name
      - events / EVENTS
      - execute (callable, sync or async)
      - actions (list of callables, sync or async)  # fallback if no execute
    """
    def __init__(self, spec: Dict[str, Any], logger: logging.Logger):
        self._spec = spec
        self.logger = logger
        # tolerant getters
        def g(*keys, default=None):
            for k in keys:
                if k in spec:
                    return spec[k]
            return default
        self.name = g("name", "NAME", default="UNKNOWN")
        self.EVENTS = g("events", "EVENTS", default=[]) or []

        self._execute = g("execute", "EXECUTE")
        self._actions = g("actions", "ACTIONS", default=[])

        if not callable(self._execute) and not self._actions:
            raise TypeError(f"Workflow {self.name} missing callable 'execute' and has no 'actions' list")

    async def _run_callable(self, fn: Callable, event: dict):
        if inspect.iscoroutinefunction(fn):
            return await fn(event)
        # Allow sync callables (run inline — they should be fast)
        return fn(event)

    async def execute(self, event: dict):
        if callable(self._execute):
            return await self._run_callable(self._execute, event)
        # Fallback: run actions in order
        result = None
        for i, act in enumerate(self._actions):
            if not callable(act):
                self.logger.warning("Workflow %s action[%d] is not callable; skipping", self.name, i)
                continue
            result = await self._run_callable(act, event)
        return result


class WorkflowRegistry:
    """
    Manages the loading and retrieval of workflow definitions.
    """
    
    def __init__(self, workflow_dir: str = "grace/workflows"):
        self.logger = logging.getLogger(__name__)
        self.workflow_dir = workflow_dir
        self.workflows: List[Any] = []
        self.logger.info(f"Workflow Registry initializing from directory: {self.workflow_dir}")
        self._load_workflows()

    def _load_module_from_path(self, path: str) -> ModuleType | None:
        mod_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(f"grace.workflows.{mod_name}", path)
        if not spec or not spec.loader:
            self.logger.warning("Skipping module (no spec): %s", path)
            return None
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            self.logger.exception("Failed to import workflow module %s: %s", path, e)
            return None

    def _load_workflows(self):
        base = os.path.abspath(self.workflow_dir)
        if not os.path.isdir(base):
            self.logger.warning("Workflow directory not found: %s", base)
            os.makedirs(base, exist_ok=True)
            return

        found = 0
        loaded = 0
        names: List[str] = []

        for fname in os.listdir(base):
            if not fname.endswith(".py") or fname == "__init__.py":
                continue
            found += 1
            path = os.path.join(base, fname)
            mod = self._load_module_from_path(path)
            if not mod:
                continue

            wf = getattr(mod, "workflow", None)
            if wf is None:
                self.logger.warning("Module %s has no `workflow` symbol; skipping", fname)
                continue

            # Normalize: dict -> adapter; object -> validate
            try:
                if isinstance(wf, dict):
                    adapter = DictWorkflowAdapter(wf, logger=self.logger)
                    self.workflows.append(adapter)
                    names.append(adapter.name)
                    self.logger.info("Registered workflow: %s", adapter.name)
                else:
                    # object must have .name and .execute
                    wname = getattr(wf, "name", wf.__class__.__name__)
                    wexec = getattr(wf, "execute", None)
                    if not callable(wexec):
                        raise TypeError(f"Workflow {wname} missing callable 'execute'")
                    # EVENTS optional but recommended
                    self.workflows.append(wf)
                    names.append(wname)
                    # Optional: show events if present
                    wevents = getattr(wf, "EVENTS", None)
                    if wevents:
                        self.logger.info("Registered workflow: %s (events: %s)", wname, wevents)
                    else:
                        self.logger.info("Registered workflow: %s", wname)
            except Exception as e:
                self.logger.exception("Skipping workflow in %s due to error: %s", fname, e)
                continue
            loaded += 1

        self.logger.info("WorkflowRegistry loaded %d/%d workflow modules.", loaded, found)
        if names:
            self.logger.info("Workflows available: %s", names)

    def find_workflows_for_event(self, event_type: str, payload: dict):
        matches = []
        for wf in self.workflows:
            events = getattr(wf, "EVENTS", [])
            if event_type in events:
                matches.append(wf)
        if matches:
            self.logger.info(
                "find_workflows_for_event: event_type=%s -> %s",
                event_type, [getattr(wf, "name", wf.__class__.__name__) for wf in matches]
            )
        else:
            self.logger.info("find_workflows_for_event: event_type=%s -> NONE", event_type)
        return matches
