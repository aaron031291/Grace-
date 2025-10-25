"""
Workflow Registry for TriggerMesh Orchestration Layer.

Loads, validates, and provides access to workflows defined in YAML files.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

logger = logging.getLogger(__name__)

# This was previously in event_router.py, moving it here as it's a core model
class EventFilter:
    """A filter to determine if an event should be processed by a workflow."""
    def __init__(self, criteria: Dict[str, Any]):
        self.criteria = criteria

    def match(self, event_data: Dict[str, Any]) -> bool:
        """Checks if the event data matches the filter criteria."""
        for key, value in self.criteria.items():
            if event_data.get(key) != value:
                return False
        return True


@dataclass
class WorkflowTrigger:
    """Defines what event triggers a workflow."""
    event_type: str
    filter_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowAction:
    """Defines a single action to be executed in a workflow."""
    action_type: str  # e.g., 'call_kernel', 'dispatch_event'
    target: str       # e.g., 'learning_kernel'
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Workflow:
    """Represents a complete, event-driven workflow."""
    name: str
    trigger: WorkflowTrigger
    actions: List[WorkflowAction]


class WorkflowRegistry:
    """
    Manages the loading and retrieval of workflow definitions.
    """
    def __init__(self, workflow_dir: str = "grace/workflows"):
        self.logger = logging.getLogger(__name__)
        self.workflow_dir = workflow_dir
        self.workflows = []
        self.logger.info(f"Workflow Registry initializing from directory: {self.workflow_dir}")
        self._load_workflows()

    def _load_module_from_path(self, path: str):
        """Load a Python module from a file path."""
        import importlib.util
        from types import ModuleType
        
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(f"grace.workflows.{name}", path)
        if not spec or not spec.loader:
            self.logger.warning(f"Skipping module (no spec): {path}")
            return None
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            self.logger.exception(f"Failed to import workflow module {path}: {e}")
            return None

    def _load_workflows(self):
        """Load all workflows from the configured directory."""
        base = os.path.abspath(self.workflow_dir)
        if not os.path.isdir(base):
            self.logger.warning(f"Workflow directory not found: {base}. Creating it.")
            os.makedirs(base, exist_ok=True)
            return

        found = 0
        loaded = 0
        
        for fname in os.listdir(base):
            # Load Python workflow modules
            if fname.endswith(".py") and fname != "__init__.py":
                found += 1
                m = self._load_module_from_path(os.path.join(base, fname))
                if not m:
                    continue
                wf = getattr(m, "workflow", None)
                if wf is None:
                    self.logger.warning(f"Module {fname} has no `workflow` symbol; skipping")
                    continue
                # Minimal interface check for dict or object forms
                if isinstance(wf, dict):
                    if "name" not in wf or "execute" not in wf:
                        self.logger.warning(f"Workflow dict in {fname} missing keys; skipping")
                        continue
                else:
                    if not hasattr(wf, "name") or not hasattr(wf, "execute"):
                        self.logger.warning(f"Workflow object in {fname} missing attributes; skipping")
                        continue
                self.workflows.append(wf)
                loaded += 1
                self.logger.info(f"Loaded Python workflow: {wf.name} for events: {wf.EVENTS}")
            
            # Load YAML workflows
            elif fname.endswith((".yml", ".yaml")):
                found += 1
                filepath = os.path.join(base, fname)
                try:
                    with open(filepath, 'r') as f:
                        workflow_data = yaml.safe_load(f)
                        if workflow_data:
                            self._register_workflow(workflow_data)
                            loaded += 1
                except Exception as e:
                    self.logger.error(f"Failed to load YAML workflow from {filepath}: {e}")

        self.logger.info(f"WorkflowRegistry loaded {loaded}/{found} workflow modules.")
        if loaded:
            workflow_names = [getattr(wf, "name", str(wf)) for wf in self.workflows]
            self.logger.info(f"Workflows available: {workflow_names}")

    def load_workflows_from_directory(self):
        """
        Loads all .yml, .yaml files and .py workflow modules from the configured directory.
        """
        if not os.path.isdir(self.workflow_dir):
            logger.warning(f"Workflow directory not found: {self.workflow_dir}. Creating it.")
            os.makedirs(self.workflow_dir, exist_ok=True)
            return

        # Load YAML workflows
        for filename in os.listdir(self.workflow_dir):
            if filename.endswith((".yml", ".yaml")):
                filepath = os.path.join(self.workflow_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        workflow_data = yaml.safe_load(f)
                        if workflow_data:  # Ensure file is not empty
                            self._register_workflow(workflow_data)
                except Exception as e:
                    logger.error(f"Failed to load workflow from {filepath}: {e}")
            
            # Load Python workflow modules
            elif filename.endswith(".py") and not filename.startswith("_"):
                try:
                    import importlib.util
                    import sys
                    
                    module_name = filename[:-3]  # Remove .py
                    filepath = os.path.join(self.workflow_dir, filename)
                    
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # Look for 'workflow' object in the module
                    if hasattr(module, 'workflow'):
                        wf = module.workflow
                        if hasattr(wf, 'name') and hasattr(wf, 'EVENTS') and hasattr(wf, 'execute'):
                            self.workflows.append(wf)
                            logger.info(f"Registered Python workflow: {wf.name} for events: {wf.EVENTS}")
                except Exception as e:
                    logger.error(f"Failed to load Python workflow from {filepath}: {e}")

    def _load_workflow_file(self, file_path: Path):
        """Load a single workflow file."""
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                if not data or "workflows" not in data:
                    logger.warning(f"No 'workflows' key found in {file_path}, skipping.")
                    return

                for workflow_data in data["workflows"]:
                    self._register_workflow(workflow_data)
        except yaml.YAMLError as e:
            logger.exception(f"Error parsing YAML file: {file_path}")
        except Exception:
            logger.exception(f"Error loading workflow file: {file_path}")

    def _register_workflow(self, data: Dict[str, Any]):
        """Parses workflow data and registers it."""
        try:
            trigger_data = data['trigger']
            trigger = WorkflowTrigger(
                event_type=trigger_data['type'],
                filter_criteria=trigger_data.get('filter', {})
            )
            
            actions = [WorkflowAction(**a) for a in data['actions']]
            
            workflow = Workflow(
                name=data['name'],
                trigger=trigger,
                actions=actions
            )
            self.workflows.append(workflow)
            logger.info(f"Registered workflow: {workflow.name}")
        except KeyError as e:
            logger.error(f"Invalid workflow definition. Missing key: {e}")

    def find_workflows_for_event(self, event_type: str, payload: Dict[str, Any]) -> List:
        """
        Finds all workflows that should be triggered by a given event.
        Handles both YAML-based workflows (with filters) and Python-based workflows (with EVENTS list).
        """
        matches = [wf for wf in self.workflows if event_type in getattr(wf, "EVENTS", [])]
        
        # Also check YAML-based workflows
        for wf in self.workflows:
            if hasattr(wf, 'trigger') and wf.trigger.event_type == event_type:
                event_filter = EventFilter(wf.trigger.filter_criteria)
                if event_filter.match(payload):
                    if wf not in matches:
                        matches.append(wf)
        
        if matches:
            self.logger.info(
                f"find_workflows_for_event: event_type={event_type} -> {[wf.name for wf in matches]}"
            )
        else:
            self.logger.info(f"find_workflows_for_event: event_type={event_type} -> NONE")
        
        return matches

    def get_all_workflows(self) -> List[Workflow]:
        """Return a list of all loaded workflows."""
        return self.workflows

    def get_workflow_by_name(self, name: str) -> Workflow | None:
        """Return a single workflow by its name."""
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow
        return None

    def get_all_trigger_event_types(self) -> Set[str]:
        """Return a set of all unique event types that can trigger workflows."""
        return {workflow.trigger.event_type for workflow in self.workflows}
