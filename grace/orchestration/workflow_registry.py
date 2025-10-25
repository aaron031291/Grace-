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
    def __init__(self, workflow_dir: str | Path):
        self.workflow_dir = Path(workflow_dir)
        self.workflows: List[Workflow] = []
        logger.info("Workflow Registry initialized.")

    def load_workflows(self):
        """Load all YAML workflow files from the specified directory."""
        logger.info(f"Loading workflows from: {self.workflow_dir}")
        self.workflows = []

        if not self.workflow_dir.is_dir():
            logger.error(f"Workflow directory not found: {self.workflow_dir}")
            return

        for yaml_file in self.workflow_dir.glob("*.yaml"):
            self._load_workflow_file(yaml_file)

        logger.info(f"Loaded {len(self.workflows)} workflows.")

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

    def find_workflows_for_event(self, event_type: str, payload: Dict[str, Any]) -> List[Workflow]:
        """
        Finds all workflows that should be triggered by a given event.
        """
        matching = []
        for wf in self.workflows:
            if wf.trigger.event_type == event_type:
                event_filter = EventFilter(wf.trigger.filter_criteria)
                if event_filter.match(payload):
                    matching.append(wf)
        return matching

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
