"""
Workflow Registry for TriggerMesh Orchestration Layer.

Loads, validates, and provides access to workflows defined in YAML files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

logger = logging.getLogger(__name__)


class WorkflowRegistry:
    """Loads and manages workflows from YAML files."""

    def __init__(self, workflow_dir: str | Path):
        self.workflow_dir = Path(workflow_dir)
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.trigger_event_map: Dict[str, List[str]] = {}
        self._stats = {"workflows_loaded": 0, "validation_errors": 0}

    def load_workflows(self):
        """Load all YAML workflow files from the specified directory."""
        logger.info(f"Loading workflows from: {self.workflow_dir}")
        self.workflows = {}
        self.trigger_event_map = {}
        self._stats = {"workflows_loaded": 0, "validation_errors": 0}

        if not self.workflow_dir.is_dir():
            logger.error(f"Workflow directory not found: {self.workflow_dir}")
            return

        for yaml_file in self.workflow_dir.glob("*.yaml"):
            self._load_workflow_file(yaml_file)

        self._build_trigger_map()
        logger.info(f"Loaded {self._stats['workflows_loaded']} workflows.")
        if self._stats["validation_errors"] > 0:
            logger.error(f"Found {self._stats['validation_errors']} validation errors in workflows.")

    def _load_workflow_file(self, file_path: Path):
        """Load a single workflow file."""
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                if not data or "workflows" not in data:
                    logger.warning(f"No 'workflows' key found in {file_path}, skipping.")
                    return

                for workflow in data["workflows"]:
                    if self.validate_workflow(workflow):
                        if workflow["name"] in self.workflows:
                            logger.warning(f"Duplicate workflow name '{workflow['name']}', overwriting.")
                        self.workflows[workflow["name"]] = workflow
                        self._stats["workflows_loaded"] += 1
                    else:
                        self._stats["validation_errors"] += 1
        except yaml.YAMLError as e:
            logger.exception(f"Error parsing YAML file: {file_path}")
            self._stats["validation_errors"] += 1
        except Exception:
            logger.exception(f"Error loading workflow file: {file_path}")
            self._stats["validation_errors"] += 1

    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """Validate the structure of a single workflow definition."""
        required_keys = ["name", "trigger_event", "actions"]
        for key in required_keys:
            if key not in workflow:
                logger.error(f"Validation failed: Missing required key '{key}' in workflow: {workflow.get('name', 'N/A')}")
                return False
        return True

    def _build_trigger_map(self):
        """Build a map from trigger event types to workflow names for fast lookups."""
        self.trigger_event_map = {}
        for name, workflow in self.workflows.items():
            event_type = workflow["trigger_event"]
            if event_type not in self.trigger_event_map:
                self.trigger_event_map[event_type] = []
            self.trigger_event_map[event_type].append(name)

    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Return a list of all loaded workflows."""
        return list(self.workflows.values())

    def get_workflow_by_name(self, name: str) -> Dict[str, Any] | None:
        """Return a single workflow by its name."""
        return self.workflows.get(name)

    def get_workflows_for_event(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all workflows that are triggered by a specific event type."""
        workflow_names = self.trigger_event_map.get(event_type, [])
        return [self.workflows[name] for name in workflow_names]

    def get_all_trigger_event_types(self) -> Set[str]:
        """Return a set of all unique event types that can trigger workflows."""
        return set(self.trigger_event_map.keys())

    def get_stats(self) -> Dict[str, int]:
        """Return loading and validation statistics."""
        return self._stats
