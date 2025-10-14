"""
Workflow Registry - Loads, validates, and manages TriggerMesh workflows.

Reads workflow definitions from YAML files and provides workflow lookup by event type.
"""

import yaml
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class WorkflowTrigger:
    """Defines when a workflow should trigger."""

    event_type: str
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowAction:
    """Defines an action to execute as part of a workflow."""

    name: str
    target_kernel: str
    action: str
    priority: str
    timeout_ms: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    on_success: List[Dict[str, Any]] = field(default_factory=list)
    on_failure: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Workflow:
    """Complete workflow definition."""

    name: str
    description: str
    enabled: bool
    version: str
    trigger: WorkflowTrigger
    actions: List[WorkflowAction]
    logging: Dict[str, Any] = field(default_factory=dict)


class WorkflowRegistry:
    """
    Registry for TriggerMesh workflows.

    Loads workflows from YAML files and provides lookup by event pattern.
    """

    def __init__(self):
        self.workflows: List[Workflow] = []
        self.event_type_index: Dict[str, List[Workflow]] = {}
        self.workflow_by_name: Dict[str, Workflow] = {}

    def load_workflows(self, workflow_dir: str):
        """Load all workflow YAML files from a directory."""
        workflow_path = Path(workflow_dir)

        if not workflow_path.exists():
            logger.warning(f"Workflow directory not found: {workflow_dir}")
            return

        yaml_files = list(workflow_path.glob("*.yaml")) + list(
            workflow_path.glob("*.yml")
        )

        if not yaml_files:
            logger.warning(f"No workflow YAML files found in {workflow_dir}")
            return

        for yaml_file in yaml_files:
            try:
                self._load_workflow_file(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load workflow file {yaml_file}: {e}")

        logger.info(
            f"Loaded {len(self.workflows)} workflows from {len(yaml_files)} files"
        )

        # Build index
        self._build_index()

    def _load_workflow_file(self, file_path: Path):
        """Load workflows from a single YAML file."""
        logger.debug(f"Loading workflow file: {file_path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        version = data.get("version", "1.0.0")
        workflows_data = data.get("workflows", [])

        for workflow_data in workflows_data:
            try:
                workflow = self._parse_workflow(workflow_data, version)
                self.workflows.append(workflow)
                self.workflow_by_name[workflow.name] = workflow
                logger.debug(f"Loaded workflow: {workflow.name}")
            except Exception as e:
                logger.error(
                    f"Failed to parse workflow {workflow_data.get('name', 'unknown')}: {e}"
                )

    def _parse_workflow(
        self, workflow_data: Dict[str, Any], version: str
    ) -> Workflow:
        """Parse a workflow from YAML data."""
        # Parse trigger
        trigger_data = workflow_data["trigger"]
        trigger = WorkflowTrigger(
            event_type=trigger_data["event_type"],
            filters=trigger_data.get("filters", {}),
        )

        # Parse actions
        actions = []
        for action_data in workflow_data.get("actions", []):
            action = WorkflowAction(
                name=action_data["name"],
                target_kernel=action_data["target_kernel"],
                action=action_data["action"],
                priority=action_data.get("priority", "NORMAL"),
                timeout_ms=action_data.get("timeout_ms", 10000),
                parameters=action_data.get("parameters", {}),
                on_success=action_data.get("on_success", []),
                on_failure=action_data.get("on_failure", []),
            )
            actions.append(action)

        # Build workflow
        workflow = Workflow(
            name=workflow_data["name"],
            description=workflow_data.get("description", ""),
            enabled=workflow_data.get("enabled", True),
            version=version,
            trigger=trigger,
            actions=actions,
            logging=workflow_data.get("logging", {}),
        )

        return workflow

    def _build_index(self):
        """Build index for fast workflow lookup by event type."""
        self.event_type_index.clear()

        for workflow in self.workflows:
            if not workflow.enabled:
                continue

            event_type = workflow.trigger.event_type

            if event_type not in self.event_type_index:
                self.event_type_index[event_type] = []

            self.event_type_index[event_type].append(workflow)

        logger.debug(
            f"Built index for {len(self.event_type_index)} unique event types"
        )

    def find_matching_workflows(self, event: Dict[str, Any]) -> List[Workflow]:
        """Find all workflows that match the given event."""
        event_type = event.get("type", "")

        # Direct lookup
        workflows = self.event_type_index.get(event_type, [])

        # Also check wildcard patterns (e.g., "kpi.*")
        for pattern, pattern_workflows in self.event_type_index.items():
            if "*" in pattern or "?" in pattern:
                if self._matches_pattern(event_type, pattern):
                    workflows.extend(pattern_workflows)

        return workflows

    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches a wildcard pattern."""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
        regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, event_type))

    def get_trigger_event_types(self) -> Set[str]:
        """Get all unique event types that workflows listen to."""
        return set(self.event_type_index.keys())

    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Get a workflow by name."""
        return self.workflow_by_name.get(name)

    def enable_workflow(self, name: str):
        """Enable a workflow."""
        workflow = self.workflow_by_name.get(name)
        if workflow:
            workflow.enabled = True
            self._build_index()
            logger.info(f"Enabled workflow: {name}")
        else:
            logger.warning(f"Workflow not found: {name}")

    def disable_workflow(self, name: str):
        """Disable a workflow."""
        workflow = self.workflow_by_name.get(name)
        if workflow:
            workflow.enabled = False
            self._build_index()
            logger.info(f"Disabled workflow: {name}")
        else:
            logger.warning(f"Workflow not found: {name}")

    def reload_workflows(self, workflow_dir: str):
        """Reload workflows from directory (hot-reload)."""
        logger.info("Reloading workflows...")
        self.workflows.clear()
        self.event_type_index.clear()
        self.workflow_by_name.clear()

        self.load_workflows(workflow_dir)
        logger.info("Workflows reloaded successfully")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        enabled_count = sum(1 for w in self.workflows if w.enabled)

        return {
            "total_workflows": len(self.workflows),
            "enabled_workflows": enabled_count,
            "disabled_workflows": len(self.workflows) - enabled_count,
            "unique_event_types": len(self.event_type_index),
            "workflows_by_kernel": self._count_by_kernel(),
        }

    def _count_by_kernel(self) -> Dict[str, int]:
        """Count workflows by target kernel."""
        counts = {}
        for workflow in self.workflows:
            if not workflow.enabled:
                continue

            for action in workflow.actions:
                kernel = action.target_kernel
                counts[kernel] = counts.get(kernel, 0) + 1

        return counts

    def list_workflows(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all workflows with summary information."""
        workflows = []
        for workflow in self.workflows:
            if enabled_only and not workflow.enabled:
                continue

            workflows.append(
                {
                    "name": workflow.name,
                    "description": workflow.description,
                    "enabled": workflow.enabled,
                    "trigger_event": workflow.trigger.event_type,
                    "actions_count": len(workflow.actions),
                    "version": workflow.version,
                }
            )

        return workflows
