"""
Grace AI - Workflow Registry for TriggerMesh Orchestration Layer
"""
import importlib.util
import logging
import os
from types import ModuleType
from collections import defaultdict
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class WorkflowRegistry:
    """
    Discovers, loads, and registers all available workflows from the filesystem.
    Provides a cache and lookup methods for the WorkflowEngine.
    """
    def __init__(self, workflow_dir: str):
        self.workflow_dir = workflow_dir
        self.event_to_workflow_map: Dict[str, List[str]] = defaultdict(list)
        self.modules: Dict[str, Any] = {}
        self.workflows: List[Dict[str, Any]] = []
        self.load_workflows()

    def load_workflows(self):
        """
        Dynamically load all workflow modules from the specified directory.
        """
        wf_path = Path(self.workflow_dir)
        if not wf_path.is_dir():
            logger.warning(f"Workflow directory not found: {wf_path}")
            return

        total_files = 0
        for file_path in wf_path.glob("*.py"):
            total_files += 1
            if file_path.name.startswith("_"):
                continue

            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(f"grace.workflows.{module_name}", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    if hasattr(module, "WORKFLOW_NAME") and hasattr(module, "TRIGGERS"):
                        self.register(module.WORKFLOW_NAME, module, module.TRIGGERS)
                except Exception as e:
                    logger.error(f"Failed to load workflow module {module_name}: {e}", exc_info=True)
        
        logger.info(f"WorkflowRegistry loaded {len(self.modules)}/{total_files} workflow modules.")
        logger.info(f"Workflows available: {list(self.modules.keys())}")

    def register(self, name: str, module: Any, events: List[str]):
        """Registers a single workflow module."""
        self.modules[name] = module
        
        # For test compatibility, maintain a simple list of workflow dicts
        entry = {'name': name, 'events': list(events) if events else []}
        if not any(w.get('name') == name for w in self.workflows):
            self.workflows.append(entry)

        for event_type in events:
            self.event_to_workflow_map[event_type].append(name)
        logger.info(f"Registered workflow: {name} (events: {events})")

    def find_workflows_for_event(self, event_type: str) -> list[str]:
        """Finds all workflow names that are triggered by a given event type."""
        return self.event_to_workflow_map.get(event_type, [])

    def get_module(self, workflow_name: str) -> Any | None:
        """Safely retrieves a loaded workflow module by name."""
        return self.modules.get(workflow_name)
