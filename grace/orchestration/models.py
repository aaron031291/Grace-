from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class WorkflowTrigger:
    """Defines what event triggers a workflow."""
    event_type: str
    filter_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowAction:
    """Defines a single action to be executed in a workflow."""
    action_type: str
    target: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Workflow:
    """Represents a complete, event-driven workflow."""
    name: str
    trigger: WorkflowTrigger
    actions: List[WorkflowAction]
