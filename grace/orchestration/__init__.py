from __future__ import annotations

"""
Grace Orchestration Module
==========================
Handles event routing, workflow execution, and task coordination.
This module provides the TriggerMesh and its supporting components.
"""

# Core Data Models for Workflows
from .models import (
    Workflow,
    WorkflowAction,
    WorkflowTrigger,
)

# Core TriggerMesh Components
from .event_router import EventRouter, EventFilter
from .workflow_engine import WorkflowEngine
from .workflow_registry import WorkflowRegistry
from .trigger_mesh import TriggerMesh

# Other Orchestration Services (if any are needed externally)
from .orchestration_service import OrchestrationService
from .scheduler.scheduler import Scheduler
from .state.state_manager import StateManager

# This is the single, authoritative list of what this module exports.
__all__ = [
    # TriggerMesh and its parts
    "TriggerMesh",
    "EventRouter",
    "EventFilter",
    "WorkflowEngine",
    "WorkflowRegistry",
    # Core Workflow Models
    "Workflow",
    "WorkflowAction",
    "WorkflowTrigger",
    # Other Services
    "OrchestrationService",
    "Scheduler",
    "StateManager",
]
