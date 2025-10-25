from __future__ import annotations

"""
Grace Orchestration Kernel - Central conductor and scheduler for all Grace kernels.

This kernel manages the lifecycle of all kernels (governance, memory, mldl, learning,
ingress, intelligence, interface, multi-OS, immune/event mesh), schedules loops,
routes events, handles errors, snapshots/rollback, and enforces governance order.

TriggerMesh Integration:
- EventRouter: Routes events to workflows based on trigger patterns
- WorkflowEngine: Executes workflow actions by calling kernel handlers
- WorkflowRegistry: Loads and manages workflow definitions from YAML files
"""

from .orchestration_service import OrchestrationService
from .scheduler.scheduler import Scheduler
from .router.router import Router
from .state.state_manager import StateManager
from .recovery.watchdog import Watchdog
from .scaling.manager import ScalingManager
from .lifecycle.manager import LifecycleManager
from .snapshots.manager import SnapshotManager

# TriggerMesh components
from .event_router import EventRouter, EventFilter
from .workflow_engine import WorkflowEngine, ParameterSubstitutor
from .workflow_registry import WorkflowRegistry, Workflow, WorkflowAction, WorkflowTrigger

__all__ = [
    "OrchestrationService",
    "Scheduler",
    "Router",
    "StateManager",
    "Watchdog",
    "ScalingManager",
    "LifecycleManager",
    "SnapshotManager",
    # TriggerMesh
    "EventRouter",
    "EventFilter",
    "WorkflowEngine",
    "ParameterSubstitutor",
    "WorkflowRegistry",
    "Workflow",
    "WorkflowAction",
    "WorkflowTrigger",
]

"""
Grace Orchestration Layer - Event-driven workflow routing and execution.
"""
from __future__ import annotations

from .event_router import EventRouter
from .workflow_engine import WorkflowEngine
from .workflow_registry import WorkflowRegistry

__all__ = ["EventRouter", "WorkflowEngine", "WorkflowRegistry"]

"""
Grace Orchestration - Scheduling, autoscaling, and heartbeat monitoring
"""

from .enhanced_scheduler import EnhancedScheduler, SchedulerLoop, SchedulerPolicy
from .autoscaler import AdvancedAutoscaler, ScalingMetrics, ScalingDecision
from .heartbeat import HeartbeatMonitor, HeartbeatRecord
from .scheduler_metrics import SchedulerMetrics, scheduler_metrics

__all__ = [
    "EnhancedScheduler",
    "SchedulerLoop",
    "SchedulerPolicy",
    "AdvancedAutoscaler",
    "ScalingMetrics",
    "ScalingDecision",
    "HeartbeatMonitor",
    "HeartbeatRecord",
    "SchedulerMetrics",
    "scheduler_metrics",
]

"""
Grace AI Orchestration Module - Workflow and event-driven execution
"""
from grace.orchestration.trigger_mesh import TriggerMesh, WorkflowRule, EventPattern

__all__ = [
    "TriggerMesh",
    "WorkflowRule",
    "EventPattern",
]
