"""
Core module initialization for Grace governance kernel.
"""

from .contracts import (
    DecisionSubject,
    EventType,
    Source,
    Evidence,
    LogicStep,
    Claim,
    ComponentSignal,
    UnifiedDecision,
    VerifiedClaims,
    LogicReport,
    GovernanceSnapshot,
    Experience,
    generate_correlation_id,
    generate_decision_id,
    generate_snapshot_id,
)
from .event_bus import EventBus
from .memory_core import MemoryCore
from .gtrace import (
    GraceTracer,
    TraceChain,
    TraceEvent,
    TraceMetadata,
    TraceLevel,
    VaultCompliance,
    TraceStatus,
    create_grace_tracer,
    trace_operation,
)
from .unified_service import create_unified_app

# Lazy import - only import when needed
def create_unified_app():
    from .unified_service import create_unified_app as _create
    return _create()

__all__ = [
    "DecisionSubject",
    "EventType",
    "Source",
    "Evidence",
    "LogicStep",
    "Claim",
    "ComponentSignal",
    "UnifiedDecision",
    "VerifiedClaims",
    "LogicReport",
    "GovernanceSnapshot",
    "Experience",
    "generate_correlation_id",
    "generate_decision_id",
    "generate_snapshot_id",
    "EventBus",
    "MemoryCore",
    "GraceTracer",
    "TraceChain",
    "TraceEvent",
    "TraceMetadata",
    "TraceLevel",
    "VaultCompliance",
    "TraceStatus",
    "create_grace_tracer",
    "trace_operation",
    "create_unified_app",
]

"""
Grace AI Core Module - Fundamental infrastructure components
"""
from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogger, TransparencyLevel
from grace.core.kpi_trust_monitor import KPITrustMonitor
from grace.core.component_registry import ComponentRegistry

__all__ = [
    "EventBus",
    "ImmutableLogger",
    "TransparencyLevel",
    "KPITrustMonitor",
    "ComponentRegistry",
]
