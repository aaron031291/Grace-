"""
Grace Clarity Framework - Eliminates ambiguity in Grace system components.

This framework provides standardized base classes, event routing, loop output formats,
and component lifecycle management to ensure clear, traceable, and consistent
behavior across all Grace kernels.

Classes addressed:
1. Structural Ambiguity - BaseComponent with standardized lifecycle
2. Signal Routing Ambiguity - Enhanced EventBus with YAML schema
3. Loop Identity Ambiguity - GraceLoopOutput standardized format
4. Subsystem Activation Ambiguity - Component manifest and orchestrator
"""

from .base_component import BaseComponent, ComponentStatus
from .loop_output import GraceLoopOutput, ReasoningChain
from .component_manifest import (
    GraceComponentManifest,
    ComponentRole,
    TrustLevel,
    ActivationState,
)
from .enhanced_event_bus import EnhancedEventBus
from .orchestrator import GraceOrchestrator

__all__ = [
    "BaseComponent",
    "ComponentStatus",
    "GraceLoopOutput",
    "ReasoningChain",
    "GraceComponentManifest",
    "ComponentRole",
    "TrustLevel",
    "ActivationState",
    "EnhancedEventBus",
    "GraceOrchestrator",
]

__version__ = "1.0.0"
