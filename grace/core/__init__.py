"""
grace.governance_kernel
=======================

Core module initialization for Grace Governance Kernel.

Exposes the primary contract types, event bus, and memory core
for use by higher-level orchestration and integration modules.
"""

from __future__ import annotations

# Contracts: canonical governance datatypes
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

# Core subsystems
from .event_bus import EventBus
from .memory_core import MemoryCore

__all__ = [
    # contracts
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
    # subsystems
    "EventBus",
    "MemoryCore",
]
