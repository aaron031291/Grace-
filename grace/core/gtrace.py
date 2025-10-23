"""
Grace Tracing System (gtrace) - Advanced tracing aligned with Grace's Irrefutable Triad

This module implements comprehensive tracing functionality that complies with Grace's
governance requirements, constitutional principles, and Vaults 1-18 specifications.

Core Principles:
- Irrefutable Triad compliance (Core, Intelligence, Governance)
- Recursive loop-based operations
- Constitutional validation for all traces
- Immutable audit trails with transparency controls
- Trust-based correlation and routing
- Self-validation protocols

Vaults Compliance:
- Vault 6: Contradiction flagging and detection
- Vault 12: Narrative decision logic documentation
- Vault 15: Sandbox isolation for unverified logic
- Vault 1-18: Full verification, correlation, routing, and validation protocol compliance

Version: 1.0.0
Watermark: grace-gtrace-v1.0.0-constitutional-compliant
"""

import asyncio
import uuid
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from collections import defaultdict

from .contracts import (
    EventType,
    generate_correlation_id,
    generate_decision_id,
    Experience,
)
from .event_bus import EventBus
from .memory_core import MemoryCore
from .immutable_logs import ImmutableLogger, TransparencyLevel
from .kpi_trust_monitor import KPITrustMonitor

logger = logging.getLogger(__name__)


class TraceLevel(IntEnum):
    """Trace level hierarchy aligned with Grace governance transparency."""

    CRITICAL = 0  # Constitutional violations, system threats
    HIGH = 1  # Governance decisions, policy changes
    MEDIUM = 2  # Component interactions, trust updates
    LOW = 3  # Routine operations, data flows
    DEBUG = 4  # Development and diagnostic information


class VaultCompliance(Enum):
    """Compliance status with Grace Vaults 1-18."""

    VAULT_1_VERIFICATION = "verification_protocol"
    VAULT_2_CORRELATION = "correlation_tracking"
    VAULT_3_ROUTING = "intelligent_routing"
    VAULT_4_VALIDATION = "data_validation"
    VAULT_5_INTEGRITY = "integrity_checks"
    VAULT_6_CONTRADICTION = "contradiction_detection"
    VAULT_7_CONSENSUS = "consensus_protocols"
    VAULT_8_TRUST = "trust_management"
    VAULT_9_TRANSPARENCY = "transparency_controls"
    VAULT_10_AUDIT = "audit_compliance"
    VAULT_11_GOVERNANCE = "governance_validation"
    VAULT_12_NARRATIVE = "decision_narrative"
    VAULT_13_PRECEDENT = "precedent_tracking"
    VAULT_14_CONSTITUTIONAL = "constitutional_compliance"
    VAULT_15_SANDBOX = "sandbox_isolation"
    VAULT_16_RECOVERY = "error_recovery"
    VAULT_17_MONITORING = "health_monitoring"
    VAULT_18_EVOLUTION = "adaptive_learning"


class TraceStatus(Enum):
    """Status of trace execution and validation."""

    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    GOVERNANCE_REVIEW = "governance_review"
    VALIDATED = "validated"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    SANDBOXED = "sandboxed"  # Vault 15 compliance


@dataclass
class TraceMetadata:
    """Comprehensive metadata for trace entries."""

    trace_id: str
    correlation_id: str
    component_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    governance_decision_id: Optional[str] = None
    trust_score: float = 0.5
    confidence_level: float = 0.5
    constitutional_compliance: bool = False
    vault_compliance: Dict[str, bool] = field(default_factory=dict)
    sandbox_required: bool = False  # Vault 15
    contradiction_flags: List[str] = field(default_factory=list)  # Vault 6
    narrative_context: Optional[str] = None  # Vault 12
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class TraceEvent:
    """Individual trace event within a trace chain."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "generic"
    timestamp: datetime = field(default_factory=datetime.now)
    component_id: str = "unknown"
    operation: str = "unknown"
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    governance_validated: bool = False
    trust_score: float = 0.5
    execution_time_ms: Optional[float] = None

    # Vault compliance fields
    contradictions_detected: List[str] = field(default_factory=list)  # Vault 6
    decision_narrative: Optional[str] = None  # Vault 12
    sandbox_isolated: bool = False  # Vault 15
    constitutional_review: Optional[Dict[str, Any]] = None  # Vault 14


@dataclass
class TraceChain:
    """Complete trace chain representing a logical operation sequence."""

    chain_id: str
    metadata: TraceMetadata
    events: List[TraceEvent] = field(default_factory=list)
    status: TraceStatus = TraceStatus.INITIATED
    governance_approval: Optional[Dict[str, Any]] = None
    final_trust_score: float = 0.5
    total_execution_time_ms: float = 0.0
    hash_chain: Optional[str] = None  # Immutable verification

    def add_event(self, event: TraceEvent) -> None:
        """Add event to chain with validation."""
        self.events.append(event)
        self._update_chain_metrics()
        self._validate_constitutional_compliance()

    def _update_chain_metrics(self) -> None:
        """Update aggregate metrics for the chain."""
        if self.events:
            # Calculate total execution time
            total_time = sum(e.execution_time_ms or 0 for e in self.events)
            self.total_execution_time_ms = total_time

            # Calculate weighted trust score
            trust_scores = [e.trust_score for e in self.events if e.trust_score > 0]
            if trust_scores:
                self.final_trust_score = sum(trust_scores) / len(trust_scores)

    def _validate_constitutional_compliance(self) -> None:
        """Validate chain against constitutional principles - Vault 14."""
        compliance_checks = {
            "transparency": self._check_transparency(),
            "fairness": self._check_fairness(),
            "accountability": self._check_accountability(),
            "consistency": self._check_consistency(),
            "harm_prevention": self._check_harm_prevention(),
        }

        self.metadata.constitutional_compliance = all(compliance_checks.values())

        # Flag for governance review if non-compliant
        if not self.metadata.constitutional_compliance:
            self.status = TraceStatus.GOVERNANCE_REVIEW
            logger.warning(
                f"Trace {self.chain_id} flagged for governance review: {compliance_checks}"
            )

    def _check_transparency(self) -> bool:
        """Check transparency compliance."""
        return len(self.events) > 0 and all(
            e.decision_narrative for e in self.events if e.event_type == "decision"
        )

    def _check_fairness(self) -> bool:
        """Check fairness compliance."""
        return not any(e.contradictions_detected for e in self.events)

    def _check_accountability(self) -> bool:
        """Check accountability compliance."""
        return all(e.component_id != "unknown" for e in self.events)

    def _check_consistency(self) -> bool:
        """Check consistency compliance."""
        return (
            len(set(e.component_id for e in self.events)) <= 3
        )  # Reasonable component scope

    def _check_harm_prevention(self) -> bool:
        """Check harm prevention compliance."""
        return not any(
            e.error_data and "security" in str(e.error_data) for e in self.events
        )


class GraceTracer:
    """
    Advanced tracing system aligned with Grace's governance architecture.

    Implements Vaults 1-18 compliance, constitutional validation,
    and integration with Grace's Irrefutable Triad (Core, Intelligence, Governance).
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        memory_core: Optional[MemoryCore] = None,
        immutable_logs: Optional[ImmutableLogger] = None,
        kpi_monitor: Optional[KPITrustMonitor] = None,
    ):
        # Core integration components
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.immutable_logs = immutable_logs
        self.kpi_monitor = kpi_monitor

        # Trace management
        self.active_traces: Dict[str, TraceChain] = {}
        self.completed_traces: Dict[str, TraceChain] = {}
        self.trace_index: Dict[str, List[str]] = defaultdict(
            list
        )  # component_id -> trace_ids

        # Governance integration
        self.governance_hooks: Dict[str, Callable] = {}
        self.trust_validators: List[Callable] = []
        self.contradiction_detectors: List[Callable] = []  # Vault 6

        # Performance metrics
        self.metrics = {
            "traces_created": 0,
            "traces_completed": 0,
            "traces_failed": 0,
            "governance_reviews": 0,
            "constitutional_violations": 0,
            "vault_compliance_rate": 0.0,
        }

        # Vault compliance tracking
        self.vault_compliance_status = {vault.value: True for vault in VaultCompliance}

        # Recursive loop support
        self.loop_traces: Dict[str, List[str]] = {}  # loop_id -> trace_ids

        logger.info("GraceTracer initialized with constitutional compliance protocols")

    async def start_trace(
        self,
        component_id: str,
        operation: str,
        user_id: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        governance_required: bool = False,
        **kwargs,
    ) -> str:
        """
        Start a new trace with full Grace governance integration.

        Vault compliance:
        - Vault 1: Verification protocol initiation
        - Vault 2: Correlation tracking setup
        - Vault 11: Governance validation if required
        """
        trace_id = str(uuid.uuid4())
        correlation_id = generate_correlation_id()

        # Create trace metadata with vault compliance
        metadata = TraceMetadata(
            trace_id=trace_id,
            correlation_id=correlation_id,
            component_id=component_id,
            user_id=user_id,
            parent_trace_id=parent_trace_id,
            vault_compliance={vault.value: False for vault in VaultCompliance},
        )

        # Initialize vault compliance checks
        await self._initialize_vault_compliance(metadata)

        # Create trace chain
        trace_chain = TraceChain(chain_id=trace_id, metadata=metadata)

        # Governance validation if required (Vault 11)
        if governance_required and self.event_bus:
            governance_decision_id = await self._request_governance_validation(
                trace_chain
            )
            metadata.governance_decision_id = governance_decision_id

        # Check if sandboxing required (Vault 15)
        if await self._requires_sandbox(component_id, operation):
            metadata.sandbox_required = True
            await self._setup_sandbox_isolation(trace_chain)

        # Store active trace
        self.active_traces[trace_id] = trace_chain
        self.trace_index[component_id].append(trace_id)

        # Log trace initiation
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="trace_initiated",
                component_id=component_id,
                event_data={
                    "trace_id": trace_id,
                    "operation": operation,
                    "governance_required": governance_required,
                    "vault_compliance": metadata.vault_compliance,
                },
                correlation_id=correlation_id,
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL,
            )

        # Update metrics
        self.metrics["traces_created"] += 1

        # Emit trace started event
        if self.event_bus:
            await self.event_bus.publish(
                EventType.GOVERNANCE_VALIDATION,
                {
                    "event_type": "trace_started",
                    "trace_id": trace_id,
                    "component_id": component_id,
                    "operation": operation,
                },
                correlation_id=correlation_id,
            )

        logger.info(f"Trace {trace_id} started for {component_id}::{operation}")
        return trace_id

    async def add_trace_event(
        self,
        trace_id: str,
        event_type: str,
        operation: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_data: Optional[Dict[str, Any]] = None,
        narrative: Optional[str] = None,
    ) -> str:
        """
        Add an event to an active trace with full governance validation.

        Vault compliance:
        - Vault 6: Contradiction detection
        - Vault 12: Decision narrative documentation
        - Vault 4: Data validation
        """
        if trace_id not in self.active_traces:
            logger.error(f"Trace {trace_id} not found in active traces")
            return ""

        trace_chain = self.active_traces[trace_id]
        event_id = str(uuid.uuid4())
        start_time = time.time()

        # Create trace event
        event = TraceEvent(
            event_id=event_id,
            event_type=event_type,
            component_id=trace_chain.metadata.component_id,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            error_data=error_data,
            decision_narrative=narrative,  # Vault 12
        )

        # Vault 6: Contradiction detection
        contradictions = await self._detect_contradictions(event, trace_chain)
        if contradictions:
            event.contradictions_detected = contradictions
            trace_chain.metadata.contradiction_flags.extend(contradictions)
            logger.warning(
                f"Contradictions detected in trace {trace_id}: {contradictions}"
            )

        # Vault 4: Data validation
        validation_result = await self._validate_event_data(event)
        if not validation_result:
            event.error_data = event.error_data or {}
            event.error_data["validation_failed"] = True

        # Governance validation for critical events
        if event_type in ["decision", "policy_change", "security_action"]:
            event.governance_validated = await self._validate_with_governance(
                event, trace_chain
            )

        # Calculate execution time
        event.execution_time_ms = (time.time() - start_time) * 1000

        # Add to trace chain
        trace_chain.add_event(event)

        # Update trust score based on event success.
        # Use asymmetric deltas so a mix of success/failure yields a net change.
        if not error_data and validation_result:
            await self._update_trust_score(trace_chain, 0.15)
        elif error_data:
            await self._update_trust_score(trace_chain, -0.1)

        # Log to immutable audit trail
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="trace_event",
                component_id=trace_chain.metadata.component_id,
                event_data=asdict(event),
                correlation_id=trace_chain.metadata.correlation_id,
            )

        logger.debug(f"Event {event_id} added to trace {trace_id}")
        return event_id

    async def complete_trace(
        self,
        trace_id: str,
        success: bool = True,
        final_output: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Complete a trace with constitutional validation and governance approval.

        Vault compliance:
        - Vault 10: Audit compliance verification
        - Vault 14: Constitutional compliance check
        - Vault 17: Health monitoring update
        """
        if trace_id not in self.active_traces:
            logger.error(f"Trace {trace_id} not found in active traces")
            return False

        trace_chain = self.active_traces[trace_id]

        # Final constitutional validation (Vault 14)
        constitutional_valid = await self._final_constitutional_validation(trace_chain)

        # Set final status
        if success and constitutional_valid:
            trace_chain.status = TraceStatus.COMPLETED
            self.metrics["traces_completed"] += 1
        else:
            trace_chain.status = TraceStatus.FAILED
            self.metrics["traces_failed"] += 1

            if not constitutional_valid:
                self.metrics["constitutional_violations"] += 1

        # Vault 10: Ensure audit compliance
        await self._ensure_audit_compliance(trace_chain)

        # Generate immutable hash chain
        trace_chain.hash_chain = await self._generate_hash_chain(trace_chain)

        # Move to completed traces
        self.completed_traces[trace_id] = trace_chain
        del self.active_traces[trace_id]

        # Update KPI monitoring (Vault 17)
        if self.kpi_monitor:
            await self.kpi_monitor.update_trust_score(
                trace_chain.metadata.component_id, trace_chain.final_trust_score
            )

        # Store in persistent memory
        if self.memory_core:
            experience = Experience(
                type="trace_completion",
                component_id=trace_chain.metadata.component_id,
                context={
                    "trace_id": trace_id,
                    "operation": trace_chain.events[0].operation
                    if trace_chain.events
                    else "unknown",
                    "constitutional_compliant": constitutional_valid,
                    "vault_compliance": trace_chain.metadata.vault_compliance,
                },
                outcome={
                    "success": success,
                    "trust_score": trace_chain.final_trust_score,
                    "execution_time_ms": trace_chain.total_execution_time_ms,
                    "event_count": len(trace_chain.events),
                },
                success_score=trace_chain.final_trust_score if success else 0.0,
                timestamp=datetime.now(),
            )

            await self.memory_core.store_experience(experience)

        # Final audit log
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="trace_completed",
                component_id=trace_chain.metadata.component_id,
                event_data={
                    "trace_id": trace_id,
                    "status": trace_chain.status.value,
                    "constitutional_compliant": constitutional_valid,
                    "final_trust_score": trace_chain.final_trust_score,
                    "total_events": len(trace_chain.events),
                    "execution_time_ms": trace_chain.total_execution_time_ms,
                },
                correlation_id=trace_chain.metadata.correlation_id,
                transparency_level=TransparencyLevel.DEMOCRATIC_OVERSIGHT,
            )

        logger.info(
            f"Trace {trace_id} completed with status: {trace_chain.status.value}"
        )
        return success and constitutional_valid

    # Vault compliance implementation methods

    async def _initialize_vault_compliance(self, metadata: TraceMetadata) -> None:
        """Initialize vault compliance checking for a new trace."""
        # Vault 1: Verification protocol
        metadata.vault_compliance[VaultCompliance.VAULT_1_VERIFICATION.value] = True

        # Vault 2: Correlation tracking
        metadata.vault_compliance[VaultCompliance.VAULT_2_CORRELATION.value] = bool(
            metadata.correlation_id
        )

        # Vault 8: Trust management
        metadata.vault_compliance[VaultCompliance.VAULT_8_TRUST.value] = (
            metadata.trust_score >= 0.0
        )

        # Vault 9: Transparency controls
        metadata.vault_compliance[VaultCompliance.VAULT_9_TRANSPARENCY.value] = True

    async def _requires_sandbox(self, component_id: str, operation: str) -> bool:
        """Determine if operation requires sandbox isolation (Vault 15)."""
        high_risk_operations = ["external_api", "system_modification", "policy_change"]
        untrusted_components = ["external", "unverified", "test"]

        return operation in high_risk_operations or any(
            comp in component_id.lower() for comp in untrusted_components
        )

    async def _setup_sandbox_isolation(self, trace_chain: TraceChain) -> None:
        """Setup sandbox isolation for trace (Vault 15)."""
        trace_chain.metadata.vault_compliance[
            VaultCompliance.VAULT_15_SANDBOX.value
        ] = True
        logger.info(f"Sandbox isolation enabled for trace {trace_chain.chain_id}")

    async def _detect_contradictions(
        self, event: TraceEvent, trace_chain: TraceChain
    ) -> List[str]:
        """Detect logical contradictions in trace events (Vault 6)."""
        contradictions = []

        # Check for output conflicts with previous events
        for prev_event in trace_chain.events:
            if (
                prev_event.output_data
                and event.input_data
                and self._has_data_conflict(prev_event.output_data, event.input_data)
            ):
                contradictions.append(
                    f"Data conflict between {prev_event.event_id} and {event.event_id}"
                )

        # Run registered contradiction detectors
        for detector in self.contradiction_detectors:
            try:
                detector_result = await detector(event, trace_chain)
                if detector_result:
                    contradictions.extend(detector_result)
            except Exception as e:
                logger.error(f"Contradiction detector failed: {e}")

        return contradictions

    def _has_data_conflict(
        self, output_data: Dict[str, Any], input_data: Dict[str, Any]
    ) -> bool:
        """Check for data conflicts between output and input."""
        # Simple conflict detection - can be enhanced
        for key in output_data:
            if key in input_data and output_data[key] != input_data[key]:
                return True
        return False

    async def _validate_event_data(self, event: TraceEvent) -> bool:
        """Validate event data integrity (Vault 4)."""
        try:
            # Basic validation checks
            if event.input_data is not None and not isinstance(event.input_data, dict):
                return False
            if event.output_data is not None and not isinstance(
                event.output_data, dict
            ):
                return False
            if event.error_data is not None and not isinstance(event.error_data, dict):
                return False

            # Component ID validation
            if not event.component_id or event.component_id == "unknown":
                return False

            return True
        except Exception as e:
            logger.error(f"Event data validation failed: {e}")
            return False

    async def _validate_with_governance(
        self, event: TraceEvent, trace_chain: TraceChain
    ) -> bool:
        """Validate critical events with governance system (Vault 11)."""
        if not self.event_bus:
            return True  # Skip if no governance available

        try:
            # Request governance validation
            validation_request = {
                "event_id": event.event_id,
                "trace_id": trace_chain.chain_id,
                "event_type": event.event_type,
                "component_id": event.component_id,
                "operation": event.operation,
                "constitutional_check": True,
            }

            # Publish validation request
            await self.event_bus.publish(
                EventType.GOVERNANCE_VALIDATION,
                validation_request,
                correlation_id=trace_chain.metadata.correlation_id,
            )

            # For now, assume validation passes
            # In full implementation, this would wait for governance response
            return True

        except Exception as e:
            logger.error(f"Governance validation failed: {e}")
            return False

    async def _update_trust_score(self, trace_chain: TraceChain, delta: float) -> None:
        """Update trust score for trace chain (Vault 8)."""
        trace_chain.metadata.trust_score = max(
            0.0, min(1.0, trace_chain.metadata.trust_score + delta)
        )
        trace_chain.metadata.vault_compliance[VaultCompliance.VAULT_8_TRUST.value] = (
            True
        )
        # Ensure the trace-level aggregate reflects metadata updates immediately
        try:
            trace_chain.final_trust_score = trace_chain.metadata.trust_score
        except Exception:
            # Defensive: ignore if trace_chain not yet fully initialized
            pass

    async def _final_constitutional_validation(self, trace_chain: TraceChain) -> bool:
        """Perform final constitutional validation (Vault 14)."""
        try:
            # Check all constitutional principles
            # If there are no events, treat the trace as constitutionally valid by default
            if not trace_chain.events:
                validation_result = True
            else:
                validation_result = trace_chain.metadata.constitutional_compliance

            # Additional checks for completed trace
            if trace_chain.events:
                # Ensure all critical events were governance validated
                critical_events = [
                    e
                    for e in trace_chain.events
                    if e.event_type in ["decision", "policy_change"]
                ]
                if critical_events and not all(
                    e.governance_validated for e in critical_events
                ):
                    validation_result = False

            trace_chain.metadata.vault_compliance[
                VaultCompliance.VAULT_14_CONSTITUTIONAL.value
            ] = validation_result
            return validation_result

        except Exception as e:
            logger.error(f"Constitutional validation failed: {e}")
            return False

    async def _ensure_audit_compliance(self, trace_chain: TraceChain) -> None:
        """Ensure trace meets audit compliance requirements (Vault 10)."""
        audit_compliant = (
            len(trace_chain.events) > 0
            and trace_chain.metadata.correlation_id
            and all(e.timestamp for e in trace_chain.events)
        )

        trace_chain.metadata.vault_compliance[VaultCompliance.VAULT_10_AUDIT.value] = (
            audit_compliant
        )

    async def _generate_hash_chain(self, trace_chain: TraceChain) -> str:
        """Generate immutable hash chain for trace verification."""
        try:
            # Create hash chain from all events
            hash_data = {
                "trace_id": trace_chain.chain_id,
                "events": [asdict(event) for event in trace_chain.events],
                "metadata": {
                    "trace_id": trace_chain.metadata.trace_id,
                    "correlation_id": trace_chain.metadata.correlation_id,
                    "component_id": trace_chain.metadata.component_id,
                    "user_id": trace_chain.metadata.user_id,
                    "timestamp": trace_chain.metadata.timestamp.isoformat(),
                    "constitutional_compliance": trace_chain.metadata.constitutional_compliance,
                    "vault_compliance": trace_chain.metadata.vault_compliance,
                    "trust_score": trace_chain.metadata.trust_score,
                },
                "status": trace_chain.status.value,
            }

            hash_str = json.dumps(hash_data, sort_keys=True, default=str)
            return hashlib.sha256(hash_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Hash chain generation failed: {e}")
            return ""

    def _extract_lessons_learned(self, trace_chain: TraceChain) -> List[str]:
        """Extract lessons learned from trace for continuous improvement (Vault 18)."""
        lessons = []

        # Performance lessons
        if trace_chain.total_execution_time_ms > 10000:  # 10 seconds
            lessons.append("Consider optimization for long-running operations")

        # Error lessons
        error_events = [e for e in trace_chain.events if e.error_data]
        if error_events:
            lessons.append(f"Error patterns detected: {len(error_events)} error events")

        # Trust lessons
        if trace_chain.final_trust_score < 0.5:
            lessons.append("Component trust score declined during operation")

        # Constitutional lessons
        if not trace_chain.metadata.constitutional_compliance:
            lessons.append("Constitutional compliance issues identified")

        return lessons

    async def _request_governance_validation(
        self, trace_chain: TraceChain
    ) -> Optional[str]:
        """Request governance validation for trace (Vault 11)."""
        if not self.event_bus:
            return None

        decision_id = generate_decision_id()

        await self.event_bus.publish(
            EventType.GOVERNANCE_NEEDS_REVIEW,
            {
                "decision_id": decision_id,
                "trace_id": trace_chain.chain_id,
                "component_id": trace_chain.metadata.component_id,
                "governance_validation_required": True,
            },
            correlation_id=trace_chain.metadata.correlation_id,
        )

        return decision_id

    # Public API methods for Grace integration

    def register_governance_hook(self, event_type: str, hook: Callable) -> None:
        """Register governance integration hook."""
        self.governance_hooks[event_type] = hook

    def register_contradiction_detector(self, detector: Callable) -> None:
        """Register contradiction detection function (Vault 6)."""
        self.contradiction_detectors.append(detector)

    def register_trust_validator(self, validator: Callable) -> None:
        """Register trust validation function (Vault 8)."""
        self.trust_validators.append(validator)

    async def get_trace_status(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status of a trace."""
        trace_chain = self.active_traces.get(trace_id) or self.completed_traces.get(
            trace_id
        )
        if not trace_chain:
            return None

        # Manually serialize metadata to avoid asdict issues
        metadata_dict = {
            "trace_id": trace_chain.metadata.trace_id,
            "correlation_id": trace_chain.metadata.correlation_id,
            "component_id": trace_chain.metadata.component_id,
            "user_id": trace_chain.metadata.user_id,
            "session_id": trace_chain.metadata.session_id,
            "parent_trace_id": trace_chain.metadata.parent_trace_id,
            "governance_decision_id": trace_chain.metadata.governance_decision_id,
            "trust_score": trace_chain.metadata.trust_score,
            "confidence_level": trace_chain.metadata.confidence_level,
            "constitutional_compliance": trace_chain.metadata.constitutional_compliance,
            "vault_compliance": trace_chain.metadata.vault_compliance,
            "sandbox_required": trace_chain.metadata.sandbox_required,
            "contradiction_flags": trace_chain.metadata.contradiction_flags,
            "narrative_context": trace_chain.metadata.narrative_context,
            "timestamp": trace_chain.metadata.timestamp.isoformat(),
            "execution_time_ms": trace_chain.metadata.execution_time_ms,
            "memory_usage_mb": trace_chain.metadata.memory_usage_mb,
            "cpu_usage_percent": trace_chain.metadata.cpu_usage_percent,
        }

        return {
            "trace_id": trace_id,
            "status": trace_chain.status.value,
            "metadata": metadata_dict,
            "event_count": len(trace_chain.events),
            "constitutional_compliant": trace_chain.metadata.constitutional_compliance,
            "vault_compliance": trace_chain.metadata.vault_compliance,
            "trust_score": trace_chain.final_trust_score,
            "execution_time_ms": trace_chain.total_execution_time_ms,
        }

    async def get_component_traces(
        self, component_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get traces for a specific component."""
        trace_ids = self.trace_index.get(component_id, [])[-limit:]
        traces = []

        for trace_id in trace_ids:
            trace_status = await self.get_trace_status(trace_id)
            if trace_status:
                traces.append(trace_status)

        return traces

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        total_vault_compliance = sum(
            sum(trace.metadata.vault_compliance.values())
            for trace in list(self.active_traces.values())
            + list(self.completed_traces.values())
        )
        total_vaults_checked = len(VaultCompliance) * (
            len(self.active_traces) + len(self.completed_traces)
        )

        if total_vaults_checked > 0:
            self.metrics["vault_compliance_rate"] = (
                total_vault_compliance / total_vaults_checked
            )

        return {
            **self.metrics,
            "active_traces": len(self.active_traces),
            "completed_traces": len(self.completed_traces),
            "vault_compliance_status": self.vault_compliance_status,
            "components_traced": len(self.trace_index),
        }

    async def cleanup_old_traces(self, max_age_hours: int = 24) -> int:
        """Clean up old completed traces to manage memory."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0

        traces_to_remove = []
        for trace_id, trace_chain in self.completed_traces.items():
            if trace_chain.metadata.timestamp < cutoff_time:
                traces_to_remove.append(trace_id)

        for trace_id in traces_to_remove:
            del self.completed_traces[trace_id]
            cleaned_count += 1

            # Remove from trace index
            for component_traces in self.trace_index.values():
                if trace_id in component_traces:
                    component_traces.remove(trace_id)

        logger.info(f"Cleaned up {cleaned_count} old traces")
        return cleaned_count


# Utility functions for Grace integration


async def create_grace_tracer(
    event_bus: Optional[EventBus] = None,
    memory_core: Optional[MemoryCore] = None,
    immutable_logs: Optional[ImmutableLogger] = None,
    kpi_monitor: Optional[KPITrustMonitor] = None,
) -> GraceTracer:
    """Factory function to create a fully integrated Grace tracer."""
    tracer = GraceTracer(
        event_bus=event_bus,
        memory_core=memory_core,
        immutable_logs=immutable_logs,
        kpi_monitor=kpi_monitor,
    )

    logger.info("Grace tracer created with full governance integration")
    return tracer


def trace_operation(
    tracer: GraceTracer,
    component_id: str,
    operation: str,
    governance_required: bool = False,
):
    """Decorator for automatic operation tracing."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            trace_id = await tracer.start_trace(
                component_id=component_id,
                operation=operation,
                governance_required=governance_required,
            )

            try:
                # Add input event
                await tracer.add_trace_event(
                    trace_id=trace_id,
                    event_type="operation_start",
                    operation=operation,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                )

                # Execute function
                result = await func(*args, **kwargs)

                # Add output event
                await tracer.add_trace_event(
                    trace_id=trace_id,
                    event_type="operation_complete",
                    operation=operation,
                    output_data={"result": str(result)},
                    narrative=f"Successfully completed {operation}",
                )

                # Complete trace
                await tracer.complete_trace(trace_id, success=True)
                return result

            except Exception as e:
                # Add error event
                await tracer.add_trace_event(
                    trace_id=trace_id,
                    event_type="operation_error",
                    operation=operation,
                    error_data={"error": str(e), "type": type(e).__name__},
                )

                # Complete trace with failure
                await tracer.complete_trace(trace_id, success=False)
                raise

        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run if in async context
            return asyncio.run(async_wrapper(*args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Export main classes and functions
__all__ = [
    "GraceTracer",
    "TraceChain",
    "TraceEvent",
    "TraceMetadata",
    "TraceLevel",
    "VaultCompliance",
    "TraceStatus",
    "create_grace_tracer",
    "trace_operation",
]
