"""
Grace Immune/AVN Controller - Adversarial signal processing, circuit breaking, and immune response orchestration.

Features:
- Adversarial test result processing
- Anomaly detection and classification
- Circuit breaking and sandboxing
- Immune response orchestration
- Integration with Resilience Kernel
- Trust and KPI score updates
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from ..contracts.message_envelope_simple import GraceMessageEnvelope, EventTypes
from ..resilience_kernel.kernel import ResilienceKernel
from ..core.utils import utc_timestamp

logger = logging.getLogger(__name__)


class ImmuneResponseType(Enum):
    """Types of immune responses."""

    MONITOR = "monitor"  # Increased monitoring
    ISOLATE = "isolate"  # Component isolation
    SANDBOX = "sandbox"  # Sandboxed execution
    CIRCUIT_BREAK = "circuit_break"  # Circuit breaker activation
    ROLLBACK = "rollback"  # System rollback
    SHUTDOWN = "shutdown"  # Component shutdown


class ThreatLevel(Enum):
    """Threat severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


@dataclass
class AdversarialSignal:
    """Adversarial test signal."""

    signal_id: str
    test_type: str  # Type of adversarial test
    component: str  # Target component
    severity: ThreatLevel
    confidence: float  # Confidence in the result (0-1)
    evidence: Dict[str, Any]  # Supporting evidence
    timestamp: datetime
    remediation_suggestions: List[str] = None


@dataclass
class ImmuneAction:
    """Immune response action."""

    action_id: str
    response_type: ImmuneResponseType
    target_component: str
    parameters: Dict[str, Any]
    duration_seconds: Optional[int] = None
    triggered_by: str = None
    created_at: str = utc_timestamp()


@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""

    component: str
    state: str  # open, closed, half_open
    failure_count: int
    failure_threshold: int
    last_failure: Optional[datetime]
    recovery_timeout: int  # seconds
    half_open_max_calls: int = 5


class GraceImmuneController:
    """
    Grace Immune/AVN Controller for adversarial response and system protection.

    Processes adversarial signals, orchestrates immune responses, and maintains
    system resilience through circuit breaking and sandboxing.
    """

    def __init__(
        self,
        event_bus=None,
        resilience_kernel: Optional[ResilienceKernel] = None,
        governance_bridge=None,
    ):
        self.event_bus = event_bus
        self.resilience_kernel = resilience_kernel or ResilienceKernel()
        self.governance_bridge = governance_bridge

        # Component state tracking
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.sandboxed_components: Dict[str, Dict[str, Any]] = {}
        self.immune_actions: Dict[str, ImmuneAction] = {}

        # Adversarial signal processing
        self.signal_history: List[AdversarialSignal] = []
        self.threat_patterns: Dict[str, List[AdversarialSignal]] = {}

        # Configuration
        self.config = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 300,  # 5 minutes
                "half_open_max_calls": 5,
            },
            "sandbox": {
                "default_timeout": 1800,  # 30 minutes
                "resource_limits": {"cpu_percent": 50, "memory_mb": 512},
            },
            "threat_correlation": {
                "time_window_seconds": 3600,  # 1 hour
                "pattern_threshold": 3,
            },
        }

        # Trust and KPI tracking
        self.trust_scores: Dict[str, float] = {}
        self.kpi_metrics: Dict[str, Dict[str, float]] = {}

        # Start event processing
        if self.event_bus:
            self._register_event_handlers()

        logger.info("Grace Immune Controller initialized")

    def _register_event_handlers(self):
        """Register event handlers for adversarial signals."""
        if hasattr(self.event_bus, "subscribe"):
            # Subscribe to adversarial test failures
            self.event_bus.subscribe(
                EventTypes.ADV_TEST_FAILED, self._handle_adversarial_test_failed
            )

            # Subscribe to anomaly detections
            self.event_bus.subscribe(
                EventTypes.ANOMALY_DETECTED, self._handle_anomaly_detected
            )

            # Subscribe to resilience events for integration
            self.event_bus.subscribe(
                EventTypes.RESILIENCE_INCIDENT_OPENED, self._handle_resilience_incident
            )

    async def _handle_adversarial_test_failed(self, message: GraceMessageEnvelope):
        """Handle adversarial test failure signals."""
        try:
            payload = message.payload

            # Create adversarial signal
            signal = AdversarialSignal(
                signal_id=f"adv_{uuid.uuid4().hex[:8]}",
                test_type=payload.get("test_type", "unknown"),
                component=payload.get("component", "unknown"),
                severity=ThreatLevel(payload.get("severity", 2)),
                confidence=payload.get("confidence", 0.8),
                evidence=payload.get("evidence", {}),
                timestamp=datetime.utcnow(),
                remediation_suggestions=payload.get("remediation_suggestions", []),
            )

            # Process the signal
            await self._process_adversarial_signal(signal)

            logger.info(f"Processed adversarial test failure: {signal.signal_id}")

        except Exception as e:
            logger.error(f"Error handling adversarial test failed: {e}")

    async def _handle_anomaly_detected(self, message: GraceMessageEnvelope):
        """Handle general anomaly detection signals."""
        try:
            payload = message.payload

            # Convert anomaly to adversarial signal format
            signal = AdversarialSignal(
                signal_id=f"anom_{uuid.uuid4().hex[:8]}",
                test_type="anomaly_detection",
                component=payload.get("component", "system"),
                severity=self._map_severity(payload.get("severity", "medium")),
                confidence=payload.get("confidence", 0.7),
                evidence=payload.get("details", {}),
                timestamp=datetime.utcnow(),
            )

            await self._process_adversarial_signal(signal)

        except Exception as e:
            logger.error(f"Error handling anomaly detected: {e}")

    async def _handle_resilience_incident(self, message: GraceMessageEnvelope):
        """Handle resilience incident events for coordination."""
        try:
            payload = message.payload
            incident_id = payload.get("incident_id")
            component = payload.get("component")

            # Check if we have immune actions for this component
            if component in self.sandboxed_components:
                # Component is sandboxed, inform resilience kernel
                await self._publish_immune_event(
                    EventTypes.IMMUNE_SANDBOXED,
                    {
                        "component": component,
                        "incident_id": incident_id,
                        "sandbox_info": self.sandboxed_components[component],
                    },
                )

            # Check circuit breaker state
            if component in self.circuit_breakers:
                breaker = self.circuit_breakers[component]
                if breaker.state == "open":
                    await self._publish_immune_event(
                        "CIRCUIT_BREAKER_ACTIVE",
                        {
                            "component": component,
                            "incident_id": incident_id,
                            "failure_count": breaker.failure_count,
                        },
                    )

        except Exception as e:
            logger.error(f"Error handling resilience incident: {e}")

    async def _process_adversarial_signal(self, signal: AdversarialSignal):
        """Process an adversarial signal and determine response."""
        try:
            # Store signal in history
            self.signal_history.append(signal)

            # Update threat patterns
            self._update_threat_patterns(signal)

            # Analyze threat level and determine response
            response_type = await self._determine_immune_response(signal)

            if response_type:
                # Execute immune response
                action = await self._execute_immune_response(signal, response_type)

                # Update trust scores
                await self._update_trust_scores(signal, action)

                # Publish immune event
                await self._publish_immune_event(
                    f"IMMUNE_{response_type.value.upper()}",
                    {
                        "signal_id": signal.signal_id,
                        "component": signal.component,
                        "action_id": action.action_id,
                        "severity": signal.severity.name,
                    },
                )

        except Exception as e:
            logger.error(f"Error processing adversarial signal {signal.signal_id}: {e}")

    def _update_threat_patterns(self, signal: AdversarialSignal):
        """Update threat patterns for correlation analysis."""
        pattern_key = f"{signal.test_type}:{signal.component}"

        if pattern_key not in self.threat_patterns:
            self.threat_patterns[pattern_key] = []

        self.threat_patterns[pattern_key].append(signal)

        # Keep only recent patterns
        time_cutoff = datetime.utcnow() - timedelta(
            seconds=self.config["threat_correlation"]["time_window_seconds"]
        )

        self.threat_patterns[pattern_key] = [
            s for s in self.threat_patterns[pattern_key] if s.timestamp > time_cutoff
        ]

    async def _determine_immune_response(
        self, signal: AdversarialSignal
    ) -> Optional[ImmuneResponseType]:
        """Determine appropriate immune response based on signal analysis."""

        # Check for patterns
        pattern_key = f"{signal.test_type}:{signal.component}"
        pattern_count = len(self.threat_patterns.get(pattern_key, []))

        # Escalate based on severity and patterns
        if signal.severity == ThreatLevel.CATASTROPHIC:
            return ImmuneResponseType.SHUTDOWN

        elif signal.severity == ThreatLevel.CRITICAL:
            if pattern_count >= self.config["threat_correlation"]["pattern_threshold"]:
                return ImmuneResponseType.ROLLBACK
            else:
                return ImmuneResponseType.CIRCUIT_BREAK

        elif signal.severity == ThreatLevel.HIGH:
            if pattern_count >= 2:
                return ImmuneResponseType.SANDBOX
            else:
                return ImmuneResponseType.CIRCUIT_BREAK

        elif signal.severity == ThreatLevel.MEDIUM:
            if pattern_count >= 3:
                return ImmuneResponseType.ISOLATE
            else:
                return ImmuneResponseType.MONITOR

        else:  # LOW
            return ImmuneResponseType.MONITOR

    async def _execute_immune_response(
        self, signal: AdversarialSignal, response_type: ImmuneResponseType
    ) -> ImmuneAction:
        """Execute the determined immune response."""

        action = ImmuneAction(
            action_id=f"immune_{uuid.uuid4().hex[:8]}",
            response_type=response_type,
            target_component=signal.component,
            parameters={},
            triggered_by=signal.signal_id,
        )

        try:
            if response_type == ImmuneResponseType.CIRCUIT_BREAK:
                await self._apply_circuit_breaker(signal.component, action)

            elif response_type == ImmuneResponseType.SANDBOX:
                await self._apply_sandboxing(signal.component, action)

            elif response_type == ImmuneResponseType.ISOLATE:
                await self._apply_isolation(signal.component, action)

            elif response_type == ImmuneResponseType.ROLLBACK:
                await self._request_rollback(signal.component, action)

            elif response_type == ImmuneResponseType.SHUTDOWN:
                await self._shutdown_component(signal.component, action)

            elif response_type == ImmuneResponseType.MONITOR:
                await self._increase_monitoring(signal.component, action)

            # Store action
            self.immune_actions[action.action_id] = action

            logger.info(
                f"Executed immune response {response_type.value} for component {signal.component}"
            )
            return action

        except Exception as e:
            logger.error(
                f"Failed to execute immune response {response_type.value}: {e}"
            )
            raise

    async def _apply_circuit_breaker(self, component: str, action: ImmuneAction):
        """Apply circuit breaker to component."""

        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState(
                component=component,
                state="closed",
                failure_count=0,
                failure_threshold=self.config["circuit_breaker"]["failure_threshold"],
                last_failure=None,
                recovery_timeout=self.config["circuit_breaker"]["recovery_timeout"],
            )

        breaker = self.circuit_breakers[component]
        breaker.failure_count += 1
        breaker.last_failure = datetime.utcnow()

        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "open"
            action.parameters = {
                "state": "open",
                "failure_count": breaker.failure_count,
                "recovery_timeout": breaker.recovery_timeout,
            }

            # Schedule recovery attempt
            asyncio.create_task(
                self._schedule_circuit_recovery(component, breaker.recovery_timeout)
            )

        # Integrate with resilience kernel
        if self.resilience_kernel:
            self.resilience_kernel.register_healing_action(
                f"circuit_breaker_{component}",
                lambda: self._check_circuit_health(component),
            )

    async def _apply_sandboxing(self, component: str, action: ImmuneAction):
        """Apply sandboxing to component."""

        sandbox_config = {
            "component": component,
            "started_at": datetime.utcnow(),
            "timeout": self.config["sandbox"]["default_timeout"],
            "resource_limits": self.config["sandbox"]["resource_limits"],
            "restrictions": ["network_isolated", "filesystem_readonly", "cpu_limited"],
        }

        self.sandboxed_components[component] = sandbox_config
        action.parameters = sandbox_config
        action.duration_seconds = sandbox_config["timeout"]

        # Schedule sandbox removal
        asyncio.create_task(
            self._schedule_sandbox_removal(component, sandbox_config["timeout"])
        )

    async def _apply_isolation(self, component: str, action: ImmuneAction):
        """Apply component isolation."""

        isolation_config = {
            "component": component,
            "isolated_at": datetime.utcnow(),
            "restrictions": ["no_external_calls", "limited_resources"],
            "monitoring_level": "high",
        }

        action.parameters = isolation_config

        # Notify governance for approval if needed
        if self.governance_bridge:
            await self.governance_bridge.request_approval(
                action_type="component_isolation",
                component=component,
                reason="Immune response to adversarial signals",
                metadata=isolation_config,
            )

    async def _request_rollback(self, component: str, action: ImmuneAction):
        """Request system rollback."""

        rollback_request = {
            "component": component,
            "rollback_type": "immune_response",
            "triggered_at": datetime.utcnow(),
            "reason": "Critical adversarial signals detected",
        }

        action.parameters = rollback_request

        # Publish rollback request event
        await self._publish_immune_event(
            EventTypes.ROLLBACK_REQUESTED,
            {
                "target": component,
                "reason": rollback_request["reason"],
                "immune_action_id": action.action_id,
            },
        )

    async def _shutdown_component(self, component: str, action: ImmuneAction):
        """Shutdown component (catastrophic response)."""

        shutdown_config = {
            "component": component,
            "shutdown_at": datetime.utcnow(),
            "reason": "catastrophic_threat",
            "approval_required": True,
        }

        action.parameters = shutdown_config

        # Require governance approval for shutdown
        if self.governance_bridge:
            await self.governance_bridge.request_emergency_approval(
                action_type="component_shutdown",
                component=component,
                reason="Catastrophic adversarial threat detected",
                priority="critical",
            )

    async def _increase_monitoring(self, component: str, action: ImmuneAction):
        """Increase monitoring for component."""

        monitoring_config = {
            "component": component,
            "level": "high",
            "metrics": ["performance", "errors", "security_events"],
            "frequency_seconds": 30,
            "alert_threshold_reduction": 0.5,
        }

        action.parameters = monitoring_config

        # Register enhanced health checks with resilience kernel
        if self.resilience_kernel:
            self.resilience_kernel.register_health_check(
                f"enhanced_monitoring_{component}",
                lambda: self._enhanced_health_check(component),
            )

    async def _update_trust_scores(
        self, signal: AdversarialSignal, action: ImmuneAction
    ):
        """Update trust scores based on immune response."""

        component = signal.component

        # Decrease trust based on signal severity
        trust_reduction = {
            ThreatLevel.LOW: 0.05,
            ThreatLevel.MEDIUM: 0.1,
            ThreatLevel.HIGH: 0.2,
            ThreatLevel.CRITICAL: 0.4,
            ThreatLevel.CATASTROPHIC: 0.8,
        }

        current_trust = self.trust_scores.get(component, 1.0)
        reduction = trust_reduction.get(signal.severity, 0.1)
        new_trust = max(0.0, current_trust - reduction)

        self.trust_scores[component] = new_trust

        # Update KPI metrics
        if component not in self.kpi_metrics:
            self.kpi_metrics[component] = {}

        self.kpi_metrics[component].update(
            {
                "trust_score": new_trust,
                "threat_events": self.kpi_metrics[component].get("threat_events", 0)
                + 1,
                "last_threat": datetime.utcnow().isoformat(),
                "immune_responses": self.kpi_metrics[component].get(
                    "immune_responses", 0
                )
                + 1,
            }
        )

        # Publish trust update event
        await self._publish_immune_event(
            EventTypes.TRUST_UPDATED,
            {
                "component": component,
                "trust_score": new_trust,
                "previous_score": current_trust,
                "update_reason": f"immune_response_{action.response_type.value}",
            },
        )

    async def _publish_immune_event(self, event_type: str, payload: Dict[str, Any]):
        """Publish immune system event."""
        if self.event_bus:
            await self.event_bus.publish(
                event_type=event_type,
                payload=payload,
                source="immune_controller",
                priority="high" if event_type.startswith("IMMUNE_") else "normal",
            )

    def _map_severity(self, severity_str: str) -> ThreatLevel:
        """Map string severity to ThreatLevel enum."""
        mapping = {
            "low": ThreatLevel.LOW,
            "medium": ThreatLevel.MEDIUM,
            "high": ThreatLevel.HIGH,
            "critical": ThreatLevel.CRITICAL,
            "catastrophic": ThreatLevel.CATASTROPHIC,
        }
        return mapping.get(severity_str.lower(), ThreatLevel.MEDIUM)

    async def _schedule_circuit_recovery(self, component: str, timeout: int):
        """Schedule circuit breaker recovery."""
        await asyncio.sleep(timeout)

        if component in self.circuit_breakers:
            breaker = self.circuit_breakers[component]
            if breaker.state == "open":
                breaker.state = "half_open"
                logger.info(f"Circuit breaker for {component} moved to half-open state")

    async def _schedule_sandbox_removal(self, component: str, timeout: int):
        """Schedule sandbox removal."""
        await asyncio.sleep(timeout)

        if component in self.sandboxed_components:
            del self.sandboxed_components[component]
            logger.info(f"Sandbox removed for component {component}")

            # Publish sandbox removal event
            await self._publish_immune_event(
                "IMMUNE_SANDBOX_REMOVED",
                {"component": component, "removed_at": datetime.utcnow().isoformat()},
            )

    def _check_circuit_health(self, component: str) -> bool:
        """Check circuit breaker health."""
        if component not in self.circuit_breakers:
            return True

        breaker = self.circuit_breakers[component]
        return breaker.state == "closed"

    def _enhanced_health_check(self, component: str) -> bool:
        """Enhanced health check for monitored component."""
        # Implement enhanced health checking logic
        # For now, return basic health status
        return component not in self.sandboxed_components

    # Public API methods

    async def inject_adversarial_signal(
        self,
        test_type: str,
        component: str,
        severity: str,
        confidence: float,
        evidence: Dict[str, Any],
    ) -> str:
        """Manually inject an adversarial signal for testing."""

        signal = AdversarialSignal(
            signal_id=f"manual_{uuid.uuid4().hex[:8]}",
            test_type=test_type,
            component=component,
            severity=self._map_severity(severity),
            confidence=confidence,
            evidence=evidence,
            timestamp=datetime.utcnow(),
        )

        await self._process_adversarial_signal(signal)
        return signal.signal_id

    def get_component_status(self, component: str) -> Dict[str, Any]:
        """Get current status of a component."""
        status = {
            "component": component,
            "trust_score": self.trust_scores.get(component, 1.0),
            "circuit_breaker": None,
            "sandboxed": component in self.sandboxed_components,
            "active_immune_actions": [],
        }

        if component in self.circuit_breakers:
            status["circuit_breaker"] = asdict(self.circuit_breakers[component])

        if component in self.sandboxed_components:
            status["sandbox_info"] = self.sandboxed_components[component]

        # Find active immune actions
        for action_id, action in self.immune_actions.items():
            if action.target_component == component:
                status["active_immune_actions"].append(asdict(action))

        return status

    def get_system_immune_status(self) -> Dict[str, Any]:
        """Get overall immune system status."""
        return {
            "total_signals_processed": len(self.signal_history),
            "active_circuit_breakers": len(
                [b for b in self.circuit_breakers.values() if b.state == "open"]
            ),
            "sandboxed_components": list(self.sandboxed_components.keys()),
            "threat_patterns": {k: len(v) for k, v in self.threat_patterns.items()},
            "trust_scores": self.trust_scores.copy(),
            "kpi_metrics": self.kpi_metrics.copy(),
            "timestamp": datetime.utcnow().isoformat(),
        }
