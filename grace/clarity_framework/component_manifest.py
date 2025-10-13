"""
Class 4: Subsystem Activation Ambiguity Resolution

GraceComponentManifest provides clear tracking of component trust flags,
active state, and role tags to eliminate ambiguity about component lifecycle.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum


class ComponentRole(Enum):
    """Standardized component roles in Grace system."""

    CORE_GOVERNANCE = "core_governance"
    CORE_MEMORY = "core_memory"
    CORE_INTELLIGENCE = "core_intelligence"
    KERNEL_LEARNING = "kernel_learning"
    KERNEL_MLDL = "kernel_mldl"
    KERNEL_INGRESS = "kernel_ingress"
    KERNEL_INTERFACE = "kernel_interface"
    KERNEL_ORCHESTRATION = "kernel_orchestration"
    KERNEL_RESILIENCE = "kernel_resilience"
    KERNEL_MULTIOS = "kernel_multios"
    BRIDGE_COMPONENT = "bridge_component"
    EVENT_HANDLER = "event_handler"
    SPECIALIST = "specialist"
    UTILITY = "utility"
    EXTERNAL = "external"


class TrustLevel(Enum):
    """Trust levels for Grace components."""

    UNTRUSTED = 0.0
    LOW_TRUST = 0.25
    MEDIUM_TRUST = 0.5
    HIGH_TRUST = 0.75
    FULL_TRUST = 1.0


class ActivationState(Enum):
    """Component activation states."""

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    DECOMMISSIONED = "decommissioned"


@dataclass
class ComponentCapability:
    """Represents a capability provided by a component."""

    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    confidence_level: float  # How confident the component is in this capability
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentDependency:
    """Represents a dependency on another component."""

    component_id: str
    component_type: str
    required_capabilities: List[str]
    is_hard_dependency: bool = True  # False for soft dependencies
    fallback_strategy: Optional[str] = None


@dataclass
class GraceComponentManifest:
    """
    Complete manifest for a Grace system component.

    Eliminates subsystem activation ambiguity by providing clear tracking
    of trust, activation state, capabilities, and dependencies.
    """

    # Basic identification
    component_id: str
    component_type: str
    component_name: str
    version: str

    # Role and trust
    role: ComponentRole
    trust_level: TrustLevel
    trust_score: float = 0.5
    trust_last_updated: datetime = field(default_factory=datetime.now)

    # Activation state
    activation_state: ActivationState = ActivationState.UNREGISTERED
    is_active: bool = False
    is_trusted: bool = False

    # Capabilities and dependencies
    capabilities: List[ComponentCapability] = field(default_factory=list)
    dependencies: List[ComponentDependency] = field(default_factory=list)

    # Configuration and metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Operational tracking
    registered_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"

    # Performance metrics
    activation_count: int = 0
    deactivation_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    average_response_time_ms: Optional[float] = None

    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        self.is_trusted = self.trust_level.value >= 0.5
        if (
            not self.registered_at
            and self.activation_state != ActivationState.UNREGISTERED
        ):
            self.registered_at = datetime.now()

    def update_trust(self, new_trust_score: float, reason: Optional[str] = None):
        """Update trust score and level."""
        self.trust_score = max(0.0, min(1.0, new_trust_score))  # Clamp to [0,1]

        # Determine trust level based on score
        if self.trust_score >= 0.9:
            self.trust_level = TrustLevel.FULL_TRUST
        elif self.trust_score >= 0.75:
            self.trust_level = TrustLevel.HIGH_TRUST
        elif self.trust_score >= 0.5:
            self.trust_level = TrustLevel.MEDIUM_TRUST
        elif self.trust_score >= 0.25:
            self.trust_level = TrustLevel.LOW_TRUST
        else:
            self.trust_level = TrustLevel.UNTRUSTED

        self.is_trusted = self.trust_level.value >= 0.5
        self.trust_last_updated = datetime.now()

        if reason:
            if "trust_updates" not in self.metadata:
                self.metadata["trust_updates"] = []
            self.metadata["trust_updates"].append(
                {
                    "timestamp": self.trust_last_updated.isoformat(),
                    "old_score": self.trust_score,
                    "new_score": new_trust_score,
                    "reason": reason,
                }
            )

    def set_activation_state(
        self, new_state: ActivationState, details: Optional[str] = None
    ):
        """Update activation state with tracking."""
        old_state = self.activation_state
        self.activation_state = new_state

        # Update active flag based on state
        self.is_active = new_state == ActivationState.ACTIVE

        # Track state changes
        if "state_changes" not in self.metadata:
            self.metadata["state_changes"] = []

        self.metadata["state_changes"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": new_state.value,
                "details": details,
            }
        )

        # Update timing based on state
        if new_state == ActivationState.REGISTERED and not self.registered_at:
            self.registered_at = datetime.now()
        elif new_state == ActivationState.ACTIVE:
            self.activated_at = datetime.now()
            self.activation_count += 1
        elif (
            old_state == ActivationState.ACTIVE and new_state != ActivationState.ACTIVE
        ):
            self.deactivation_count += 1

    def add_capability(
        self,
        name: str,
        description: str,
        input_types: List[str],
        output_types: List[str],
        confidence_level: float,
    ):
        """Add a capability to this component."""
        capability = ComponentCapability(
            name=name,
            description=description,
            input_types=input_types,
            output_types=output_types,
            confidence_level=confidence_level,
        )
        self.capabilities.append(capability)

    def add_dependency(
        self,
        component_id: str,
        component_type: str,
        required_capabilities: List[str],
        is_hard: bool = True,
        fallback_strategy: Optional[str] = None,
    ):
        """Add a dependency on another component."""
        dependency = ComponentDependency(
            component_id=component_id,
            component_type=component_type,
            required_capabilities=required_capabilities,
            is_hard_dependency=is_hard,
            fallback_strategy=fallback_strategy,
        )
        self.dependencies.append(dependency)

    def record_error(self, error_description: str):
        """Record an error for this component."""
        self.error_count += 1
        self.last_error = error_description

        if "errors" not in self.metadata:
            self.metadata["errors"] = []

        self.metadata["errors"].append(
            {"timestamp": datetime.now().isoformat(), "description": error_description}
        )

        # Keep only last 50 errors
        if len(self.metadata["errors"]) > 50:
            self.metadata["errors"] = self.metadata["errors"][-50:]

    def update_health(self, status: str, details: Optional[Dict[str, Any]] = None):
        """Update component health status."""
        self.health_status = status
        self.last_health_check = datetime.now()

        if details:
            self.metadata["last_health_details"] = details

    def can_activate(self) -> tuple[bool, str]:
        """Check if component can be activated."""
        if not self.is_trusted:
            return False, f"Component trust level too low: {self.trust_level.name}"

        if self.activation_state in [
            ActivationState.ACTIVE,
            ActivationState.INITIALIZING,
        ]:
            return (
                False,
                f"Component already active or initializing: {self.activation_state.value}",
            )

        if self.activation_state == ActivationState.DECOMMISSIONED:
            return False, "Component has been decommissioned"

        # Check hard dependencies (simplified - would need actual component registry)
        hard_deps = [dep for dep in self.dependencies if dep.is_hard_dependency]
        if hard_deps:
            # In real implementation, would check if dependencies are active
            pass

        return True, "Component can be activated"

    def get_capability_by_name(self, name: str) -> Optional[ComponentCapability]:
        """Get a capability by name."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def has_capability(self, name: str, min_confidence: float = 0.5) -> bool:
        """Check if component has a capability with minimum confidence."""
        cap = self.get_capability_by_name(name)
        return cap is not None and cap.confidence_level >= min_confidence

    def get_summary(self) -> Dict[str, Any]:
        """Get component summary information."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_name": self.component_name,
            "version": self.version,
            "role": self.role.value,
            "activation_state": self.activation_state.value,
            "is_active": self.is_active,
            "is_trusted": self.is_trusted,
            "trust_level": self.trust_level.name,
            "trust_score": self.trust_score,
            "health_status": self.health_status,
            "capabilities_count": len(self.capabilities),
            "dependencies_count": len(self.dependencies),
            "activation_count": self.activation_count,
            "error_count": self.error_count,
            "tags": list(self.tags),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary format."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_name": self.component_name,
            "version": self.version,
            "role": self.role.value,
            "trust_level": self.trust_level.value,
            "trust_score": self.trust_score,
            "trust_last_updated": self.trust_last_updated.isoformat(),
            "activation_state": self.activation_state.value,
            "is_active": self.is_active,
            "is_trusted": self.is_trusted,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_types": cap.input_types,
                    "output_types": cap.output_types,
                    "confidence_level": cap.confidence_level,
                    "performance_metrics": cap.performance_metrics,
                }
                for cap in self.capabilities
            ],
            "dependencies": [
                {
                    "component_id": dep.component_id,
                    "component_type": dep.component_type,
                    "required_capabilities": dep.required_capabilities,
                    "is_hard_dependency": dep.is_hard_dependency,
                    "fallback_strategy": dep.fallback_strategy,
                }
                for dep in self.dependencies
            ],
            "configuration": self.configuration,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "registered_at": self.registered_at.isoformat()
            if self.registered_at
            else None,
            "activated_at": self.activated_at.isoformat()
            if self.activated_at
            else None,
            "last_health_check": self.last_health_check.isoformat()
            if self.last_health_check
            else None,
            "health_status": self.health_status,
            "activation_count": self.activation_count,
            "deactivation_count": self.deactivation_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "average_response_time_ms": self.average_response_time_ms,
        }
