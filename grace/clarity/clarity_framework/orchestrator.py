"""
Grace Component Orchestrator for managing component lifecycle and dependencies.

Works with GraceComponentManifest to provide clear activation/deactivation
tracking and dependency resolution.
"""

import asyncio
from typing import Dict, List, Optional, Set, Any
import logging
from enum import Enum

from .base_component import BaseComponent, ComponentStatus
from .component_manifest import (
    GraceComponentManifest,
    ActivationState,
    ComponentDependency,
)
from .enhanced_event_bus import EnhancedEventBus


logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Strategies for component orchestration."""

    SEQUENTIAL = "sequential"  # Start/stop components one by one
    PARALLEL = "parallel"  # Start/stop components in parallel
    DEPENDENCY_ORDERED = "dependency_ordered"  # Respect dependency order
    TRUST_ORDERED = "trust_ordered"  # Start higher trust components first


class GraceOrchestrator:
    """
    Orchestrates Grace component lifecycle with clear dependency tracking.

    Resolves subsystem activation ambiguity by providing centralized component
    management with trust-based activation and dependency resolution.
    """

    def __init__(self, event_bus: Optional[EnhancedEventBus] = None):
        self.event_bus = event_bus or EnhancedEventBus()

        # Component tracking
        self.registered_components: Dict[str, BaseComponent] = {}
        self.component_manifests: Dict[str, GraceComponentManifest] = {}
        self.activation_order: List[str] = []
        self.dependency_graph: Dict[str, Set[str]] = {}

        # Orchestration state
        self.orchestration_active = False
        self.orchestration_strategy = OrchestrationStrategy.DEPENDENCY_ORDERED
        self.activation_timeout_sec = 30.0
        self.health_check_interval_sec = 60.0

        # Metrics and monitoring
        self.orchestration_metrics = {
            "components_registered": 0,
            "components_active": 0,
            "components_failed": 0,
            "total_activations": 0,
            "total_deactivations": 0,
            "dependency_violations": 0,
        }

        self._health_check_task = None

    def register_component(
        self, component: BaseComponent, manifest: GraceComponentManifest
    ) -> bool:
        """Register a component with the orchestrator."""
        try:
            component_id = manifest.component_id

            # Validate component matches manifest
            if component.metadata.component_type != manifest.component_type:
                logger.error(
                    f"Component type mismatch: {component.metadata.component_type} != {manifest.component_type}"
                )
                return False

            # Store component and manifest
            self.registered_components[component_id] = component
            self.component_manifests[component_id] = manifest

            # Update manifest state
            manifest.set_activation_state(
                ActivationState.REGISTERED, "Registered with orchestrator"
            )

            # Update dependency graph
            self._update_dependency_graph(component_id, manifest.dependencies)

            # Register with event bus
            if self.event_bus:
                self.event_bus.register_component(
                    component_id,
                    manifest.component_type,
                    [cap.name for cap in manifest.capabilities],
                    manifest.trust_level.name,
                )

            self.orchestration_metrics["components_registered"] += 1

            logger.info(
                f"Registered component: {component_id} ({manifest.component_type})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register component {manifest.component_id}: {e}")
            return False

    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from the orchestrator."""
        try:
            if component_id not in self.registered_components:
                logger.warning(f"Component not registered: {component_id}")
                return False

            # Deactivate if active
            component = self.registered_components[component_id]
            if component.get_status() == ComponentStatus.ACTIVE:
                asyncio.create_task(self.deactivate_component(component_id))

            # Update manifest
            manifest = self.component_manifests[component_id]
            manifest.set_activation_state(
                ActivationState.DECOMMISSIONED, "Unregistered from orchestrator"
            )

            # Remove from tracking
            del self.registered_components[component_id]
            del self.component_manifests[component_id]

            # Clean up dependency graph
            if component_id in self.dependency_graph:
                del self.dependency_graph[component_id]

            # Remove from activation order
            if component_id in self.activation_order:
                self.activation_order.remove(component_id)

            self.orchestration_metrics["components_registered"] -= 1

            logger.info(f"Unregistered component: {component_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister component {component_id}: {e}")
            return False

    def _update_dependency_graph(
        self, component_id: str, dependencies: List[ComponentDependency]
    ):
        """Update the dependency graph for a component."""
        if component_id not in self.dependency_graph:
            self.dependency_graph[component_id] = set()

        # Add hard dependencies to graph
        for dep in dependencies:
            if dep.is_hard_dependency:
                self.dependency_graph[component_id].add(dep.component_id)

    def get_activation_order(
        self, component_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Determine optimal activation order based on dependencies and trust."""
        if component_ids is None:
            component_ids = list(self.registered_components.keys())

        if self.orchestration_strategy == OrchestrationStrategy.DEPENDENCY_ORDERED:
            return self._topological_sort(component_ids)
        elif self.orchestration_strategy == OrchestrationStrategy.TRUST_ORDERED:
            return self._trust_based_order(component_ids)
        else:
            return component_ids  # Sequential or parallel don't need special ordering

    def _topological_sort(self, component_ids: List[str]) -> List[str]:
        """Perform topological sort on components based on dependencies."""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []

        def visit(component_id: str):
            if component_id in temp_visited:
                logger.warning(f"Circular dependency detected involving {component_id}")
                self.orchestration_metrics["dependency_violations"] += 1
                return

            if component_id in visited:
                return

            temp_visited.add(component_id)

            # Visit dependencies first
            for dep_id in self.dependency_graph.get(component_id, set()):
                if (
                    dep_id in component_ids
                ):  # Only consider components we're actually sorting
                    visit(dep_id)

            temp_visited.remove(component_id)
            visited.add(component_id)
            result.append(component_id)

        for component_id in component_ids:
            visit(component_id)

        return result

    def _trust_based_order(self, component_ids: List[str]) -> List[str]:
        """Order components by trust level (highest trust first)."""

        def get_trust_score(component_id: str) -> float:
            manifest = self.component_manifests.get(component_id)
            return manifest.trust_score if manifest else 0.0

        return sorted(component_ids, key=get_trust_score, reverse=True)

    async def activate_component(self, component_id: str) -> bool:
        """Activate a single component."""
        try:
            if component_id not in self.registered_components:
                logger.error(f"Component not registered: {component_id}")
                return False

            component = self.registered_components[component_id]
            manifest = self.component_manifests[component_id]

            # Check if component can be activated
            can_activate, reason = manifest.can_activate()
            if not can_activate:
                logger.error(f"Cannot activate {component_id}: {reason}")
                return False

            # Check dependencies
            if not await self._check_dependencies(component_id):
                logger.error(f"Dependencies not satisfied for {component_id}")
                return False

            # Update manifest state
            manifest.set_activation_state(
                ActivationState.INITIALIZING, "Starting activation"
            )

            # Activate component with timeout
            try:
                activation_task = component.safe_activate()
                success = await asyncio.wait_for(
                    activation_task, timeout=self.activation_timeout_sec
                )
            except asyncio.TimeoutError:
                logger.error(f"Activation timeout for {component_id}")
                manifest.record_error(
                    f"Activation timeout ({self.activation_timeout_sec}s)"
                )
                return False

            if success:
                manifest.set_activation_state(
                    ActivationState.ACTIVE, "Successfully activated"
                )
                self.orchestration_metrics["components_active"] += 1
                self.orchestration_metrics["total_activations"] += 1

                # Add to activation order tracking
                if component_id not in self.activation_order:
                    self.activation_order.append(component_id)

                # Publish activation event
                if self.event_bus:
                    await self.event_bus.publish(
                        "COMPONENT_ACTIVATED",
                        {
                            "component_id": component_id,
                            "component_type": manifest.component_type,
                            "trust_level": manifest.trust_level.name,
                        },
                        source_component_id="orchestrator",
                    )

                logger.info(f"Successfully activated component: {component_id}")
                return True
            else:
                manifest.record_error("Component activation returned False")
                self.orchestration_metrics["components_failed"] += 1
                return False

        except Exception as e:
            logger.error(f"Error activating component {component_id}: {e}")
            manifest = self.component_manifests.get(component_id)
            if manifest:
                manifest.record_error(f"Activation error: {str(e)}")
            self.orchestration_metrics["components_failed"] += 1
            return False

    async def deactivate_component(self, component_id: str) -> bool:
        """Deactivate a single component."""
        try:
            if component_id not in self.registered_components:
                logger.error(f"Component not registered: {component_id}")
                return False

            component = self.registered_components[component_id]
            manifest = self.component_manifests[component_id]

            # Check if other components depend on this one
            dependents = self._find_dependents(component_id)
            if dependents:
                logger.warning(
                    f"Component {component_id} has active dependents: {dependents}"
                )
                # Could implement cascade deactivation here

            # Deactivate component
            success = await component.safe_deactivate()

            if success:
                manifest.set_activation_state(
                    ActivationState.REGISTERED, "Successfully deactivated"
                )
                self.orchestration_metrics["components_active"] -= 1
                self.orchestration_metrics["total_deactivations"] += 1

                # Remove from activation order
                if component_id in self.activation_order:
                    self.activation_order.remove(component_id)

                # Publish deactivation event
                if self.event_bus:
                    await self.event_bus.publish(
                        "COMPONENT_DEACTIVATED",
                        {
                            "component_id": component_id,
                            "component_type": manifest.component_type,
                        },
                        source_component_id="orchestrator",
                    )

                logger.info(f"Successfully deactivated component: {component_id}")
                return True
            else:
                manifest.record_error("Component deactivation returned False")
                return False

        except Exception as e:
            logger.error(f"Error deactivating component {component_id}: {e}")
            manifest = self.component_manifests.get(component_id)
            if manifest:
                manifest.record_error(f"Deactivation error: {str(e)}")
            return False

    async def _check_dependencies(self, component_id: str) -> bool:
        """Check if all dependencies for a component are satisfied."""
        manifest = self.component_manifests.get(component_id)
        if not manifest:
            return False

        for dep in manifest.dependencies:
            if not dep.is_hard_dependency:
                continue  # Skip soft dependencies

            dep_component_id = dep.component_id

            # Check if dependency is registered and active
            if dep_component_id not in self.registered_components:
                if dep.fallback_strategy:
                    logger.info(
                        f"Using fallback strategy for missing dependency: {dep_component_id}"
                    )
                    continue
                logger.error(f"Hard dependency not registered: {dep_component_id}")
                return False

            dep_component = self.registered_components[dep_component_id]
            if dep_component.get_status() != ComponentStatus.ACTIVE:
                logger.error(f"Hard dependency not active: {dep_component_id}")
                return False

        return True

    def _find_dependents(self, component_id: str) -> List[str]:
        """Find components that depend on the given component."""
        dependents = []
        for comp_id, deps in self.dependency_graph.items():
            if component_id in deps:
                # Check if the dependent is currently active
                component = self.registered_components.get(comp_id)
                if component and component.get_status() == ComponentStatus.ACTIVE:
                    dependents.append(comp_id)
        return dependents

    async def start_orchestration(self) -> bool:
        """Start the orchestration system."""
        try:
            if self.orchestration_active:
                logger.warning("Orchestration already active")
                return True

            self.orchestration_active = True

            # Start health checking
            if self.health_check_interval_sec > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info("Orchestration system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start orchestration: {e}")
            return False

    async def stop_orchestration(self) -> bool:
        """Stop the orchestration system."""
        try:
            self.orchestration_active = False

            # Stop health checking
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Deactivate all active components
            active_components = [
                comp_id
                for comp_id, comp in self.registered_components.items()
                if comp.get_status() == ComponentStatus.ACTIVE
            ]

            # Deactivate in reverse activation order
            for component_id in reversed(active_components):
                await self.deactivate_component(component_id)

            logger.info("Orchestration system stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop orchestration: {e}")
            return False

    async def _health_check_loop(self):
        """Continuous health checking loop."""
        while self.orchestration_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval_sec)

    async def _perform_health_checks(self):
        """Perform health checks on all active components."""
        active_components = [
            (comp_id, comp)
            for comp_id, comp in self.registered_components.items()
            if comp.get_status() == ComponentStatus.ACTIVE
        ]

        for component_id, component in active_components:
            try:
                health_info = await component.health_check()
                manifest = self.component_manifests[component_id]
                manifest.update_health("healthy", health_info)

            except Exception as e:
                logger.warning(f"Health check failed for {component_id}: {e}")
                manifest = self.component_manifests[component_id]
                manifest.update_health("unhealthy", {"error": str(e)})
                manifest.record_error(f"Health check failed: {str(e)}")

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        active_components = [
            comp_id
            for comp_id, comp in self.registered_components.items()
            if comp.get_status() == ComponentStatus.ACTIVE
        ]

        return {
            "orchestration_active": self.orchestration_active,
            "strategy": self.orchestration_strategy.value,
            "total_registered": len(self.registered_components),
            "active_components": len(active_components),
            "activation_order": self.activation_order.copy(),
            "metrics": self.orchestration_metrics.copy(),
            "component_summary": {
                comp_id: manifest.get_summary()
                for comp_id, manifest in self.component_manifests.items()
            },
        }
