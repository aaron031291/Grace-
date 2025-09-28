"""
Grace Orchestration Lifecycle Manager - Startup/shutdown, health, and upgrades.

Manages the complete lifecycle of the orchestration system including
initialization, health checks, graceful shutdown, and system upgrades.
"""

import asyncio
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class LifecyclePhase(Enum):
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    UPGRADING = "upgrading"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ComponentStatus(Enum):
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


@dataclass
class HealthCheck:
    """Health check configuration and results."""
    name: str
    check_function: Callable
    interval_seconds: int
    timeout_seconds: int
    retries: int
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_checks == 0:
            return 1.0
        return (self.total_checks - self.total_failures) / self.total_checks


class ManagedComponent:
    """Represents a component managed by the lifecycle manager."""
    
    def __init__(self, name: str, component: Any, 
                 start_method: str = "start",
                 stop_method: str = "stop",
                 health_method: str = "get_status",
                 dependencies: List[str] = None,
                 startup_timeout: int = 60,
                 shutdown_timeout: int = 30):
        self.name = name
        self.component = component
        self.start_method = start_method
        self.stop_method = stop_method
        self.health_method = health_method
        self.dependencies = dependencies or []
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        
        self.status = ComponentStatus.UNKNOWN
        self.started_at = None
        self.stopped_at = None
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_health_check = None
        self.error_count = 0
    
    async def start(self) -> bool:
        """Start the component."""
        try:
            logger.info(f"Starting component: {self.name}")
            self.status = ComponentStatus.STARTING
            
            if hasattr(self.component, self.start_method):
                start_func = getattr(self.component, self.start_method)
                
                if asyncio.iscoroutinefunction(start_func):
                    await asyncio.wait_for(start_func(), timeout=self.startup_timeout)
                else:
                    start_func()
            
            self.status = ComponentStatus.HEALTHY
            self.started_at = datetime.now()
            logger.info(f"Successfully started component: {self.name}")
            return True
            
        except Exception as e:
            self.status = ComponentStatus.UNHEALTHY
            self.error_count += 1
            logger.error(f"Failed to start component {self.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the component."""
        try:
            logger.info(f"Stopping component: {self.name}")
            
            if hasattr(self.component, self.stop_method):
                stop_func = getattr(self.component, self.stop_method)
                
                if asyncio.iscoroutinefunction(stop_func):
                    await asyncio.wait_for(stop_func(), timeout=self.shutdown_timeout)
                else:
                    stop_func()
            
            self.status = ComponentStatus.STOPPED
            self.stopped_at = datetime.now()
            logger.info(f"Successfully stopped component: {self.name}")
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to stop component {self.name}: {e}")
            return False
    
    async def check_health(self) -> Dict[str, Any]:
        """Check component health."""
        try:
            if hasattr(self.component, self.health_method):
                health_func = getattr(self.component, self.health_method)
                
                if asyncio.iscoroutinefunction(health_func):
                    result = await health_func()
                else:
                    result = health_func()
                
                self.last_health_check = datetime.now()
                return result
            
            return {"status": "unknown", "message": "No health check method"}
            
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def add_health_check(self, check: HealthCheck):
        """Add a health check."""
        self.health_checks[check.name] = check
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "dependencies": self.dependencies,
            "error_count": self.error_count,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_checks": {
                name: {
                    "last_result": check.last_result,
                    "consecutive_failures": check.consecutive_failures,
                    "success_rate": check.success_rate(),
                    "last_check": check.last_check.isoformat() if check.last_check else None
                }
                for name, check in self.health_checks.items()
            }
        }


class LifecycleManager:
    """Orchestration lifecycle and health management."""
    
    def __init__(self, event_publisher=None):
        self.event_publisher = event_publisher
        
        # Lifecycle state
        self.phase = LifecyclePhase.INITIALIZING
        self.phase_since = datetime.now()
        
        # Components
        self.components: Dict[str, ManagedComponent] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_health = ComponentStatus.UNKNOWN
        
        # Configuration
        self.startup_timeout = 300  # 5 minutes
        self.shutdown_timeout = 120 # 2 minutes
        self.health_check_interval = 30  # seconds
        
        # Control
        self.running = False
        self._health_task = None
        self._shutdown_requested = False
        
        # Statistics
        self.startup_time = None
        self.uptime_start = None
        self.restart_count = 0
        
        # Signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self._shutdown_requested = True
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def register_component(self, component: ManagedComponent):
        """Register a component for lifecycle management."""
        self.components[component.name] = component
        logger.debug(f"Registered component: {component.name}")
    
    def set_startup_order(self, order: List[str]):
        """Set the order for component startup."""
        # Validate all components exist
        for name in order:
            if name not in self.components:
                raise ValueError(f"Unknown component: {name}")
        
        self.startup_order = order
        self.shutdown_order = list(reversed(order))  # Reverse order for shutdown
        logger.info(f"Set startup order: {' -> '.join(order)}")
    
    def add_health_check(self, check: HealthCheck):
        """Add a global health check."""
        self.health_checks[check.name] = check
    
    async def startup(self) -> bool:
        """Start the orchestration system."""
        try:
            logger.info("Starting orchestration system lifecycle...")
            start_time = time.time()
            
            await self._transition_phase(LifecyclePhase.STARTING)
            
            # Start components in dependency order
            for component_name in self.startup_order:
                component = self.components[component_name]
                
                # Check dependencies
                if not await self._check_dependencies(component):
                    logger.error(f"Dependencies not met for {component_name}")
                    await self._transition_phase(LifecyclePhase.FAILED)
                    return False
                
                # Start component
                if not await component.start():
                    logger.error(f"Failed to start component: {component_name}")
                    await self._transition_phase(LifecyclePhase.FAILED)
                    return False
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            # Mark as running
            await self._transition_phase(LifecyclePhase.RUNNING)
            
            self.startup_time = time.time() - start_time
            self.uptime_start = datetime.now()
            self.running = True
            
            logger.info(f"Orchestration system started successfully in {self.startup_time:.2f}s")
            
            # Publish startup event
            if self.event_publisher:
                await self.event_publisher("ORCH_SYSTEM_STARTED", {
                    "startup_time_seconds": self.startup_time,
                    "components_started": len(self.startup_order),
                    "timestamp": datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            logger.error(f"System startup failed: {e}", exc_info=True)
            await self._transition_phase(LifecyclePhase.FAILED)
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the orchestration system gracefully."""
        try:
            logger.info("Initiating graceful shutdown...")
            await self._transition_phase(LifecyclePhase.STOPPING)
            
            self.running = False
            
            # Stop health monitoring
            if self._health_task:
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass
            
            # Stop components in reverse order
            for component_name in self.shutdown_order:
                component = self.components[component_name]
                await component.stop()
            
            await self._transition_phase(LifecyclePhase.STOPPED)
            
            logger.info("Orchestration system shutdown completed")
            
            # Publish shutdown event
            if self.event_publisher:
                await self.event_publisher("ORCH_SYSTEM_STOPPED", {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": (datetime.now() - self.uptime_start).total_seconds() if self.uptime_start else 0
                })
            
            return True
            
        except Exception as e:
            logger.error(f"System shutdown failed: {e}", exc_info=True)
            return False
    
    async def restart(self) -> bool:
        """Restart the orchestration system."""
        logger.info("Restarting orchestration system...")
        
        if self.running:
            if not await self.shutdown():
                return False
        
        # Wait a moment before restart
        await asyncio.sleep(2)
        
        self.restart_count += 1
        return await self.startup()
    
    async def upgrade(self, upgrade_plan: Dict[str, Any]) -> bool:
        """Perform system upgrade."""
        try:
            logger.info("Starting system upgrade...")
            await self._transition_phase(LifecyclePhase.UPGRADING)
            
            # Implementation would depend on upgrade strategy:
            # - Rolling upgrade
            # - Blue-green deployment
            # - Canary deployment
            
            # Simplified upgrade process
            upgrade_steps = upgrade_plan.get("steps", [])
            
            for step in upgrade_steps:
                logger.info(f"Executing upgrade step: {step.get('name', 'Unknown')}")
                
                # Execute upgrade step
                if not await self._execute_upgrade_step(step):
                    logger.error(f"Upgrade step failed: {step.get('name')}")
                    return False
            
            # Restart system with new version
            if not await self.restart():
                logger.error("Failed to restart system after upgrade")
                return False
            
            logger.info("System upgrade completed successfully")
            
            # Publish upgrade event
            if self.event_publisher:
                await self.event_publisher("ORCH_SYSTEM_UPGRADED", {
                    "upgrade_plan": upgrade_plan,
                    "timestamp": datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            logger.error(f"System upgrade failed: {e}", exc_info=True)
            await self._transition_phase(LifecyclePhase.FAILED)
            return False
    
    async def _execute_upgrade_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single upgrade step."""
        # Placeholder for upgrade step execution
        step_type = step.get("type", "unknown")
        
        if step_type == "component_update":
            component_name = step.get("component")
            if component_name in self.components:
                # Stop component, update, restart
                component = self.components[component_name]
                await component.stop()
                # Update logic would go here
                await asyncio.sleep(1)  # Simulate update
                return await component.start()
        
        elif step_type == "configuration_update":
            # Update configuration
            await asyncio.sleep(0.5)  # Simulate config update
            return True
        
        elif step_type == "database_migration":
            # Run database migrations
            await asyncio.sleep(2)  # Simulate migration
            return True
        
        return True
    
    async def _check_dependencies(self, component: ManagedComponent) -> bool:
        """Check if component dependencies are satisfied."""
        for dep_name in component.dependencies:
            if dep_name not in self.components:
                logger.warning(f"Dependency {dep_name} not found for {component.name} - will proceed anyway")
                continue
            
            dep_component = self.components[dep_name]
            if dep_component.status != ComponentStatus.HEALTHY:
                logger.warning(f"Dependency {dep_name} not healthy for {component.name} - will proceed anyway")
                continue
        
        return True
    
    async def _start_health_monitoring(self):
        """Start health monitoring task."""
        if self._health_task:
            self._health_task.cancel()
        
        self._health_task = asyncio.create_task(self._health_monitoring_loop())
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop."""
        try:
            while self.running:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
        except asyncio.CancelledError:
            logger.debug("Health monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Health monitoring error: {e}", exc_info=True)
    
    async def _perform_health_checks(self):
        """Perform all health checks."""
        try:
            # Check component health
            healthy_components = 0
            total_components = len(self.components)
            
            for component in self.components.values():
                try:
                    health_result = await component.check_health()
                    
                    if health_result.get("status") in ["healthy", "running"]:
                        if component.status == ComponentStatus.UNHEALTHY:
                            logger.info(f"Component {component.name} recovered")
                        component.status = ComponentStatus.HEALTHY
                        healthy_components += 1
                    else:
                        if component.status == ComponentStatus.HEALTHY:
                            logger.warning(f"Component {component.name} became unhealthy")
                        component.status = ComponentStatus.UNHEALTHY
                        
                except Exception as e:
                    logger.error(f"Health check failed for {component.name}: {e}")
                    component.status = ComponentStatus.UNHEALTHY
            
            # Execute global health checks
            for check in self.health_checks.values():
                await self._execute_health_check(check)
            
            # Update overall health
            if healthy_components == total_components:
                self.overall_health = ComponentStatus.HEALTHY
            elif healthy_components > total_components * 0.5:
                self.overall_health = ComponentStatus.DEGRADED
            else:
                self.overall_health = ComponentStatus.UNHEALTHY
                
        except Exception as e:
            logger.error(f"Health check execution failed: {e}")
            self.overall_health = ComponentStatus.UNHEALTHY
    
    async def _execute_health_check(self, check: HealthCheck):
        """Execute a single health check."""
        try:
            check.total_checks += 1
            
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(), 
                    timeout=check.timeout_seconds
                )
            else:
                result = check.check_function()
            
            check.last_check = datetime.now()
            
            if result:
                check.last_result = True
                check.consecutive_failures = 0
            else:
                check.last_result = False
                check.consecutive_failures += 1
                check.total_failures += 1
                
        except Exception as e:
            logger.error(f"Health check {check.name} failed: {e}")
            check.last_result = False
            check.consecutive_failures += 1
            check.total_failures += 1
            check.last_check = datetime.now()
    
    async def _transition_phase(self, new_phase: LifecyclePhase):
        """Transition to a new lifecycle phase."""
        old_phase = self.phase
        self.phase = new_phase
        self.phase_since = datetime.now()
        
        logger.info(f"Lifecycle phase transition: {old_phase.value} -> {new_phase.value}")
        
        # Publish phase transition event
        if self.event_publisher:
            await self.event_publisher("ORCH_PHASE_TRANSITION", {
                "from_phase": old_phase.value,
                "to_phase": new_phase.value,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle status."""
        uptime_seconds = 0
        if self.uptime_start:
            uptime_seconds = (datetime.now() - self.uptime_start).total_seconds()
        
        return {
            "phase": self.phase.value,
            "phase_since": self.phase_since.isoformat(),
            "overall_health": self.overall_health.value,
            "running": self.running,
            "startup_time_seconds": self.startup_time,
            "uptime_seconds": uptime_seconds,
            "restart_count": self.restart_count,
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
            "startup_order": self.startup_order,
            "health_checks": {
                name: {
                    "last_result": check.last_result,
                    "consecutive_failures": check.consecutive_failures,
                    "success_rate": check.success_rate(),
                    "last_check": check.last_check.isoformat() if check.last_check else None
                }
                for name, check in self.health_checks.items()
            },
            "configuration": {
                "startup_timeout": self.startup_timeout,
                "shutdown_timeout": self.shutdown_timeout,
                "health_check_interval": self.health_check_interval
            }
        }