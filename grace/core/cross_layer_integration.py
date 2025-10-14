"""
Cross-Layer Integration System - Seamless kernel connectivity.

Implements comprehensive cross-layer integration as specified in the missing
components requirements. Features:
- Unified kernel interface protocol
- Inter-kernel communication bus
- State synchronization mechanisms
- Cross-layer event routing
- Dependency management
- Health monitoring across layers
- Performance optimization coordination
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid

logger = logging.getLogger(__name__)


class KernelType(Enum):
    """Types of kernels in the Grace system."""

    GOVERNANCE = "governance"
    ORCHESTRATION = "orchestration"
    INTELLIGENCE = "intelligence"
    MEMORY = "memory"
    RESILIENCE = "resilience"
    INGRESS = "ingress"
    INTERFACE = "interface"
    LEARNING = "learning"
    MLDL = "mldl"
    MLT = "mlt"
    MTL = "mtl"
    MULTI_OS = "multi_os"
    EVENT_MESH = "event_mesh"
    AUDIT_LOGS = "audit_logs"


class IntegrationStatus(Enum):
    """Status of kernel integration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCHRONIZING = "synchronizing"
    SYNCHRONIZED = "synchronized"
    DEGRADED = "degraded"
    FAILED = "failed"


class MessageType(Enum):
    """Types of inter-kernel messages."""

    HEARTBEAT = "heartbeat"
    STATE_SYNC = "state_sync"
    EVENT_NOTIFICATION = "event_notification"
    REQUEST = "request"
    RESPONSE = "response"
    COMMAND = "command"
    STATUS_UPDATE = "status_update"
    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"


@dataclass
class KernelCapability:
    """Represents a capability provided by a kernel."""

    capability_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"


@dataclass
class IntegrationMessage:
    """Message structure for inter-kernel communication."""

    message_id: str
    source_kernel: str
    target_kernel: str
    message_type: MessageType
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source_kernel": self.source_kernel,
            "target_kernel": self.target_kernel,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationMessage":
        return cls(
            message_id=data["message_id"],
            source_kernel=data["source_kernel"],
            target_kernel=data["target_kernel"],
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data.get("priority", 0),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )


@dataclass
class KernelHealthStatus:
    """Health status of a kernel."""

    kernel_id: str
    kernel_type: KernelType
    status: IntegrationStatus
    last_heartbeat: datetime
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    error_count: int
    uptime_seconds: float
    version: str
    capabilities: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if kernel is in healthy state."""
        return (
            self.status in [IntegrationStatus.CONNECTED, IntegrationStatus.SYNCHRONIZED]
            and (datetime.now() - self.last_heartbeat).total_seconds() < 30
            and self.response_time_ms < 1000
            and self.cpu_usage_percent < 90
        )


class KernelInterface(ABC):
    """Abstract interface that all kernels must implement for integration."""

    def __init__(self, kernel_id: str, kernel_type: KernelType):
        self.kernel_id = kernel_id
        self.kernel_type = kernel_type
        self.capabilities: Dict[str, KernelCapability] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.integration_bus: Optional["CrossLayerIntegrationBus"] = None

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the kernel."""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Gracefully shutdown the kernel."""
        pass

    @abstractmethod
    async def get_health_status(self) -> KernelHealthStatus:
        """Get current health status."""
        pass

    @abstractmethod
    async def handle_message(
        self, message: IntegrationMessage
    ) -> Optional[IntegrationMessage]:
        """Handle an inter-kernel message."""
        pass

    def register_capability(self, capability: KernelCapability) -> None:
        """Register a capability provided by this kernel."""
        self.capabilities[capability.capability_id] = capability
        logger.info(
            f"Registered capability {capability.capability_id} for kernel {self.kernel_id}"
        )

    def register_message_handler(
        self, message_type: MessageType, handler: Callable
    ) -> None:
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(
            f"Registered {message_type.value} handler for kernel {self.kernel_id}"
        )

    async def send_message(
        self,
        target_kernel: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """Send a message to another kernel."""
        if not self.integration_bus:
            raise RuntimeError("Kernel not connected to integration bus")

        message = IntegrationMessage(
            message_id=str(uuid.uuid4()),
            source_kernel=self.kernel_id,
            target_kernel=target_kernel,
            message_type=message_type,
            payload=payload,
            priority=priority,
        )

        await self.integration_bus.route_message(message)
        return message.message_id

    async def broadcast_message(
        self, message_type: MessageType, payload: Dict[str, Any]
    ) -> None:
        """Broadcast a message to all connected kernels."""
        if not self.integration_bus:
            raise RuntimeError("Kernel not connected to integration bus")

        await self.integration_bus.broadcast_message(
            self.kernel_id, message_type, payload
        )


class DependencyResolver:
    """Resolves and manages kernel dependencies."""

    def __init__(self):
        self.dependencies: Dict[
            str, Set[str]
        ] = {}  # kernel_id -> set of dependency kernel_ids
        self.dependents: Dict[
            str, Set[str]
        ] = {}  # kernel_id -> set of dependent kernel_ids
        self.initialization_order: List[str] = []

    def add_dependency(self, kernel_id: str, dependency_id: str) -> None:
        """Add a dependency relationship."""
        if kernel_id not in self.dependencies:
            self.dependencies[kernel_id] = set()
        if dependency_id not in self.dependents:
            self.dependents[dependency_id] = set()

        self.dependencies[kernel_id].add(dependency_id)
        self.dependents[dependency_id].add(kernel_id)

        # Invalidate initialization order
        self.initialization_order = []

    def get_initialization_order(self, kernel_ids: List[str]) -> List[str]:
        """Get the correct initialization order based on dependencies."""
        if self.initialization_order:
            return [k for k in self.initialization_order if k in kernel_ids]

        # Topological sort for dependency resolution
        in_degree = {k: 0 for k in kernel_ids}

        # Calculate in-degrees
        for kernel_id in kernel_ids:
            for dep in self.dependencies.get(kernel_id, set()):
                if dep in kernel_ids:
                    in_degree[kernel_id] += 1

        # Initialize queue with kernels that have no dependencies
        queue = [k for k in kernel_ids if in_degree[k] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Update in-degrees of dependents
            for dependent in self.dependents.get(current, set()):
                if dependent in kernel_ids:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for circular dependencies
        if len(result) != len(kernel_ids):
            remaining = set(kernel_ids) - set(result)
            logger.warning(f"Circular dependency detected among kernels: {remaining}")
            result.extend(remaining)  # Add remaining kernels anyway

        self.initialization_order = result
        return result

    def get_shutdown_order(self, kernel_ids: List[str]) -> List[str]:
        """Get the correct shutdown order (reverse of initialization)."""
        init_order = self.get_initialization_order(kernel_ids)
        return list(reversed(init_order))


class StateManager:
    """Manages state synchronization across kernels."""

    def __init__(self):
        self.kernel_states: Dict[str, Dict[str, Any]] = {}
        self.state_locks: Dict[str, asyncio.Lock] = {}
        self.state_subscribers: Dict[
            str, Set[str]
        ] = {}  # state_key -> set of kernel_ids
        self.sync_handlers: Dict[str, Callable] = {}

    async def register_state(
        self, kernel_id: str, state_key: str, initial_state: Any
    ) -> None:
        """Register a state that needs to be synchronized."""
        if state_key not in self.state_locks:
            self.state_locks[state_key] = asyncio.Lock()

        async with self.state_locks[state_key]:
            if state_key not in self.kernel_states:
                self.kernel_states[state_key] = {}

            self.kernel_states[state_key][kernel_id] = initial_state

    async def update_state(
        self,
        kernel_id: str,
        state_key: str,
        new_state: Any,
        notify_subscribers: bool = True,
    ) -> None:
        """Update kernel state and optionally notify subscribers."""
        if state_key not in self.state_locks:
            logger.warning(f"State key {state_key} not registered")
            return

        async with self.state_locks[state_key]:
            if state_key not in self.kernel_states:
                self.kernel_states[state_key] = {}

            old_state = self.kernel_states[state_key].get(kernel_id)
            self.kernel_states[state_key][kernel_id] = new_state

            if notify_subscribers and state_key in self.state_subscribers:
                for subscriber_id in self.state_subscribers[state_key]:
                    if (
                        subscriber_id != kernel_id
                        and subscriber_id in self.sync_handlers
                    ):
                        try:
                            await self.sync_handlers[subscriber_id](
                                state_key, kernel_id, old_state, new_state
                            )
                        except Exception as e:
                            logger.error(f"State sync failed for {subscriber_id}: {e}")

    async def get_state(self, state_key: str, kernel_id: Optional[str] = None) -> Any:
        """Get current state value."""
        if state_key not in self.state_locks:
            return None

        async with self.state_locks[state_key]:
            states = self.kernel_states.get(state_key, {})
            if kernel_id:
                return states.get(kernel_id)
            return states

    def subscribe_to_state(
        self, subscriber_id: str, state_key: str, sync_handler: Callable
    ) -> None:
        """Subscribe to state changes."""
        if state_key not in self.state_subscribers:
            self.state_subscribers[state_key] = set()

        self.state_subscribers[state_key].add(subscriber_id)
        self.sync_handlers[subscriber_id] = sync_handler

        logger.debug(f"Kernel {subscriber_id} subscribed to state {state_key}")


class CrossLayerIntegrationBus:
    """Central integration bus connecting all kernels."""

    def __init__(self, event_bus=None, memory_core=None):
        self.event_bus = event_bus
        self.memory_core = memory_core

        # Kernel management
        self.kernels: Dict[str, KernelInterface] = {}
        self.kernel_health: Dict[str, KernelHealthStatus] = {}

        # Integration infrastructure
        self.dependency_resolver = DependencyResolver()
        self.state_manager = StateManager()

        # Message routing
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.message_handlers: Dict[str, asyncio.Task] = {}
        self.broadcast_subscribers: Dict[MessageType, Set[str]] = {}

        # Performance monitoring
        self.message_stats = {
            "total_messages": 0,
            "messages_by_type": {},
            "avg_latency_ms": 0.0,
            "error_count": 0,
            "start_time": datetime.now(),
        }

        # System state
        self.running = False
        self.integration_status = IntegrationStatus.DISCONNECTED

    async def start(self) -> None:
        """Start the integration bus."""
        if self.running:
            return

        self.running = True
        self.integration_status = IntegrationStatus.CONNECTING

        # Start health monitoring
        health_monitor = asyncio.create_task(self._health_monitor())
        self.message_handlers["health_monitor"] = health_monitor

        # Start performance monitoring
        perf_monitor = asyncio.create_task(self._performance_monitor())
        self.message_handlers["perf_monitor"] = perf_monitor

        self.integration_status = IntegrationStatus.CONNECTED
        logger.info("Cross-layer integration bus started")

    async def stop(self) -> None:
        """Stop the integration bus."""
        if not self.running:
            return

        self.running = False
        self.integration_status = IntegrationStatus.DISCONNECTED

        # Shutdown all kernels in correct order
        kernel_ids = list(self.kernels.keys())
        shutdown_order = self.dependency_resolver.get_shutdown_order(kernel_ids)

        for kernel_id in shutdown_order:
            try:
                kernel = self.kernels[kernel_id]
                await kernel.shutdown()
                logger.info(f"Shutdown kernel: {kernel_id}")
            except Exception as e:
                logger.error(f"Error shutting down kernel {kernel_id}: {e}")

        # Cancel all message handlers
        for task in self.message_handlers.values():
            task.cancel()

        await asyncio.gather(*self.message_handlers.values(), return_exceptions=True)

        logger.info("Cross-layer integration bus stopped")

    async def register_kernel(self, kernel: KernelInterface) -> None:
        """Register a kernel with the integration bus."""
        kernel_id = kernel.kernel_id

        # Connect kernel to bus
        kernel.integration_bus = self

        # Create message queue for kernel
        self.message_queues[kernel_id] = asyncio.Queue(maxsize=1000)

        # Start message handler for kernel
        handler = asyncio.create_task(self._kernel_message_handler(kernel))
        self.message_handlers[kernel_id] = handler

        # Register kernel
        self.kernels[kernel_id] = kernel

        # Initialize kernel health status
        self.kernel_health[kernel_id] = await kernel.get_health_status()

        logger.info(f"Registered kernel: {kernel_id} ({kernel.kernel_type.value})")

        # If all kernels are registered, perform initialization
        await self._check_and_initialize()

    async def unregister_kernel(self, kernel_id: str) -> None:
        """Unregister a kernel from the integration bus."""
        if kernel_id not in self.kernels:
            return

        # Cancel message handler
        if kernel_id in self.message_handlers:
            self.message_handlers[kernel_id].cancel()
            del self.message_handlers[kernel_id]

        # Clean up message queue
        if kernel_id in self.message_queues:
            del self.message_queues[kernel_id]

        # Remove kernel
        del self.kernels[kernel_id]

        if kernel_id in self.kernel_health:
            del self.kernel_health[kernel_id]

        logger.info(f"Unregistered kernel: {kernel_id}")

    async def route_message(self, message: IntegrationMessage) -> None:
        """Route a message to its target kernel."""
        target_kernel = message.target_kernel

        if target_kernel not in self.message_queues:
            logger.warning(f"Target kernel {target_kernel} not registered")
            return

        try:
            # Check if message has expired
            if message.expires_at and datetime.now() > message.expires_at:
                logger.debug(f"Message {message.message_id} expired")
                return

            # Add to target kernel's message queue
            await self.message_queues[target_kernel].put(message)

            # Update statistics
            self.message_stats["total_messages"] += 1
            msg_type = message.message_type.value
            if msg_type not in self.message_stats["messages_by_type"]:
                self.message_stats["messages_by_type"][msg_type] = 0
            self.message_stats["messages_by_type"][msg_type] += 1

        except asyncio.QueueFull:
            logger.warning(f"Message queue full for kernel {target_kernel}")
            self.message_stats["error_count"] += 1
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            self.message_stats["error_count"] += 1

    async def broadcast_message(
        self, source_kernel: str, message_type: MessageType, payload: Dict[str, Any]
    ) -> None:
        """Broadcast a message to all subscribed kernels."""
        if message_type not in self.broadcast_subscribers:
            return

        for target_kernel in self.broadcast_subscribers[message_type]:
            if target_kernel != source_kernel:
                message = IntegrationMessage(
                    message_id=str(uuid.uuid4()),
                    source_kernel=source_kernel,
                    target_kernel=target_kernel,
                    message_type=message_type,
                    payload=payload,
                )
                await self.route_message(message)

    def subscribe_to_broadcasts(
        self, kernel_id: str, message_type: MessageType
    ) -> None:
        """Subscribe a kernel to broadcast messages of a specific type."""
        if message_type not in self.broadcast_subscribers:
            self.broadcast_subscribers[message_type] = set()

        self.broadcast_subscribers[message_type].add(kernel_id)
        logger.debug(
            f"Kernel {kernel_id} subscribed to {message_type.value} broadcasts"
        )

    async def _kernel_message_handler(self, kernel: KernelInterface) -> None:
        """Handle messages for a specific kernel."""
        kernel_id = kernel.kernel_id
        message_queue = self.message_queues[kernel_id]

        logger.debug(f"Started message handler for kernel {kernel_id}")

        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(message_queue.get(), timeout=5.0)

                start_time = datetime.now()

                # Handle message
                response = await kernel.handle_message(message)

                # Calculate latency
                latency = (datetime.now() - start_time).total_seconds() * 1000

                # Update performance metrics
                self._update_latency_stats(latency)

                # Send response if provided
                if response:
                    await self.route_message(response)

            except asyncio.TimeoutError:
                # Timeout is normal, continue
                continue
            except Exception as e:
                logger.error(f"Message handler error for kernel {kernel_id}: {e}")
                self.message_stats["error_count"] += 1

        logger.debug(f"Stopped message handler for kernel {kernel_id}")

    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update average latency statistics."""
        total_msgs = self.message_stats["total_messages"]
        current_avg = self.message_stats["avg_latency_ms"]

        # Exponential moving average
        if total_msgs == 0:
            self.message_stats["avg_latency_ms"] = latency_ms
        else:
            alpha = 0.1  # Smoothing factor
            self.message_stats["avg_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * current_avg
            )

    async def _health_monitor(self) -> None:
        """Monitor health of all registered kernels."""
        logger.info("Started integration health monitor")

        while self.running:
            try:
                healthy_kernels = 0
                total_kernels = len(self.kernels)

                for kernel_id, kernel in self.kernels.items():
                    try:
                        # Get health status
                        health_status = await kernel.get_health_status()
                        self.kernel_health[kernel_id] = health_status

                        if health_status.is_healthy():
                            healthy_kernels += 1
                        else:
                            logger.warning(
                                f"Kernel {kernel_id} is unhealthy: {health_status.status.value}"
                            )

                    except Exception as e:
                        logger.error(f"Health check failed for kernel {kernel_id}: {e}")
                        # Mark as failed
                        if kernel_id in self.kernel_health:
                            self.kernel_health[
                                kernel_id
                            ].status = IntegrationStatus.FAILED

                # Update overall integration status
                if total_kernels == 0:
                    self.integration_status = IntegrationStatus.DISCONNECTED
                elif healthy_kernels == total_kernels:
                    self.integration_status = IntegrationStatus.SYNCHRONIZED
                elif healthy_kernels > total_kernels * 0.5:
                    self.integration_status = IntegrationStatus.DEGRADED
                else:
                    self.integration_status = IntegrationStatus.FAILED

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)

    async def _performance_monitor(self) -> None:
        """Monitor performance metrics and log statistics."""
        logger.info("Started integration performance monitor")

        while self.running:
            try:
                # Log performance statistics every 60 seconds
                await asyncio.sleep(60)

                stats = self.message_stats
                uptime = (datetime.now() - stats["start_time"]).total_seconds()

                if stats["total_messages"] > 0:
                    throughput = stats["total_messages"] / uptime
                    error_rate = stats["error_count"] / stats["total_messages"]

                    logger.info(
                        f"Integration stats: {stats['total_messages']} messages, "
                        f"{throughput:.1f} msg/sec, {stats['avg_latency_ms']:.1f}ms avg latency, "
                        f"{error_rate:.2%} error rate"
                    )

            except Exception as e:
                logger.error(f"Performance monitor error: {e}")

    async def _check_and_initialize(self) -> None:
        """Check if all required kernels are registered and initialize system."""
        # This could be enhanced to wait for specific required kernels
        kernel_ids = list(self.kernels.keys())

        if len(kernel_ids) >= 3:  # Minimum viable set
            initialization_order = self.dependency_resolver.get_initialization_order(
                kernel_ids
            )

            for kernel_id in initialization_order:
                try:
                    kernel = self.kernels[kernel_id]
                    success = await kernel.initialize()
                    if success:
                        logger.info(f"Initialized kernel: {kernel_id}")
                    else:
                        logger.error(f"Failed to initialize kernel: {kernel_id}")
                except Exception as e:
                    logger.error(f"Error initializing kernel {kernel_id}: {e}")

            self.integration_status = IntegrationStatus.SYNCHRONIZING

    def add_kernel_dependency(self, kernel_id: str, dependency_id: str) -> None:
        """Add a dependency between kernels."""
        self.dependency_resolver.add_dependency(kernel_id, dependency_id)

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        kernel_statuses = {}
        for kernel_id, health in self.kernel_health.items():
            kernel_statuses[kernel_id] = {
                "type": health.kernel_type.value,
                "status": health.status.value,
                "healthy": health.is_healthy(),
                "last_heartbeat": health.last_heartbeat.isoformat(),
                "response_time_ms": health.response_time_ms,
                "uptime_seconds": health.uptime_seconds,
            }

        return {
            "overall_status": self.integration_status.value,
            "total_kernels": len(self.kernels),
            "healthy_kernels": sum(
                1 for h in self.kernel_health.values() if h.is_healthy()
            ),
            "kernel_statuses": kernel_statuses,
            "message_stats": self.message_stats,
            "uptime_seconds": (
                datetime.now() - self.message_stats["start_time"]
            ).total_seconds(),
        }


# Global integration bus instance
integration_bus = CrossLayerIntegrationBus()
