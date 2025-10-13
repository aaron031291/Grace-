"""
Grace Orchestration Watchdog - Error isolation, recovery, and kernel respawn.

Monitors tasks and kernels for failures, implements isolation strategies,
and manages kernel lifecycle for resilient operation.
"""

import asyncio
from datetime import datetime, timedelta

# Optional psutil import for system metrics
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import Dict, List, Any, Callable
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class IsolationStrategy(Enum):
    NONE = "none"
    CIRCUIT_BREAKER = "circuit_breaker"
    SANDBOX = "sandbox"
    QUARANTINE = "quarantine"
    RESTART = "restart"


class KernelStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    ISOLATED = "isolated"
    RESTARTING = "restarting"
    OFFLINE = "offline"


@dataclass
class HealthMetrics:
    """Health metrics for a monitored component."""

    cpu_usage: float
    memory_usage: float
    response_time_ms: float
    error_rate: float
    throughput: float
    last_heartbeat: datetime

    def is_healthy(self, thresholds: Dict[str, float]) -> bool:
        """Check if metrics indicate healthy state."""
        return (
            self.cpu_usage <= thresholds.get("cpu_max", 90.0)
            and self.memory_usage <= thresholds.get("memory_max", 85.0)
            and self.response_time_ms <= thresholds.get("response_time_max", 5000.0)
            and self.error_rate <= thresholds.get("error_rate_max", 0.05)
            and (datetime.now() - self.last_heartbeat).total_seconds()
            <= thresholds.get("heartbeat_max", 60.0)
        )


class MonitoredTask:
    """Task being monitored by the watchdog."""

    def __init__(
        self, task_id: str, loop_id: str, kernel: str, timeout_minutes: int = 30
    ):
        self.task_id = task_id
        self.loop_id = loop_id
        self.kernel = kernel
        self.timeout_minutes = timeout_minutes

        self.started_at = datetime.now()
        self.last_heartbeat = self.started_at
        self.status = "running"
        self.failure_count = 0
        self.isolation_applied = False
        self.callbacks: List[Callable] = []

    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        return (datetime.now() - self.started_at).total_seconds() > (
            self.timeout_minutes * 60
        )

    def is_stale(self, heartbeat_timeout: int = 120) -> bool:
        """Check if task heartbeat is stale."""
        return (
            datetime.now() - self.last_heartbeat
        ).total_seconds() > heartbeat_timeout

    def heartbeat(self):
        """Update last heartbeat."""
        self.last_heartbeat = datetime.now()

    def add_callback(self, callback: Callable):
        """Add callback for task events."""
        self.callbacks.append(callback)

    async def trigger_callbacks(self, event: str, context: Dict[str, Any]):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, self.task_id, context)
                else:
                    callback(event, self.task_id, context)
            except Exception as e:
                logger.error(f"Callback error for task {self.task_id}: {e}")


class KernelMonitor:
    """Monitor for a specific kernel."""

    def __init__(self, kernel_name: str, health_thresholds: Dict[str, float] = None):
        self.kernel_name = kernel_name
        self.health_thresholds = health_thresholds or {
            "cpu_max": 90.0,
            "memory_max": 85.0,
            "response_time_max": 5000.0,
            "error_rate_max": 0.05,
            "heartbeat_max": 60.0,
        }

        self.status = KernelStatus.HEALTHY
        self.last_health_check = datetime.now()
        self.failure_count = 0
        self.consecutive_failures = 0
        self.total_restarts = 0
        self.isolation_strategy = IsolationStrategy.CIRCUIT_BREAKER

        # Health metrics history
        self.health_history: List[HealthMetrics] = []
        self.max_history_size = 100

        # Isolation state
        self.isolated_since = None
        self.isolation_reason = None

    def update_health_metrics(self, metrics: HealthMetrics):
        """Update health metrics for the kernel."""
        self.health_history.append(metrics)
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size :]

        self.last_health_check = datetime.now()

        # Update status based on metrics
        if metrics.is_healthy(self.health_thresholds):
            if self.status in [KernelStatus.DEGRADED, KernelStatus.FAILING]:
                self.consecutive_failures = 0
                self.status = KernelStatus.HEALTHY
                logger.info(f"Kernel {self.kernel_name} recovered to healthy state")
        else:
            self.consecutive_failures += 1
            self.failure_count += 1

            if self.consecutive_failures >= 3:
                self.status = KernelStatus.FAILING
                logger.warning(
                    f"Kernel {self.kernel_name} is failing (consecutive failures: {self.consecutive_failures})"
                )
            elif self.consecutive_failures >= 1:
                self.status = KernelStatus.DEGRADED
                logger.warning(f"Kernel {self.kernel_name} is degraded")

    def should_isolate(self) -> bool:
        """Check if kernel should be isolated."""
        return (
            self.status == KernelStatus.FAILING
            and self.consecutive_failures >= 5
            and not self.is_isolated()
        )

    def is_isolated(self) -> bool:
        """Check if kernel is currently isolated."""
        return self.status == KernelStatus.ISOLATED

    def isolate(self, reason: str):
        """Isolate the kernel."""
        self.status = KernelStatus.ISOLATED
        self.isolated_since = datetime.now()
        self.isolation_reason = reason
        logger.warning(f"Isolated kernel {self.kernel_name}: {reason}")

    def release_isolation(self):
        """Release kernel from isolation."""
        if self.is_isolated():
            self.status = KernelStatus.HEALTHY
            self.isolated_since = None
            self.isolation_reason = None
            self.consecutive_failures = 0
            logger.info(f"Released kernel {self.kernel_name} from isolation")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for the kernel."""
        latest_metrics = self.health_history[-1] if self.health_history else None

        return {
            "kernel": self.kernel_name,
            "status": self.status.value,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "total_restarts": self.total_restarts,
            "isolated": self.is_isolated(),
            "isolation_reason": self.isolation_reason,
            "isolated_since": self.isolated_since.isoformat()
            if self.isolated_since
            else None,
            "last_health_check": self.last_health_check.isoformat(),
            "latest_metrics": {
                "cpu_usage": latest_metrics.cpu_usage if latest_metrics else 0,
                "memory_usage": latest_metrics.memory_usage if latest_metrics else 0,
                "response_time_ms": latest_metrics.response_time_ms
                if latest_metrics
                else 0,
                "error_rate": latest_metrics.error_rate if latest_metrics else 0,
                "last_heartbeat": latest_metrics.last_heartbeat.isoformat()
                if latest_metrics
                else None,
            }
            if latest_metrics
            else None,
        }


class Watchdog:
    """Orchestration watchdog for error isolation and recovery."""

    def __init__(self, event_publisher=None, kernel_registry=None):
        self.event_publisher = event_publisher
        self.kernel_registry = kernel_registry

        # Monitoring state
        self.monitored_tasks: Dict[str, MonitoredTask] = {}
        self.kernel_monitors: Dict[str, KernelMonitor] = {}

        # Configuration
        self.health_check_interval = 30  # seconds
        self.task_check_interval = 10  # seconds
        self.isolation_timeout = 300  # seconds (5 minutes)

        # Statistics
        self.total_tasks_monitored = 0
        self.total_failures_detected = 0
        self.total_isolations = 0
        self.total_recoveries = 0

        # Control
        self.running = False
        self._monitoring_tasks: List[asyncio.Task] = []

        # Initialize kernel monitors for known kernels
        self._initialize_kernel_monitors()

    def _initialize_kernel_monitors(self):
        """Initialize monitors for known Grace kernels."""
        known_kernels = [
            "orchestration",
            "governance",
            "memory",
            "learning",
            "intelligence",
            "interface",
            "ingress",
            "mlt",
            "mtl",
            "multi_os",
            "resilience",
            "event_mesh",
        ]

        for kernel in known_kernels:
            self.kernel_monitors[kernel] = KernelMonitor(kernel)

    async def start(self):
        """Start the watchdog."""
        if self.running:
            logger.warning("Watchdog already running")
            return

        logger.info("Starting orchestration watchdog...")
        self.running = True

        # Start monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._task_monitor_loop()),
            asyncio.create_task(self._health_monitor_loop()),
            asyncio.create_task(self._isolation_manager_loop()),
        ]

        logger.info("Orchestration watchdog started")

    async def stop(self):
        """Stop the watchdog."""
        if not self.running:
            return

        logger.info("Stopping orchestration watchdog...")
        self.running = False

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        self._monitoring_tasks.clear()

        logger.info("Orchestration watchdog stopped")

    def monitor(
        self,
        task_id: str,
        loop_id: str = None,
        kernel: str = None,
        callback: Callable = None,
        timeout_minutes: int = 30,
    ):
        """Start monitoring a task."""
        if not kernel:
            kernel = "unknown"

        task = MonitoredTask(task_id, loop_id, kernel, timeout_minutes)

        if callback:
            task.add_callback(callback)

        self.monitored_tasks[task_id] = task
        self.total_tasks_monitored += 1

        logger.debug(f"Monitoring task {task_id} on kernel {kernel}")

    def unmonitor(self, task_id: str):
        """Stop monitoring a task."""
        if task_id in self.monitored_tasks:
            del self.monitored_tasks[task_id]
            logger.debug(f"Stopped monitoring task {task_id}")

    def heartbeat_task(self, task_id: str):
        """Update task heartbeat."""
        if task_id in self.monitored_tasks:
            self.monitored_tasks[task_id].heartbeat()

    async def isolate(
        self,
        kernel: str,
        strategy: IsolationStrategy = IsolationStrategy.CIRCUIT_BREAKER,
        reason: str = "Manual isolation",
    ):
        """Isolate a kernel."""
        if kernel not in self.kernel_monitors:
            self.kernel_monitors[kernel] = KernelMonitor(kernel)

        monitor = self.kernel_monitors[kernel]
        monitor.isolation_strategy = strategy
        monitor.isolate(reason)

        self.total_isolations += 1

        # Publish isolation event
        if self.event_publisher:
            await self.event_publisher(
                "KERNEL_ISOLATED",
                {
                    "kernel": kernel,
                    "strategy": strategy.value,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        logger.warning(
            f"Isolated kernel {kernel} with strategy {strategy.value}: {reason}"
        )

    async def respawn(self, kernel: str) -> bool:
        """Respawn a kernel."""
        try:
            logger.info(f"Attempting to respawn kernel: {kernel}")

            if kernel not in self.kernel_monitors:
                self.kernel_monitors[kernel] = KernelMonitor(kernel)

            monitor = self.kernel_monitors[kernel]
            monitor.status = KernelStatus.RESTARTING
            monitor.total_restarts += 1

            # In a real implementation, this would:
            # 1. Stop the kernel gracefully
            # 2. Clean up resources
            # 3. Restart the kernel process
            # 4. Verify successful startup

            # Simulate restart process
            await asyncio.sleep(2)

            # Update status
            monitor.status = KernelStatus.HEALTHY
            monitor.consecutive_failures = 0
            monitor.release_isolation()

            # Publish respawn event
            if self.event_publisher:
                await self.event_publisher(
                    "KERNEL_RESPAWNED",
                    {
                        "kernel": kernel,
                        "restart_count": monitor.total_restarts,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            self.total_recoveries += 1
            logger.info(f"Successfully respawned kernel: {kernel}")
            return True

        except Exception as e:
            logger.error(f"Failed to respawn kernel {kernel}: {e}")
            return False

    def update_kernel_health(self, kernel: str, metrics: HealthMetrics):
        """Update health metrics for a kernel."""
        if kernel not in self.kernel_monitors:
            self.kernel_monitors[kernel] = KernelMonitor(kernel)

        self.kernel_monitors[kernel].update_health_metrics(metrics)

    async def _task_monitor_loop(self):
        """Monitor tasks for failures and timeouts."""
        try:
            while self.running:
                current_time = datetime.now()
                expired_tasks = []
                stale_tasks = []

                for task_id, task in self.monitored_tasks.items():
                    if task.is_expired():
                        expired_tasks.append(task_id)
                    elif task.is_stale():
                        stale_tasks.append(task_id)

                # Handle expired tasks
                for task_id in expired_tasks:
                    await self._handle_expired_task(task_id)

                # Handle stale tasks
                for task_id in stale_tasks:
                    await self._handle_stale_task(task_id)

                await asyncio.sleep(self.task_check_interval)

        except asyncio.CancelledError:
            logger.debug("Task monitor loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Task monitor loop error: {e}", exc_info=True)

    async def _health_monitor_loop(self):
        """Monitor kernel health and detect failures."""
        try:
            while self.running:
                for kernel_name, monitor in self.kernel_monitors.items():
                    # Collect health metrics (simplified - would integrate with actual kernel APIs)
                    try:
                        metrics = await self._collect_kernel_metrics(kernel_name)
                        monitor.update_health_metrics(metrics)

                        # Check if isolation is needed
                        if monitor.should_isolate():
                            await self.isolate(
                                kernel_name,
                                monitor.isolation_strategy,
                                f"Consecutive failures: {monitor.consecutive_failures}",
                            )

                    except Exception as e:
                        logger.error(f"Error monitoring kernel {kernel_name}: {e}")

                await asyncio.sleep(self.health_check_interval)

        except asyncio.CancelledError:
            logger.debug("Health monitor loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Health monitor loop error: {e}", exc_info=True)

    async def _isolation_manager_loop(self):
        """Manage isolated kernels and recovery attempts."""
        try:
            while self.running:
                current_time = datetime.now()

                for kernel_name, monitor in self.kernel_monitors.items():
                    if monitor.is_isolated():
                        # Check if isolation timeout has passed
                        isolation_duration = (
                            current_time - monitor.isolated_since
                        ).total_seconds()

                        if isolation_duration > self.isolation_timeout:
                            # Attempt recovery
                            logger.info(
                                f"Attempting recovery for isolated kernel: {kernel_name}"
                            )
                            success = await self.respawn(kernel_name)

                            if not success:
                                # Reset isolation timer
                                monitor.isolated_since = current_time

                await asyncio.sleep(30)  # Check every 30 seconds

        except asyncio.CancelledError:
            logger.debug("Isolation manager loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Isolation manager loop error: {e}", exc_info=True)

    async def _handle_expired_task(self, task_id: str):
        """Handle an expired task."""
        task = self.monitored_tasks.get(task_id)
        if not task:
            return

        logger.warning(f"Task {task_id} expired after {task.timeout_minutes} minutes")

        task.status = "expired"
        self.total_failures_detected += 1

        # Trigger callbacks
        await task.trigger_callbacks(
            "expired", {"timeout_minutes": task.timeout_minutes, "kernel": task.kernel}
        )

        # Update kernel failure count
        if task.kernel in self.kernel_monitors:
            self.kernel_monitors[task.kernel].failure_count += 1

        # Remove from monitoring
        del self.monitored_tasks[task_id]

        # Publish event
        if self.event_publisher:
            await self.event_publisher(
                "TASK_EXPIRED",
                {
                    "task_id": task_id,
                    "loop_id": task.loop_id,
                    "kernel": task.kernel,
                    "timeout_minutes": task.timeout_minutes,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def _handle_stale_task(self, task_id: str):
        """Handle a stale task (missing heartbeats)."""
        task = self.monitored_tasks.get(task_id)
        if not task:
            return

        logger.warning(f"Task {task_id} is stale (no heartbeat)")

        # Trigger callbacks
        await task.trigger_callbacks(
            "stale",
            {"last_heartbeat": task.last_heartbeat.isoformat(), "kernel": task.kernel},
        )

    async def _collect_kernel_metrics(self, kernel_name: str) -> HealthMetrics:
        """Collect health metrics for a kernel (simplified implementation)."""
        # In a real implementation, this would:
        # 1. Call kernel health endpoints
        # 2. Collect system metrics
        # 3. Measure response times
        # 4. Calculate error rates

        # Simulate metrics collection
        try:
            # Get system metrics if psutil is available
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
            else:
                # Fallback values when psutil not available
                cpu_percent = 15.0
                memory_percent = 45.0

            # Simulate kernel-specific metrics
            response_time = 100.0  # ms
            error_rate = 0.01  # 1%
            throughput = 100.0  # requests/sec

            return HealthMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                response_time_ms=response_time,
                error_rate=error_rate,
                throughput=throughput,
                last_heartbeat=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to collect metrics for {kernel_name}: {e}")
            # Return degraded metrics on collection failure
            return HealthMetrics(
                cpu_usage=100.0,
                memory_usage=100.0,
                response_time_ms=10000.0,
                error_rate=1.0,
                throughput=0.0,
                last_heartbeat=datetime.now() - timedelta(minutes=10),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get watchdog status and metrics."""
        return {
            "running": self.running,
            "monitored_tasks": len(self.monitored_tasks),
            "kernel_monitors": len(self.kernel_monitors),
            "statistics": {
                "total_tasks_monitored": self.total_tasks_monitored,
                "total_failures_detected": self.total_failures_detected,
                "total_isolations": self.total_isolations,
                "total_recoveries": self.total_recoveries,
            },
            "kernel_health": {
                kernel: monitor.get_health_summary()
                for kernel, monitor in self.kernel_monitors.items()
            },
            "active_tasks": [
                {
                    "task_id": task.task_id,
                    "loop_id": task.loop_id,
                    "kernel": task.kernel,
                    "status": task.status,
                    "started_at": task.started_at.isoformat(),
                    "last_heartbeat": task.last_heartbeat.isoformat(),
                }
                for task in self.monitored_tasks.values()
            ],
        }
