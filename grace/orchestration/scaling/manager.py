"""
Grace Orchestration Scaling Manager - Multi-instance orchestration and load balancing.

Manages scaling of orchestration loops across multiple instances,
implements load balancing strategies, and handles instance lifecycle.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    NONE = "none"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ADAPTIVE = "adaptive"


class LoadBalanceAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"


class InstanceStatus(Enum):
    STARTING = "starting"
    ACTIVE = "active"
    DRAINING = "draining"
    RETIRING = "retiring"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class InstanceMetrics:
    """Metrics for a running instance."""

    cpu_usage: float
    memory_usage: float
    active_tasks: int
    completed_tasks: int
    response_time_avg: float
    error_rate: float
    last_heartbeat: datetime

    def get_load_score(self) -> float:
        """Calculate load score for load balancing (lower is better)."""
        return (
            self.cpu_usage * 0.3
            + self.memory_usage * 0.3
            + (self.active_tasks * 10) * 0.2
            + self.response_time_avg * 0.1
            + self.error_rate * 100 * 0.1
        )


class OrchestrationInstance:
    """Represents a running orchestration instance."""

    def __init__(self, instance_id: str, loop_id: str, parent_id: str = None):
        self.instance_id = instance_id
        self.loop_id = loop_id
        self.parent_id = parent_id

        self.status = InstanceStatus.STARTING
        self.created_at = datetime.now()
        self.started_at = None
        self.stopped_at = None

        self.metrics = InstanceMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            active_tasks=0,
            completed_tasks=0,
            response_time_avg=0.0,
            error_rate=0.0,
            last_heartbeat=datetime.now(),
        )

        # Configuration
        self.max_tasks = 10
        self.weight = 1.0

        # Lifecycle callbacks
        self.on_status_change = None

    def update_metrics(self, metrics: InstanceMetrics):
        """Update instance metrics."""
        self.metrics = metrics

    def can_accept_tasks(self) -> bool:
        """Check if instance can accept new tasks."""
        return (
            self.status == InstanceStatus.ACTIVE
            and self.metrics.active_tasks < self.max_tasks
        )

    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        if self.status != InstanceStatus.ACTIVE:
            return False

        heartbeat_age = (datetime.now() - self.metrics.last_heartbeat).total_seconds()
        return (
            heartbeat_age < 120  # 2 minutes
            and self.metrics.error_rate < 0.1
            and self.metrics.cpu_usage < 95
        )

    async def transition_status(self, new_status: InstanceStatus):
        """Transition to new status."""
        old_status = self.status
        self.status = new_status

        if new_status == InstanceStatus.ACTIVE:
            self.started_at = datetime.now()
        elif new_status in [InstanceStatus.STOPPED, InstanceStatus.FAILED]:
            self.stopped_at = datetime.now()

        if self.on_status_change:
            await self.on_status_change(self, old_status, new_status)

        logger.info(
            f"Instance {self.instance_id} transitioned: {old_status.value} -> {new_status.value}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "instance_id": self.instance_id,
            "loop_id": self.loop_id,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "metrics": {
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "active_tasks": self.metrics.active_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "response_time_avg": self.metrics.response_time_avg,
                "error_rate": self.metrics.error_rate,
                "load_score": self.metrics.get_load_score(),
                "last_heartbeat": self.metrics.last_heartbeat.isoformat(),
            },
            "max_tasks": self.max_tasks,
            "weight": self.weight,
        }


class LoadBalancer:
    """Load balancer for orchestration instances."""

    def __init__(
        self, algorithm: LoadBalanceAlgorithm = LoadBalanceAlgorithm.LEAST_CONNECTIONS
    ):
        self.algorithm = algorithm
        self.round_robin_index = 0

    def select_instance(
        self, instances: List[OrchestrationInstance]
    ) -> Optional[OrchestrationInstance]:
        """Select the best instance for a new task."""
        # Filter healthy instances that can accept tasks
        available_instances = [
            instance
            for instance in instances
            if instance.can_accept_tasks() and instance.is_healthy()
        ]

        if not available_instances:
            return None

        if self.algorithm == LoadBalanceAlgorithm.ROUND_ROBIN:
            return self._round_robin_selection(available_instances)
        elif self.algorithm == LoadBalanceAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_instances)
        elif self.algorithm == LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_instances)
        elif self.algorithm == LoadBalanceAlgorithm.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_instances)

        return available_instances[0]  # Fallback

    def _round_robin_selection(
        self, instances: List[OrchestrationInstance]
    ) -> OrchestrationInstance:
        """Round-robin selection."""
        if not instances:
            return None

        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return instance

    def _least_connections_selection(
        self, instances: List[OrchestrationInstance]
    ) -> OrchestrationInstance:
        """Select instance with least active connections."""
        return min(instances, key=lambda i: i.metrics.active_tasks)

    def _weighted_round_robin_selection(
        self, instances: List[OrchestrationInstance]
    ) -> OrchestrationInstance:
        """Weighted round-robin selection."""
        # Simplified implementation - can be enhanced with proper weight tracking
        weighted_instances = []
        for instance in instances:
            weight_factor = max(1, int(instance.weight))
            weighted_instances.extend([instance] * weight_factor)

        return self._round_robin_selection(weighted_instances)

    def _least_response_time_selection(
        self, instances: List[OrchestrationInstance]
    ) -> OrchestrationInstance:
        """Select instance with lowest response time."""
        return min(instances, key=lambda i: i.metrics.response_time_avg)


class ScalingManager:
    """Multi-instance orchestration and load balancing manager."""

    def __init__(self, event_publisher=None, scheduler=None):
        self.event_publisher = event_publisher
        self.scheduler = scheduler

        # Instance management
        self.instances: Dict[str, OrchestrationInstance] = {}
        self.load_balancer = LoadBalancer()

        # Scaling configuration
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self.default_scaling_policy = {
            "min_instances": 1,
            "max_instances": 5,
            "target_cpu_utilization": 70.0,
            "target_task_load": 8,
            "scale_up_threshold": 80.0,
            "scale_down_threshold": 30.0,
            "cooldown_minutes": 5,
        }

        # State
        self.running = False
        self._scaling_task = None
        self._last_scaling_actions: Dict[str, datetime] = {}

        # Statistics
        self.total_instances_created = 0
        self.total_instances_retired = 0
        self.total_scaling_actions = 0

    async def start(self):
        """Start the scaling manager."""
        if self.running:
            logger.warning("Scaling manager already running")
            return

        logger.info("Starting orchestration scaling manager...")
        self.running = True

        # Start scaling loop
        self._scaling_task = asyncio.create_task(self._scaling_loop())

        logger.info("Orchestration scaling manager started")

    async def stop(self):
        """Stop the scaling manager."""
        if not self.running:
            return

        logger.info("Stopping orchestration scaling manager...")
        self.running = False

        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass

        # Gracefully stop all instances
        for instance in list(self.instances.values()):
            await self.retire_instance(instance.instance_id)

        logger.info("Orchestration scaling manager stopped")

    def set_scaling_policy(self, loop_id: str, policy: Dict[str, Any]):
        """Set scaling policy for a specific loop."""
        self.scaling_policies[loop_id] = {**self.default_scaling_policy, **policy}
        logger.info(f"Set scaling policy for loop {loop_id}")

    async def spawn_instance(self, loop_id: str, parent_id: str = None) -> str:
        """Spawn a new orchestration instance."""
        instance_id = f"orch_{loop_id}_{uuid.uuid4().hex[:8]}"

        instance = OrchestrationInstance(instance_id, loop_id, parent_id)
        instance.on_status_change = self._on_instance_status_change

        self.instances[instance_id] = instance
        self.total_instances_created += 1

        # Start instance (simplified - would create actual process/container)
        await self._start_instance(instance)

        logger.info(f"Spawned instance {instance_id} for loop {loop_id}")

        # Publish event
        if self.event_publisher:
            await self.event_publisher(
                "ORCH_INSTANCE_SPAWNED",
                {"instance_id": instance_id, "loop_id": loop_id, "parent": parent_id},
            )

        return instance_id

    async def retire_instance(self, instance_id: str) -> bool:
        """Retire an orchestration instance."""
        if instance_id not in self.instances:
            return False

        instance = self.instances[instance_id]

        # Start draining process
        await instance.transition_status(InstanceStatus.DRAINING)

        # Wait for tasks to complete (simplified)
        max_wait_time = 300  # 5 minutes
        start_time = time.time()

        while (
            instance.metrics.active_tasks > 0
            and (time.time() - start_time) < max_wait_time
        ):
            await asyncio.sleep(5)
            # In real implementation, would check actual task status

        # Force stop if tasks didn't complete
        if instance.metrics.active_tasks > 0:
            logger.warning(
                f"Force retiring instance {instance_id} with {instance.metrics.active_tasks} active tasks"
            )

        await instance.transition_status(InstanceStatus.RETIRING)
        await self._stop_instance(instance)
        await instance.transition_status(InstanceStatus.STOPPED)

        # Remove from active instances
        del self.instances[instance_id]
        self.total_instances_retired += 1

        logger.info(f"Retired instance {instance_id}")

        # Publish event
        if self.event_publisher:
            await self.event_publisher(
                "ORCH_INSTANCE_RETIRED",
                {
                    "instance_id": instance_id,
                    "loop_id": instance.loop_id,
                    "ts": datetime.now().isoformat(),
                },
            )

        return True

    async def _start_instance(self, instance: OrchestrationInstance):
        """Start an orchestration instance."""
        # In real implementation, would:
        # 1. Create process/container
        # 2. Configure networking
        # 3. Start orchestration loop
        # 4. Wait for health check

        # Simulate startup
        await asyncio.sleep(2)
        await instance.transition_status(InstanceStatus.ACTIVE)

    async def _stop_instance(self, instance: OrchestrationInstance):
        """Stop an orchestration instance."""
        # In real implementation, would:
        # 1. Send shutdown signal
        # 2. Wait for graceful shutdown
        # 3. Clean up resources

        # Simulate shutdown
        await asyncio.sleep(1)

    async def _on_instance_status_change(
        self,
        instance: OrchestrationInstance,
        old_status: InstanceStatus,
        new_status: InstanceStatus,
    ):
        """Handle instance status changes."""
        logger.debug(
            f"Instance {instance.instance_id} status: {old_status.value} -> {new_status.value}"
        )

        # Handle failed instances
        if new_status == InstanceStatus.FAILED:
            logger.error(
                f"Instance {instance.instance_id} failed, will attempt replacement"
            )
            asyncio.create_task(self._handle_failed_instance(instance))

    async def _handle_failed_instance(self, instance: OrchestrationInstance):
        """Handle a failed instance."""
        # Remove failed instance
        if instance.instance_id in self.instances:
            del self.instances[instance.instance_id]

        # Check if we need to spawn replacement
        loop_instances = self.get_loop_instances(instance.loop_id)
        policy = self.scaling_policies.get(
            instance.loop_id, self.default_scaling_policy
        )

        if len(loop_instances) < policy["min_instances"]:
            logger.info(
                f"Spawning replacement instance for failed {instance.instance_id}"
            )
            await self.spawn_instance(instance.loop_id)

    def get_loop_instances(self, loop_id: str) -> List[OrchestrationInstance]:
        """Get all instances for a specific loop."""
        return [
            instance
            for instance in self.instances.values()
            if instance.loop_id == loop_id
        ]

    def select_instance_for_task(self, loop_id: str) -> Optional[OrchestrationInstance]:
        """Select best instance for a new task."""
        loop_instances = self.get_loop_instances(loop_id)
        return self.load_balancer.select_instance(loop_instances)

    async def _scaling_loop(self):
        """Main scaling decision loop."""
        try:
            while self.running:
                # Evaluate scaling decisions for each loop
                for loop_id in set(
                    instance.loop_id for instance in self.instances.values()
                ):
                    await self._evaluate_scaling(loop_id)

                await asyncio.sleep(30)  # Check every 30 seconds

        except asyncio.CancelledError:
            logger.debug("Scaling loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Scaling loop error: {e}", exc_info=True)

    async def _evaluate_scaling(self, loop_id: str):
        """Evaluate scaling needs for a specific loop."""
        instances = self.get_loop_instances(loop_id)
        policy = self.scaling_policies.get(loop_id, self.default_scaling_policy)

        if not instances:
            # No instances - spawn minimum required
            for _ in range(policy["min_instances"]):
                await self.spawn_instance(loop_id)
            return

        # Calculate metrics
        active_instances = [i for i in instances if i.status == InstanceStatus.ACTIVE]
        if not active_instances:
            return

        avg_cpu = sum(i.metrics.cpu_usage for i in active_instances) / len(
            active_instances
        )
        avg_task_load = sum(i.metrics.active_tasks for i in active_instances) / len(
            active_instances
        )
        total_tasks = sum(i.metrics.active_tasks for i in active_instances)

        # Check cooldown
        last_action = self._last_scaling_actions.get(loop_id)
        if last_action:
            cooldown = timedelta(minutes=policy["cooldown_minutes"])
            if datetime.now() - last_action < cooldown:
                return

        # Scale up decision
        should_scale_up = len(active_instances) < policy["max_instances"] and (
            avg_cpu > policy["scale_up_threshold"]
            or avg_task_load > policy["target_task_load"]
        )

        # Scale down decision
        should_scale_down = (
            len(active_instances) > policy["min_instances"]
            and avg_cpu < policy["scale_down_threshold"]
            and avg_task_load < policy["target_task_load"] * 0.5
        )

        if should_scale_up:
            await self.spawn_instance(loop_id)
            self._last_scaling_actions[loop_id] = datetime.now()
            self.total_scaling_actions += 1
            logger.info(
                f"Scaled up loop {loop_id}: CPU={avg_cpu:.1f}%, tasks={avg_task_load:.1f}"
            )

        elif should_scale_down and len(active_instances) > 1:
            # Find least loaded instance to retire
            instance_to_retire = min(
                active_instances, key=lambda i: i.metrics.get_load_score()
            )
            await self.retire_instance(instance_to_retire.instance_id)
            self._last_scaling_actions[loop_id] = datetime.now()
            self.total_scaling_actions += 1
            logger.info(
                f"Scaled down loop {loop_id}: CPU={avg_cpu:.1f}%, tasks={avg_task_load:.1f}"
            )

    def update_instance_metrics(self, instance_id: str, metrics: InstanceMetrics):
        """Update metrics for a specific instance."""
        if instance_id in self.instances:
            self.instances[instance_id].update_metrics(metrics)

    def get_status(self) -> Dict[str, Any]:
        """Get scaling manager status."""
        instances_by_loop = {}
        for instance in self.instances.values():
            if instance.loop_id not in instances_by_loop:
                instances_by_loop[instance.loop_id] = []
            instances_by_loop[instance.loop_id].append(instance.to_dict())

        return {
            "running": self.running,
            "total_instances": len(self.instances),
            "instances_by_loop": instances_by_loop,
            "scaling_policies": self.scaling_policies,
            "load_balancer_algorithm": self.load_balancer.algorithm.value,
            "statistics": {
                "total_instances_created": self.total_instances_created,
                "total_instances_retired": self.total_instances_retired,
                "total_scaling_actions": self.total_scaling_actions,
            },
            "last_scaling_actions": {
                loop_id: timestamp.isoformat()
                for loop_id, timestamp in self._last_scaling_actions.items()
            },
        }
