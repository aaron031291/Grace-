"""
Enhanced scheduler with metrics, snapshots, and restoration
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
import asyncio
import json
import logging
from pathlib import Path

from grace.orchestration.scheduler_metrics import scheduler_metrics

logger = logging.getLogger(__name__)


class SchedulerLoop:
    """Represents a scheduled loop"""
    
    def __init__(
        self,
        loop_id: str,
        callback: Callable,
        interval: float,
        priority: int = 1,
        max_retries: int = 3,
        timeout: Optional[float] = None
    ):
        self.loop_id = loop_id
        self.callback = callback
        self.interval = interval
        self.priority = priority
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.enabled = True
        self.last_execution: Optional[datetime] = None
        self.next_execution: Optional[datetime] = None
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.queue_depth = 0
        
        self._task: Optional[asyncio.Task] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize loop configuration"""
        return {
            "loop_id": self.loop_id,
            "interval": self.interval,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }


class SchedulerPolicy:
    """Represents a scheduling policy"""
    
    def __init__(
        self,
        policy_id: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: str,
        description: str = ""
    ):
        self.policy_id = policy_id
        self.condition = condition
        self.action = action
        self.description = description
        self.evaluation_count = 0
        self.allowed_count = 0
        self.denied_count = 0
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate policy condition"""
        try:
            self.evaluation_count += 1
            result = self.condition(context)
            
            if result:
                self.allowed_count += 1
            else:
                self.denied_count += 1
            
            scheduler_metrics.record_policy_evaluation(self.policy_id, result)
            return result
        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")
            self.denied_count += 1
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy configuration"""
        return {
            "policy_id": self.policy_id,
            "action": self.action,
            "description": self.description,
            "evaluation_count": self.evaluation_count,
            "allowed_count": self.allowed_count,
            "denied_count": self.denied_count
        }


class EnhancedScheduler:
    """
    Production scheduler with metrics, snapshots, and restoration
    """
    
    def __init__(self, scheduler_id: str = "main"):
        self.scheduler_id = scheduler_id
        self.loops: Dict[str, SchedulerLoop] = {}
        self.policies: Dict[str, SchedulerPolicy] = {}
        
        self.running = False
        self.start_time: Optional[datetime] = None
        self._main_task: Optional[asyncio.Task] = None
        
        logger.info(f"Enhanced scheduler initialized: {scheduler_id}")
    
    async def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        scheduler_metrics.update_scheduler_status(self.scheduler_id, True)
        
        # Start main scheduling loop
        self._main_task = asyncio.create_task(self._main_loop())
        
        # Start metrics update loop
        asyncio.create_task(self._metrics_loop())
        
        logger.info(f"Scheduler started: {self.scheduler_id}")
    
    async def stop(self):
        """Stop the scheduler"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all loop tasks
        for loop in self.loops.values():
            if loop._task and not loop._task.done():
                loop._task.cancel()
        
        # Cancel main task
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
        
        scheduler_metrics.update_scheduler_status(self.scheduler_id, False)
        
        logger.info(f"Scheduler stopped: {self.scheduler_id}")
    
    def register_loop(
        self,
        loop_id: str,
        callback: Callable,
        interval: float,
        priority: int = 1,
        max_retries: int = 3,
        timeout: Optional[float] = None
    ) -> SchedulerLoop:
        """Register a new loop"""
        if loop_id in self.loops:
            raise ValueError(f"Loop already registered: {loop_id}")
        
        loop = SchedulerLoop(
            loop_id=loop_id,
            callback=callback,
            interval=interval,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self.loops[loop_id] = loop
        
        logger.info(f"Registered loop: {loop_id} (interval={interval}s)")
        return loop
    
    def register_policy(
        self,
        policy_id: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: str,
        description: str = ""
    ) -> SchedulerPolicy:
        """Register a scheduling policy"""
        if policy_id in self.policies:
            raise ValueError(f"Policy already registered: {policy_id}")
        
        policy = SchedulerPolicy(
            policy_id=policy_id,
            condition=condition,
            action=action,
            description=description
        )
        
        self.policies[policy_id] = policy
        
        logger.info(f"Registered policy: {policy_id}")
        return policy
    
    async def _main_loop(self):
        """Main scheduling loop"""
        while self.running:
            try:
                await self._tick()
                await asyncio.sleep(0.1)  # Fast tick rate
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler main loop error: {e}")
                scheduler_metrics.record_error("main_loop_error")
                await asyncio.sleep(1)
    
    async def _tick(self):
        """Process one scheduling tick"""
        now = datetime.now(timezone.utc)
        
        for loop in self.loops.values():
            if not loop.enabled:
                continue
            
            # Check if loop should execute
            if loop.next_execution is None or now >= loop.next_execution:
                # Check policies
                if not self._check_policies(loop):
                    continue
                
                # Execute loop
                loop._task = asyncio.create_task(self._execute_loop(loop))
                
                # Schedule next execution
                loop.next_execution = now + timedelta(seconds=loop.interval)
    
    def _check_policies(self, loop: SchedulerLoop) -> bool:
        """Check if policies allow loop execution"""
        context = {
            "loop_id": loop.loop_id,
            "execution_count": loop.execution_count,
            "success_rate": loop.success_count / max(1, loop.execution_count),
            "queue_depth": loop.queue_depth
        }
        
        for policy in self.policies.values():
            if not policy.evaluate(context):
                logger.debug(f"Policy denied loop execution: {loop.loop_id}")
                return False
        
        return True
    
    async def _execute_loop(self, loop: SchedulerLoop):
        """Execute a loop with metrics"""
        start_time = time.time()
        loop.execution_count += 1
        loop.queue_depth += 1
        
        scheduler_metrics.update_queue_depth(loop.loop_id, loop.queue_depth)
        
        status = "success"
        
        try:
            # Execute with timeout if specified
            if loop.timeout:
                await asyncio.wait_for(loop.callback(), timeout=loop.timeout)
            else:
                await loop.callback()
            
            loop.success_count += 1
            loop.last_execution = datetime.now(timezone.utc)
            
        except asyncio.TimeoutError:
            status = "timeout"
            loop.failure_count += 1
            logger.warning(f"Loop timeout: {loop.loop_id}")
            scheduler_metrics.record_error("loop_timeout")
            
        except Exception as e:
            status = "failure"
            loop.failure_count += 1
            logger.error(f"Loop execution error ({loop.loop_id}): {e}")
            scheduler_metrics.record_error("loop_execution_error")
        
        finally:
            duration = time.time() - start_time
            loop.queue_depth -= 1
            
            # Record metrics
            scheduler_metrics.record_loop_execution(loop.loop_id, duration, status)
            scheduler_metrics.update_queue_depth(loop.loop_id, loop.queue_depth)
    
    async def _metrics_loop(self):
        """Update metrics periodically"""
        while self.running:
            try:
                self._update_metrics()
                await asyncio.sleep(5)  # Update every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
    
    def _update_metrics(self):
        """Update Prometheus metrics"""
        # Active loops
        active_count = sum(1 for loop in self.loops.values() if loop.enabled)
        scheduler_metrics.update_active_loops(active_count)
        
        # Global queue size
        total_queue = sum(loop.queue_depth for loop in self.loops.values())
        scheduler_metrics.update_global_queue_size(total_queue)
        
        # Uptime
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            scheduler_metrics.update_uptime(self.scheduler_id, uptime)
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of scheduler state"""
        try:
            snapshot = {
                "scheduler_id": self.scheduler_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "running": self.running,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "loops": {
                    loop_id: loop.to_dict()
                    for loop_id, loop in self.loops.items()
                },
                "policies": {
                    policy_id: policy.to_dict()
                    for policy_id, policy in self.policies.items()
                }
            }
            
            scheduler_metrics.record_snapshot_operation("create", True)
            logger.info(f"Created scheduler snapshot: {len(self.loops)} loops, {len(self.policies)} policies")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            scheduler_metrics.record_snapshot_operation("create", False)
            raise
    
    async def restore_from_snapshot(
        self,
        snapshot: Dict[str, Any],
        loop_callbacks: Dict[str, Callable],
        policy_conditions: Dict[str, Callable]
    ):
        """Restore scheduler from snapshot"""
        try:
            logger.info(f"Restoring scheduler from snapshot...")
            
            # Stop current scheduler
            if self.running:
                await self.stop()
            
            # Clear current state
            self.loops.clear()
            self.policies.clear()
            
            # Restore loops
            await self._restore_loops_and_policies(
                snapshot,
                loop_callbacks,
                policy_conditions
            )
            
            # Restart scheduler with restored config
            await self._restart_scheduler_with_config(snapshot)
            
            scheduler_metrics.record_snapshot_operation("restore", True)
            logger.info(f"Scheduler restored successfully")
            
        except Exception as e:
            logger.error(f"Error restoring snapshot: {e}")
            scheduler_metrics.record_snapshot_operation("restore", False)
            raise
    
    async def _restore_loops_and_policies(
        self,
        snapshot: Dict[str, Any],
        loop_callbacks: Dict[str, Callable],
        policy_conditions: Dict[str, Callable]
    ):
        """Restore loops and policies from snapshot"""
        # Restore loops
        for loop_id, loop_data in snapshot.get("loops", {}).items():
            if loop_id not in loop_callbacks:
                logger.warning(f"No callback found for loop: {loop_id}, skipping")
                continue
            
            loop = self.register_loop(
                loop_id=loop_id,
                callback=loop_callbacks[loop_id],
                interval=loop_data["interval"],
                priority=loop_data.get("priority", 1),
                max_retries=loop_data.get("max_retries", 3),
                timeout=loop_data.get("timeout")
            )
            
            # Restore statistics
            loop.enabled = loop_data.get("enabled", True)
            loop.execution_count = loop_data.get("execution_count", 0)
            loop.success_count = loop_data.get("success_count", 0)
            loop.failure_count = loop_data.get("failure_count", 0)
            
            if loop_data.get("last_execution"):
                loop.last_execution = datetime.fromisoformat(loop_data["last_execution"])
            
            logger.info(f"Restored loop: {loop_id}")
        
        # Restore policies
        for policy_id, policy_data in snapshot.get("policies", {}).items():
            if policy_id not in policy_conditions:
                logger.warning(f"No condition found for policy: {policy_id}, skipping")
                continue
            
            policy = self.register_policy(
                policy_id=policy_id,
                condition=policy_conditions[policy_id],
                action=policy_data["action"],
                description=policy_data.get("description", "")
            )
            
            # Restore statistics
            policy.evaluation_count = policy_data.get("evaluation_count", 0)
            policy.allowed_count = policy_data.get("allowed_count", 0)
            policy.denied_count = policy_data.get("denied_count", 0)
            
            logger.info(f"Restored policy: {policy_id}")
    
    async def _restart_scheduler_with_config(self, snapshot: Dict[str, Any]):
        """Restart scheduler with restored configuration"""
        # Apply any scheduler-level config from snapshot
        self.scheduler_id = snapshot.get("scheduler_id", self.scheduler_id)
        
        # Start scheduler if it was running
        if snapshot.get("running", True):
            await self.start()
            logger.info("Scheduler restarted with restored configuration")
