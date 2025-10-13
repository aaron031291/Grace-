"""
Grace Orchestration Scheduler - Priority-based loop and task scheduler.

Manages the execution of orchestration loops with priority queues, deadlines,
and fairness algorithms. Coordinates OODA, Homeostasis, Antifragility,
Governance Adaptation, Meta-Learning, and Value Generation loops.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import heapq
import logging

logger = logging.getLogger(__name__)


class LoopPriority(Enum):
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 2
    BACKGROUND = 1


class SchedulerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"


class LoopDefinition:
    """Orchestration loop definition with scheduling metadata."""
    
    def __init__(self, loop_id: str, name: str, priority: int, interval_s: int,
                 kernels: List[str], policies: Dict[str, Any], enabled: bool = True):
        self.loop_id = loop_id
        self.name = name
        self.priority = priority
        self.interval_s = interval_s
        self.kernels = kernels
        self.policies = policies
        self.enabled = enabled
        
        # Scheduling metadata
        self.last_run = 0.0
        self.next_run = time.time() + interval_s
        self.run_count = 0
        self.failure_count = 0
        self.average_duration = 0.0
        
    def should_run(self) -> bool:
        """Check if loop should run based on schedule and enabled state."""
        return self.enabled and time.time() >= self.next_run
    
    def schedule_next_run(self):
        """Schedule the next execution time."""
        self.next_run = time.time() + self.interval_s
    
    def record_execution(self, duration: float, success: bool):
        """Record execution metrics."""
        self.last_run = time.time()
        self.run_count += 1
        if not success:
            self.failure_count += 1
        
        # Update average duration with exponential moving average
        alpha = 0.3
        self.average_duration = alpha * duration + (1 - alpha) * self.average_duration
    
    def get_failure_rate(self) -> float:
        """Get failure rate (0-1)."""
        return self.failure_count / max(1, self.run_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "loop_id": self.loop_id,
            "name": self.name,
            "priority": self.priority,
            "interval_s": self.interval_s,
            "kernels": self.kernels,
            "policies": self.policies,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "average_duration": self.average_duration
        }


class TaskRequest:
    """Scheduled task request with priority and deadline."""
    
    def __init__(self, task_id: str, loop_id: str, stage: str, inputs: Dict[str, Any],
                 priority: int = 5, deadline: Optional[datetime] = None):
        self.task_id = task_id
        self.loop_id = loop_id
        self.stage = stage
        self.inputs = inputs
        self.priority = priority
        self.deadline = deadline or datetime.now() + timedelta(minutes=10)
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.status = "pending"
        self.outputs = {}
        self.error = None
    
    def __lt__(self, other):
        """Priority queue ordering: higher priority first, then by deadline."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.deadline < other.deadline
    
    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline."""
        return datetime.now() > self.deadline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "loop_id": self.loop_id,
            "stage": self.stage,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "status": self.status,
            "error": self.error,
            "ts": self.completed_at.isoformat() if self.completed_at else self.created_at.isoformat()
        }


class Scheduler:
    """Priority-based scheduler for orchestration loops and tasks."""
    
    def __init__(self, state_manager=None, event_publisher=None):
        self.state_manager = state_manager
        self.event_publisher = event_publisher
        
        # Scheduling state
        self.state = SchedulerState.STOPPED
        self.loops: Dict[str, LoopDefinition] = {}
        self.task_queue: List[TaskRequest] = []
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.completed_tasks: Dict[str, TaskRequest] = {}
        
        # Configuration
        self.tick_interval = 1.0  # 1 second scheduling tick
        self.max_concurrent_tasks = 10
        self.task_timeout_minutes = 30
        
        # Metrics
        self.total_tasks_scheduled = 0
        self.total_tasks_completed = 0
        self.total_loops_executed = 0
        
        # Internal
        self._scheduler_task = None
        # keep references to per-task futures started by the scheduler
        self._task_futures: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the scheduler."""
        if self.state != SchedulerState.STOPPED:
            logger.warning("Scheduler already running")
            return
        
        logger.info("Starting orchestration scheduler...")
        self.state = SchedulerState.STARTING
        
        # Start main scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.state = SchedulerState.RUNNING
        logger.info("Orchestration scheduler started")
        
    async def stop(self):
        """Stop the scheduler gracefully."""
        if self.state == SchedulerState.STOPPED:
            return
            
        logger.info("Stopping orchestration scheduler...")
        self.state = SchedulerState.STOPPING
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        # Cancel any outstanding per-task futures created by _process_task_queue
        for fut in list(self._task_futures.values()):
            if fut and not fut.done():
                fut.cancel()
                try:
                    await fut
                except asyncio.CancelledError:
                    pass

        self.state = SchedulerState.STOPPED
        logger.info("Orchestration scheduler stopped")
    
    async def register_loop(self, loop_def: Dict[str, Any]) -> str:
        """Register a new orchestration loop."""
        async with self._lock:
            loop = LoopDefinition(
                loop_id=loop_def["loop_id"],
                name=loop_def["name"],
                priority=loop_def["priority"],
                interval_s=loop_def["interval_s"],
                kernels=loop_def["kernels"],
                policies=loop_def["policies"],
                enabled=loop_def.get("enabled", True)
            )
            
            self.loops[loop.loop_id] = loop
            logger.info(f"Registered loop: {loop.loop_id} ({loop.name})")
            return loop.loop_id
    
    async def enable_loop(self, loop_id: str, enabled: bool = True) -> bool:
        """Enable or disable a loop."""
        async with self._lock:
            if loop_id not in self.loops:
                return False
            
            self.loops[loop_id].enabled = enabled
            logger.info(f"Loop {loop_id} {'enabled' if enabled else 'disabled'}")
            return True
    
    async def dispatch_task(self, loop_id: str, inputs: Dict[str, Any], 
                          priority: int = 5) -> str:
        """Dispatch a new task for execution."""
        if loop_id not in self.loops:
            raise ValueError(f"Unknown loop: {loop_id}")
        
        task_id = f"task_{int(time.time() * 1000)}_{len(self.active_tasks)}"
        task = TaskRequest(
            task_id=task_id,
            loop_id=loop_id,
            stage="start",
            inputs=inputs,
            priority=priority
        )
        
        async with self._lock:
            heapq.heappush(self.task_queue, task)
            self.total_tasks_scheduled += 1
            
        logger.info(f"Dispatched task {task_id} for loop {loop_id}")
        
        # Publish event
        if self.event_publisher:
            await self.event_publisher("ORCH_TASK_DISPATCHED", {"task": task.to_dict()})
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task status."""
        async with self._lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].to_dict()
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()
            
            # Check pending tasks
            for task in self.task_queue:
                if task.task_id == task_id:
                    return task.to_dict()
        
        return None
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        try:
            while self.state == SchedulerState.RUNNING:
                await self.tick()
                await asyncio.sleep(self.tick_interval)
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}", exc_info=True)
            self.state = SchedulerState.STOPPED
    
    async def tick(self):
        """Single scheduler tick - check loops and dispatch tasks."""
        current_time = time.time()
        
        async with self._lock:
            # 1. Check for expired tasks
            await self._handle_expired_tasks()
            
            # 2. Check loops that need to run
            for loop in self.loops.values():
                if loop.should_run():
                    await self._schedule_loop_execution(loop)
            
            # 3. Process pending tasks
            await self._process_task_queue()
            
            # 4. Update metrics
            await self._update_metrics()
    
    async def _schedule_loop_execution(self, loop: LoopDefinition):
        """Schedule execution of a loop."""
        try:
            # Create task for loop execution
            task_id = await self.dispatch_task(
                loop_id=loop.loop_id,
                inputs={"loop_execution": True},
                priority=loop.priority
            )
            
            loop.schedule_next_run()
            self.total_loops_executed += 1
            
            logger.debug(f"Scheduled loop execution: {loop.loop_id}")
            
            # Publish event
            if self.event_publisher:
                await self.event_publisher("ORCH_LOOP_STARTED", {
                    "loop_id": loop.loop_id,
                    "ts": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Failed to schedule loop {loop.loop_id}: {e}")
    
    async def _process_task_queue(self):
        """Process pending tasks from the queue."""
        # Only process if we have capacity
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return
        
        # Get next highest priority task
        if not self.task_queue:
            return
        
        task = heapq.heappop(self.task_queue)
        
        # Check if task expired
        if task.is_expired():
            logger.warning(f"Task {task.task_id} expired before execution")
            task.status = "failed"
            task.error = {"code": "EXPIRED", "message": "Task expired before execution"}
            self.completed_tasks[task.task_id] = task
            return
        
        # Start task execution
        task.status = "running"
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        
        # Execute task asynchronously and keep a reference so we can cancel on stop
        task_future = asyncio.create_task(self._execute_task(task))
        # store future keyed by task id for cancellation during shutdown
        self._task_futures[task.task_id] = task_future
    
    async def _execute_task(self, task: TaskRequest):
        """Execute a task (placeholder for actual loop execution)."""
        try:
            start_time = time.time()
            
            # Get loop definition
            loop = self.loops.get(task.loop_id)
            if not loop:
                raise ValueError(f"Unknown loop: {task.loop_id}")
            
            # Simulate loop execution (replace with actual loop runner)
            await asyncio.sleep(0.1)  # Simulate work
            
            # Record success
            duration = time.time() - start_time
            loop.record_execution(duration, True)
            
            # Complete task
            task.status = "succeeded"
            task.completed_at = datetime.now()
            task.outputs = {"result": "success", "duration": duration}
            
            logger.debug(f"Task {task.task_id} completed successfully")
            
            # Publish event
            if self.event_publisher:
                await self.event_publisher("ORCH_TASK_COMPLETED", {
                    "task_id": task.task_id,
                    "status": "succeeded",
                    "outputs": task.outputs
                })
        
        except Exception as e:
            # Record failure
            if task.loop_id in self.loops:
                duration = time.time() - (task.started_at.timestamp() if task.started_at else time.time())
                self.loops[task.loop_id].record_execution(duration, False)
            
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error = {
                "code": "EXECUTION_ERROR",
                "message": str(e),
                "details": {"type": type(e).__name__}
            }
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Publish error event
            if self.event_publisher:
                await self.event_publisher("ORCH_ERROR", {
                    "task_id": task.task_id,
                    "loop_id": task.loop_id,
                    "error": task.error
                })
        
        finally:
            # Move from active to completed
            async with self._lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
                self.total_tasks_completed += 1
    
    async def _handle_expired_tasks(self):
        """Handle tasks that have exceeded their timeout."""
        current_time = datetime.now()
        timeout_delta = timedelta(minutes=self.task_timeout_minutes)
        
        expired_tasks = []
        for task_id, task in self.active_tasks.items():
            if task.started_at and (current_time - task.started_at) > timeout_delta:
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            task = self.active_tasks[task_id]
            task.status = "failed"
            task.completed_at = current_time
            task.error = {
                "code": "TIMEOUT",
                "message": f"Task exceeded timeout of {self.task_timeout_minutes} minutes"
            }
            
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            logger.warning(f"Task {task_id} timed out after {self.task_timeout_minutes} minutes")
    
    async def _update_metrics(self):
        """Update scheduler metrics."""
        # Could update metrics here for monitoring
        pass
    
    async def run_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute a specific loop manually (for testing or manual triggers)."""
        start_time = time.time()
        
        try:
            logger.info(f"Executing loop: {loop.loop_id} ({loop.name})")
            
            # Validate kernel dependencies
            if not await self._validate_kernel_dependencies(loop):
                raise ValueError(f"Kernel dependencies not met for loop {loop.loop_id}")
            
            # Execute loop stages
            result = await self._execute_loop_stages(loop)
            
            duration = time.time() - start_time
            loop.record_execution(duration, True)
            
            logger.info(f"Loop {loop.loop_id} completed in {duration:.2f}s")
            return {"status": "success", "duration": duration, "result": result}
            
        except Exception as e:
            duration = time.time() - start_time
            loop.record_execution(duration, False)
            
            logger.error(f"Loop {loop.loop_id} failed: {e}")
            return {"status": "failed", "duration": duration, "error": str(e)}
    
    async def _validate_kernel_dependencies(self, loop: LoopDefinition) -> bool:
        """Validate that required kernels are available."""
        # Placeholder for kernel dependency validation
        # In real implementation, would check with kernel registry
        return True
    
    async def _execute_loop_stages(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute the stages of a loop."""
        # Placeholder for actual loop execution logic
        # Different loop types would have different execution patterns
        
        if loop.name == "ooda":
            return await self._execute_ooda_loop(loop)
        elif loop.name == "homeostasis":
            return await self._execute_homeostasis_loop(loop)
        elif loop.name == "antifragility":
            return await self._execute_antifragility_loop(loop)
        elif loop.name == "governance_adaptation":
            return await self._execute_governance_adaptation_loop(loop)
        elif loop.name == "meta_learning":
            return await self._execute_meta_learning_loop(loop)
        elif loop.name == "value_generation":
            return await self._execute_value_generation_loop(loop)
        else:
            return {"message": f"Unknown loop type: {loop.name}"}
    
    async def _execute_ooda_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute OODA (Observe-Orient-Decide-Act) loop."""
        # Observe - gather intelligence
        observations = {"timestamp": datetime.now().isoformat(), "stage": "observe"}
        
        # Orient - analyze situation
        orientation = {"timestamp": datetime.now().isoformat(), "stage": "orient"}
        
        # Decide - make decisions
        decisions = {"timestamp": datetime.now().isoformat(), "stage": "decide"}
        
        # Act - execute actions
        actions = {"timestamp": datetime.now().isoformat(), "stage": "act"}
        
        return {
            "loop_type": "ooda",
            "stages": [observations, orientation, decisions, actions]
        }
    
    async def _execute_homeostasis_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute homeostasis loop for system stability."""
        return {"loop_type": "homeostasis", "status": "maintaining_equilibrium"}
    
    async def _execute_antifragility_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute antifragility loop for system strengthening."""
        return {"loop_type": "antifragility", "status": "strengthening_system"}
    
    async def _execute_governance_adaptation_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute governance adaptation loop."""
        return {"loop_type": "governance_adaptation", "status": "adapting_governance"}
    
    async def _execute_meta_learning_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute meta-learning loop."""
        return {"loop_type": "meta_learning", "status": "learning_to_learn"}
    
    async def _execute_value_generation_loop(self, loop: LoopDefinition) -> Dict[str, Any]:
        """Execute value generation loop."""
        return {"loop_type": "value_generation", "status": "generating_value"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and metrics."""
        return {
            "state": self.state.value,
            "loops": {loop_id: loop.to_dict() for loop_id, loop in self.loops.items()},
            "task_queue_size": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_tasks_scheduled": self.total_tasks_scheduled,
            "total_tasks_completed": self.total_tasks_completed,
            "total_loops_executed": self.total_loops_executed,
            "max_concurrent_tasks": self.max_concurrent_tasks
        }