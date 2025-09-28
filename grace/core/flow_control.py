"""
Enhanced Backpressure & Flow Control system for Grace Event Infrastructure.

Provides throttling, batching, priority queueing, and adaptive flow control 
to prevent event floods from overwhelming downstream consumers.
"""
import asyncio
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class FlowControlStatus(Enum):
    """Flow control status levels."""
    NORMAL = "normal"
    THROTTLED = "throttled"
    BACKPRESSURE = "backpressure"
    OVERLOADED = "overloaded"


class ThrottleStrategy(Enum):
    """Throttling strategies."""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    DROP_LOWEST_PRIORITY = "drop_lowest_priority"
    ADAPTIVE_BATCH = "adaptive_batch"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class FlowControlConfig:
    """Configuration for flow control parameters."""
    # Queue size limits
    max_queue_size: int = 10000
    high_water_mark: int = 8000
    low_water_mark: int = 2000
    critical_queue_size: int = 1000
    
    # Rate limits (events per second)
    max_events_per_second: int = 1000
    burst_capacity: int = 2000
    
    # Batching
    max_batch_size: int = 100
    batch_timeout_ms: int = 50
    enable_adaptive_batching: bool = True
    
    # Backpressure
    backpressure_threshold_ms: float = 100.0
    recovery_threshold_ms: float = 50.0
    throttle_strategy: ThrottleStrategy = ThrottleStrategy.ADAPTIVE_BATCH
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30
    
    # Consumer monitoring  
    slow_consumer_threshold_ms: float = 200.0
    consumer_timeout_seconds: int = 10.0


@dataclass
class EventBatch:
    """Batch of events for processing."""
    events: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: int = 3  # Normal priority by default
    batch_id: str = field(default_factory=lambda: f"batch_{int(time.time() * 1000)}")
    
    def add_event(self, event: Any, priority: int = 3):
        """Add event to batch."""
        self.events.append(event)
        # Promote batch priority if higher priority event added
        if priority < self.priority:
            self.priority = priority
    
    def is_ready(self, config: FlowControlConfig) -> bool:
        """Check if batch is ready for processing."""
        return (
            len(self.events) >= config.max_batch_size or
            (datetime.utcnow() - self.created_at).total_seconds() * 1000 >= config.batch_timeout_ms
        )


@dataclass
class ConsumerMetrics:
    """Metrics for tracking consumer performance."""
    consumer_id: str
    events_processed: int = 0
    events_failed: int = 0
    total_processing_time_ms: float = 0.0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    average_processing_time_ms: float = 0.0
    is_slow: bool = False
    is_healthy: bool = True
    
    def update_metrics(self, processing_time_ms: float, success: bool = True):
        """Update consumer metrics."""
        self.last_activity = datetime.utcnow()
        self.total_processing_time_ms += processing_time_ms
        
        if success:
            self.events_processed += 1
        else:
            self.events_failed += 1
        
        # Calculate average processing time
        total_events = self.events_processed + self.events_failed
        if total_events > 0:
            self.average_processing_time_ms = self.total_processing_time_ms / total_events
    
    def get_success_rate(self) -> float:
        """Get consumer success rate."""
        total = self.events_processed + self.events_failed
        return self.events_processed / max(1, total)


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, rate_limit: int, burst_capacity: int):
        self.rate_limit = rate_limit
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        with self._lock:
            now = time.time()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(
                self.burst_capacity,
                self.tokens + elapsed * self.rate_limit
            )
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available."""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.rate_limit


class PriorityEventQueue:
    """Priority-based event queue with backpressure support."""
    
    def __init__(self, config: FlowControlConfig):
        self.config = config
        self._queues = {
            1: deque(),  # Critical
            2: deque(),  # High 
            3: deque(),  # Normal
            4: deque()   # Low
        }
        self._total_size = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "dropped": 0,
            "batched": 0
        }
    
    async def put(self, event: Any, priority: int = 3) -> bool:
        """Add event to queue. Returns False if dropped due to backpressure."""
        async with self._lock:
            # Check if queue is at capacity
            if self._total_size >= self.config.max_queue_size:
                return await self._handle_backpressure(event, priority)
            
            # Add to appropriate priority queue
            self._queues[priority].append(event)
            self._total_size += 1
            self._stats["enqueued"] += 1
            
            # Notify waiting consumers
            self._not_empty.notify()
            return True
    
    async def get(self) -> Tuple[Any, int]:
        """Get next event from highest priority queue."""
        async with self._not_empty:
            while self._total_size == 0:
                await self._not_empty.wait()
            
            # Get from highest priority queue first
            for priority in sorted(self._queues.keys()):
                if self._queues[priority]:
                    event = self._queues[priority].popleft()
                    self._total_size -= 1
                    self._stats["dequeued"] += 1
                    return event, priority
            
            # Should not reach here if _total_size > 0
            raise RuntimeError("Queue corruption detected")
    
    async def get_batch(self, max_size: int = None) -> List[Tuple[Any, int]]:
        """Get a batch of events, prioritizing higher priority items."""
        if max_size is None:
            max_size = self.config.max_batch_size
        
        batch = []
        async with self._lock:
            # Collect from priority queues
            for priority in sorted(self._queues.keys()):
                queue = self._queues[priority]
                while queue and len(batch) < max_size:
                    event = queue.popleft()
                    batch.append((event, priority))
                    self._total_size -= 1
                    self._stats["dequeued"] += 1
                    
                if len(batch) >= max_size:
                    break
        
        self._stats["batched"] += len(batch)
        return batch
    
    async def _handle_backpressure(self, event: Any, priority: int) -> bool:
        """Handle backpressure by applying configured strategy."""
        strategy = self.config.throttle_strategy
        
        if strategy == ThrottleStrategy.DROP_OLDEST:
            # Drop oldest event from lowest priority queue
            space_made = False
            for p in reversed(sorted(self._queues.keys())):
                if self._queues[p]:
                    self._queues[p].popleft()
                    self._total_size -= 1
                    self._stats["dropped"] += 1
                    space_made = True
                    break
            
            if not space_made:
                # No events to drop, drop the new event
                self._stats["dropped"] += 1
                return False
        
        elif strategy == ThrottleStrategy.DROP_NEWEST:
            # Drop the incoming event
            self._stats["dropped"] += 1
            return False
        
        elif strategy == ThrottleStrategy.DROP_LOWEST_PRIORITY:
            # Drop from lowest priority queue
            space_made = False
            for p in reversed(sorted(self._queues.keys())):
                if self._queues[p]:
                    self._queues[p].popleft()
                    self._total_size -= 1 
                    self._stats["dropped"] += 1
                    space_made = True
                    break
            
            if not space_made:
                # No events to drop, drop the new event
                self._stats["dropped"] += 1
                return False
        
        # Add the new event after making space
        self._queues[priority].append(event)
        self._total_size += 1
        self._stats["enqueued"] += 1
        return True
    
    def size(self) -> int:
        """Get total queue size."""
        return self._total_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "current_size": self._total_size,
            "queue_sizes": {p: len(q) for p, q in self._queues.items()}
        }


class FlowControlManager:
    """
    Enhanced flow control manager providing adaptive backpressure,
    throttling, batching, and consumer monitoring.
    """
    
    def __init__(self, config: FlowControlConfig = None, event_bus=None):
        self.config = config or FlowControlConfig()
        self.event_bus = event_bus
        
        # Core components
        self.event_queue = PriorityEventQueue(self.config)
        self.token_bucket = TokenBucket(
            self.config.max_events_per_second,
            self.config.burst_capacity
        )
        
        # Consumer tracking
        self.consumers: Dict[str, ConsumerMetrics] = {}
        self.consumer_handlers: Dict[str, Callable] = {}
        self.slow_consumers: Set[str] = set()
        
        # Batching
        self.pending_batches: Dict[str, EventBatch] = {}
        self.batch_processor_task: Optional[asyncio.Task] = None
        
        # Circuit breaker state
        self.circuit_breaker_failures: Dict[str, int] = defaultdict(int)
        self.circuit_breaker_recovery: Dict[str, datetime] = {}
        
        # Flow control state
        self.current_status = FlowControlStatus.NORMAL
        self.status_history: deque = deque(maxlen=100)
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance metrics
        self.metrics = {
            "events_processed": 0,
            "events_dropped": 0,
            "batches_created": 0,
            "backpressure_events": 0,
            "throttle_events": 0,
            "start_time": datetime.utcnow()
        }
    
    async def start(self):
        """Start the flow control manager."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting Flow Control Manager...")
        
        # Start background tasks
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Flow Control Manager started")
    
    async def stop(self):
        """Stop the flow control manager."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping Flow Control Manager...")
        
        # Cancel background tasks
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self.batch_processor_task,
            self.monitoring_task,
            return_exceptions=True
        )
        
        logger.info("Flow Control Manager stopped")
    
    async def submit_event(self, event: Any, priority: int = 3, consumer_id: str = None) -> bool:
        """
        Submit event for processing with flow control.
        Returns False if event was dropped due to backpressure.
        """
        if not self.running:
            return False
        
        # Rate limiting check
        if not self.token_bucket.consume():
            wait_time = self.token_bucket.get_wait_time()
            if wait_time > 0.1:  # Don't wait more than 100ms
                self.metrics["throttle_events"] += 1
                logger.debug(f"Rate limit exceeded, dropping event")
                return False
            else:
                await asyncio.sleep(wait_time)
        
        # Check consumer circuit breaker
        if consumer_id and self._is_circuit_breaker_open(consumer_id):
            logger.debug(f"Circuit breaker open for consumer {consumer_id}")
            return False
        
        # Submit to queue
        accepted = await self.event_queue.put(event, priority)
        
        if accepted:
            self.metrics["events_processed"] += 1
            await self._update_flow_control_status()
        else:
            self.metrics["events_dropped"] += 1
        
        return accepted
    
    async def register_consumer(self, consumer_id: str, handler: Callable):
        """Register a consumer for event processing."""
        self.consumers[consumer_id] = ConsumerMetrics(consumer_id)
        self.consumer_handlers[consumer_id] = handler
        logger.info(f"Registered consumer: {consumer_id}")
    
    async def unregister_consumer(self, consumer_id: str):
        """Unregister a consumer."""
        if consumer_id in self.consumers:
            del self.consumers[consumer_id]
        if consumer_id in self.consumer_handlers:
            del self.consumer_handlers[consumer_id]
        self.slow_consumers.discard(consumer_id)
        logger.info(f"Unregistered consumer: {consumer_id}")
    
    async def _batch_processor_loop(self):
        """Background loop for processing event batches."""
        while self.running:
            try:
                # Get batch of events
                batch_events = await self.event_queue.get_batch()
                
                if not batch_events:
                    await asyncio.sleep(0.01)  # Brief pause if no events
                    continue
                
                # Process batch
                await self._process_event_batch(batch_events)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)  # Brief recovery pause
    
    async def _process_event_batch(self, batch_events: List[Tuple[Any, int]]):
        """Process a batch of events."""
        if not batch_events:
            return
        
        # Group events by consumer
        consumer_batches = defaultdict(list)
        
        for event, priority in batch_events:
            # Determine target consumers (simple round-robin for now)
            target_consumers = list(self.consumer_handlers.keys())
            if target_consumers:
                # Simple load balancing - pick least loaded consumer
                target = min(target_consumers, 
                           key=lambda c: self.consumers[c].events_processed)
                consumer_batches[target].append((event, priority))
        
        # Process batches per consumer
        tasks = []
        for consumer_id, events in consumer_batches.items():
            task = asyncio.create_task(
                self._process_consumer_batch(consumer_id, events)
            )
            tasks.append(task)
        
        # Wait for all consumer batches to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.metrics["batches_created"] += 1
    
    async def _process_consumer_batch(self, consumer_id: str, events: List[Tuple[Any, int]]):
        """Process events for a specific consumer."""
        if consumer_id not in self.consumer_handlers:
            return
        
        handler = self.consumer_handlers[consumer_id]
        metrics = self.consumers[consumer_id]
        
        start_time = time.time()
        
        try:
            # Call handler with batch
            if asyncio.iscoroutinefunction(handler):
                await handler([event for event, priority in events])
            else:
                handler([event for event, priority in events])
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            metrics.update_metrics(processing_time_ms, success=True)
            
            # Reset circuit breaker on success
            if consumer_id in self.circuit_breaker_failures:
                self.circuit_breaker_failures[consumer_id] = 0
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            metrics.update_metrics(processing_time_ms, success=False)
            
            # Update circuit breaker
            self.circuit_breaker_failures[consumer_id] += 1
            if self.circuit_breaker_failures[consumer_id] >= self.config.failure_threshold:
                self.circuit_breaker_recovery[consumer_id] = (
                    datetime.utcnow() + timedelta(seconds=self.config.recovery_timeout_seconds)
                )
                logger.warning(f"Circuit breaker opened for consumer {consumer_id}")
            
            logger.error(f"Consumer {consumer_id} processing error: {e}")
    
    def _is_circuit_breaker_open(self, consumer_id: str) -> bool:
        """Check if circuit breaker is open for consumer."""
        recovery_time = self.circuit_breaker_recovery.get(consumer_id)
        if recovery_time and datetime.utcnow() < recovery_time:
            return True
        elif recovery_time and datetime.utcnow() >= recovery_time:
            # Recovery time passed, reset circuit breaker
            del self.circuit_breaker_recovery[consumer_id]
            self.circuit_breaker_failures[consumer_id] = 0
        return False
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                await self._monitor_consumers()
                await self._update_flow_control_status()
                await asyncio.sleep(1.0)  # Monitor more frequently for tests
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(0.5)
    
    async def _monitor_consumers(self):
        """Monitor consumer performance and health."""
        slow_threshold = self.config.slow_consumer_threshold_ms
        timeout_threshold = timedelta(seconds=self.config.consumer_timeout_seconds)
        now = datetime.utcnow()
        
        for consumer_id, metrics in self.consumers.items():
            # Check if consumer is slow
            if metrics.average_processing_time_ms > slow_threshold:
                if consumer_id not in self.slow_consumers:
                    self.slow_consumers.add(consumer_id)
                    logger.warning(f"Consumer {consumer_id} marked as slow (avg: {metrics.average_processing_time_ms:.2f}ms)")
                metrics.is_slow = True
            else:
                if consumer_id in self.slow_consumers:
                    self.slow_consumers.discard(consumer_id)
                    logger.info(f"Consumer {consumer_id} recovered from slow status")
                metrics.is_slow = False
            
            # Check if consumer is healthy (recent activity)
            time_since_activity = now - metrics.last_activity
            metrics.is_healthy = time_since_activity < timeout_threshold
    
    async def _update_flow_control_status(self):
        """Update flow control status based on current metrics."""
        queue_size = self.event_queue.size()
        
        # Determine status based on queue size
        if queue_size >= self.config.max_queue_size:
            new_status = FlowControlStatus.OVERLOADED
        elif queue_size >= self.config.high_water_mark:
            new_status = FlowControlStatus.BACKPRESSURE
        elif queue_size >= self.config.low_water_mark:
            new_status = FlowControlStatus.THROTTLED
        else:
            new_status = FlowControlStatus.NORMAL
        
        # Update status if changed
        if new_status != self.current_status:
            self.status_history.append({
                "from_status": self.current_status.value,
                "to_status": new_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "queue_size": queue_size
            })
            
            logger.info(f"Flow control status changed: {self.current_status.value} -> {new_status.value}")
            self.current_status = new_status
            
            # Publish status change event
            if self.event_bus:
                await self.event_bus.publish("flow_control_status_changed", {
                    "old_status": self.current_status.value,
                    "new_status": new_status.value,
                    "queue_size": queue_size,
                    "metrics": self.get_metrics()
                })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive flow control metrics."""
        uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
        
        return {
            "status": self.current_status.value,
            "uptime_seconds": uptime,
            "queue_stats": self.event_queue.get_stats(),
            "processing_stats": self.metrics.copy(),
            "consumer_metrics": {
                consumer_id: {
                    "events_processed": metrics.events_processed,
                    "events_failed": metrics.events_failed,
                    "success_rate": metrics.get_success_rate(),
                    "avg_processing_time_ms": metrics.average_processing_time_ms,
                    "is_slow": metrics.is_slow,
                    "is_healthy": metrics.is_healthy
                } for consumer_id, metrics in self.consumers.items()
            },
            "slow_consumers": list(self.slow_consumers),
            "circuit_breaker_failures": dict(self.circuit_breaker_failures),
            "status_history": list(self.status_history)[-10:]  # Last 10 status changes
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring."""
        queue_size = self.event_queue.size()
        healthy_consumers = sum(1 for m in self.consumers.values() if m.is_healthy)
        total_consumers = len(self.consumers)
        
        is_healthy = (
            self.running and
            self.current_status in [FlowControlStatus.NORMAL, FlowControlStatus.THROTTLED] and
            queue_size < self.config.high_water_mark and
            (total_consumers == 0 or healthy_consumers / total_consumers >= 0.8)
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "running": self.running,
            "flow_control_status": self.current_status.value,
            "queue_size": queue_size,
            "healthy_consumers": healthy_consumers,
            "total_consumers": total_consumers,
            "issues": self._get_health_issues()
        }
    
    def _get_health_issues(self) -> List[str]:
        """Get list of current health issues."""
        issues = []
        
        if not self.running:
            issues.append("Flow control manager not running")
        
        if self.current_status == FlowControlStatus.OVERLOADED:
            issues.append("System overloaded - dropping events")
        elif self.current_status == FlowControlStatus.BACKPRESSURE:
            issues.append("High backpressure - throttling events")
        
        if self.slow_consumers:
            issues.append(f"Slow consumers detected: {list(self.slow_consumers)}")
        
        unhealthy_consumers = [
            consumer_id for consumer_id, metrics in self.consumers.items()
            if not metrics.is_healthy
        ]
        if unhealthy_consumers:
            issues.append(f"Unhealthy consumers: {unhealthy_consumers}")
        
        return issues