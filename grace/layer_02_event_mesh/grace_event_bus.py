"""Enhanced Grace Event Bus with GME support, DLQ, retries, and backpressure."""

import asyncio
import json
import logging
from asyncio import Queue
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
import uuid
import hashlib

from .message_envelope import GraceMessageEnvelope, GMEHeaders, EventTypes

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class BackpressureConfig:
    """Configuration for backpressure management."""
    def __init__(self,
                 max_queue_size: int = 10000,
                 high_water_mark: int = 8000,
                 low_water_mark: int = 2000,
                 drop_policy: str = "oldest"):  # oldest, newest, priority
        self.max_queue_size = max_queue_size
        self.high_water_mark = high_water_mark
        self.low_water_mark = low_water_mark
        self.drop_policy = drop_policy


class DeadLetterQueue:
    """Dead Letter Queue for failed messages."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.messages: deque = deque(maxlen=max_size)
        self.message_index: Dict[str, Dict[str, Any]] = {}
    
    def add_message(self, message: GraceMessageEnvelope, failure_reason: str):
        """Add a failed message to the DLQ."""
        dlq_entry = {
            "message": message,
            "failure_reason": failure_reason,
            "failed_at": datetime.utcnow(),
            "original_timestamp": message.timestamp,
            "retry_count": message.retry_count
        }
        
        self.messages.append(dlq_entry)
        self.message_index[message.msg_id] = dlq_entry
        
        logger.warning(f"Message {message.msg_id} sent to DLQ: {failure_reason}")
    
    def get_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages from DLQ."""
        return list(self.messages)[-limit:]
    
    def replay_message(self, msg_id: str) -> Optional[GraceMessageEnvelope]:
        """Get message for replay."""
        entry = self.message_index.get(msg_id)
        if entry:
            return entry["message"]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        failure_reasons = defaultdict(int)
        for entry in self.messages:
            failure_reasons[entry["failure_reason"]] += 1
        
        return {
            "total_messages": len(self.messages),
            "max_size": self.max_size,
            "failure_reasons": dict(failure_reasons)
        }


class GraceEventBus:
    """
    Enhanced Grace Event Bus with:
    - GME message envelope support
    - Dead Letter Queue (DLQ)
    - Exponential backoff with jitter
    - Backpressure management
    - Message deduplication
    - OpenTelemetry tracing
    """
    
    def __init__(self, 
                 retry_config: Optional[RetryConfig] = None,
                 backpressure_config: Optional[BackpressureConfig] = None,
                 enable_deduplication: bool = True):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: Queue = Queue()
        self.processing_workers: List[asyncio.Task] = []
        self.running = False
        
        # Configuration
        self.retry_config = retry_config or RetryConfig()
        self.backpressure_config = backpressure_config or BackpressureConfig()
        self.enable_deduplication = enable_deduplication
        
        # Components
        self.dlq = DeadLetterQueue()
        self.schema_registry: Dict[str, Dict[str, Any]] = {}
        
        # State tracking
        self.seen_messages: Set[str] = set()  # For deduplication
        self.processing_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_deduplicated": 0,
            "messages_dropped": 0,
            "start_time": datetime.utcnow()
        }
        
        self._register_default_schemas()
    
    def _register_default_schemas(self):
        """Register default event schemas."""
        # Register all event types from EventTypes class
        for attr_name in dir(EventTypes):
            if not attr_name.startswith('_'):
                event_type = getattr(EventTypes, attr_name)
                if isinstance(event_type, str):
                    self.schema_registry[event_type] = {
                        "schema_version": "1.0.0",
                        "registered_at": datetime.utcnow().isoformat()
                    }
    
    async def start(self):
        """Start the event bus."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting Grace Event Bus...")
        
        # Start processing workers
        worker_count = 4  # Configure based on load
        for i in range(worker_count):
            worker = asyncio.create_task(self._message_processor(f"worker-{i}"))
            self.processing_workers.append(worker)
        
        logger.info(f"Started {len(self.processing_workers)} processing workers")
    
    async def stop(self):
        """Stop the event bus."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping Grace Event Bus...")
        
        # Cancel all workers
        for worker in self.processing_workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.processing_workers, return_exceptions=True)
        self.processing_workers.clear()
        
        logger.info("Grace Event Bus stopped")
    
    def register_schema(self, event_type: str, schema: Dict[str, Any]):
        """Register a schema for an event type."""
        self.schema_registry[event_type] = {
            "schema": schema,
            "registered_at": datetime.utcnow().isoformat()
        }
        logger.info(f"Registered schema for event type: {event_type}")
    
    def subscribe(self, event_pattern: str, handler: Callable) -> str:
        """Subscribe to events matching a pattern."""
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
        self.subscribers[event_pattern].append({
            "id": subscription_id,
            "handler": handler,
            "subscribed_at": datetime.utcnow()
        })
        logger.info(f"Subscribed to {event_pattern} (ID: {subscription_id})")
        return subscription_id
    
    def unsubscribe(self, event_pattern: str, subscription_id: str):
        """Unsubscribe from events."""
        self.subscribers[event_pattern] = [
            sub for sub in self.subscribers[event_pattern]
            if sub["id"] != subscription_id
        ]
        logger.info(f"Unsubscribed from {event_pattern} (ID: {subscription_id})")
    
    async def publish(self, 
                     event_type: str, 
                     payload: Dict[str, Any],
                     source: str,
                     correlation_id: Optional[str] = None,
                     priority: str = "normal",
                     traceparent: Optional[str] = None) -> str:
        """Publish an event using GME format."""
        
        # Create GME
        gme = GraceMessageEnvelope.create_event(
            event_type=event_type,
            payload=payload,
            source=source,
            priority=priority,
            traceparent=traceparent
        )
        
        if correlation_id:
            gme.payload["correlation_id"] = correlation_id
        
        return await self._enqueue_message(gme)
    
    async def publish_gme(self, gme: GraceMessageEnvelope) -> str:
        """Publish a pre-constructed GME message."""
        return await self._enqueue_message(gme)
    
    async def _enqueue_message(self, gme: GraceMessageEnvelope) -> str:
        """Enqueue message for processing."""
        
        # Check if message is expired
        if gme.is_expired():
            logger.warning(f"Message {gme.msg_id} expired, discarding")
            return gme.msg_id
        
        # Deduplication check
        if self.enable_deduplication:
            if gme.idempotency_key in self.seen_messages:
                self.processing_stats["messages_deduplicated"] += 1
                logger.debug(f"Deduplicated message {gme.msg_id}")
                return gme.msg_id
            self.seen_messages.add(gme.idempotency_key)
        
        # Backpressure management
        queue_size = self.message_queue.qsize()
        if queue_size >= self.backpressure_config.max_queue_size:
            await self._handle_backpressure()
        
        # Enqueue message
        await self.message_queue.put(gme)
        return gme.msg_id
    
    async def _handle_backpressure(self):
        """Handle backpressure by dropping messages or applying flow control."""
        current_size = self.message_queue.qsize()
        target_size = self.backpressure_config.low_water_mark
        messages_to_drop = current_size - target_size
        
        if self.backpressure_config.drop_policy == "oldest":
            # Drop oldest messages
            dropped = 0
            temp_messages = []
            
            # Drain queue to list
            while not self.message_queue.empty() and dropped < messages_to_drop:
                try:
                    message = self.message_queue.get_nowait()
                    if dropped < messages_to_drop:
                        self.processing_stats["messages_dropped"] += 1
                        dropped += 1
                    else:
                        temp_messages.append(message)
                except asyncio.QueueEmpty:
                    break
            
            # Put remaining messages back
            for msg in temp_messages:
                await self.message_queue.put(msg)
        
        logger.warning(f"Backpressure: dropped {messages_to_drop} messages")
    
    async def _message_processor(self, worker_name: str):
        """Process messages from the queue."""
        logger.info(f"Message processor {worker_name} started")
        
        while self.running:
            try:
                # Get message from queue
                gme = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process the message
                await self._process_message(gme, worker_name)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processor {worker_name} error: {e}")
        
        logger.info(f"Message processor {worker_name} stopped")
    
    async def _process_message(self, gme: GraceMessageEnvelope, worker_name: str):
        """Process a single message."""
        try:
            # Find matching subscribers
            handlers = self._find_matching_handlers(gme.headers.event_type)
            
            if not handlers:
                logger.debug(f"No handlers for event type: {gme.headers.event_type}")
                return
            
            # Execute handlers
            tasks = []
            for handler_info in handlers:
                handler = handler_info["handler"]
                task = asyncio.create_task(self._execute_handler(handler, gme))
                tasks.append(task)
            
            # Wait for all handlers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for failures
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                # Some handlers failed, determine if retry is needed
                if gme.retry_count < self.retry_config.max_retries:
                    await self._schedule_retry(gme)
                else:
                    self.dlq.add_message(gme, f"Max retries exceeded: {failures}")
                    self.processing_stats["messages_failed"] += 1
            else:
                self.processing_stats["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing message {gme.msg_id}: {e}")
            if gme.retry_count < self.retry_config.max_retries:
                await self._schedule_retry(gme)
            else:
                self.dlq.add_message(gme, f"Processing error: {str(e)}")
                self.processing_stats["messages_failed"] += 1
    
    def _find_matching_handlers(self, event_type: str) -> List[Dict[str, Any]]:
        """Find handlers that match the event type."""
        matching_handlers = []
        
        for pattern, handlers in self.subscribers.items():
            if self._matches_pattern(event_type, pattern):
                matching_handlers.extend(handlers)
        
        return matching_handlers
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches subscription pattern."""
        if pattern == "*":
            return True
        if "|" in pattern:
            patterns = [p.strip() for p in pattern.split("|")]
            return event_type in patterns
        if pattern.endswith("*"):
            return event_type.startswith(pattern[:-1])
        return event_type == pattern
    
    async def _execute_handler(self, handler: Callable, gme: GraceMessageEnvelope):
        """Execute a single event handler."""
        if asyncio.iscoroutinefunction(handler):
            await handler(gme)
        else:
            handler(gme)
    
    async def _schedule_retry(self, gme: GraceMessageEnvelope):
        """Schedule message for retry with exponential backoff."""
        gme.increment_retry()
        
        # Calculate delay with exponential backoff and jitter
        delay = min(
            self.retry_config.base_delay * (
                self.retry_config.exponential_base ** (gme.retry_count - 1)
            ),
            self.retry_config.max_delay
        )
        
        # Add jitter
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        logger.info(f"Scheduling retry {gme.retry_count} for message {gme.msg_id} in {delay:.2f}s")
        
        # Schedule retry
        asyncio.create_task(self._retry_after_delay(gme, delay))
    
    async def _retry_after_delay(self, gme: GraceMessageEnvelope, delay: float):
        """Retry message after delay."""
        await asyncio.sleep(delay)
        await self._enqueue_message(gme)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        uptime = (datetime.utcnow() - self.processing_stats["start_time"]).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "queue_size": self.message_queue.qsize(),
            "processing_stats": self.processing_stats.copy(),
            "subscribers": {
                pattern: len(handlers) 
                for pattern, handlers in self.subscribers.items()
            },
            "registered_schemas": len(self.schema_registry),
            "dlq_stats": self.dlq.get_stats(),
            "deduplication_cache_size": len(self.seen_messages)
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        queue_size = self.message_queue.qsize()
        is_healthy = (
            self.running and
            queue_size < self.backpressure_config.high_water_mark and
            len(self.processing_workers) > 0
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "running": self.running,
            "queue_size": queue_size,
            "active_workers": len(self.processing_workers),
            "issues": []
        }