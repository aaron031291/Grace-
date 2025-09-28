"""
Trigger Mesh - Enhanced event router with priority handling and latency recording.
Part of Phase 5: Trigger Mesh MVP implementation.
"""
import asyncio
import time
from typing import Dict, List, Callable, Any, Optional, Set
from datetime import datetime, timedelta
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict
import heapq
from collections import defaultdict

from ..contracts.message_envelope import GraceMessageEnvelope, EventTypes
from ..core.immutable_logs import ImmutableLogs, TransparencyLevel
from ..config.environment import get_grace_config

logger = logging.getLogger(__name__)


class RoutingPriority(Enum):
    CRITICAL = 1  # Immediate processing
    HIGH = 2     # High priority queue
    NORMAL = 3   # Normal priority queue  
    LOW = 4      # Low priority queue


class RoutingMode(Enum):
    UNICAST = "unicast"      # One-to-one
    MULTICAST = "multicast"  # One-to-many
    BROADCAST = "broadcast"  # One-to-all
    MIRROR = "mirror"        # Duplicate to multiple instances


@dataclass
class RoutingRule:
    """Defines how events should be routed."""
    event_pattern: str  # Event pattern to match (supports wildcards)
    target_components: List[str]
    priority: RoutingPriority
    mode: RoutingMode
    filter_conditions: Dict[str, Any]
    transformation_rules: Dict[str, Any]
    timeout_ms: int = 5000
    max_retries: int = 3


@dataclass
class PriorityEvent:
    """Event wrapper for priority queue processing."""
    priority: int  # Lower number = higher priority
    timestamp: float
    event_id: str
    event: Dict[str, Any]
    routing_rule: Optional[RoutingRule] = None
    
    def __lt__(self, other):
        """For heap queue sorting - higher priority first, then by timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class RoutingMetrics:
    """Tracks routing performance metrics."""
    total_routed: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    total_latency_ms: float = 0.0
    priority_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.priority_counts is None:
            self.priority_counts = {
                "critical": 0,
                "high": 0,
                "normal": 0,
                "low": 0
            }
    avg_latency_ms: float = 0.0
    last_updated: datetime = datetime.now()


class TriggerMesh:
    """
    Enhanced event router with priority handling, latency recording, and constitutional validation.
    Part of Phase 5: Trigger Mesh MVP implementation.
    """
    
    def __init__(self, event_bus=None, immutable_logs: Optional[ImmutableLogs] = None):
        self.config = get_grace_config()
        self.event_bus = event_bus
        self.immutable_logs = immutable_logs
        
        # Routing configuration
        self.routing_rules: List[RoutingRule] = []
        self.component_registry: Dict[str, Dict[str, Any]] = {}
        self.routing_metrics = RoutingMetrics()
        
        # Priority queue system for event processing
        self.priority_queue: List[PriorityEvent] = []  # Using heapq
        self.processing_workers: List[asyncio.Task] = []
        self.worker_count = self.config["event_routing"]["priority_workers"]["normal"]
        
        # Performance tracking
        self.routing_cache: Dict[str, List[RoutingRule]] = {}
        self.latency_history: defaultdict[str, List[float]] = defaultdict(list)
        
        # System state
        self.running = False
        self.started_at: Optional[datetime] = None
        
        # Initialize default routing rules
        self._initialize_default_rules()
        
        logger.info("TriggerMesh initialized with priority routing")
    
    async def start(self):
        """Start the trigger mesh with priority processing workers."""
        if self.running:
            logger.warning("TriggerMesh already running")
            return
            
        self.running = True
        self.started_at = datetime.now()
        
        # Start priority processing workers
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._priority_worker(f"worker_{i}"))
            self.processing_workers.append(worker)
        
        # Log startup
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="trigger_mesh_started",
                component_id="trigger_mesh",
                event_data={
                    "worker_count": self.worker_count,
                    "routing_rules": len(self.routing_rules)
                },
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL
            )
        
        logger.info(f"TriggerMesh started with {self.worker_count} workers")
    
    async def stop(self):
        """Stop the trigger mesh and workers."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop all workers
        for worker in self.processing_workers:
            worker.cancel()
        
        await asyncio.gather(*self.processing_workers, return_exceptions=True)
        self.processing_workers.clear()
        
        # Log shutdown with metrics
        if self.immutable_logs:
            await self.immutable_logs.log_event(
                event_type="trigger_mesh_stopped",
                component_id="trigger_mesh",
                event_data={
                    "total_routed": self.routing_metrics.total_routed,
                    "success_rate": self._calculate_success_rate(),
                    "avg_latency_ms": self._calculate_avg_latency()
                },
                transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL
            )
        
        logger.info("TriggerMesh stopped")
    
    async def route_event(self, event: Dict[str, Any], priority: RoutingPriority = RoutingPriority.NORMAL,
                         correlation_id: Optional[str] = None) -> str:
        """
        Route an event with priority handling and latency tracking.
        This is the main method specified in the requirements.
        """
        start_time = time.time()
        event_id = f"route_{int(start_time * 1000000)}"
        
        # Enhance event with routing metadata
        enhanced_event = event.copy()
        enhanced_event.update({
            "event_id": event_id,
            "correlation_id": correlation_id or event.get("correlation_id"),
            "priority": priority.name.lower(),
            "routed_at": datetime.now().isoformat(),
            "routing_metadata": {
                "mesh_version": "1.0.0",
                "source_component": "trigger_mesh"
            }
        })
        
        # Find applicable routing rules
        matching_rules = self._find_matching_rules(enhanced_event)
        
        if not matching_rules:
            logger.warning(f"No routing rules found for event {event.get('type', 'unknown')}")
            return event_id
        
        # Create priority event for processing
        priority_event = PriorityEvent(
            priority=priority.value,
            timestamp=start_time,
            event_id=event_id,
            event=enhanced_event,
            routing_rule=matching_rules[0] if matching_rules else None
        )
        
        # Add to priority queue
        heapq.heappush(self.priority_queue, priority_event)
        
        # Record in metrics
        self.routing_metrics.total_routed += 1
        self.routing_metrics.priority_counts[priority.name.lower()] += 1
        
        # Log high-priority events immediately
        if priority in [RoutingPriority.CRITICAL, RoutingPriority.HIGH] and self.immutable_logs:
            await self.immutable_logs.log_system_performance(
                metric_name="event_routed",
                metric_value=1.0,
                component_id="trigger_mesh",
                tags={
                    "priority": priority.name.lower(),
                    "event_type": event.get("type", "unknown")
                },
                correlation_id=correlation_id
            )
        
        logger.debug(f"Event {event_id} queued with {priority.name} priority")
        
        return event_id
    
    async def _priority_worker(self, worker_id: str):
        """Background worker for processing priority events."""
        logger.info(f"Priority worker {worker_id} started")
        
        while self.running:
            try:
                # Process events from priority queue
                if self.priority_queue:
                    priority_event = heapq.heappop(self.priority_queue)
                    await self._process_priority_event(priority_event, worker_id)
                else:
                    # No events - brief sleep to prevent busy waiting
                    await asyncio.sleep(0.001)  # 1ms
                    
            except Exception as e:
                logger.error(f"Error in priority worker {worker_id}: {e}")
                await asyncio.sleep(0.1)  # Longer sleep on error
        
        logger.info(f"Priority worker {worker_id} stopped")
    
    async def _process_priority_event(self, priority_event: PriorityEvent, worker_id: str):
        """Process a single priority event with latency tracking."""
        processing_start = time.time()
        event = priority_event.event
        rule = priority_event.routing_rule
        
        try:
            # Calculate queueing latency
            queue_latency_ms = (processing_start - priority_event.timestamp) * 1000
            
            # Record queueing latency
            if self.immutable_logs and queue_latency_ms > 10:  # Log if > 10ms queue time
                await self.immutable_logs.log_system_performance(
                    metric_name="event_queue_latency",
                    metric_value=queue_latency_ms,
                    component_id="trigger_mesh",
                    tags={
                        "priority": RoutingPriority(priority_event.priority).name.lower(),
                        "worker_id": worker_id
                    }
                )
            
            # Apply routing rule if available
            if rule:
                await self._apply_routing_rule(event, rule)
            else:
                # Fallback: broadcast to event bus
                if self.event_bus:
                    await self.event_bus.publish(
                        event.get("type", "unknown"),
                        event,
                        correlation_id=event.get("correlation_id"),
                        priority=RoutingPriority(priority_event.priority).name.lower()
                    )
            
            # Calculate total processing latency
            total_latency_ms = (time.time() - priority_event.timestamp) * 1000
            
            # Update metrics
            self.routing_metrics.successful_deliveries += 1
            self.routing_metrics.total_latency_ms += total_latency_ms
            
            # Track latency history for the event type
            event_type = event.get("type", "unknown")
            self.latency_history[event_type].append(total_latency_ms)
            if len(self.latency_history[event_type]) > 1000:  # Keep last 1000 samples
                self.latency_history[event_type] = self.latency_history[event_type][-1000:]
            
            # Log latency for critical/high priority events or slow events
            priority_name = RoutingPriority(priority_event.priority).name.lower()
            if priority_name in ["critical", "high"] or total_latency_ms > 100:
                if self.immutable_logs:
                    await self.immutable_logs.log_system_performance(
                        metric_name="event_processing_latency",
                        metric_value=total_latency_ms,
                        component_id="trigger_mesh",
                        tags={
                            "priority": priority_name,
                            "event_type": event_type,
                            "worker_id": worker_id
                        },
                        correlation_id=event.get("correlation_id")
                    )
            
            logger.debug(f"Processed {priority_event.event_id} in {total_latency_ms:.2f}ms")
            
        except Exception as e:
            self.routing_metrics.failed_deliveries += 1
            
            # Log processing failure
            if self.immutable_logs:
                await self.immutable_logs.log_event(
                    event_type="event_processing_failed",
                    component_id="trigger_mesh",
                    event_data={
                        "event_id": priority_event.event_id,
                        "error": str(e),
                        "worker_id": worker_id,
                        "priority": RoutingPriority(priority_event.priority).name.lower()
                    },
                    correlation_id=event.get("correlation_id"),
                    transparency_level=TransparencyLevel.GOVERNANCE_INTERNAL
                )
            
            logger.error(f"Failed to process event {priority_event.event_id}: {e}")
    
    async def _apply_routing_rule(self, event: Dict[str, Any], rule: RoutingRule):
        """Apply a specific routing rule to an event."""
        # Apply transformations if specified
        if rule.transformation_rules:
            event = self._transform_event(event, rule.transformation_rules)
        
        # Route based on mode
        if rule.mode == RoutingMode.UNICAST:
            # Route to first available target
            if rule.target_components:
                await self._route_to_component(event, rule.target_components[0])
        
        elif rule.mode == RoutingMode.MULTICAST:
            # Route to all specified targets
            tasks = []
            for component in rule.target_components:
                tasks.append(self._route_to_component(event, component))
            await asyncio.gather(*tasks, return_exceptions=True)
        
        elif rule.mode == RoutingMode.BROADCAST:
            # Route to all registered components
            tasks = []
            for component_id in self.component_registry:
                tasks.append(self._route_to_component(event, component_id))
            await asyncio.gather(*tasks, return_exceptions=True)
        
        elif rule.mode == RoutingMode.MIRROR:
            # Route to primary and mirror targets
            tasks = []
            for component in rule.target_components:
                tasks.append(self._route_to_component(event, component))
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _route_to_component(self, event: Dict[str, Any], component_id: str):
        """Route an event to a specific component."""
        try:
            if component_id in self.component_registry:
                component_info = self.component_registry[component_id]
                handler = component_info.get("handler")
                
                if handler:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                else:
                    # Fallback: publish to event bus with component-specific topic
                    if self.event_bus:
                        await self.event_bus.publish(
                            f"component.{component_id}",
                            event,
                            correlation_id=event.get("correlation_id")
                        )
            else:
                logger.warning(f"Component {component_id} not registered")
                
        except Exception as e:
            logger.error(f"Failed to route to component {component_id}: {e}")
            raise
    
    def _find_matching_rules(self, event: Dict[str, Any]) -> List[RoutingRule]:
        """Find routing rules that match the event."""
        event_type = event.get("type", "")
        
        # Check cache first
        if event_type in self.routing_cache:
            return self.routing_cache[event_type]
        
        matching_rules = []
        for rule in self.routing_rules:
            if self._event_matches_pattern(event_type, rule.event_pattern):
                if self._event_matches_filters(event, rule.filter_conditions):
                    matching_rules.append(rule)
        
        # Cache the result
        self.routing_cache[event_type] = matching_rules
        
        return matching_rules
    
    def _event_matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches the routing pattern."""
        # Simple pattern matching - could be enhanced with regex
        if pattern == "*":
            return True
        if pattern == event_type:
            return True
        if pattern.endswith("*") and event_type.startswith(pattern[:-1]):
            return True
        return False
    
    def _event_matches_filters(self, event: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if event matches filter conditions."""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key not in event or event[key] != expected_value:
                return False
        
        return True
    
    def _transform_event(self, event: Dict[str, Any], transformations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformations to an event."""
        transformed = event.copy()
        
        # Simple transformation rules - could be enhanced
        for key, transformation in transformations.items():
            if transformation == "remove":
                transformed.pop(key, None)
            elif isinstance(transformation, dict) and "rename_to" in transformation:
                if key in transformed:
                    transformed[transformation["rename_to"]] = transformed.pop(key)
        
        return transformed
    
    def register_component(self, component_id: str, handler: Optional[Callable] = None, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Register a component for event routing."""
        self.component_registry[component_id] = {
            "handler": handler,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat()
        }
        
        # Clear routing cache when components change
        self.routing_cache.clear()
        
        logger.info(f"Registered component: {component_id}")
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule."""
        self.routing_rules.append(rule)
        
        # Clear routing cache when rules change
        self.routing_cache.clear()
        
        logger.info(f"Added routing rule for pattern: {rule.event_pattern}")
    
    def _initialize_default_rules(self):
        """Initialize default routing rules for governance events."""
        # Critical system events
        self.add_routing_rule(RoutingRule(
            event_pattern="constitutional_violation",
            target_components=["governance_engine", "audit_system"],
            priority=RoutingPriority.CRITICAL,
            mode=RoutingMode.MULTICAST,
            filter_conditions={},
            transformation_rules={},
            timeout_ms=1000
        ))
        
        # High priority governance events
        self.add_routing_rule(RoutingRule(
            event_pattern="governance_*",
            target_components=["governance_engine"],
            priority=RoutingPriority.HIGH,
            mode=RoutingMode.UNICAST,
            filter_conditions={},
            transformation_rules={},
            timeout_ms=2000
        ))
        
        # Normal system events
        self.add_routing_rule(RoutingRule(
            event_pattern="system_*",
            target_components=["system_monitor"],
            priority=RoutingPriority.NORMAL,
            mode=RoutingMode.UNICAST,
            filter_conditions={},
            transformation_rules={},
            timeout_ms=5000
        ))
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics."""
        return {
            "total_routed": self.routing_metrics.total_routed,
            "successful_deliveries": self.routing_metrics.successful_deliveries,
            "failed_deliveries": self.routing_metrics.failed_deliveries,
            "success_rate": self._calculate_success_rate(),
            "average_latency_ms": self._calculate_avg_latency(),
            "priority_distribution": self.routing_metrics.priority_counts,
            "registered_components": len(self.component_registry),
            "routing_rules": len(self.routing_rules),
            "queue_size": len(self.priority_queue),
            "workers_active": len([w for w in self.processing_workers if not w.done()])
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate routing success rate."""
        total = self.routing_metrics.successful_deliveries + self.routing_metrics.failed_deliveries
        if total == 0:
            return 1.0
        return self.routing_metrics.successful_deliveries / total
    
    def _calculate_avg_latency(self) -> float:
        """Calculate average processing latency."""
        if self.routing_metrics.successful_deliveries == 0:
            return 0.0
        return self.routing_metrics.total_latency_ms / self.routing_metrics.successful_deliveries
    
    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics by event type."""
        stats = {}
        
        for event_type, latencies in self.latency_history.items():
            if latencies:
                stats[event_type] = {
                    "count": len(latencies),
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies)
                }
        
        return stats