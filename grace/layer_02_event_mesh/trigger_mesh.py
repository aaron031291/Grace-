"""
Trigger Mesh - Event router for governance messages and system-wide event routing.
"""
import asyncio
from typing import Dict, List, Callable, Any, Optional, Set
from datetime import datetime, timedelta
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict

from ..contracts.message_envelope import GraceMessageEnvelope, EventTypes


logger = logging.getLogger(__name__)


class RoutingPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


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
class RoutingMetrics:
    """Tracks routing performance metrics."""
    total_routed: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    avg_latency_ms: float = 0.0
    last_updated: datetime = datetime.now()


class TriggerMesh:
    """
    Advanced event router for governance messages with sub-millisecond routing,
    priority classes, constitutional validators, and mirror/shadow capabilities.
    """
    
    def __init__(self, event_bus, memory_core=None):
        self.event_bus = event_bus  # Should be GraceEventBus instance
        self.memory_core = memory_core
        self.routing_rules: List[RoutingRule] = []
        self.component_registry: Dict[str, Dict[str, Any]] = {}
        self.routing_metrics: Dict[str, RoutingMetrics] = {}
        self.constitutional_validators: List[Callable] = []
        self.shadow_targets: Dict[str, List[str]] = {}
        
        # Performance optimization
        self.routing_cache: Dict[str, List[RoutingRule]] = {}
        self.priority_queues: Dict[RoutingPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in RoutingPriority
        }
        
        self._initialize_default_rules()
        asyncio.create_task(self._start_routing_workers())
    
    def _initialize_default_rules(self):
        """Initialize default routing rules for governance events."""
        default_rules = [
            # Governance validation flow
            RoutingRule(
                event_pattern=EventTypes.GOVERNANCE_VALIDATION,
                target_components=["governance_engine"],
                priority=RoutingPriority.HIGH,
                mode=RoutingMode.UNICAST,
                filter_conditions={},
                transformation_rules={},
                timeout_ms=2000
            ),
            
            # Governance decisions - broadcast to interested parties
            RoutingRule(
                event_pattern=f"{EventTypes.GOVERNANCE_APPROVED}|{EventTypes.GOVERNANCE_REJECTED}",
                target_components=["audit_logger", "notification_service", "memory_core"],
                priority=RoutingPriority.HIGH,
                mode=RoutingMode.MULTICAST,
                filter_conditions={},
                transformation_rules={},
                timeout_ms=1000
            ),
            
            # Parliament reviews
            RoutingRule(
                event_pattern=EventTypes.GOVERNANCE_NEEDS_REVIEW,
                target_components=["parliament", "notification_service"],
                priority=RoutingPriority.NORMAL,
                mode=RoutingMode.MULTICAST,
                filter_conditions={},
                transformation_rules={},
                timeout_ms=3000
            ),
            
            # Critical system events
            RoutingRule(
                event_pattern=f"{EventTypes.GOVERNANCE_ROLLBACK}|{EventTypes.ANOMALY_DETECTED}",
                target_components=["governance_engine", "avn_core", "alert_system"],
                priority=RoutingPriority.CRITICAL,
                mode=RoutingMode.BROADCAST,
                filter_conditions={},
                transformation_rules={},
                timeout_ms=500
            ),
            
            # Learning experiences
            RoutingRule(
                event_pattern="LEARNING_EXPERIENCE",
                target_components=["learning_engine", "immutable_logs"],
                priority=RoutingPriority.LOW,
                mode=RoutingMode.MULTICAST,
                filter_conditions={},
                transformation_rules={},
                timeout_ms=10000
            ),
            
            # Trust updates
            RoutingRule(
                event_pattern="TRUST_UPDATED",
                target_components=["trust_core", "governance_engine"],
                priority=RoutingPriority.NORMAL,
                mode=RoutingMode.MULTICAST,
                filter_conditions={},
                transformation_rules={},
                timeout_ms=2000
            )
        ]
        
        self.routing_rules.extend(default_rules)
    
    async def _start_routing_workers(self):
        """Start priority-based routing workers."""
        # Create workers for each priority level
        for priority in RoutingPriority:
            worker_count = {
                RoutingPriority.CRITICAL: 4,
                RoutingPriority.HIGH: 3,
                RoutingPriority.NORMAL: 2,
                RoutingPriority.LOW: 1
            }.get(priority, 1)
            
            for i in range(worker_count):
                asyncio.create_task(
                    self._routing_worker(priority, f"{priority.value}_worker_{i}")
                )
    
    async def _routing_worker(self, priority: RoutingPriority, worker_name: str):
        """Worker coroutine for processing routing tasks."""
        queue = self.priority_queues[priority]
        
        while True:
            try:
                # Get routing task
                routing_task = await queue.get()
                
                # Process the routing
                start_time = datetime.now()
                await self._execute_routing(routing_task)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Update metrics
                await self._update_routing_metrics(
                    routing_task["event_type"], 
                    processing_time, 
                    True
                )
                
                # Mark task as done
                queue.task_done()
                
            except Exception as e:
                logger.error(f"Routing worker {worker_name} error: {e}")
                await self._update_routing_metrics(
                    routing_task.get("event_type", "unknown"), 
                    0, 
                    False
                )
    
    async def route_event(self, event_type: str, payload: Dict[str, Any],
                         correlation_id: str, source_component: str = "unknown") -> bool:
        """
        Route an event through the trigger mesh.
        
        Args:
            event_type: Type of event to route
            payload: Event payload
            correlation_id: Correlation ID for tracing
            source_component: Component that originated the event
            
        Returns:
            True if routing was initiated successfully
        """
        try:
            # Find matching routing rules
            matching_rules = await self._find_matching_rules(event_type, payload)
            
            if not matching_rules:
                logger.warning(f"No routing rules found for event: {event_type}")
                return False
            
            # Apply constitutional validation
            if not await self._validate_constitutional_compliance(event_type, payload):
                logger.warning(f"Constitutional validation failed for event: {event_type}")
                return False
            
            # Create routing tasks for each matching rule
            for rule in matching_rules:
                routing_task = {
                    "event_type": event_type,
                    "payload": payload,
                    "correlation_id": correlation_id,
                    "source_component": source_component,
                    "rule": rule,
                    "created_at": datetime.now()
                }
                
                # Add to appropriate priority queue
                await self.priority_queues[rule.priority].put(routing_task)
            
            logger.debug(f"Queued {len(matching_rules)} routing tasks for {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error routing event {event_type}: {e}")
            return False
    
    async def _find_matching_rules(self, event_type: str, 
                                 payload: Dict[str, Any]) -> List[RoutingRule]:
        """Find routing rules that match the event."""
        # Check cache first
        cache_key = f"{event_type}:{hash(json.dumps(payload, sort_keys=True))}"
        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]
        
        matching_rules = []
        
        for rule in self.routing_rules:
            if await self._matches_pattern(event_type, rule.event_pattern):
                if await self._matches_filter_conditions(payload, rule.filter_conditions):
                    matching_rules.append(rule)
        
        # Cache the result
        self.routing_cache[cache_key] = matching_rules
        
        # Limit cache size
        if len(self.routing_cache) > 10000:
            # Remove oldest 25% of cache entries
            keys_to_remove = list(self.routing_cache.keys())[:2500]
            for key in keys_to_remove:
                del self.routing_cache[key]
        
        return matching_rules
    
    async def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern (supports wildcards and OR)."""
        # Handle OR patterns
        if "|" in pattern:
            patterns = pattern.split("|")
            return any(await self._matches_pattern(event_type, p.strip()) for p in patterns)
        
        # Handle wildcards
        if "*" in pattern:
            # Convert pattern to regex-like matching
            pattern_parts = pattern.split("*")
            current_pos = 0
            
            for i, part in enumerate(pattern_parts):
                if not part:  # Empty part from leading/trailing *
                    continue
                
                pos = event_type.find(part, current_pos)
                if pos == -1:
                    return False
                
                if i == 0 and pos != 0:  # First part must match at start if no leading *
                    if not pattern.startswith("*"):
                        return False
                
                current_pos = pos + len(part)
            
            # Check if last part must match at end
            if not pattern.endswith("*") and current_pos != len(event_type):
                return False
            
            return True
        
        # Exact match
        return event_type == pattern
    
    async def _matches_filter_conditions(self, payload: Dict[str, Any],
                                       conditions: Dict[str, Any]) -> bool:
        """Check if payload matches filter conditions."""
        if not conditions:
            return True
        
        for key, expected_value in conditions.items():
            # Navigate nested payload
            value = payload
            for part in key.split("."):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return False
            
            # Check condition
            if isinstance(expected_value, dict):
                # Complex conditions (gt, lt, in, etc.)
                for op, op_value in expected_value.items():
                    if op == "gt" and value <= op_value:
                        return False
                    elif op == "lt" and value >= op_value:
                        return False
                    elif op == "in" and value not in op_value:
                        return False
                    elif op == "eq" and value != op_value:
                        return False
            else:
                # Simple equality
                if value != expected_value:
                    return False
        
        return True
    
    async def _validate_constitutional_compliance(self, event_type: str,
                                                payload: Dict[str, Any]) -> bool:
        """Validate event against constitutional constraints."""
        for validator in self.constitutional_validators:
            try:
                if not await validator(event_type, payload):
                    return False
            except Exception as e:
                logger.error(f"Constitutional validator error: {e}")
                return False
        
        return True
    
    async def _execute_routing(self, routing_task: Dict[str, Any]):
        """Execute a routing task."""
        rule = routing_task["rule"]
        event_type = routing_task["event_type"]
        payload = routing_task["payload"]
        correlation_id = routing_task["correlation_id"]
        
        # Apply transformations
        transformed_payload = await self._apply_transformations(payload, rule.transformation_rules)
        
        # Route based on mode
        if rule.mode == RoutingMode.UNICAST:
            await self._route_unicast(rule, event_type, transformed_payload, correlation_id)
        elif rule.mode == RoutingMode.MULTICAST:
            await self._route_multicast(rule, event_type, transformed_payload, correlation_id)
        elif rule.mode == RoutingMode.BROADCAST:
            await self._route_broadcast(rule, event_type, transformed_payload, correlation_id)
        elif rule.mode == RoutingMode.MIRROR:
            await self._route_mirror(rule, event_type, transformed_payload, correlation_id)
    
    async def _apply_transformations(self, payload: Dict[str, Any],
                                   transformation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation rules to payload."""
        if not transformation_rules:
            return payload
        
        transformed = payload.copy()
        
        for rule_type, rule_config in transformation_rules.items():
            if rule_type == "add_fields":
                transformed.update(rule_config)
            elif rule_type == "remove_fields":
                for field in rule_config:
                    transformed.pop(field, None)
            elif rule_type == "rename_fields":
                for old_field, new_field in rule_config.items():
                    if old_field in transformed:
                        transformed[new_field] = transformed.pop(old_field)
            elif rule_type == "filter_fields":
                # Keep only specified fields
                filtered = {field: transformed.get(field) for field in rule_config}
                transformed = filtered
        
        return transformed
    
    async def _route_unicast(self, rule: RoutingRule, event_type: str,
                           payload: Dict[str, Any], correlation_id: str):
        """Route event to a single target (load balanced)."""
        if not rule.target_components:
            return
        
        # Simple round-robin load balancing
        target = rule.target_components[0]  # Simplified for now
        await self._deliver_to_component(target, event_type, payload, correlation_id, rule.timeout_ms)
    
    async def _route_multicast(self, rule: RoutingRule, event_type: str,
                             payload: Dict[str, Any], correlation_id: str):
        """Route event to multiple specific targets."""
        delivery_tasks = []
        
        for target in rule.target_components:
            task = self._deliver_to_component(target, event_type, payload, correlation_id, rule.timeout_ms)
            delivery_tasks.append(task)
        
        # Wait for all deliveries (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*delivery_tasks, return_exceptions=True),
                timeout=rule.timeout_ms / 1000.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Multicast delivery timeout for {event_type}")
    
    async def _route_broadcast(self, rule: RoutingRule, event_type: str,
                             payload: Dict[str, Any], correlation_id: str):
        """Route event to all registered components."""
        delivery_tasks = []
        
        # Send to all registered components
        for component_id in self.component_registry.keys():
            task = self._deliver_to_component(component_id, event_type, payload, correlation_id, rule.timeout_ms)
            delivery_tasks.append(task)
        
        # Fire and forget for broadcasts
        asyncio.create_task(asyncio.gather(*delivery_tasks, return_exceptions=True))
    
    async def _route_mirror(self, rule: RoutingRule, event_type: str,
                          payload: Dict[str, Any], correlation_id: str):
        """Route event to mirror/shadow instances."""
        # Check for shadow targets
        shadow_targets = self.shadow_targets.get(event_type, [])
        
        delivery_tasks = []
        
        # Deliver to primary targets
        for target in rule.target_components:
            task = self._deliver_to_component(target, event_type, payload, correlation_id, rule.timeout_ms)
            delivery_tasks.append(task)
        
        # Deliver to shadow targets (non-blocking)
        for shadow_target in shadow_targets:
            # Mark as shadow delivery
            shadow_payload = payload.copy()
            shadow_payload["__shadow_delivery__"] = True
            
            task = self._deliver_to_component(
                shadow_target, event_type, shadow_payload, 
                f"shadow_{correlation_id}", rule.timeout_ms
            )
            delivery_tasks.append(task)
        
        await asyncio.gather(*delivery_tasks, return_exceptions=True)
    
    async def _deliver_to_component(self, component_id: str, event_type: str,
                                  payload: Dict[str, Any], correlation_id: str,
                                  timeout_ms: int):
        """Deliver event to a specific component."""
        try:
            # Check if component is registered
            if component_id not in self.component_registry:
                logger.warning(f"Component {component_id} not registered")
                return False
            
            component_info = self.component_registry[component_id]
            
            # Create event for delivery
            event = {
                "type": event_type,
                "payload": payload,
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "target_component": component_id,
                "delivery_id": f"del_{datetime.now().strftime('%H%M%S')}_{component_id}"
            }
            
            # Use event bus for delivery (assuming GraceEventBus)
            if hasattr(self.event_bus, 'publish_gme'):
                # Create GME and publish
                gme = GraceMessageEnvelope.create_event(
                    event_type=event_type,
                    payload=payload,
                    source="trigger_mesh",
                    correlation_id=correlation_id
                )
                await asyncio.wait_for(
                    self.event_bus.publish_gme(gme),
                    timeout=timeout_ms / 1000.0
                )
            else:
                # Fallback to legacy publish method
                await asyncio.wait_for(
                    self.event_bus.publish(event_type, payload, correlation_id),
                    timeout=timeout_ms / 1000.0
                )
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Delivery timeout to {component_id} for {event_type}")
            return False
        except Exception as e:
            logger.error(f"Delivery error to {component_id}: {e}")
            return False
    
    def register_component(self, component_id: str, component_type: str,
                         event_subscriptions: List[str],
                         metadata: Optional[Dict[str, Any]] = None):
        """Register a component with the trigger mesh."""
        self.component_registry[component_id] = {
            "component_type": component_type,
            "event_subscriptions": event_subscriptions,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        
        logger.info(f"Registered component {component_id} ({component_type})")
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule."""
        self.routing_rules.append(rule)
        # Clear cache since rules changed
        self.routing_cache.clear()
        
        logger.info(f"Added routing rule for pattern: {rule.event_pattern}")
    
    def add_constitutional_validator(self, validator: Callable[[str, Dict[str, Any]], bool]):
        """Add a constitutional validator function."""
        self.constitutional_validators.append(validator)
        logger.info("Added constitutional validator")
    
    def add_shadow_target(self, event_type: str, shadow_component: str):
        """Add a shadow target for mirrored routing."""
        if event_type not in self.shadow_targets:
            self.shadow_targets[event_type] = []
        
        if shadow_component not in self.shadow_targets[event_type]:
            self.shadow_targets[event_type].append(shadow_component)
            logger.info(f"Added shadow target {shadow_component} for {event_type}")
    
    def remove_shadow_target(self, event_type: str, shadow_component: str):
        """Remove a shadow target."""
        if event_type in self.shadow_targets:
            self.shadow_targets[event_type] = [
                target for target in self.shadow_targets[event_type]
                if target != shadow_component
            ]
            logger.info(f"Removed shadow target {shadow_component} for {event_type}")
    
    async def _update_routing_metrics(self, event_type: str, latency_ms: float, success: bool):
        """Update routing metrics."""
        if event_type not in self.routing_metrics:
            self.routing_metrics[event_type] = RoutingMetrics()
        
        metrics = self.routing_metrics[event_type]
        metrics.total_routed += 1
        
        if success:
            metrics.successful_deliveries += 1
        else:
            metrics.failed_deliveries += 1
        
        # Update rolling average latency
        alpha = 0.1  # Smoothing factor
        if metrics.avg_latency_ms == 0:
            metrics.avg_latency_ms = latency_ms
        else:
            metrics.avg_latency_ms = alpha * latency_ms + (1 - alpha) * metrics.avg_latency_ms
        
        metrics.last_updated = datetime.now()
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        return {
            "event_metrics": {
                event_type: asdict(metrics)
                for event_type, metrics in self.routing_metrics.items()
            },
            "queue_sizes": {
                priority.value: queue.qsize()
                for priority, queue in self.priority_queues.items()
            },
            "registered_components": len(self.component_registry),
            "routing_rules": len(self.routing_rules),
            "cache_size": len(self.routing_cache)
        }
    
    def get_component_registry(self) -> Dict[str, Any]:
        """Get registered components."""
        return self.component_registry.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the trigger mesh."""
        return {
            "status": "healthy",
            "uptime": "running",  # Could track actual uptime
            "routing_metrics": self.get_routing_metrics(),
            "registered_components": len(self.component_registry),
            "constitutional_validators": len(self.constitutional_validators),
            "shadow_targets": {
                event_type: len(targets)
                for event_type, targets in self.shadow_targets.items()
            }
        }