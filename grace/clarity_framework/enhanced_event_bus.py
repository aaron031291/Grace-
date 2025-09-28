"""
Class 2: Signal Routing Ambiguity Resolution

Enhanced EventBus with declarative YAML schema support for clear event routing.
Eliminates uncertainty about who listens/responds to what events.
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional, Set
from datetime import datetime
import json
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field

from ..core.contracts import generate_correlation_id


logger = logging.getLogger(__name__)


@dataclass
class EventRoute:
    """Declarative event routing configuration."""
    event_pattern: str  # Event name or pattern (supports wildcards)
    target_components: List[str]  # Component IDs that should receive this event
    priority: int = 5  # 1=highest, 10=lowest
    async_delivery: bool = True
    retry_attempts: int = 3
    timeout_ms: int = 5000
    filter_conditions: Dict[str, Any] = field(default_factory=dict)
    transform_payload: Optional[Dict[str, str]] = None  # Field mappings for payload transformation


@dataclass
class ComponentSubscription:
    """Component subscription information."""
    component_id: str
    handler: Callable
    event_patterns: Set[str]
    is_active: bool = True
    subscription_metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedEventBus:
    """
    Enhanced event bus with declarative YAML routing and advanced features.
    
    Resolves signal routing ambiguity by providing clear, declarative configuration
    for event routing and component subscriptions.
    """
    
    def __init__(self, routing_config_path: Optional[str] = None):
        # Core event handling
        self.subscribers: Dict[str, List[ComponentSubscription]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.correlation_tracking: Dict[str, List[str]] = {}
        
        # Enhanced routing features
        self.routing_rules: List[EventRoute] = []
        self.component_registry: Dict[str, Dict[str, Any]] = {}
        self.event_metrics: Dict[str, Dict[str, int]] = {}
        
        # Configuration
        self.routing_config_path = routing_config_path
        if routing_config_path:
            self.load_routing_config(routing_config_path)
    
    def load_routing_config(self, config_path: str):
        """Load declarative routing configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Routing config file not found: {config_path}")
                self.create_default_routing_config(config_path)
                return
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse routing rules
            self.routing_rules = []
            for rule_data in config_data.get('routing_rules', []):
                route = EventRoute(
                    event_pattern=rule_data['event_pattern'],
                    target_components=rule_data['target_components'],
                    priority=rule_data.get('priority', 5),
                    async_delivery=rule_data.get('async_delivery', True),
                    retry_attempts=rule_data.get('retry_attempts', 3),
                    timeout_ms=rule_data.get('timeout_ms', 5000),
                    filter_conditions=rule_data.get('filter_conditions', {}),
                    transform_payload=rule_data.get('transform_payload')
                )
                self.routing_rules.append(route)
            
            # Parse component definitions
            for comp_data in config_data.get('components', []):
                self.component_registry[comp_data['id']] = comp_data
            
            logger.info(f"Loaded {len(self.routing_rules)} routing rules and "
                       f"{len(self.component_registry)} component definitions")
            
        except Exception as e:
            logger.error(f"Failed to load routing config: {e}")
            self.create_default_routing_config(config_path)
    
    def create_default_routing_config(self, config_path: str):
        """Create a default routing configuration file."""
        default_config = {
            'routing_rules': [
                {
                    'event_pattern': 'GOVERNANCE_*',
                    'target_components': ['governance_engine', 'immutable_logs'],
                    'priority': 1,
                    'async_delivery': True,
                    'retry_attempts': 3
                },
                {
                    'event_pattern': 'LEARNING_*',
                    'target_components': ['learning_kernel', 'memory_core'],
                    'priority': 2,
                    'async_delivery': True
                },
                {
                    'event_pattern': 'HEALTH_*',
                    'target_components': ['avn_core', 'orchestrator'],
                    'priority': 1,
                    'async_delivery': False
                },
                {
                    'event_pattern': '*',
                    'target_components': ['event_logger'],
                    'priority': 10,
                    'async_delivery': True
                }
            ],
            'components': [
                {
                    'id': 'governance_engine',
                    'type': 'governance',
                    'capabilities': ['constitutional_validation', 'policy_enforcement'],
                    'trust_level': 'high'
                },
                {
                    'id': 'learning_kernel',
                    'type': 'learning',
                    'capabilities': ['pattern_detection', 'adaptation'],
                    'trust_level': 'medium'
                },
                {
                    'id': 'avn_core',
                    'type': 'health',
                    'capabilities': ['anomaly_detection', 'health_monitoring'],
                    'trust_level': 'high'
                },
                {
                    'id': 'event_logger',
                    'type': 'utility',
                    'capabilities': ['event_logging'],
                    'trust_level': 'low'
                }
            ]
        }
        
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            logger.info(f"Created default routing config at: {config_path}")
            self.routing_rules = [EventRoute(**rule) for rule in default_config['routing_rules']]
            
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
    
    def register_component(self, component_id: str, component_type: str,
                          capabilities: List[str], trust_level: str = "medium"):
        """Register a component with the event bus."""
        self.component_registry[component_id] = {
            'id': component_id,
            'type': component_type,
            'capabilities': capabilities,
            'trust_level': trust_level,
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"Registered component: {component_id} ({component_type})")
    
    async def subscribe(self, component_id: str, event_pattern: str, 
                       handler: Callable, metadata: Optional[Dict[str, Any]] = None):
        """Subscribe a component handler to event patterns."""
        if event_pattern not in self.subscribers:
            self.subscribers[event_pattern] = []
        
        # Check if component already subscribed to this pattern
        for subscription in self.subscribers[event_pattern]:
            if subscription.component_id == component_id:
                logger.warning(f"Component {component_id} already subscribed to {event_pattern}")
                return
        
        subscription = ComponentSubscription(
            component_id=component_id,
            handler=handler,
            event_patterns={event_pattern},
            subscription_metadata=metadata or {}
        )
        
        self.subscribers[event_pattern].append(subscription)
        logger.info(f"Component {component_id} subscribed to {event_pattern}")
    
    async def unsubscribe(self, component_id: str, event_pattern: str):
        """Unsubscribe a component from an event pattern."""
        if event_pattern in self.subscribers:
            self.subscribers[event_pattern] = [
                sub for sub in self.subscribers[event_pattern]
                if sub.component_id != component_id
            ]
            logger.info(f"Component {component_id} unsubscribed from {event_pattern}")
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return event_type == pattern
        
        # Simple wildcard matching (* at end of pattern)
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return event_type.startswith(prefix)
        
        return event_type == pattern
    
    def _find_routing_rules(self, event_type: str) -> List[EventRoute]:
        """Find routing rules that match the event type."""
        matching_rules = []
        for rule in self.routing_rules:
            if self._matches_pattern(event_type, rule.event_pattern):
                matching_rules.append(rule)
        
        # Sort by priority (lower number = higher priority)
        return sorted(matching_rules, key=lambda r: r.priority)
    
    def _filter_event(self, event: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Apply filter conditions to determine if event should be delivered."""
        if not conditions:
            return True
        
        payload = event.get('payload', {})
        
        for field, expected_value in conditions.items():
            if field not in payload:
                return False
            
            actual_value = payload[field]
            
            # Support for different comparison operators
            if isinstance(expected_value, dict):
                if 'equals' in expected_value:
                    if actual_value != expected_value['equals']:
                        return False
                elif 'contains' in expected_value:
                    if expected_value['contains'] not in str(actual_value):
                        return False
                elif 'greater_than' in expected_value:
                    if actual_value <= expected_value['greater_than']:
                        return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _transform_payload(self, payload: Dict[str, Any], 
                          transform_rules: Dict[str, str]) -> Dict[str, Any]:
        """Transform event payload according to transformation rules."""
        if not transform_rules:
            return payload
        
        transformed = payload.copy()
        
        for target_field, source_field in transform_rules.items():
            if source_field in payload:
                transformed[target_field] = payload[source_field]
            elif '.' in source_field:
                # Support nested field access
                parts = source_field.split('.')
                value = payload
                try:
                    for part in parts:
                        value = value[part]
                    transformed[target_field] = value
                except (KeyError, TypeError):
                    logger.warning(f"Failed to transform field {source_field} to {target_field}")
        
        return transformed
    
    async def publish(self, event_type: str, payload: Dict[str, Any],
                     correlation_id: Optional[str] = None,
                     source_component_id: Optional[str] = None) -> str:
        """
        Publish an event using declarative routing rules.
        """
        if correlation_id is None:
            correlation_id = generate_correlation_id()
        
        event = {
            "type": event_type,
            "payload": payload,
            "correlation_id": correlation_id,
            "source_component_id": source_component_id,
            "timestamp": datetime.now().isoformat(),
            "id": f"evt_{len(self.message_history):06d}"
        }
        
        # Store in history
        self.message_history.append(event)
        
        # Track correlation
        if correlation_id not in self.correlation_tracking:
            self.correlation_tracking[correlation_id] = []
        self.correlation_tracking[correlation_id].append(event["id"])
        
        # Update metrics
        if event_type not in self.event_metrics:
            self.event_metrics[event_type] = {"published": 0, "delivered": 0, "failed": 0}
        self.event_metrics[event_type]["published"] += 1
        
        # Find applicable routing rules
        routing_rules = self._find_routing_rules(event_type)
        
        # Deliver via routing rules
        delivery_tasks = []
        delivered_to = set()
        
        for rule in routing_rules:
            # Apply filters
            if not self._filter_event(event, rule.filter_conditions):
                continue
            
            # Transform payload if needed
            transformed_payload = self._transform_payload(payload, rule.transform_payload)
            if transformed_payload != payload:
                transformed_event = event.copy()
                transformed_event["payload"] = transformed_payload
            else:
                transformed_event = event
            
            # Deliver to target components
            for component_id in rule.target_components:
                if component_id in delivered_to:
                    continue  # Avoid duplicate delivery
                
                delivered_to.add(component_id)
                
                # Find component subscriptions
                component_handlers = []
                for pattern, subscriptions in self.subscribers.items():
                    if self._matches_pattern(event_type, pattern):
                        for subscription in subscriptions:
                            if subscription.component_id == component_id and subscription.is_active:
                                component_handlers.append(subscription.handler)
                
                # Create delivery tasks
                for handler in component_handlers:
                    task = self._safe_deliver(
                        handler, transformed_event, rule.async_delivery,
                        rule.retry_attempts, rule.timeout_ms
                    )
                    delivery_tasks.append(task)
        
        # Also deliver to direct subscribers (fallback for components not in routing rules)
        for pattern, subscriptions in self.subscribers.items():
            if self._matches_pattern(event_type, pattern):
                for subscription in subscriptions:
                    if subscription.component_id not in delivered_to and subscription.is_active:
                        task = self._safe_deliver(subscription.handler, event, True, 3, 5000)
                        delivery_tasks.append(task)
        
        # Execute deliveries
        if delivery_tasks:
            results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            # Update metrics
            successful_deliveries = sum(1 for r in results if r is True)
            failed_deliveries = len(results) - successful_deliveries
            
            self.event_metrics[event_type]["delivered"] += successful_deliveries
            self.event_metrics[event_type]["failed"] += failed_deliveries
        
        logger.info(f"Published {event_type} event to {len(delivery_tasks)} handlers "
                   f"with correlation_id {correlation_id}")
        return correlation_id
    
    async def _safe_deliver(self, handler: Callable, event: Dict[str, Any],
                           async_delivery: bool, retry_attempts: int,
                           timeout_ms: int) -> bool:
        """Safely deliver event to handler with retry logic and timeout."""
        for attempt in range(retry_attempts):
            try:
                if async_delivery and asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(event), timeout=timeout_ms / 1000)
                elif async_delivery:
                    # Run sync handler in executor for async delivery
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, event)
                else:
                    # Synchronous delivery
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"Handler timeout (attempt {attempt + 1}/{retry_attempts})")
                if attempt == retry_attempts - 1:
                    logger.error(f"Handler delivery failed after {retry_attempts} attempts (timeout)")
            except Exception as e:
                logger.warning(f"Handler error (attempt {attempt + 1}/{retry_attempts}): {e}")
                if attempt == retry_attempts - 1:
                    logger.error(f"Handler delivery failed after {retry_attempts} attempts: {e}")
        
        return False
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get event routing metrics."""
        return {
            "event_metrics": self.event_metrics,
            "total_routing_rules": len(self.routing_rules),
            "registered_components": len(self.component_registry),
            "active_subscriptions": sum(
                len([sub for sub in subs if sub.is_active])
                for subs in self.subscribers.values()
            ),
            "message_history_size": len(self.message_history)
        }
    
    def get_component_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific component."""
        if component_id not in self.component_registry:
            return None
        
        component_info = self.component_registry[component_id].copy()
        
        # Add subscription information
        subscriptions = []
        for pattern, subs in self.subscribers.items():
            for sub in subs:
                if sub.component_id == component_id:
                    subscriptions.append({
                        "pattern": pattern,
                        "is_active": sub.is_active,
                        "metadata": sub.subscription_metadata
                    })
        
        component_info["subscriptions"] = subscriptions
        component_info["subscription_count"] = len(subscriptions)
        
        return component_info
    
    async def clear_history(self, keep_recent: int = 1000):
        """Clear old events, keeping only the most recent ones."""
        if len(self.message_history) > keep_recent:
            self.message_history = self.message_history[-keep_recent:]
            logger.info(f"Cleared event history, kept {keep_recent} recent events")