"""
TriggerMesh - Event routing and subscription management
"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import asyncio
import fnmatch

from grace.schemas.events import GraceEvent, EventPriority

logger = logging.getLogger(__name__)


class RouteFilter:
    """Base class for route filters"""
    
    @staticmethod
    def evaluate(event: GraceEvent, filter_config: Dict[str, Any]) -> bool:
        """Evaluate if event passes filter"""
        filter_type = filter_config.get("type")
        
        if filter_type == "trust_threshold":
            threshold = filter_config.get("threshold", 0.5)
            return event.trust_score >= threshold
        
        elif filter_type == "source_in":
            sources = filter_config.get("sources", [])
            return event.source in sources
        
        elif filter_type == "trust_change_significant":
            threshold = filter_config.get("threshold", 0.1)
            change = event.payload.get("trust_change", 0)
            return abs(change) >= threshold
        
        elif filter_type == "authorized":
            # Would check actual authorization
            return True
        
        # Unknown filter type passes by default
        return True


class TriggerMeshRoute:
    """Represents a single route in the mesh"""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("name", "unnamed")
        self.pattern = config.get("pattern", "*")
        self.targets = config.get("targets", [])
        self.filters = config.get("filters", [])
        self.actions = config.get("actions", [])
        self.priority = config.get("priority", "normal")
        self.timeout_seconds = config.get("timeout_seconds", None)
    
    def matches(self, event: GraceEvent) -> bool:
        """Check if event matches this route pattern"""
        return fnmatch.fnmatch(event.event_type, self.pattern)
    
    def apply_filters(self, event: GraceEvent) -> bool:
        """Apply all filters to event"""
        for filter_config in self.filters:
            if not RouteFilter.evaluate(event, filter_config):
                return False
        return True
    
    async def execute_actions(self, event: GraceEvent):
        """Execute route actions"""
        for action in self.actions:
            action_type = action.get("type")
            
            if action_type == "log":
                level = action.get("level", "info")
                getattr(logger, level)(
                    f"Route action: {self.name}",
                    extra={"event_id": event.event_id}
                )
            
            elif action_type == "metric":
                metric_name = action.get("name")
                logger.debug(f"Metric: {metric_name} +1")
            
            elif action_type == "alert":
                channel = action.get("channel")
                logger.warning(f"Alert to {channel}: {event.event_type}")


class TriggerMesh:
    """
    TriggerMesh - Central event routing system
    
    Loads configuration from YAML and manages event routing
    """
    
    def __init__(self, event_bus, config_path: Optional[str] = None):
        self.event_bus = event_bus
        self.config_path = config_path or "config/trigger_mesh.yaml"
        
        self.routes: List[TriggerMeshRoute] = []
        self.subscriptions: Dict[str, List[str]] = {}
        self.config: Dict[str, Any] = {}
        
        self._loaded = False
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"TriggerMesh config not found: {self.config_path}")
                logger.info("Using default configuration")
                self._load_defaults()
                return
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Loaded TriggerMesh config from {self.config_path}")
            
            # Load routes
            for route_config in self.config.get("routes", []):
                route = TriggerMeshRoute(route_config)
                self.routes.append(route)
                logger.debug(f"Loaded route: {route.name} ({route.pattern})")
            
            # Load subscriptions
            for sub_config in self.config.get("subscriptions", []):
                subscriber = sub_config.get("subscriber")
                patterns = sub_config.get("patterns", [])
                self.subscriptions[subscriber] = patterns
                logger.debug(f"Loaded subscription: {subscriber} -> {patterns}")
            
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load TriggerMesh config: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.config = {
            "version": "1.0",
            "defaults": {
                "ttl_seconds": 300,
                "priority": "normal"
            },
            "routes": [],
            "subscriptions": {}
        }
        self._loaded = True
    
    def bind_subscriptions(self):
        """Bind subscriptions to event bus"""
        if not self._loaded:
            logger.warning("TriggerMesh not loaded, cannot bind subscriptions")
            return
        
        for subscriber, patterns in self.subscriptions.items():
            for pattern in patterns:
                # Create a handler for this subscription
                async def handler(event: GraceEvent, sub=subscriber, pat=pattern):
                    if fnmatch.fnmatch(event.event_type, pat):
                        logger.debug(
                            f"Subscription match: {sub} <- {event.event_type}",
                            extra={"subscriber": sub, "pattern": pat}
                        )
                
                # Subscribe to the pattern
                # Note: EventBus doesn't support wildcard patterns directly,
                # so we'd need to subscribe to base event types
                base_type = pattern.split('.')[0] if '.' in pattern else pattern
                
                try:
                    self.event_bus.subscribe(base_type, handler)
                    logger.info(f"Bound subscription: {subscriber} -> {pattern}")
                except Exception as e:
                    logger.error(f"Failed to bind subscription {subscriber}: {e}")
    
    async def emit(
        self,
        event: GraceEvent,
        apply_routes: bool = True
    ) -> bool:
        """
        Emit event through mesh with routing
        
        Args:
            event: Event to emit
            apply_routes: Whether to apply route filters and actions
        
        Returns:
            True if emitted successfully
        """
        if apply_routes:
            # Find matching routes
            matched_routes = []
            for route in self.routes:
                if route.matches(event) and route.apply_filters(event):
                    matched_routes.append(route)
            
            # Apply route actions
            for route in matched_routes:
                try:
                    await route.execute_actions(event)
                    
                    # Add route targets to event
                    for target in route.targets:
                        if target not in event.targets:
                            event.targets.append(target)
                
                except Exception as e:
                    logger.error(f"Route action failed: {route.name}: {e}")
        
        # Emit through event bus
        return await self.event_bus.emit(event)
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        subscriber_name: Optional[str] = None
    ):
        """
        Subscribe to event type
        
        Args:
            event_type: Event type or pattern
            handler: Callback function
            subscriber_name: Optional name for tracking
        """
        self.event_bus.subscribe(event_type, handler)
        
        if subscriber_name:
            logger.info(
                f"Subscription added: {subscriber_name} -> {event_type}",
                extra={"subscriber": subscriber_name, "event_type": event_type}
            )
    
    async def wait_for(
        self,
        event_type: str,
        predicate: Callable[[GraceEvent], bool],
        timeout: float = 5.0
    ) -> Optional[GraceEvent]:
        """
        Wait for event matching predicate
        
        Args:
            event_type: Event type to wait for
            predicate: Function to test if event matches
            timeout: Timeout in seconds
        
        Returns:
            Matching event or None if timeout
        """
        return await self.event_bus.wait_for(event_type, predicate, timeout)
    
    async def request_response(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[GraceEvent]:
        """
        Request/response pattern with mesh routing
        
        Args:
            event_type: Event type
            payload: Event payload
            timeout: Timeout (uses route config if not specified)
            **kwargs: Additional event parameters
        
        Returns:
            Response event or None if timeout
        """
        # Find matching route to get timeout
        if timeout is None:
            for route in self.routes:
                if fnmatch.fnmatch(event_type, route.pattern):
                    timeout = route.timeout_seconds
                    break
        
        timeout = timeout or 5.0
        
        return await self.event_bus.request_response(
            event_type,
            payload,
            timeout=timeout,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        return {
            "loaded": self._loaded,
            "config_path": self.config_path,
            "routes_count": len(self.routes),
            "subscriptions_count": len(self.subscriptions),
            "routes": [
                {
                    "name": r.name,
                    "pattern": r.pattern,
                    "targets": r.targets,
                    "priority": r.priority
                }
                for r in self.routes
            ],
            "event_bus_stats": self.event_bus.get_metrics()
        }


# Global instance
_trigger_mesh: Optional[TriggerMesh] = None


def get_trigger_mesh(event_bus=None) -> TriggerMesh:
    """Get global TriggerMesh instance"""
    global _trigger_mesh
    
    if _trigger_mesh is None:
        if event_bus is None:
            from grace.integration.event_bus import get_event_bus
            event_bus = get_event_bus()
        
        _trigger_mesh = TriggerMesh(event_bus)
        _trigger_mesh.load_config()
        _trigger_mesh.bind_subscriptions()
    
    return _trigger_mesh
