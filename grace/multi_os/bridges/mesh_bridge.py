"""
Multi-OS Mesh Bridge - Integration with Event Mesh system using GME standard.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import Grace Communications Module
try:
    from grace.comms import create_envelope, MessageKind, Priority, QoSClass
    GME_AVAILABLE = True
except ImportError:
    # Fallback for backward compatibility
    GME_AVAILABLE = False


logger = logging.getLogger(__name__)


class MeshBridge:
    """
    Bridge to connect Multi-OS kernel with the Event Mesh system.
    Handles event publishing, subscription, and routing.
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.subscriptions = {}  # event_name -> callback
        self.published_events = []  # Track published events
        self.routing_rules = self._default_routing_rules()
        
        logger.info("Multi-OS Mesh Bridge initialized")
    
    def _default_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Default event routing rules."""
        return {
            "MOS_HOST_REGISTERED": {
                "targets": ["inventory", "governance", "mlt"],
                "priority": "high",
                "persist": True
            },
            "MOS_TASK_SUBMITTED": {
                "targets": ["scheduler", "telemetry"],
                "priority": "normal",
                "persist": True
            },
            "MOS_TASK_STARTED": {
                "targets": ["telemetry", "audit"],
                "priority": "normal",
                "persist": True
            },
            "MOS_TASK_COMPLETED": {
                "targets": ["telemetry", "mlt", "audit"],
                "priority": "normal",
                "persist": True
            },
            "MOS_HOST_HEALTH": {
                "targets": ["inventory", "telemetry"],
                "priority": "normal",
                "persist": False,  # Health updates are frequent
                "aggregate": True
            },
            "MOS_AGENT_ROLLING_UPDATE": {
                "targets": ["governance", "audit"],
                "priority": "high",
                "persist": True
            },
            "MOS_SNAPSHOT_CREATED": {
                "targets": ["audit", "mlt"],
                "priority": "high",
                "persist": True
            },
            "ROLLBACK_REQUESTED": {
                "targets": ["governance", "audit", "all_kernels"],
                "priority": "critical",
                "persist": True
            },
            "ROLLBACK_COMPLETED": {
                "targets": ["governance", "audit", "all_kernels"],
                "priority": "high",
                "persist": True
            },
            "MOS_EXPERIENCE": {
                "targets": ["mlt"],
                "priority": "low",
                "persist": True
            }
        }
    
    async def publish_event(self, event_name: str, payload: Dict[str, Any], 
                          host_id: Optional[str] = None, 
                          metadata: Optional[Dict] = None) -> str:
        """
        Publish an event to the mesh using GME standard.
        
        Args:
            event_name: Name of the event
            payload: Event payload data
            host_id: Optional host identifier
            metadata: Optional event metadata
            
        Returns:
            Message ID (GME format) or Event ID (legacy format)
        """
        try:
            if GME_AVAILABLE:
                # Use GME standard
                envelope = create_envelope(
                    kind=MessageKind.EVENT,
                    domain="multi_os",
                    name=event_name,
                    payload=payload,
                    priority=self._get_priority_for_event(event_name),
                    qos=self._get_qos_for_event(event_name),
                    partition_key=host_id or f"mos_default",
                    rbac=["mos.publish"]
                )
                
                # Publish to event bus if available
                if self.event_bus:
                    await self.event_bus.publish(event_name, envelope.model_dump(), envelope.headers.correlation_id)
                
                # Store locally for tracking  
                self.published_events.append(envelope.model_dump())
                
                logger.info(f"Published GME event {event_name} with msg_id {envelope.msg_id}")
                return envelope.msg_id
                
            else:
                # Legacy format fallback
                import uuid
                event_id = str(uuid.uuid4())
                
                event = {
                    "event_id": event_id,
                    "event_name": event_name,
                    "payload": payload,
                    "host_id": host_id,
                    "metadata": metadata or {},
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "multi_os_kernel"
                }
                
                # Apply routing rules
                routing = self.routing_rules.get(event_name, {})
                event["routing"] = routing
                
                # Publish to event bus if available
                if self.event_bus:
                    await self.event_bus.publish(event_name, event)
                
                # Store locally for tracking
                self.published_events.append(event)
            
            # Maintain size limit
            if len(self.published_events) > 1000:
                self.published_events = self.published_events[-500:]  # Keep last 500
            
            logger.info(f"Published event {event_name} with ID {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to publish event {event_name}: {e}")
            return ""
    
    def subscribe(self, event_name: str, callback) -> bool:
        """
        Subscribe to events from the mesh.
        
        Args:
            event_name: Event name to subscribe to
            callback: Callback function to handle events
            
        Returns:
            True if subscription was successful
        """
        try:
            self.subscriptions[event_name] = callback
            
            # Register with event bus if available
            if self.event_bus:
                self.event_bus.subscribe(event_name, callback)
            
            logger.info(f"Subscribed to event {event_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {event_name}: {e}")
            return False
    
    def unsubscribe(self, event_name: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            event_name: Event name to unsubscribe from
            
        Returns:
            True if unsubscription was successful
        """
        try:
            if event_name in self.subscriptions:
                del self.subscriptions[event_name]
                
                # Unregister from event bus if available
                if self.event_bus:
                    self.event_bus.unsubscribe(event_name)
                
                logger.info(f"Unsubscribed from event {event_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {event_name}: {e}")
            return False
    
    async def handle_incoming_event(self, event: Dict[str, Any]) -> None:
        """
        Handle incoming events from the mesh.
        
        Args:
            event: Event data
        """
        try:
            event_name = event.get("event_name", "")
            
            if event_name in self.subscriptions:
                callback = self.subscriptions[event_name]
                
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                
                logger.debug(f"Handled incoming event {event_name}")
            else:
                logger.debug(f"No subscription for event {event_name}")
                
        except Exception as e:
            logger.error(f"Failed to handle incoming event: {e}")
    
    async def publish_host_registered(self, host_descriptor: Dict[str, Any]) -> str:
        """Publish host registration event."""
        return await self.publish_event(
            "MOS_HOST_REGISTERED",
            {"host": host_descriptor},
            host_id=host_descriptor.get("host_id")
        )
    
    async def publish_task_submitted(self, task: Dict[str, Any], host_id: str) -> str:
        """Publish task submission event."""
        return await self.publish_event(
            "MOS_TASK_SUBMITTED", 
            {"task": task, "host_id": host_id},
            host_id=host_id
        )
    
    async def publish_task_started(self, task_id: str, pid: int, host_id: str) -> str:
        """Publish task started event."""
        return await self.publish_event(
            "MOS_TASK_STARTED",
            {"task_id": task_id, "pid": pid, "host_id": host_id},
            host_id=host_id
        )
    
    async def publish_task_completed(self, task_id: str, status: str, exit_code: int,
                                   outputs: List[str], logs_uri: str, duration_ms: int,
                                   host_id: str) -> str:
        """Publish task completion event."""
        return await self.publish_event(
            "MOS_TASK_COMPLETED",
            {
                "task_id": task_id,
                "status": status,
                "exit_code": exit_code,
                "outputs": outputs,
                "logs_uri": logs_uri,
                "duration_ms": duration_ms
            },
            host_id=host_id
        )
    
    async def publish_host_health(self, host_id: str, status: str, metrics: Dict[str, Any]) -> str:
        """Publish host health update."""
        return await self.publish_event(
            "MOS_HOST_HEALTH",
            {"host_id": host_id, "status": status, **metrics},
            host_id=host_id
        )
    
    async def publish_agent_rollout(self, from_version: str, to_version: str, 
                                  mode: str, progress: int) -> str:
        """Publish agent rollout progress."""
        return await self.publish_event(
            "MOS_AGENT_ROLLING_UPDATE",
            {
                "from": from_version,
                "to": to_version, 
                "mode": mode,
                "progress": progress
            }
        )
    
    async def publish_snapshot_created(self, snapshot_id: str, scope: str, 
                                     uri: str, host_id: Optional[str] = None) -> str:
        """Publish snapshot creation event."""
        return await self.publish_event(
            "MOS_SNAPSHOT_CREATED",
            {
                "snapshot_id": snapshot_id,
                "scope": scope,
                "uri": uri,
                "host_id": host_id
            },
            host_id=host_id
        )
    
    async def publish_rollback_requested(self, target: str, to_snapshot: str) -> str:
        """Publish rollback request event."""
        return await self.publish_event(
            "ROLLBACK_REQUESTED",
            {"target": target, "to_snapshot": to_snapshot}
        )
    
    async def publish_rollback_completed(self, target: str, snapshot_id: str) -> str:
        """Publish rollback completion event."""
        return await self.publish_event(
            "ROLLBACK_COMPLETED",
            {
                "target": target,
                "snapshot_id": snapshot_id,
                "at": datetime.utcnow().isoformat()
            }
        )
    
    async def publish_experience(self, experience: Dict[str, Any]) -> str:
        """Publish multi-OS experience for meta-learning."""
        return await self.publish_event(
            "MOS_EXPERIENCE",
            {
                "schema_version": "1.0.0",
                "experience": experience
            },
            host_id=experience.get("host_id")
        )
    
    def get_published_events(self, event_name: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get published events for monitoring.
        
        Args:
            event_name: Optional filter by event name
            limit: Maximum number of events to return
            
        Returns:
            List of published events
        """
        events = self.published_events
        
        if event_name:
            events = [e for e in events if e.get("event_name") == event_name]
        
        # Return most recent events first
        return list(reversed(events[-limit:]))
    
    def _get_priority_for_event(self, event_name: str) -> Priority:
        """Map event names to GME priorities based on Multi-OS operations."""
        # Critical system events
        if event_name in ["MOS_HOST_CRITICAL", "MOS_SECURITY_ALERT", "ROLLBACK_REQUESTED"]:
            return Priority.P0
        
        # Important operational events
        elif event_name in ["MOS_TASK_FAILED", "MOS_HOST_HEALTH", "ROLLBACK_COMPLETED"]:
            return Priority.P1
            
        # Standard events
        elif event_name in ["MOS_TASK_STARTED", "MOS_TASK_COMPLETED"]:
            return Priority.P2
            
        # Background/bulk events
        else:
            return Priority.P3
    
    def _get_qos_for_event(self, event_name: str) -> QoSClass:
        """Map event names to GME QoS classes based on Multi-OS requirements."""
        # Real-time events need immediate processing
        if event_name in ["MOS_HOST_CRITICAL", "MOS_SECURITY_ALERT"]:
            return QoSClass.REALTIME
        
        # Most operational events use standard processing
        elif event_name in ["MOS_TASK_STARTED", "MOS_TASK_COMPLETED", "MOS_HOST_HEALTH"]:
            return QoSClass.STANDARD
            
        # Background and batch events
        else:
            return QoSClass.BULK
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        event_counts = {}
        target_counts = {}
        
        for event in self.published_events:
            event_name = event.get("event_name", "unknown")
            event_counts[event_name] = event_counts.get(event_name, 0) + 1
            
            routing = event.get("routing", {})
            targets = routing.get("targets", [])
            
            for target in targets:
                target_counts[target] = target_counts.get(target, 0) + 1
        
        return {
            "total_events_published": len(self.published_events),
            "events_by_type": event_counts,
            "events_by_target": target_counts,
            "active_subscriptions": len(self.subscriptions),
            "routing_rules": len(self.routing_rules)
        }
    
    def update_routing_rule(self, event_name: str, rule: Dict[str, Any]) -> None:
        """Update routing rule for an event type."""
        self.routing_rules[event_name] = rule
        logger.info(f"Updated routing rule for {event_name}")
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status information."""
        return {
            "connected": self.event_bus is not None,
            "subscriptions": list(self.subscriptions.keys()),
            "events_published": len(self.published_events),
            "routing_rules_count": len(self.routing_rules),
            "last_event": self.published_events[-1] if self.published_events else None
        }