"""
Ingress-Event Mesh Bridge - Connects Ingress Kernel to Event Mesh/Trigger System.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

from grace.contracts.ingress_events import IngressEvent, IngressEventType


logger = logging.getLogger(__name__)


class IngressMeshBridge:
    """Bridge between Ingress Kernel and Event Mesh."""
    
    def __init__(self, ingress_kernel, event_bus=None, trigger_mesh=None):
        """
        Initialize the bridge.
        
        Args:
            ingress_kernel: The Ingress Kernel instance
            event_bus: Event bus for system communication
            trigger_mesh: Trigger mesh for event routing
        """
        self.ingress_kernel = ingress_kernel
        self.event_bus = event_bus
        self.trigger_mesh = trigger_mesh
        
        # Event routing configuration
        self.routing_rules = {
            IngressEventType.SOURCE_REGISTERED: ["governance", "audit"],
            IngressEventType.CAPTURED_RAW: ["audit", "monitoring"],
            IngressEventType.NORMALIZED: ["specialists", "feature_store"],
            IngressEventType.VALIDATION_FAILED: ["governance", "alert_manager"],
            IngressEventType.PERSISTED: ["audit", "data_catalog"],
            IngressEventType.PUBLISHED: ["subscribers", "monitoring"],
            IngressEventType.SOURCE_HEALTH: ["monitoring", "ops"],
            IngressEventType.EXPERIENCE: ["mlt_kernel"]
        }
        
        self.running = False
        
    async def start(self):
        """Start the bridge."""
        if self.running:
            return
        
        logger.info("Starting Ingress-Event Mesh Bridge...")
        self.running = True
        
        # Start event processing
        if self.event_bus:
            asyncio.create_task(self._process_events())
        
        logger.info("Ingress-Event Mesh Bridge started")
    
    async def stop(self):
        """Stop the bridge."""
        logger.info("Stopping Ingress-Event Mesh Bridge...")
        self.running = False
    
    async def publish_event(self, event: IngressEvent):
        """
        Publish an ingress event to the event mesh.
        
        Args:
            event: Ingress event to publish
        """
        try:
            # Add bridge metadata
            event_dict = event.dict()
            event_dict["bridge"] = {
                "source": "ingress_mesh_bridge",
                "timestamp": datetime.utcnow().isoformat(),
                "routing_rules": self.routing_rules.get(event.event_type, [])
            }
            
            # Route to appropriate systems
            await self._route_event(event_dict)
            
            # Publish to general event bus
            if self.event_bus:
                await self.event_bus.publish(event_dict)
            
            # Trigger mesh routing
            if self.trigger_mesh:
                await self._trigger_mesh_route(event_dict)
            
            logger.debug(f"Published event: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_type}: {e}")
    
    async def _process_events(self):
        """Process events from the event bus."""
        while self.running:
            try:
                if hasattr(self.event_bus, 'subscribe'):
                    async for event in self.event_bus.subscribe("ingress.*"):
                        await self._handle_external_event(event)
                else:
                    # Fallback for mock event bus
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_external_event(self, event: Dict[str, Any]):
        """Handle external events that affect ingress operations."""
        event_type = event.get("event_type")
        
        try:
            if event_type == "ROLLBACK_REQUESTED":
                await self._handle_rollback_request(event)
            elif event_type == "SOURCE_CONFIGURATION_UPDATED":
                await self._handle_source_config_update(event)
            elif event_type == "GOVERNANCE_POLICY_CHANGED":
                await self._handle_policy_change(event)
            else:
                logger.debug(f"Ignoring external event: {event_type}")
                
        except Exception as e:
            logger.error(f"Failed to handle external event {event_type}: {e}")
    
    async def _route_event(self, event: Dict[str, Any]):
        """Route event to specific systems based on type."""
        event_type = event.get("event_type")
        targets = self.routing_rules.get(event_type, [])
        
        for target in targets:
            try:
                await self._send_to_target(target, event)
            except Exception as e:
                logger.error(f"Failed to route event to {target}: {e}")
    
    async def _send_to_target(self, target: str, event: Dict[str, Any]):
        """Send event to specific target system."""
        # Mock implementation - in real system would use proper routing
        logger.debug(f"Routing event {event['event_type']} to {target}")
        
        if target == "governance" and hasattr(self.ingress_kernel, 'governance_bridge'):
            # Route to governance system
            pass
        elif target == "mlt_kernel" and hasattr(self.ingress_kernel, 'mlt_bridge'):
            # Route to MLT kernel
            pass
        elif target == "specialists":
            # Route to specialist systems
            pass
        elif target == "audit":
            # Route to audit system
            pass
        elif target == "monitoring":
            # Route to monitoring system
            pass
    
    async def _trigger_mesh_route(self, event: Dict[str, Any]):
        """Route event through trigger mesh."""
        if not self.trigger_mesh:
            return
        
        try:
            # Create trigger from event
            trigger = {
                "trigger_id": str(uuid.uuid4()),
                "source": "ingress_kernel",
                "event_type": event["event_type"],
                "timestamp": datetime.utcnow().isoformat(),
                "payload": event["payload"],
                "correlation_id": event.get("correlation_id")
            }
            
            await self.trigger_mesh.process_trigger(trigger)
            
        except Exception as e:
            logger.error(f"Trigger mesh routing failed: {e}")
    
    async def _handle_rollback_request(self, event: Dict[str, Any]):
        """Handle rollback request from governance."""
        payload = event.get("payload", {})
        
        if payload.get("target") == "ingress":
            snapshot_id = payload.get("to_snapshot")
            
            # Mock rollback implementation
            logger.info(f"Initiating rollback to snapshot: {snapshot_id}")
            
            # Emit rollback completed event
            rollback_event = IngressEvent(
                event_type=IngressEventType.ROLLBACK_COMPLETED,
                correlation_id=event.get("correlation_id", str(uuid.uuid4())),
                payload={
                    "target": "ingress",
                    "snapshot_id": snapshot_id,
                    "at": datetime.utcnow().isoformat()
                }
            )
            
            await self.publish_event(rollback_event)
    
    async def _handle_source_config_update(self, event: Dict[str, Any]):
        """Handle source configuration updates."""
        payload = event.get("payload", {})
        source_id = payload.get("source_id")
        
        logger.info(f"Source configuration update for: {source_id}")
        # Implementation would update source configuration
    
    async def _handle_policy_change(self, event: Dict[str, Any]):
        """Handle governance policy changes."""
        payload = event.get("payload", {})
        policy_type = payload.get("policy_type")
        
        logger.info(f"Policy change detected: {policy_type}")
        
        # Update kernel configuration based on policy changes
        if policy_type == "pii_policy":
            new_policy = payload.get("new_value")
            if self.ingress_kernel.config.get("validation"):
                self.ingress_kernel.config["validation"]["pii_policy"] = new_policy
        elif policy_type == "trust_threshold":
            new_threshold = payload.get("new_value")
            if self.ingress_kernel.config.get("validation"):
                self.ingress_kernel.config["validation"]["min_trust"] = new_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "running": self.running,
            "routing_rules": len(self.routing_rules),
            "event_types_handled": list(self.routing_rules.keys())
        }