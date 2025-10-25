"""
Grace AI Trigger Mesh - Event-driven orchestration engine
Routes events to appropriate kernels and services based on declarative rules
All kernel actions flow through and are recorded in the Core Truth Layer
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
import uuid

from .workflow_registry import WorkflowRegistry
from .workflow_engine import WorkflowEngine
from .event_router import EventRouter

logger = logging.getLogger(__name__)

class EventPattern(Enum):
    """Event pattern types for routing."""
    EXACT = "exact"
    PREFIX = "prefix"
    WILDCARD = "wildcard"
    REGEX = "regex"

class WorkflowRule:
    """Declarative rule for routing events to handlers."""
    
    def __init__(
        self,
        rule_id: str,
        event_pattern: str,
        pattern_type: EventPattern,
        handlers: List[str],
        condition: Optional[Callable] = None,
        priority: int = 0
    ):
        self.rule_id = rule_id
        self.event_pattern = event_pattern
        self.pattern_type = pattern_type
        self.handlers = handlers
        self.condition = condition
        self.priority = priority
        self.created_at = datetime.now().isoformat()
        self.match_count = 0

class TriggerMesh:
    """
    The TriggerMesh is the central nervous system for event-driven workflows in Grace.
    It listens for events, matches them against workflow triggers, and executes the
    corresponding actions.
    """
    def __init__(self, service_registry=None, workflow_dir: str = "grace/workflows"):
        """
        Initializes the TriggerMesh.
        Args:
            service_registry: The global service registry.
            workflow_dir: Directory containing workflow YAML files.
        """
        self.service_registry = service_registry
        self.workflow_registry = WorkflowRegistry(workflow_dir=workflow_dir)
        self.workflow_engine = WorkflowEngine(service_registry)
        self.event_router = EventRouter(self.workflow_registry, self.workflow_engine)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TriggerMesh initialized with service registry and workflow directory: {workflow_dir}")

    async def dispatch_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Receives an event and routes it for processing.
        This is the main entry point for the TriggerMesh.
        Step 4: Ensure events carry an id and standardized shape.
        """
        import uuid
        import time
        import json
        
        event = {
            "id": payload.get("event_id") or str(uuid.uuid4()),
            "type": event_type,
            "ts": time.time(),
            "payload": payload,
        }
        
        self.logger.info(f"Dispatching event: {event_type}, id: {event['id']}")
        
        # Step 4: Write RECEIVED phase to immutable log
        immutable_logger = self.service_registry.get('immutable_logger') if self.service_registry else None
        if immutable_logger:
            immutable_logger.append_phase(
                event, 
                phase="RECEIVED", 
                status="ok", 
                metadata={"size": len(json.dumps(payload))}
            )
        
        await self.event_router.route_event(event)

    def load_workflows_from_directory(self, directory_path: str):
        """Loads all workflow YAML files from a given directory."""
        self.workflow_registry.load_workflows_from_directory(directory_path)
        self.logger.info(f"Workflows loaded from {directory_path}. Total: {len(self.workflow_registry.workflows)}.")
