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
    Event-driven orchestration engine that routes events to appropriate kernels.
    - Declaratively defines which events trigger which kernel actions
    - Ensures all actions are recorded in the Core Truth Layer
    - Manages workflow execution and state
    """
    
    def __init__(self, event_bus, truth_layer):
        self.event_bus = event_bus
        self.truth_layer = truth_layer
        self.rules: Dict[str, WorkflowRule] = {}
        self.handlers: Dict[str, Callable] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    def register_rule(self, rule: WorkflowRule) -> bool:
        """Register a workflow rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Registered trigger rule: {rule.rule_id} (pattern: {rule.event_pattern})")
        return True
    
    def register_handler(self, handler_id: str, handler_func: Callable) -> bool:
        """Register an event handler (kernel action)."""
        self.handlers[handler_id] = handler_func
        logger.info(f"Registered handler: {handler_id}")
        return True
    
    async def dispatch_event(self, event_type: str, event_data: Dict[str, Any]) -> List[str]:
        """
        Dispatch an event through the trigger mesh.
        Returns list of workflow IDs that were activated.
        """
        workflow_ids = []
        matching_rules = self._find_matching_rules(event_type)
        
        for rule in sorted(matching_rules, key=lambda r: r.priority, reverse=True):
            # Check condition if present
            if rule.condition and not rule.condition(event_data):
                continue
            
            # Create workflow execution
            workflow_id = await self._execute_workflow(rule, event_type, event_data)
            workflow_ids.append(workflow_id)
            rule.match_count += 1
        
        return workflow_ids
    
    async def _execute_workflow(self, rule: WorkflowRule, event_type: str, event_data: Dict[str, Any]) -> str:
        """Execute a workflow based on a matching rule."""
        workflow_id = str(uuid.uuid4())[:8]
        
        self.active_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "rule_id": rule.rule_id,
            "event_type": event_type,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "actions": []
        }
        
        # Record event in truth layer
        await self.truth_layer.immutable_log.record_event(
            event_type=f"trigger_mesh.workflow_started",
            component="trigger_mesh",
            data={
                "workflow_id": workflow_id,
                "rule_id": rule.rule_id,
                "event": event_type
            },
            correlation_id=workflow_id
        )
        
        # Execute all handlers in the rule
        for handler_id in rule.handlers:
            handler = self.handlers.get(handler_id)
            if not handler:
                logger.warning(f"Handler not found: {handler_id}")
                continue
            
            try:
                logger.info(f"Executing handler {handler_id} in workflow {workflow_id}")
                
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(event_data)
                else:
                    result = handler(event_data)
                
                # Record handler execution
                await self.truth_layer.immutable_log.record_event(
                    event_type=f"trigger_mesh.handler_executed",
                    component="trigger_mesh",
                    data={
                        "handler_id": handler_id,
                        "status": "success"
                    },
                    correlation_id=workflow_id
                )
                
                self.active_workflows[workflow_id]["actions"].append({
                    "handler": handler_id,
                    "status": "completed",
                    "result": str(result)
                })
                
            except Exception as e:
                logger.error(f"Handler {handler_id} failed: {str(e)}")
                
                await self.truth_layer.immutable_log.record_event(
                    event_type=f"trigger_mesh.handler_failed",
                    component="trigger_mesh",
                    data={
                        "handler_id": handler_id,
                        "error": str(e)
                    },
                    correlation_id=workflow_id
                )
                
                self.active_workflows[workflow_id]["actions"].append({
                    "handler": handler_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Mark workflow as complete
        self.active_workflows[workflow_id]["status"] = "completed"
        self.active_workflows[workflow_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Workflow completed: {workflow_id}")
        return workflow_id
    
    def _find_matching_rules(self, event_type: str) -> List[WorkflowRule]:
        """Find all rules that match an event type."""
        matching = []
        
        for rule in self.rules.values():
            if rule.pattern_type == EventPattern.EXACT:
                if rule.event_pattern == event_type:
                    matching.append(rule)
            elif rule.pattern_type == EventPattern.PREFIX:
                if event_type.startswith(rule.event_pattern):
                    matching.append(rule)
            elif rule.pattern_type == EventPattern.WILDCARD:
                import fnmatch
                if fnmatch.fnmatch(event_type, rule.event_pattern):
                    matching.append(rule)
            elif rule.pattern_type == EventPattern.REGEX:
                import re
                if re.match(rule.event_pattern, event_type):
                    matching.append(rule)
        
        return matching
    
    def get_execution_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        return list(self.active_workflows.values())[-limit:]
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """Get statistics on rule matching."""
        return {
            "total_rules": len(self.rules),
            "total_handlers": len(self.handlers),
            "rule_matches": {rule_id: rule.match_count for rule_id, rule in self.rules.items()},
            "active_workflows": len([w for w in self.active_workflows.values() if w["status"] == "running"])
        }
