"""
Grace AI Immune System - AVN (Autonomous Healing Network) Module
Autonomous self-healing and recovery mechanisms
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class HealingAction:
    """Represents a healing action to be executed."""
    
    def __init__(self, action_id: str, action_type: str, target: str, parameters: Dict[str, Any]):
        self.action_id = action_id
        self.action_type = action_type
        self.target = target
        self.parameters = parameters
        self.status = "pending"
        self.created_at = datetime.now().isoformat()
        self.executed_at: Optional[str] = None

class AVNHealer:
    """Autonomous Healing Network - executes self-healing actions."""
    
    def __init__(self, event_bus=None, resilience_kernel=None):
        self.event_bus = event_bus
        self.resilience_kernel = resilience_kernel
        self.healing_actions: Dict[str, HealingAction] = {}
        self.healing_history: List[Dict[str, Any]] = []
        self.healing_handlers: Dict[str, Callable] = {}
    
    def register_healing_handler(self, action_type: str, handler: Callable):
        """Register a handler for a specific healing action type."""
        self.healing_handlers[action_type] = handler
        logger.info(f"Registered healing handler: {action_type}")
    
    async def create_healing_action(
        self,
        action_type: str,
        target: str,
        parameters: Dict[str, Any] = None
    ) -> str:
        """Create a new healing action."""
        import uuid
        action_id = str(uuid.uuid4())[:8]
        
        action = HealingAction(action_id, action_type, target, parameters or {})
        self.healing_actions[action_id] = action
        
        logger.info(f"Healing action created: {action_type} on {target}")
        return action_id
    
    async def execute_healing_action(self, action_id: str) -> bool:
        """Execute a healing action."""
        action = self.healing_actions.get(action_id)
        if not action:
            logger.warning(f"Healing action not found: {action_id}")
            return False
        
        handler = self.healing_handlers.get(action.action_type)
        if not handler:
            logger.warning(f"No handler for healing action type: {action.action_type}")
            return False
        
        try:
            logger.info(f"Executing healing action: {action_id}")
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(action.target, action.parameters)
            else:
                result = handler(action.target, action.parameters)
            
            action.status = "completed"
            action.executed_at = datetime.now().isoformat()
            
            self.healing_history.append({
                "action_id": action_id,
                "action_type": action.action_type,
                "target": action.target,
                "status": "success",
                "executed_at": action.executed_at
            })
            
            logger.info(f"Healing action completed: {action_id}")
            
            if self.event_bus:
                await self.event_bus.publish("avn.healing_completed", {
                    "action_id": action_id,
                    "target": action.target
                })
            
            return True
        
        except Exception as e:
            action.status = "failed"
            action.executed_at = datetime.now().isoformat()
            
            logger.error(f"Healing action failed: {action_id} - {str(e)}")
            
            self.healing_history.append({
                "action_id": action_id,
                "action_type": action.action_type,
                "target": action.target,
                "status": "failed",
                "error": str(e),
                "executed_at": action.executed_at
            })
            
            if self.event_bus:
                await self.event_bus.publish("avn.healing_failed", {
                    "action_id": action_id,
                    "error": str(e)
                })
            
            return False
    
    async def execute_cascade_healing(self, target: str, action_types: List[str]) -> List[str]:
        """Execute multiple healing actions in sequence (cascade)."""
        action_ids = []
        
        for action_type in action_types:
            action_id = await self.create_healing_action(action_type, target)
            success = await self.execute_healing_action(action_id)
            action_ids.append(action_id)
            
            if not success:
                logger.warning(f"Cascade healing stopped at {action_type}")
                break
        
        return action_ids
    
    def get_healing_status(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a healing action."""
        action = self.healing_actions.get(action_id)
        if action:
            return {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "target": action.target,
                "status": action.status,
                "created_at": action.created_at,
                "executed_at": action.executed_at
            }
        return None
    
    def get_healing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get healing history."""
        return self.healing_history[-limit:]
