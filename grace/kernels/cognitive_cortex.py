"""
Grace AI Cognitive Cortex - Strategic decision-making and reasoning
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CognitiveCortex:
    """The strategic brain of Grace - responsible for high-level decision-making."""
    
    def __init__(
        self,
        event_bus,
        task_manager,
        communication_channel,
        sandbox_manager,
        llm_service=None
    ):
        self.event_bus = event_bus
        self.task_manager = task_manager
        self.communication_channel = communication_channel
        self.sandbox_manager = sandbox_manager
        self.llm_service = llm_service
    
    async def synthesize_and_decide(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the complete system state and make strategic decisions.
        This is the core of Grace's reasoning.
        """
        kpis = system_state.get("kpis", {})
        trust = system_state.get("trust", 0)
        num_tasks = system_state.get("tasks", 0)
        
        logger.info(f"Synthesizing system state. Trust: {trust}, KPIs: {kpis}, Active Tasks: {num_tasks}")
        
        # Decision Logic
        if trust < 30:
            # Low trust - critical issues need immediate attention
            decision = {
                "action": "escalate_healing",
                "urgency": "critical",
                "reason": "Low trust score detected"
            }
            logger.info(f"Decision: {decision['action']}")
            return decision
        
        elif num_tasks > 0:
            # There are user-defined tasks - prioritize them
            tasks = self.task_manager.get_open_tasks()
            decision = {
                "action": "execute_user_task",
                "urgency": "high",
                "reason": f"Prioritizing user-defined task",
                "task_id": tasks[0].task_id if tasks else None
            }
            logger.info(f"Decision: {decision['action']}")
            return decision
        
        elif trust > 70 and kpis.get("stability", 0) > 95:
            # System is stable and trust is good - propose self-improvement
            decision = {
                "action": "propose_self_improvement",
                "urgency": "low",
                "reason": "System stable - seeking proactive improvement opportunities"
            }
            logger.info(f"Decision: {decision['action']}")
            return decision
        
        else:
            # System is stable, no immediate action needed
            decision = {
                "action": "observe",
                "urgency": "low",
                "reason": "System stable - continuing to observe"
            }
            logger.info(f"Decision: {decision['action']}")
            return decision
    
    async def propose_self_improvement_task(self) -> Optional[str]:
        """Grace proactively identifies and proposes a self-improvement task."""
        message = """I've reviewed my current operational status and everything appears stable. 
        I'd like to proactively start a new self-improvement cycle to analyze my own codebase for potential optimizations.
        I will create a new task to track this initiative."""
        
        await self.communication_channel.send_to_user(message, context="proactive_improvement", urgency="info")
        
        task_id = self.task_manager.create_task(
            title="Proactive Self-Improvement Cycle",
            description="Grace identified this as a proactive improvement opportunity",
            created_by="grace"
        )
        
        return task_id
    
    async def execute_task(self, task_id: str):
        """Execute a task by delegating to the SandboxManager."""
        task = self.task_manager.get_task(task_id)
        if task:
            logger.info(f"Executing task: {task.title}")
            await self.sandbox_manager.initiate_improvement_cycle(
                reason=task.description,
                details={"task_id": task_id},
                correlation_id=task_id
            )
