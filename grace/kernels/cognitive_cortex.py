"""
Grace AI Cognitive Cortex - Strategic decision-making and reasoning
"""
import logging
from typing import Dict, Any, Optional

from grace.kernels.base_kernel import BaseKernel

logger = logging.getLogger(__name__)

class CognitiveCortex(BaseKernel):
    """The strategic brain of Grace - responsible for high-level decision-making."""
    
    def __init__(
        self,
        service_registry=None
    ):
        super().__init__("cognitive_cortex", service_registry)
        self.task_manager = self.get_service('task_manager')
        self.llm_service = self.get_service('llm_service')
        self.trigger_mesh = self.get_service('trigger_mesh')
        self.tasks_processed = 0
        logger.info("Cognitive Cortex initialized and services wired.")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a cognitive task, such as planning, reasoning, or delegating.

        Args:
            task: A dictionary defining the task.
                  Example: {'type': 'plan', 'goal': 'Organize project files.'}
        """
        task_type = task.get('type', 'unknown')
        logger.info(f"Executing cognitive task of type: {task_type}")

        if not all([self.task_manager, self.llm_service, self.trigger_mesh]):
            error_msg = "One or more required services are not available."
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        try:
            if task_type == 'plan':
                result = await self._create_plan(task)
            elif task_type == 'reason':
                result = await self._reason_about(task)
            else:
                result = {'success': False, 'error': f"Unknown task type: {task_type}"}

            self.tasks_processed += 1
            return result
        except Exception as e:
            logger.error(f"Error during cognitive execution: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _create_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to create a plan and delegate tasks."""
        goal = task.get('goal', 'No goal specified.')
        logger.info(f"Creating plan for goal: {goal}")

        # 1. Use LLM to break down the goal
        prompt = f"Break down the goal '{goal}' into a series of simple, actionable steps."
        llm_response = await self.llm_service.query(prompt)
        plan_steps = llm_response.get('response', '').split('\n')

        # 2. Create tasks in TaskManager
        task_ids = []
        for step in plan_steps:
            if step.strip():
                new_task = self.task_manager.create_task(description=step)
                task_ids.append(new_task['id'])
                # 3. Dispatch event via TriggerMesh
                await self.trigger_mesh.dispatch_event('task_created', {'task_id': new_task['id'], 'description': step})

        logger.info(f"Created plan with {len(task_ids)} tasks.")
        return {'success': True, 'plan_created': True, 'task_ids': task_ids}

    async def _reason_about(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to reason about a specific query."""
        query = task.get('query', 'No query specified.')
        logger.info(f"Reasoning about query: {query}")

        llm_response = await self.llm_service.query(query)
        conclusion = llm_response.get('response', 'No conclusion reached.')

        return {'success': True, 'reasoning_complete': True, 'conclusion': conclusion}

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

    async def health_check(self) -> Dict[str, Any]:
        """Return the health status of the kernel."""
        return {
            'name': self.name,
            'running': self.is_running,
            'services': {
                'task_manager': 'wired' if self.task_manager else 'missing',
                'llm_service': 'wired' if self.llm_service else 'missing',
                'trigger_mesh': 'wired' if self.trigger_mesh else 'missing',
            },
            'tasks_processed': self.tasks_processed,
        }
