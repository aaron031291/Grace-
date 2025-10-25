"""
Grace AI Swarm Kernel - Distributed coordination and multi-agent collaboration
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from grace.kernels.base_kernel import BaseKernel

logger = logging.getLogger(__name__)

class Agent:
    """Represents an agent in the swarm."""
    
    def __init__(self, agent_id: str, name: str, role: str):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.status = "idle"
        self.created_at = datetime.now().isoformat()
        self.current_task: Optional[str] = None

class SwarmKernel(BaseKernel):
    """
    The distributed coordination and communication kernel for Grace.
    It integrates with CommunicationChannel and TriggerMesh to manage
    distributed tasks and agent communication.
    """

    def __init__(self, service_registry=None):
        super().__init__("swarm_kernel", service_registry)
        self.comm_channel = self.get_service('communication_channel')
        self.trigger_mesh = self.get_service('trigger_mesh')
        self.active_agents = {}
        self.messages_relayed = 0
        self.logger.info("Swarm Kernel initialized and services wired.")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a swarm-related task, like agent coordination or broadcasting.

        Args:
            task: A dictionary defining the task.
                  Example: {'type': 'broadcast', 'event': 'new_target', 'data': {...}}
                           {'type': 'register_agent', 'agent_id': 'agent-123'}
        """
        task_type = task.get('type', 'unknown')
        self.logger.info(f"Executing swarm task of type: {task_type}")

        if not all([self.comm_channel, self.trigger_mesh]):
            error_msg = "One or more required services are not available."
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        try:
            if task_type == 'broadcast':
                result = await self._broadcast_to_swarm(task)
            elif task_type == 'register_agent':
                result = self._register_agent(task)
            elif task_type == 'delegate_to_agent':
                result = await self._delegate_to_agent(task)
            else:
                result = {'success': False, 'error': f"Unknown task type: {task_type}"}

            return result
        except Exception as e:
            self.logger.error(f"Error during swarm execution: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _broadcast_to_swarm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message to all agents in the swarm."""
        event = task.get('event', 'generic_message')
        data = task.get('data', {})
        self.logger.info(f"Broadcasting event '{event}' to swarm.")

        await self.comm_channel.broadcast(event=event, data=data)
        self.messages_relayed += len(self.active_agents)
        return {'success': True, 'broadcast_sent': True, 'event': event}

    def _register_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent with the swarm."""
        agent_id = task.get('agent_id')
        if not agent_id:
            return {'success': False, 'error': 'agent_id is required.'}

        self.active_agents[agent_id] = {'status': 'active', 'registered_at': asyncio.get_event_loop().time()}
        self.logger.info(f"Registered new agent: {agent_id}")
        return {'success': True, 'agent_registered': True, 'agent_id': agent_id}

    async def _delegate_to_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a specific task to a registered agent."""
        agent_id = task.get('agent_id')
        agent_task = task.get('agent_task', {})
        if not agent_id or not agent_task:
            return {'success': False, 'error': 'agent_id and agent_task are required.'}
        if agent_id not in self.active_agents:
            return {'success': False, 'error': f'Agent {agent_id} not registered.'}

        self.logger.info(f"Delegating task to agent {agent_id}.")
        # In a real system, this would send a direct message.
        # Here, we'll use the trigger mesh to simulate dispatching a targeted task.
        await self.trigger_mesh.dispatch_event(
            event_type='agent_task',
            payload={'target_agent': agent_id, 'task': agent_task}
        )
        self.messages_relayed += 1
        return {'success': True, 'task_delegated': True, 'agent_id': agent_id}

    async def health_check(self) -> Dict[str, Any]:
        """Return the health status of the kernel."""
        return {
            'name': self.name,
            'running': self.is_running,
            'services': {
                'comm_channel': 'wired' if self.comm_channel else 'missing',
                'trigger_mesh': 'wired' if self.trigger_mesh else 'missing',
            },
            'active_agents': len(self.active_agents),
            'messages_relayed': self.messages_relayed,
        }
    
    def register_agent(self, agent_id: str, name: str, role: str) -> Agent:
        """Register a new agent in the swarm."""
        agent = Agent(agent_id, name, role)
        self.agents[agent_id] = agent
        logger.info(f"Agent registered: {name} ({role})")
        return agent
    
    async def dispatch_task(self, task_id: str, task_description: str, required_roles: List[str]) -> bool:
        """Dispatch a task to appropriate agents."""
        # Find agents with required roles
        available_agents = [
            a for a in self.agents.values() 
            if a.role in required_roles and a.status == "idle"
        ]
        
        if not available_agents:
            logger.warning(f"No available agents for task {task_id}")
            return False
        
        # Assign task to first available agent
        agent = available_agents[0]
        agent.status = "busy"
        agent.current_task = task_id
        
        self.tasks[task_id] = {
            "description": task_description,
            "assigned_to": agent.agent_id,
            "status": "running",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Task {task_id} dispatched to agent {agent.agent_id}")
        
        if self.event_bus:
            await self.event_bus.publish("swarm.task_dispatched", {
                "task_id": task_id,
                "agent_id": agent.agent_id
            })
        
        return True
    
    async def report_task_completion(self, task_id: str, agent_id: str, result: Dict[str, Any]):
        """Report completion of a task."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result
        
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = "idle"
            agent.current_task = None
        
        logger.info(f"Task {task_id} completed by agent {agent_id}")
        
        if self.event_bus:
            await self.event_bus.publish("swarm.task_completed", {
                "task_id": task_id,
                "agent_id": agent_id
            })
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an agent."""
        agent = self.agents.get(agent_id)
        if agent:
            return {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "role": agent.role,
                "status": agent.status,
                "current_task": agent.current_task
            }
        return None
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get overall swarm status."""
        total_agents = len(self.agents)
        idle_agents = len([a for a in self.agents.values() if a.status == "idle"])
        busy_agents = len([a for a in self.agents.values() if a.status == "busy"])
        
        return {
            "total_agents": total_agents,
            "idle_agents": idle_agents,
            "busy_agents": busy_agents,
            "total_tasks": len(self.tasks),
            "running_tasks": len([t for t in self.tasks.values() if t["status"] == "running"])
        }
