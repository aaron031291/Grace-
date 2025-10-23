"""
Grace AI Swarm Kernel - Distributed coordination and multi-agent collaboration
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

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

class SwarmKernel:
    """Coordinates multiple agents in a swarm for distributed task execution."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.coordination_log: List[Dict[str, Any]] = []
    
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
