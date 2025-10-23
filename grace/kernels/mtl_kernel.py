"""
Grace AI MTL Kernel - Multi-Task Learning orchestration
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TaskLearner:
    """Represents a learner for a specific task in MTL."""
    
    def __init__(self, task_id: str, task_description: str):
        self.task_id = task_id
        self.task_description = task_description
        self.model = None
        self.performance = 0.0
        self.shared_representations = {}

class MTLKernel:
    """Multi-Task Learning kernel - allows Grace to learn multiple related tasks simultaneously."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.tasks: Dict[str, TaskLearner] = {}
        self.shared_layer_weights = {}
        self.task_relationships = {}
    
    async def register_task(
        self,
        task_id: str,
        task_description: str,
        related_tasks: List[str] = None
    ) -> Dict[str, Any]:
        """Register a new task for multi-task learning."""
        logger.info(f"MTLKernel: Registering task '{task_id}'")
        
        learner = TaskLearner(task_id, task_description)
        self.tasks[task_id] = learner
        
        if related_tasks:
            self.task_relationships[task_id] = related_tasks
        
        return {
            "task_id": task_id,
            "status": "registered",
            "related_tasks": related_tasks or []
        }
    
    async def learn_task(
        self,
        task_id: str,
        training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn a specific task while sharing knowledge with related tasks."""
        logger.info(f"MTLKernel: Learning task '{task_id}' with {len(training_data)} samples")
        
        if task_id not in self.tasks:
            return {"error": f"Task '{task_id}' not registered"}
        
        learner = self.tasks[task_id]
        
        # Simulate multi-task learning
        result = {
            "task_id": task_id,
            "training_samples": len(training_data),
            "status": "learning_complete",
            "performance": 0.88,
            "shared_knowledge_utilized": True,
            "related_tasks_performance": {}
        }
        
        learner.performance = result["performance"]
        
        # Publish event
        if self.event_bus:
            await self.event_bus.publish("mtl.task_learned", {
                "task_id": task_id,
                "performance": result["performance"]
            })
        
        return result
    
    async def transfer_knowledge(
        self,
        source_task_id: str,
        target_task_id: str
    ) -> Dict[str, Any]:
        """Transfer knowledge from one task to another."""
        logger.info(f"MTLKernel: Transferring knowledge from '{source_task_id}' to '{target_task_id}'")
        
        if source_task_id not in self.tasks or target_task_id not in self.tasks:
            return {"error": "One or both tasks not found"}
        
        source_learner = self.tasks[source_task_id]
        target_learner = self.tasks[target_task_id]
        
        # Simulate knowledge transfer
        performance_improvement = source_learner.performance * 0.1  # 10% improvement
        target_learner.performance += performance_improvement
        
        return {
            "source_task": source_task_id,
            "target_task": target_task_id,
            "performance_improvement": performance_improvement,
            "new_target_performance": target_learner.performance,
            "status": "knowledge_transferred"
        }
    
    async def get_shared_representation(self, task_id: str) -> Dict[str, Any]:
        """Get the shared representation learned across all tasks."""
        logger.info(f"MTLKernel: Retrieving shared representation for task '{task_id}'")
        
        return {
            "task_id": task_id,
            "shared_layers": len(self.shared_layer_weights),
            "representation_dim": 256,
            "status": "success"
        }
    
    def get_mtl_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-task learning."""
        avg_performance = sum(t.performance for t in self.tasks.values()) / max(1, len(self.tasks))
        
        return {
            "total_tasks": len(self.tasks),
            "average_performance": avg_performance,
            "task_relationships_count": len(self.task_relationships),
            "shared_layers": len(self.shared_layer_weights)
        }
