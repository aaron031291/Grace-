"""
Grace AI - Layer 0: Runtime & Infrastructure
Containers, queues, storage, runners - the physical execution environment
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class RuntimeEnvironment(Enum):
    """Runtime execution environments."""
    LOCAL = "local"
    CONTAINER = "container"
    DISTRIBUTED = "distributed"

class ExecutionQueue:
    """Queue for task execution."""
    
    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self.running: Dict[str, Dict[str, Any]] = {}
        self.completed: List[Dict[str, Any]] = []
    
    async def enqueue(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """Enqueue a task for execution."""
        self.queue.append({
            "task_id": task_id,
            "data": task_data,
            "enqueued_at": datetime.now().isoformat(),
            "status": "queued"
        })
        logger.info(f"Task enqueued: {task_id}")
        return True
    
    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Dequeue a task for execution."""
        if self.queue:
            task = self.queue.pop(0)
            task["status"] = "running"
            task["started_at"] = datetime.now().isoformat()
            self.running[task["task_id"]] = task
            return task
        return None
    
    async def mark_complete(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as completed."""
        if task_id in self.running:
            task = self.running.pop(task_id)
            task["result"] = result
            task["completed_at"] = datetime.now().isoformat()
            task["status"] = "completed"
            self.completed.append(task)
            logger.info(f"Task completed: {task_id}")

class StorageBackend:
    """Abstraction for storage (local, remote, cloud)."""
    
    def __init__(self):
        self.storage: Dict[str, Any] = {}
    
    async def store(self, key: str, value: Any) -> bool:
        """Store a value."""
        self.storage[key] = {
            "value": value,
            "stored_at": datetime.now().isoformat()
        }
        logger.info(f"Stored: {key}")
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value."""
        if key in self.storage:
            return self.storage[key]["value"]
        return None
    
    async def delete(self, key: str) -> bool:
        """Delete a value."""
        if key in self.storage:
            del self.storage[key]
            logger.info(f"Deleted: {key}")
            return True
        return False

class RuntimeInfrastructure:
    """
    Layer 0: Runtime & Infrastructure
    Manages containers, queues, storage, and execution runners
    """
    
    def __init__(self):
        self.environment = RuntimeEnvironment.LOCAL
        self.execution_queue = ExecutionQueue()
        self.storage = StorageBackend()
        self.runners: Dict[str, Any] = {}
    
    async def initialize_environment(self, env_type: RuntimeEnvironment):
        """Initialize the runtime environment."""
        self.environment = env_type
        logger.info(f"Runtime infrastructure initialized: {env_type.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get infrastructure status."""
        return {
            "environment": self.environment.value,
            "queued_tasks": len(self.execution_queue.queue),
            "running_tasks": len(self.execution_queue.running),
            "completed_tasks": len(self.execution_queue.completed),
            "storage_items": len(self.storage.storage)
        }
