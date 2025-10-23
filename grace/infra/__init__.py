"""
Grace AI L0 - Runtime/Infrastructure Layer
Containers, queues, storage, runners
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ContainerType(Enum):
    """Types of containers."""
    KERNEL = "kernel"
    SERVICE = "service"
    WORKER = "worker"

class Container:
    """Represents a containerized component."""
    
    def __init__(self, container_id: str, component_name: str, container_type: ContainerType):
        self.container_id = container_id
        self.component_name = component_name
        self.container_type = container_type
        self.status = "initializing"
        self.created_at = datetime.now().isoformat()

class Queue:
    """In-memory queue for async message passing."""
    
    def __init__(self, queue_id: str, max_size: int = 10000):
        self.queue_id = queue_id
        self.messages: List[Dict[str, Any]] = []
        self.max_size = max_size
    
    async def enqueue(self, message: Dict[str, Any]) -> bool:
        """Add message to queue."""
        if len(self.messages) >= self.max_size:
            logger.warning(f"Queue {self.queue_id} is full")
            return False
        
        self.messages.append(message)
        return True
    
    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Remove and return first message."""
        if self.messages:
            return self.messages.pop(0)
        return None
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.messages)

class Storage:
    """In-memory persistent storage."""
    
    def __init__(self, storage_id: str):
        self.storage_id = storage_id
        self.data: Dict[str, Any] = {}
    
    async def write(self, key: str, value: Any) -> bool:
        """Write to storage."""
        self.data[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Storage {self.storage_id}: wrote {key}")
        return True
    
    async def read(self, key: str) -> Optional[Any]:
        """Read from storage."""
        if key in self.data:
            return self.data[key]["value"]
        return None
    
    async def delete(self, key: str) -> bool:
        """Delete from storage."""
        if key in self.data:
            del self.data[key]
            logger.info(f"Storage {self.storage_id}: deleted {key}")
            return True
        return False

class Runner:
    """Executes tasks asynchronously."""
    
    def __init__(self, runner_id: str, max_workers: int = 10):
        self.runner_id = runner_id
        self.max_workers = max_workers
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def submit_task(self, task_id: str, task_func, *args, **kwargs) -> bool:
        """Submit a task for execution."""
        if len(self.active_tasks) >= self.max_workers:
            logger.warning(f"Runner {self.runner_id} is at capacity")
            return False
        
        self.active_tasks[task_id] = {
            "status": "running",
            "submitted_at": datetime.now().isoformat()
        }
        
        logger.info(f"Runner {self.runner_id}: submitted task {task_id}")
        return True

class InfrastructureLayer:
    """L0 - Runtime/Infrastructure layer."""
    
    def __init__(self):
        self.containers: Dict[str, Container] = {}
        self.queues: Dict[str, Queue] = {}
        self.storage: Dict[str, Storage] = {}
        self.runners: Dict[str, Runner] = {}
    
    def create_container(self, container_id: str, component_name: str, container_type: ContainerType) -> Container:
        """Create a container."""
        container = Container(container_id, component_name, container_type)
        self.containers[container_id] = container
        logger.info(f"Created container: {container_id}")
        return container
    
    def create_queue(self, queue_id: str) -> Queue:
        """Create a queue."""
        queue = Queue(queue_id)
        self.queues[queue_id] = queue
        logger.info(f"Created queue: {queue_id}")
        return queue
    
    def create_storage(self, storage_id: str) -> Storage:
        """Create storage."""
        storage = Storage(storage_id)
        self.storage[storage_id] = storage
        logger.info(f"Created storage: {storage_id}")
        return storage
    
    def create_runner(self, runner_id: str) -> Runner:
        """Create a runner."""
        runner = Runner(runner_id)
        self.runners[runner_id] = runner
        logger.info(f"Created runner: {runner_id}")
        return runner
    
    def get_infra_status(self) -> Dict[str, Any]:
        """Get infrastructure status."""
        return {
            "containers": len(self.containers),
            "queues": len(self.queues),
            "storage_instances": len(self.storage),
            "runners": len(self.runners),
            "timestamp": datetime.now().isoformat()
        }
