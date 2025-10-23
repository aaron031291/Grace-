"""
Grace AI Task Manager - Project and task management for collaborative missions
"""
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Task:
    """Represents a single task."""
    
    def __init__(self, task_id: str, title: str, description: str, created_by: str):
        self.task_id = task_id
        self.title = title
        self.description = description
        self.created_by = created_by
        self.status = TaskStatus.OPEN
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.sub_tasks: List[str] = []
        self.associated_files: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "created_by": self.created_by,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sub_tasks": self.sub_tasks,
            "associated_files": self.associated_files
        }

class TaskManager:
    """Manages tasks and projects."""
    
    def __init__(self, storage_path: str = "tasks.json"):
        self.storage_path = Path(storage_path)
        self.tasks: Dict[str, Task] = {}
        self._load_tasks()
    
    def create_task(self, title: str, description: str, created_by: str = "user") -> str:
        """Create a new task."""
        task_id = str(uuid.uuid4())[:8]
        task = Task(task_id, title, description, created_by)
        self.tasks[task_id] = task
        self._save_tasks()
        logger.info(f"Task created: {task_id} - {title}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update a task's status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.now().isoformat()
            self._save_tasks()
            logger.info(f"Task {task_id} status updated to {status.value}")
    
    def add_sub_task(self, parent_task_id: str, sub_task_title: str) -> str:
        """Add a sub-task to a task."""
        sub_task_id = self.create_task(sub_task_title, f"Sub-task of {parent_task_id}", "grace")
        if parent_task_id in self.tasks:
            self.tasks[parent_task_id].sub_tasks.append(sub_task_id)
            self._save_tasks()
        return sub_task_id
    
    def associate_file(self, task_id: str, file_path: str):
        """Associate a file with a task."""
        if task_id in self.tasks:
            self.tasks[task_id].associated_files.append(file_path)
            self._save_tasks()
    
    def get_open_tasks(self) -> List[Task]:
        """Get all open tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.OPEN]
    
    def _load_tasks(self):
        """Load tasks from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for task_data in data:
                        task = Task(task_data['task_id'], task_data['title'], task_data['description'], task_data['created_by'])
                        task.status = TaskStatus[task_data['status']]
                        task.created_at = task_data['created_at']
                        task.updated_at = task_data['updated_at']
                        task.sub_tasks = task_data.get('sub_tasks', [])
                        task.associated_files = task_data.get('associated_files', [])
                        self.tasks[task.task_id] = task
            except Exception as e:
                logger.error(f"Error loading tasks: {str(e)}")
    
    def _save_tasks(self):
        """Save tasks to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump([t.to_dict() for t in self.tasks.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tasks: {str(e)}")
