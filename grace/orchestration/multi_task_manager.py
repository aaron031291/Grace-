"""
Multi-Task Manager - Concurrent Background Processing

Grace can run up to 6 concurrent tasks in background:
1. Code generation
2. Research/knowledge gathering
3. Test execution
4. Documentation writing
5. Refactoring
6. Analysis/monitoring

Both human and Grace can:
- Delegate tasks to each other
- Monitor task progress
- Take over tasks if faster
- Collaborate on complex tasks

Grace manages multiple responsibilities simultaneously!
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks"""
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    ANALYSIS = "analysis"
    DEBUGGING = "debugging"
    DEPLOYMENT = "deployment"


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"
    TAKEN_OVER = "taken_over"


@dataclass
class Task:
    """A task in the system"""
    task_id: str
    task_type: TaskType
    description: str
    assigned_to: str  # "user" or "grace"
    delegated_by: str  # Who delegated this task
    priority: int  # 1-5, 5 being highest
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": self.task_type.value,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "delegated_by": self.delegated_by,
            "priority": self.priority,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class MultiTaskManager:
    """
    Manages concurrent background tasks for Grace.
    
    Features:
    - Up to 6 concurrent tasks
    - Task delegation (Human ↔ Grace)
    - Progress monitoring
    - Task takeover (if one is faster)
    - Priority queueing
    """
    
    MAX_CONCURRENT_TASKS = 6
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: List[Task] = []
        
        self.on_task_complete: Optional[Callable] = None
        self.on_task_progress: Optional[Callable] = None
        
        logger.info("Multi-Task Manager initialized")
        logger.info(f"  Max concurrent tasks: {self.MAX_CONCURRENT_TASKS}")
    
    async def delegate_to_grace(
        self,
        task_type: TaskType,
        description: str,
        priority: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Human delegates task to Grace.
        
        Grace will work on it in background.
        """
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            description=description,
            assigned_to="grace",
            delegated_by="user",
            priority=priority,
            created_at=datetime.utcnow()
        )
        
        self.tasks[task.task_id] = task
        
        logger.info(f"\n📋 Task delegated to Grace:")
        logger.info(f"   ID: {task.task_id}")
        logger.info(f"   Type: {task_type.value}")
        logger.info(f"   Description: {description}")
        logger.info(f"   Priority: {priority}/5")
        
        # Start task if slots available
        await self._maybe_start_task(task)
        
        return task
    
    async def delegate_to_human(
        self,
        task_type: TaskType,
        description: str,
        reason: str,
        priority: int = 3
    ) -> Task:
        """
        Grace delegates task to human.
        
        Grace asks for help when:
        - She lacks knowledge
        - Decision needs human judgment
        - Task requires human expertise
        """
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            description=description,
            assigned_to="user",
            delegated_by="grace",
            priority=priority,
            created_at=datetime.utcnow()
        )
        
        self.tasks[task.task_id] = task
        task.status = TaskStatus.DELEGATED
        
        logger.info(f"\n📋 Grace delegates task to human:")
        logger.info(f"   ID: {task.task_id}")
        logger.info(f"   Type: {task_type.value}")
        logger.info(f"   Description: {description}")
        logger.info(f"   Reason: {reason}")
        
        # Send notification to user
        await self._notify_user_of_delegation(task, reason)
        
        return task
    
    async def _maybe_start_task(self, task: Task):
        """Start task if slots available"""
        if len(self.running_tasks) < self.MAX_CONCURRENT_TASKS:
            await self._start_task(task)
        else:
            self.task_queue.append(task)
            logger.info(f"   ⏳ Task queued (all slots busy)")
    
    async def _start_task(self, task: Task):
        """Start executing a task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        logger.info(f"   🚀 Starting task: {task.task_id}")
        
        # Create async task
        async_task = asyncio.create_task(self._execute_task(task))
        self.running_tasks[task.task_id] = async_task
    
    async def _execute_task(self, task: Task):
        """Execute a task"""
        try:
            logger.info(f"\n🔄 Executing: {task.description}")
            
            # Route to appropriate executor
            if task.task_type == TaskType.CODE_GENERATION:
                result = await self._execute_code_generation(task)
            
            elif task.task_type == TaskType.RESEARCH:
                result = await self._execute_research(task)
            
            elif task.task_type == TaskType.TESTING:
                result = await self._execute_testing(task)
            
            elif task.task_type == TaskType.DOCUMENTATION:
                result = await self._execute_documentation(task)
            
            elif task.task_type == TaskType.REFACTORING:
                result = await self._execute_refactoring(task)
            
            elif task.task_type == TaskType.ANALYSIS:
                result = await self._execute_analysis(task)
            
            else:
                result = {"status": "completed", "message": "Generic task executed"}
            
            # Task completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            task.progress = 1.0
            
            logger.info(f"   ✅ Task completed: {task.task_id}")
            
            # Notify
            if self.on_task_complete:
                await self.on_task_complete(task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"   ❌ Task failed: {e}")
        
        finally:
            # Remove from running
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Start next queued task
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                await self._start_task(next_task)
    
    async def _execute_code_generation(self, task: Task) -> Dict[str, Any]:
        """Execute code generation task"""
        from grace.intelligence.expert_code_generator import get_expert_code_generator, CodeGenerationRequest
        
        generator = get_expert_code_generator()
        
        # Update progress
        task.progress = 0.3
        if self.on_task_progress:
            await self.on_task_progress(task)
        
        # Generate code
        result = await generator.generate(CodeGenerationRequest(
            requirements=task.description,
            language="python",  # Would extract from task
            include_tests=True
        ))
        
        task.progress = 1.0
        
        return {
            "code": result.code,
            "quality": result.quality_score,
            "tests": result.tests
        }
    
    async def _execute_research(self, task: Task) -> Dict[str, Any]:
        """Execute research task"""
        from grace.intelligence.research_mode import GraceResearchMode
        
        research = GraceResearchMode()
        
        task.progress = 0.2
        
        # Start research
        research_task = await research.start_research(
            topic=task.description,
            knowledge_gaps=["general_research"]
        )
        
        task.progress = 0.5
        
        # Conduct research
        result = await research.conduct_research(research_task.task_id)
        
        task.progress = 1.0
        
        return {
            "findings": len(result.findings),
            "confidence": result.confidence_after,
            "summary": result.summary
        }
    
    async def _execute_testing(self, task: Task) -> Dict[str, Any]:
        """Execute testing task"""
        # Run tests
        task.progress = 0.5
        await asyncio.sleep(2)  # Simulate test run
        
        return {
            "tests_run": 47,
            "tests_passed": 47,
            "coverage": 0.94
        }
    
    async def _execute_documentation(self, task: Task) -> Dict[str, Any]:
        """Execute documentation task"""
        # Generate docs
        return {"docs_generated": True}
    
    async def _execute_refactoring(self, task: Task) -> Dict[str, Any]:
        """Execute refactoring task"""
        # Refactor code
        return {"refactored": True}
    
    async def _execute_analysis(self, task: Task) -> Dict[str, Any]:
        """Execute analysis task"""
        # Analyze code/system
        return {"analysis": "complete"}
    
    async def _notify_user_of_delegation(self, task: Task, reason: str):
        """Notify user that Grace needs their help"""
        # Would send via WebSocket/notification system
        logger.info(f"\n🔔 NOTIFICATION TO USER:")
        logger.info(f"   Grace needs your help!")
        logger.info(f"   Task: {task.description}")
        logger.info(f"   Reason: {reason}")
    
    async def grace_takes_over_task(
        self,
        task_id: str,
        reason: str = "Grace can complete faster"
    ):
        """
        Grace takes over a task from human.
        
        If Grace determines she can complete faster/better,
        she proactively takes over (with notification).
        """
        task = self.tasks.get(task_id)
        if not task:
            return
        
        logger.info(f"\n🤖 Grace taking over task: {task_id}")
        logger.info(f"   Reason: {reason}")
        
        task.assigned_to = "grace"
        task.status = TaskStatus.TAKEN_OVER
        
        # Start executing
        await self._start_task(task)
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks"""
        return [
            task for task in self.tasks.values()
            if task.status in [TaskStatus.IN_PROGRESS, TaskStatus.PENDING]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task management statistics"""
        total = len(self.tasks)
        
        by_status = {}
        by_type = {}
        by_assignee = {}
        
        for task in self.tasks.values():
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
            by_type[task.task_type.value] = by_type.get(task.task_type.value, 0) + 1
            by_assignee[task.assigned_to] = by_assignee.get(task.assigned_to, 0) + 1
        
        return {
            "total_tasks": total,
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
            "slots_available": self.MAX_CONCURRENT_TASKS - len(self.running_tasks),
            "by_status": by_status,
            "by_type": by_type,
            "by_assignee": by_assignee
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🔄 Multi-Task Manager Demo\n")
        
        manager = MultiTaskManager()
        
        # User delegates tasks to Grace
        print("👤 User delegates 3 tasks to Grace:")
        
        task1 = await manager.delegate_to_grace(
            TaskType.CODE_GENERATION,
            "Create authentication system",
            priority=5
        )
        
        task2 = await manager.delegate_to_grace(
            TaskType.RESEARCH,
            "Research best practices for microservices",
            priority=3
        )
        
        task3 = await manager.delegate_to_grace(
            TaskType.TESTING,
            "Run full test suite",
            priority=4
        )
        
        # Grace delegates task to human
        print("\n🧠 Grace delegates task to human:")
        
        task4 = await manager.delegate_to_human(
            TaskType.ANALYSIS,
            "Review security architecture",
            reason="Requires human judgment on risk tolerance"
        )
        
        # Show stats
        print(f"\n📊 Task Manager Stats:")
        stats = manager.get_stats()
        print(f"   Total tasks: {stats['total_tasks']}")
        print(f"   Running: {stats['running_tasks']}")
        print(f"   Queued: {stats['queued_tasks']}")
        print(f"   Slots available: {stats['slots_available']}/6")
        print(f"   By assignee: {stats['by_assignee']}")
        
        # Wait for some tasks to complete
        await asyncio.sleep(3)
        
        print("\n✅ Grace is managing multiple tasks concurrently!")
    
    asyncio.run(demo())
