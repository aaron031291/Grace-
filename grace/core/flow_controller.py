"""
Flow Controller - Manages async task execution flow
"""

from typing import Optional, List, Any
from datetime import datetime
import asyncio
import inspect
import logging

logger = logging.getLogger(__name__)


class BaseComponent:
    """Base component class"""
    
    def __init__(self):
        self.loop_index: Optional[int] = None
        self._active_tasks: List[Any] = []


class FlowController(BaseComponent):
    """
    Controls the flow of async tasks
    
    Fixed issues:
    - Proper task awaiting
    - Type safety
    - Error handling
    """
    
    def __init__(self):
        super().__init__()
        self.loop_index = 0
    
    async def tick(self, timestamp: Optional[datetime] = None) -> None:
        """
        Execute one tick of the flow controller
        
        Args:
            timestamp: Optional timestamp for this tick
        """
        timestamp = timestamp or datetime.utcnow()
        self.loop_index = int(self.loop_index or 0) + 1
        
        # Filter for actual awaitables
        tasks = [t for t in self._active_tasks if inspect.isawaitable(t)]
        
        if tasks:
            try:
                # Gather all tasks with proper error handling
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {i} failed: {result}")
                
            except Exception as e:
                logger.error(f"Flow controller tick failed: {e}")
        
        # Clear completed tasks
        self._active_tasks = []
    
    def add_task(self, task: Any) -> None:
        """Add a task to be executed"""
        if inspect.isawaitable(task):
            self._active_tasks.append(task)
    
    def get_stats(self) -> dict:
        """Get controller statistics"""
        return {
            "loop_index": self.loop_index,
            "active_tasks": len(self._active_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
