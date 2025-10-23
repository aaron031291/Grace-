"""
Periodic task scheduler
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PeriodicTask:
    """Represents a periodic task"""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        interval_seconds: float,
        run_immediately: bool = False
    ):
        self.name = name
        self.func = func
        self.interval_seconds = interval_seconds
        self.run_immediately = run_immediately
        
        self.task: Optional[asyncio.Task] = None
        self.running = False
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0
    
    async def _loop(self):
        """Task execution loop"""
        self.running = True
        
        if not self.run_immediately:
            await asyncio.sleep(self.interval_seconds)
        
        while self.running:
            try:
                logger.debug(f"Running periodic task: {self.name}")
                
                if asyncio.iscoroutinefunction(self.func):
                    await self.func()
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.func)
                
                self.last_run = datetime.utcnow()
                self.run_count += 1
            
            except Exception as e:
                self.error_count += 1
                logger.exception(f"Periodic task {self.name} error: {e}")
            
            await asyncio.sleep(self.interval_seconds)
    
    def start(self):
        """Start the periodic task"""
        if self.task and not self.task.done():
            logger.warning(f"Task {self.name} already running")
            return
        
        self.task = asyncio.create_task(self._loop())
        logger.info(f"Started periodic task: {self.name} (interval: {self.interval_seconds}s)")
    
    async def stop(self):
        """Stop the periodic task"""
        self.running = False
        
        if self.task:
            self.task.cancel()
            try:
                await asyncio.wait_for(self.task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        logger.info(f"Stopped periodic task: {self.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics"""
        return {
            "name": self.name,
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_run": self.last_run.isoformat() if self.last_run else None
        }


class GraceScheduler:
    """
    Scheduler for periodic tasks
    """
    
    def __init__(self):
        self.tasks: Dict[str, PeriodicTask] = {}
    
    def schedule(
        self,
        name: str,
        func: Callable,
        interval_seconds: float,
        run_immediately: bool = False
    ) -> PeriodicTask:
        """Schedule a periodic task"""
        if name in self.tasks:
            logger.warning(f"Task {name} already scheduled, replacing")
            asyncio.create_task(self.tasks[name].stop())
        
        task = PeriodicTask(name, func, interval_seconds, run_immediately)
        self.tasks[name] = task
        task.start()
        
        logger.info(f"Scheduled task: {name}")
        return task
    
    async def stop_task(self, name: str):
        """Stop a specific task"""
        if name in self.tasks:
            await self.tasks[name].stop()
            del self.tasks[name]
    
    async def stop_all(self):
        """Stop all tasks"""
        logger.info("Stopping all scheduled tasks")
        
        for task in list(self.tasks.values()):
            await task.stop()
        
        self.tasks.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": sum(1 for t in self.tasks.values() if t.running),
            "tasks": [t.get_stats() for t in self.tasks.values()]
        }


# Global scheduler instance
_scheduler: Optional[GraceScheduler] = None


def get_scheduler() -> GraceScheduler:
    """Get global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = GraceScheduler()
    return _scheduler
