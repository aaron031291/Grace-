"""Orchestration kernel - main run loop and task dispatch."""
import asyncio
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime


class OrchestrationKernel:
    """Main orchestration kernel managing system run loop."""
    
    def __init__(self):
        self.running = False
        self.tick_count = 0
        self.last_tick = None
        self.registered_tasks = []
    
    def register_task(self, name: str, callback: Callable, interval_seconds: int = 60):
        """Register a periodic task."""
        self.registered_tasks.append({
            "name": name,
            "callback": callback,
            "interval": interval_seconds,
            "last_run": 0
        })
    
    async def run_loop(self):
        """Main orchestration run loop."""
        self.running = True
        self.last_tick = time.time()
        
        while self.running:
            await self.tick()
            await asyncio.sleep(5)  # 5 second tick interval
    
    async def tick(self):
        """Single orchestration tick."""
        self.tick_count += 1
        self.last_tick = time.time()
        
        # Process registered tasks
        for task in self.registered_tasks:
            if (self.last_tick - task["last_run"]) >= task["interval"]:
                try:
                    if asyncio.iscoroutinefunction(task["callback"]):
                        await task["callback"]()
                    else:
                        task["callback"]()
                    task["last_run"] = self.last_tick
                except Exception as e:
                    # Log error but continue
                    pass
    
    def stop(self):
        """Stop the orchestration loop."""
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get orchestration statistics."""
        return {
            "running": self.running,
            "tick_count": self.tick_count,
            "last_tick": datetime.fromtimestamp(self.last_tick).isoformat() if self.last_tick else None,
            "registered_tasks": len(self.registered_tasks),
            "uptime_seconds": time.time() - (self.last_tick or time.time())
        }