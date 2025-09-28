"""
Process Management Adapter - Cross-platform process operations.
"""
import logging
import asyncio
from typing import Dict, Any, Optional
from .base import ProcessAdapter


logger = logging.getLogger(__name__)


class ProcessManager(ProcessAdapter):
    """Cross-platform process management implementation."""
    
    def __init__(self, os_adapter):
        self.os_adapter = os_adapter
        self.active_processes = {}  # pid -> process info
        logger.info("Process Manager initialized")
    
    async def exec(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using OS-specific adapter."""
        result = await self.os_adapter.exec(task)
        
        if result.get("success") and "pid" in result:
            self.active_processes[result["pid"]] = {
                "task_id": task.get("task_id"),
                "started_at": asyncio.get_event_loop().time(),
                "command": task.get("command")
            }
        
        return result
    
    async def kill(self, pid: int) -> bool:
        """Kill process and clean up tracking."""
        success = await self.os_adapter.kill(pid)
        
        if success and pid in self.active_processes:
            del self.active_processes[pid]
        
        return success
    
    async def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get process information."""
        return await self.os_adapter.get_process_info(pid)
    
    def get_active_processes(self) -> Dict[int, Dict[str, Any]]:
        """Get all active processes."""
        return self.active_processes.copy()