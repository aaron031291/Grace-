"""
Filesystem Management Adapter - Cross-platform filesystem operations.
"""
import logging
from typing import Dict, Any, List
from .base import FSAdapter


logger = logging.getLogger(__name__)


class FSManager(FSAdapter):
    """Cross-platform filesystem management implementation."""
    
    def __init__(self, os_adapter):
        self.os_adapter = os_adapter
        self.operation_log = []  # Track operations for auditing
        logger.info("Filesystem Manager initialized")
    
    async def apply(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filesystem action with logging."""
        result = await self.os_adapter.apply(action)
        
        # Log the operation
        self.operation_log.append({
            "action_id": action.get("action_id"),
            "type": action.get("type"),
            "path": action.get("path"),
            "success": result.get("success", False),
            "timestamp": logging.time.time()
        })
        
        # Keep log size manageable
        if len(self.operation_log) > 1000:
            self.operation_log = self.operation_log[-500:]
        
        return result
    
    async def read(self, path: str, encoding: str = "utf-8") -> bytes:
        """Read file with proper error handling."""
        return await self.os_adapter.read(path, encoding)
    
    async def write(self, path: str, content: bytes) -> bool:
        """Write file with proper error handling.""" 
        return await self.os_adapter.write(path, content)
    
    async def list_dir(self, path: str, recursive: bool = False) -> List[str]:
        """List directory contents."""
        return await self.os_adapter.list_dir(path, recursive)
    
    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        return await self.os_adapter.exists(path)
    
    async def delete(self, path: str, recursive: bool = False) -> bool:
        """Delete file or directory."""
        return await self.os_adapter.delete(path, recursive)
    
    def get_operation_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent filesystem operations."""
        return self.operation_log[-limit:]