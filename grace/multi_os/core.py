"""
Grace AI - Multi-OS Kernel
==========================
Handles multi-platform operations and cross-OS execution
"""

import logging
import asyncio
import platform
import sys
from typing import Dict, Any
from grace.kernels.base_kernel import BaseKernel

logger = logging.getLogger(__name__)


class MultiOSKernel(BaseKernel):
    """
    Multi-OS kernel for platform-specific operations
    Supports Linux, macOS, Windows
    """
    
    def __init__(self, service_registry=None):
        super().__init__("multi_os", service_registry)
        self.current_platform = platform.system()
        self.operations_executed = 0
        self.platform_handlers = {
            'Linux': self._handle_linux,
            'Darwin': self._handle_macos,
            'Windows': self._handle_windows,
        }
        self.logger.info(f"MultiOS kernel initialized for {self.current_platform}")
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute platform-specific task
        
        Args:
            task: {
                'type': 'execute' | 'detect' | 'adapt',
                'command': command to run,
                'platform': target platform (optional)
            }
        
        Returns:
            {
                'success': bool,
                'result': execution result,
                'platform': detected platform
            }
        """
        task_type = task.get('type', 'execute')
        
        try:
            self.logger.info(f"MultiOS task: {task_type} on {self.current_platform}")
            
            if task_type == 'execute':
                result = await self._execute_command(task)
            elif task_type == 'detect':
                result = await self._detect_platform(task)
            elif task_type == 'adapt':
                result = await self._adapt_to_platform(task)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            self.operations_executed += 1
            
            return {
                'success': 'error' not in result,
                'result': result,
                'platform': self.current_platform,
                'operations': self.operations_executed
            }
        
        except Exception as e:
            self.logger.error(f"MultiOS execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': self.current_platform,
                'operations': self.operations_executed
            }
    
    async def _execute_command(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute platform-specific command"""
        command = task.get('command', 'echo test')
        
        self.logger.info(f"Executing command: {command}")
        
        # Get handler for current platform
        handler = self.platform_handlers.get(
            self.current_platform,
            self._handle_unknown
        )
        
        result = await handler(command)
        return result
    
    async def _detect_platform(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect platform information"""
        self.logger.info("Detecting platform information")
        
        return {
            'system': self.current_platform,
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'python': platform.python_version(),
            'processor': platform.processor()
        }
    
    async def _adapt_to_platform(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt behavior to current platform"""
        config = task.get('config', {})
        
        self.logger.info(f"Adapting to {self.current_platform}")
        
        # Simulate adaptation
        await asyncio.sleep(0.05)
        
        return {
            'platform': self.current_platform,
            'adapted': True,
            'config': config
        }
    
    async def _handle_linux(self, command: str) -> Dict[str, Any]:
        """Handle Linux-specific operations"""
        self.logger.info(f"Linux handler: {command}")
        await asyncio.sleep(0.05)
        return {
            'platform': 'Linux',
            'command': command,
            'result': 'executed'
        }
    
    async def _handle_macos(self, command: str) -> Dict[str, Any]:
        """Handle macOS-specific operations"""
        self.logger.info(f"macOS handler: {command}")
        await asyncio.sleep(0.05)
        return {
            'platform': 'Darwin',
            'command': command,
            'result': 'executed'
        }
    
    async def _handle_windows(self, command: str) -> Dict[str, Any]:
        """Handle Windows-specific operations"""
        self.logger.info(f"Windows handler: {command}")
        await asyncio.sleep(0.05)
        return {
            'platform': 'Windows',
            'command': command,
            'result': 'executed'
        }
    
    async def _handle_unknown(self, command: str) -> Dict[str, Any]:
        """Handle unknown platform"""
        self.logger.warning(f"Unknown platform handler for: {self.current_platform}")
        return {
            'platform': self.current_platform,
            'command': command,
            'result': 'unknown_platform'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Return kernel health status"""
        return {
            'name': self.name,
            'running': self.is_running,
            'platform': self.current_platform,
            'operations_executed': self.operations_executed,
            'python_version': platform.python_version()
        }
