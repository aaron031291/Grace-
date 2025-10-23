"""
Grace AI - Base Kernel Implementation
====================================
All kernels inherit from this to ensure proper structure
and service integration
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseKernel(ABC):
    """
    Base class for all Grace kernels
    Provides common interface and service integration
    """
    
    def __init__(self, name: str, service_registry=None):
        self.name = name
        self.service_registry = service_registry
        self.is_running = False
        self.logger = logging.getLogger(f"grace.kernels.{name}")
        self.logger.info(f"Initializing {name} kernel")
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute kernel-specific logic"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return kernel health status"""
        pass
    
    def get_service(self, service_name: str):
        """Get a service from the registry"""
        if not self.service_registry:
            self.logger.warning(f"⚠ No service registry for {self.name}")
            return None
        return self.service_registry.get_service(service_name)
    
    async def start(self):
        """Start the kernel"""
        self.is_running = True
        self.logger.info(f"✓ Started {self.name} kernel")
    
    async def stop(self):
        """Stop the kernel"""
        self.is_running = False
        self.logger.info(f"✓ Stopped {self.name} kernel")
    
    async def run_loop(self, interval: float = 1.0):
        """
        Run main kernel loop
        Subclasses can override for custom behavior
        """
        await self.start()
        
        try:
            while self.is_running:
                health = await self.health_check()
                self.logger.debug(f"Health: {health}")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self.logger.info(f"Cancelled {self.name} kernel")
        finally:
            await self.stop()
