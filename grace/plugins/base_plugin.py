"""
Base Plugin Interface

This module defines the base interface that all BusinessOps plugins must implement
to ensure consistent behavior and sandboxed execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import time


class PluginStatus(Enum):
    """Status of plugin execution"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class PluginResult:
    """Result of plugin execution"""
    plugin_name: str
    action: str
    status: PluginStatus
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class BasePlugin(ABC):
    """
    Base class for all BusinessOps plugins
    
    This class defines the interface that all plugins must implement to work
    with the BusinessOps Kernel's sandbox execution environment.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the plugin"""
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin and any required resources
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, action: str, parameters: Dict[str, Any]) -> PluginResult:
        """
        Execute a specific action with given parameters
        
        Args:
            action: The action to perform
            parameters: Parameters for the action
            
        Returns:
            PluginResult: Result of the execution
        """
        pass
    
    @abstractmethod
    def get_supported_actions(self) -> list[str]:
        """
        Get list of actions supported by this plugin
        
        Returns:
            list: List of supported action names
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the plugin
        
        Returns:
            Dict containing health status
        """
        return {
            "healthy": self.enabled and self.initialized,
            "name": self.name,
            "enabled": self.enabled,
            "initialized": self.initialized
        }
    
    def validate_parameters(self, action: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters for a given action
        
        Args:
            action: The action to validate parameters for
            parameters: Parameters to validate
            
        Returns:
            bool: True if parameters are valid
        """
        # Default implementation - subclasses should override for specific validation
        return action in self.get_supported_actions()
    
    def shutdown(self):
        """Cleanup plugin resources"""
        self.enabled = False
        self.initialized = False