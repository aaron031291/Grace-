"""
Plugin Registry and Sandbox Management

This module manages the registry of available plugins and provides sandboxed
execution capabilities to ensure plugins run safely without breaking system rules.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .plugins.base_plugin import BasePlugin, PluginResult, PluginStatus


class PluginRegistry:
    """
    Registry for managing and executing plugins in a sandboxed environment
    """
    
    def __init__(self, max_workers: int = 5, default_timeout: float = 30.0):
        """Initialize the plugin registry"""
        self.logger = logging.getLogger(__name__)
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_locks: Dict[str, threading.Lock] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the plugin registry"""
        try:
            self.logger.info("Initializing Plugin Registry...")
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Plugin Registry initialization failed: {str(e)}")
            return False
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """
        Register a new plugin
        
        Args:
            plugin: The plugin instance to register
            
        Returns:
            bool: True if registration successful
        """
        try:
            if plugin.name in self.plugins:
                self.logger.warning(f"Plugin {plugin.name} already registered, replacing...")
            
            # Initialize the plugin
            if not plugin.initialize():
                self.logger.error(f"Failed to initialize plugin {plugin.name}")
                return False
            
            self.plugins[plugin.name] = plugin
            self.plugin_locks[plugin.name] = threading.Lock()
            
            self.logger.info(f"Plugin {plugin.name} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin.name}: {str(e)}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        try:
            if plugin_name not in self.plugins:
                self.logger.warning(f"Plugin {plugin_name} not found for unregistration")
                return False
            
            plugin = self.plugins[plugin_name]
            plugin.shutdown()
            
            del self.plugins[plugin_name]
            del self.plugin_locks[plugin_name]
            
            self.logger.info(f"Plugin {plugin_name} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {str(e)}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a plugin by name
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            BasePlugin instance or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """
        Get list of registered plugin names
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def execute_plugin_safely(
        self, 
        plugin_name: str, 
        action: str, 
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> PluginResult:
        """
        Execute a plugin action safely in a sandboxed environment
        
        Args:
            plugin_name: Name of the plugin to execute
            action: Action to perform
            parameters: Parameters for the action
            timeout: Execution timeout in seconds
            
        Returns:
            PluginResult: Result of the execution
        """
        if not self.initialized:
            return PluginResult(
                plugin_name=plugin_name,
                action=action,
                status=PluginStatus.FAILED,
                message="Plugin registry not initialized"
            )
        
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return PluginResult(
                plugin_name=plugin_name,
                action=action,
                status=PluginStatus.FAILED,
                message=f"Plugin '{plugin_name}' not found"
            )
        
        if not plugin.enabled:
            return PluginResult(
                plugin_name=plugin_name,
                action=action,
                status=PluginStatus.FAILED,
                message=f"Plugin '{plugin_name}' is disabled"
            )
        
        # Validate action and parameters
        if not plugin.validate_parameters(action, parameters):
            return PluginResult(
                plugin_name=plugin_name,
                action=action,
                status=PluginStatus.FAILED,
                message=f"Invalid action '{action}' or parameters for plugin '{plugin_name}'"
            )
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            # Execute plugin in thread with timeout (sandbox)
            with self.plugin_locks[plugin_name]:
                future = self.executor.submit(
                    self._execute_plugin_with_monitoring,
                    plugin, action, parameters
                )
                
                result = future.result(timeout=timeout)
                result.execution_time = time.time() - start_time
                
                self.logger.info(
                    f"Plugin {plugin_name}.{action} executed successfully "
                    f"in {result.execution_time:.2f}s"
                )
                
                return result
                
        except FutureTimeoutError:
            self.logger.error(f"Plugin {plugin_name}.{action} timed out after {timeout}s")
            return PluginResult(
                plugin_name=plugin_name,
                action=action,
                status=PluginStatus.TIMEOUT,
                message=f"Execution timed out after {timeout} seconds",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Plugin {plugin_name}.{action} execution failed: {str(e)}")
            return PluginResult(
                plugin_name=plugin_name,
                action=action,
                status=PluginStatus.FAILED,
                message=f"Execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _execute_plugin_with_monitoring(
        self, 
        plugin: BasePlugin, 
        action: str, 
        parameters: Dict[str, Any]
    ) -> PluginResult:
        """
        Execute plugin with monitoring and resource constraints
        
        This method runs in a separate thread and implements the actual sandbox
        constraints and monitoring.
        """
        try:
            # In a real implementation, this would set up resource limits,
            # network restrictions, file system sandboxing, etc.
            
            # Execute the plugin
            result = plugin.execute(action, parameters)
            
            # Validate the result
            if not isinstance(result, PluginResult):
                return PluginResult(
                    plugin_name=plugin.name,
                    action=action,
                    status=PluginStatus.FAILED,
                    message="Plugin returned invalid result type"
                )
            
            return result
            
        except Exception as e:
            return PluginResult(
                plugin_name=plugin.name,
                action=action,
                status=PluginStatus.FAILED,
                message=f"Plugin execution error: {str(e)}"
            )
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dict containing plugin information or None if not found
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return None
        
        return {
            "name": plugin.name,
            "enabled": plugin.enabled,
            "initialized": plugin.initialized,
            "supported_actions": plugin.get_supported_actions(),
            "health": plugin.health_check()
        }
    
    def shutdown(self):
        """Shutdown the plugin registry and cleanup resources"""
        self.logger.info("Shutting down Plugin Registry...")
        
        # Shutdown all plugins
        for plugin_name, plugin in self.plugins.items():
            try:
                plugin.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down plugin {plugin_name}: {str(e)}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.plugins.clear()
        self.plugin_locks.clear()
        self.initialized = False
        
        self.logger.info("Plugin Registry shutdown complete")