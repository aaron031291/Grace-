"""
Grace AI - Core Service Registry & Unified Service Interface
===========================================================
Replaces the non-existent grace.core.unified_service
Provides proper dependency injection and service management
"""

from typing import Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = None
    dependencies: list = None


class ServiceRegistry:
    """
    Central service registry for dependency injection
    Manages all services and their dependencies
    """
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._config: Dict[str, ServiceConfig] = {}
        self._initialized: bool = False
        logger.info("Service Registry initialized.")
    
    def register_factory(self, name: str, factory: Callable, config: ServiceConfig = None, overwrite: bool = False):
        """
        Registers a factory function for creating a service.
        The factory function should take the registry instance as its only argument.
        """
        if name in self._factories and not overwrite:
            logger.warning(f"Service factory for '{name}' is already registered. Skipping.")
            return
        self._factories[name] = factory
        if config:
            self._config[name] = config
        logger.info(f"✓ Registered service factory: {name}")
    
    def register_instance(self, name: str, instance: Any, config: ServiceConfig = None):
        """Register a service instance directly"""
        self._instances[name] = instance
        logger.info(f"✓ Registered service instance: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Retrieves a service instance by name.
        If the service has not been created yet, it will be instantiated using
        its registered factory.
        """
        # First, check if an instance already exists.
        instance = self._instances.get(name)
        if instance:
            logger.debug(f"Returning existing instance of service '{name}'.")
            return instance

        # If not, create it using the factory.
        factory = self._factories.get(name)
        if not factory:
            logger.error(f"No factory registered for service '{name}'.")
            return None

        logger.info(f"Creating new instance of service '{name}'...")
        try:
            # Create and store the new instance
            self._instances[name] = factory(self)  # Pass registry for DI
            logger.info(f"Service '{name}' created successfully.")
            return self._instances[name]
        except Exception as e:
            logger.error(f"Failed to create service '{name}': {e}", exc_info=True)
            # Remove from instances if creation failed to allow retry
            if name in self._instances:
                del self._instances[name]
            return None
    
    def initialize(self):
        """
        Marks the registry as initialized. In this lazy-loading implementation,
        this method doesn't need to pre-load services.
        """
        logger.info("Service registry is now marked as initialized.")
        self._initialized = True
        logger.info("✓ All services are ready to be loaded on demand.")
    
    def shutdown(self):
        """Shutdown all services"""
        logger.info("Shutting down services...")
        for name in list(self._instances.keys()):
            service = self._instances[name]
            if hasattr(service, 'shutdown'):
                try:
                    service.shutdown()
                    logger.info(f"✓ Shutdown: {name}")
                except Exception as e:
                    logger.error(f"✗ Shutdown error for {name}: {e}")
        
        self._instances.clear()
        logger.info("✓ All services shut down")
    
    def is_initialized(self) -> bool:
        """Check if registry is initialized"""
        return self._initialized
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all service instances"""
        return self._instances.copy()
    
    def get_config(self, name: str) -> Optional[ServiceConfig]:
        """Get service configuration"""
        return self._config.get(name)


# Global registry instance
_global_registry: Optional[ServiceRegistry] = None


def get_global_registry() -> ServiceRegistry:
    """Get or create global service registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def initialize_global_registry():
    """Initialize the global registry"""
    registry = get_global_registry()
    registry.initialize()
    return registry
