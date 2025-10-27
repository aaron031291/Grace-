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
    
    _instance = None

    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        logger.info("Service Registry initialized.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ServiceRegistry()
        return cls._instance

    def has(self, name: str) -> bool:
        return name in self._instances

    def get_optional(self, name: str):
        try:
            return self.get(name)
        except Exception:
            return None

    def register_factory(self, name: str, factory: Callable, overwrite: bool = False):
        """
        Registers a factory function for creating a service.
        The factory function should take the registry instance as its only argument.
        """
        if name in self._factories and not overwrite:
            logger.warning("Service factory for '%s' is already registered. Skipping.", name)
            return
        self._factories[name] = factory
        logger.info("✓ Registered service factory: %s", name)

    def get(self, name: str):
        """
        Retrieves a service instance by name.
        If the service has not been created yet, it will be instantiated using
        its registered factory.
        """
        if name in self._instances:
            return self._instances[name]
        if name not in self._factories:
            logger.error("No factory registered for service '%s'.", name)
            raise ValueError(f"No factory registered for service '{name}'.")

        logger.info("Creating new instance of service '%s'...", name)
        try:
            factory = self._factories[name]
            # Pass the registry to the factory if it accepts an argument
            import inspect
            sig = inspect.signature(factory)
            if len(sig.parameters) > 0:
                instance = factory(self) # Pass registry for DI
            else:
                instance = factory()

            self._instances[name] = instance
            logger.info("Service '%s' created successfully.", name)
            return instance
        except Exception as e:
            logger.error("Failed to create service '%s': %s", name, e, exc_info=True)
            raise

    def register_instance(self, name: str, instance: Any):
        """Register a service instance directly"""
        self._instances[name] = instance
        logger.info("✓ Registered service instance: %s", name)

    def get_all_services(self) -> Dict[str, Any]:
        """Get all service instances"""
        return self._instances.copy()

    def initialize(self):
        """
        Marks the registry as initialized. In this lazy-loading implementation,
        this method doesn't need to pre-load services.
        """
        logger.info("Service registry is now marked as initialized.")
        logger.info("✓ All services are ready to be loaded on demand.")

    def shutdown(self):
        """Shutdown all services"""
        logger.info("Shutting down all services...")
        for name, service in self._instances.items():
            if hasattr(service, 'shutdown'):
                try:
                    service.shutdown()
                    logger.info("✓ Service '%s' shut down.", name)
                except Exception as e:
                    logger.error("Error shutting down service '%s': %s", name, e)
        self._instances.clear()
        logger.info("All services shut down.")


# Global registry instance
_global_registry: ServiceRegistry | None = None


def initialize_global_registry() -> ServiceRegistry:
    """Initialize the global registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def get_global_registry() -> ServiceRegistry:
    """Get or create global service registry"""
    if _global_registry is None:
        raise RuntimeError("Global service registry has not been initialized.")
    return _global_registry
