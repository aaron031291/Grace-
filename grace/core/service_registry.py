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
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._config: Dict[str, ServiceConfig] = {}
        self._initialized = False
        logger.info("ServiceRegistry initialized")
    
    def register_factory(self, name: str, factory: Callable, config: ServiceConfig = None):
        """Register a service factory"""
        self._factories[name] = factory
        self._config[name] = config or ServiceConfig(name=name)
        logger.info(f"✓ Registered service factory: {name}")
    
    def register_instance(self, name: str, instance: Any, config: ServiceConfig = None):
        """Register a service instance directly"""
        self._services[name] = instance
        self._config[name] = config or ServiceConfig(name=name)
        logger.info(f"✓ Registered service instance: {name}")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance, creating if needed"""
        # Check if already instantiated
        if name in self._services:
            return self._services[name]
        
        # Check if factory exists
        if name not in self._factories:
            logger.warning(f"⚠ Service not found: {name}")
            return None
        
        # Create from factory
        try:
            factory = self._factories[name]
            instance = factory(self)  # Pass registry for DI
            self._services[name] = instance
            logger.info(f"✓ Created service instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"✗ Failed to create service {name}: {e}")
            return None
    
    def initialize(self):
        """Initialize all registered services"""
        logger.info("Initializing all services...")
        for name, config in self._config.items():
            if not config.enabled:
                logger.info(f"⊘ Service disabled: {name}")
                continue
            
            service = self.get_service(name)
            if service is None:
                logger.warning(f"⚠ Failed to initialize: {name}")
        
        self._initialized = True
        logger.info("✓ All services initialized")
    
    def shutdown(self):
        """Shutdown all services"""
        logger.info("Shutting down services...")
        for name in list(self._services.keys()):
            service = self._services[name]
            if hasattr(service, 'shutdown'):
                try:
                    service.shutdown()
                    logger.info(f"✓ Shutdown: {name}")
                except Exception as e:
                    logger.error(f"✗ Shutdown error for {name}: {e}")
        
        self._services.clear()
        logger.info("✓ All services shut down")
    
    def is_initialized(self) -> bool:
        """Check if registry is initialized"""
        return self._initialized
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all service instances"""
        return self._services.copy()
    
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
