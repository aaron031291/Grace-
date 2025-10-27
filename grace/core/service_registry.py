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
__all__ = ["ServiceRegistry", "initialize_global_registry", "get_global_registry"]

@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = None
    dependencies: list = None


class ServiceRegistry:
    """
    Global singleton registry for services/factories.
    Previous behavior allowed multiple instances which led to workflows
    seeing an empty registry (e.g., trust_ledger factory "not registered").
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Guard double-init
        if getattr(self, "_initialized", False):
            return
        self._factories = {}
        self._services = {}
        self._initialized = True
        logger.info("Service Registry initialized.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_factory(self, name, factory_fn):
        self._factories[name] = factory_fn
        logger.info("✓ Registered service factory: %s", name)

    def get(self, name):
        if name in self._services:
            return self._services[name]
        if name not in self._factories:
            logger.error("No factory registered for service '%s'.", name)
            raise KeyError(name)
        
        # Pass the registry to the factory if it accepts an argument
        import inspect
        factory = self._factories[name]
        sig = inspect.signature(factory)
        if len(sig.parameters) > 0:
            svc = factory(self) # Pass registry for DI
        else:
            svc = factory()

        self._services[name] = svc
        return svc

    def get_optional(self, name: str):
        try:
            return self.get(name)
        except KeyError:
            return None

    # --- Back-compat lifecycle shims ------------------------------------------
    def initialize(self):
        """
        Legacy launcher used registry.initialize().
        Newer design auto-inits on first use; this remains as a safe no-op that
        marks the registry as ready and emits a helpful log line.
        """
        if not getattr(self, "_initialized", False):
            self._initialized = True
            logger.info("Service registry is now marked as initialized.")
        logger.info("✓ All services are ready to be loaded on demand.")

    def is_initialized(self) -> bool:
        """Back-compat convenience accessor."""
        return getattr(self, "_initialized", False)

# --- Back-compat / global accessors ------------------------------------------------
# Some callers (e.g., launcher) import initialize_global_registry / get_global_registry.
# Provide no-op shims that return the singleton.
def initialize_global_registry():
    """
    Back-compat: historically set up a process-global registry.
    Now the class is a singleton; this function simply returns it.
    """
    return ServiceRegistry.get_instance()

def get_global_registry():
    """Back-compat alias for ServiceRegistry.get_instance()."""
    return ServiceRegistry.get_instance()

# Ensure a singleton exists as soon as the module loads so that imports
# across modules always share the same instance (and same factory table).
if ServiceRegistry._instance is None:
    ServiceRegistry._instance = ServiceRegistry()
