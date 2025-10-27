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

# Ensure the launcher-created instance becomes the singleton
# (Importing this module elsewhere will share the same instance.)
ServiceRegistry._instance = ServiceRegistry()
