import logging

class BaseService:
    """
    A base class for all services in the Grace system.
    Provides common functionality like logging and service registry access.
    """
    def __init__(self, name: str, service_registry):
        self.name = name
        self.service_registry = service_registry
        self.logger = logging.getLogger(f"service.{self.name}")
        self.logger.info(f"Service '{self.name}' is initializing.")

    def get_name(self) -> str:
        """Returns the name of the service."""
        return self.name

    def get_service(self, service_name: str):
        """
        Retrieves another service from the registry.
        """
        return self.service_registry.get(service_name)
