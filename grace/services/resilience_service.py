from .base_service import BaseService
from ..resilience.immune_system import ImmuneSystem

class ResilienceService(BaseService):
    def __init__(self, service_registry):
        super().__init__("resilience_service", service_registry)
        self.immune_system = ImmuneSystem(service_registry)
        self.logger.info("Resilience Service initialized, wrapping Immune System.")

    async def report_threat(self, threat_info: dict):
        """Report a threat to the immune system."""
        return await self.immune_system.report_threat(threat_info)

    async def get_system_health_score(self) -> float:
        """Get the current system health score."""
        return await self.immune_system.get_system_health_score()

    def get_name(self) -> str:
        return self.immune_system.get_name()
