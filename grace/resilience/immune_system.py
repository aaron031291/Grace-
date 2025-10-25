import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ImmuneSystem:
    """
    Monitors system health, detects threats, and coordinates healing actions.
    This is a placeholder implementation.
    """
    def __init__(self, service_registry):
        self.service_registry = service_registry
        self.health_score = 100.0
        self.threats = []
        logger.info("Immune System initialized.")

    async def report_threat(self, threat_info: Dict[str, Any]):
        """Logs a threat and adjusts the health score."""
        logger.warning(f"Threat reported: {threat_info}")
        self.threats.append(threat_info)
        
        # Decrease health score based on threat level
        level = threat_info.get('level', 'low')
        if level == 'critical':
            self.health_score -= 25
        elif level == 'high':
            self.health_score -= 15
        elif level == 'medium':
            self.health_score -= 10
        else:
            self.health_score -= 5
        
        self.health_score = max(0, self.health_score)
        logger.info(f"System health score is now: {self.health_score}")
        return {"status": "threat_logged", "new_health_score": self.health_score}

    async def get_system_health_score(self) -> float:
        """Returns the current system health score."""
        return self.health_score

    def get_name(self) -> str:
        return "immune_system"
