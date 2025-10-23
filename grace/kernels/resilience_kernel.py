"""
Grace AI Resilience Kernel - Self-healing and fault tolerance
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ResilienceKernel:
    """Manages system resilience, self-healing, and fault tolerance."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.failure_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[str, callable] = {}
    
    async def check_component_health(self, component_name: str) -> HealthStatus:
        """Check the health of a component."""
        health_data = self.component_health.get(component_name, {})
        status = health_data.get("status", HealthStatus.HEALTHY)
        logger.info(f"Health check for {component_name}: {status.value}")
        return status
    
    async def report_failure(self, component_name: str, error: str, severity: str = "medium"):
        """Report a component failure."""
        failure = {
            "component": component_name,
            "error": error,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        self.failure_history.append(failure)
        logger.warning(f"Failure reported: {component_name} - {error}")
        
        # Attempt recovery
        await self.attempt_recovery(component_name, severity)
    
    async def attempt_recovery(self, component_name: str, severity: str):
        """Attempt to recover from a failure."""
        recovery_strategy = self.recovery_strategies.get(component_name)
        
        if severity == "critical":
            logger.error(f"Critical failure in {component_name}, escalating")
            if self.event_bus:
                await self.event_bus.publish("resilience.critical_failure", {
                    "component": component_name
                })
        elif recovery_strategy:
            logger.info(f"Attempting recovery for {component_name}")
            try:
                await recovery_strategy()
                self.component_health[component_name] = {
                    "status": HealthStatus.HEALTHY,
                    "recovered_at": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Recovery failed for {component_name}: {str(e)}")
    
    def register_recovery_strategy(self, component_name: str, strategy: callable):
        """Register a recovery strategy for a component."""
        self.recovery_strategies[component_name] = strategy
        logger.info(f"Registered recovery strategy for {component_name}")
    
    def get_failure_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failure history."""
        return self.failure_history[-limit:]
