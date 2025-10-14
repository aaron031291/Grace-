"""Resilience kernel - health checking and healing."""

import time
from typing import Dict
from datetime import datetime


class ResilienceKernel:
    """Handles system health monitoring and healing."""

    def __init__(self):
        self.health_checks = {}
        self.healing_actions = {}
        self.last_check = None

    def register_health_check(self, name: str, check_func):
        """Register a health check function."""
        self.health_checks[name] = {
            "function": check_func,
            "last_result": None,
            "last_check": None,
            "failure_count": 0,
        }

    def register_healing_action(self, name: str, heal_func):
        """Register a healing action."""
        self.healing_actions[name] = heal_func

    def check_health(self) -> Dict:
        """Run all health checks."""
        self.last_check = time.time()
        results = {}

        for name, check_info in self.health_checks.items():
            try:
                result = check_info["function"]()
                check_info["last_result"] = result
                check_info["last_check"] = self.last_check

                if result:
                    check_info["failure_count"] = 0
                    results[name] = "healthy"
                else:
                    check_info["failure_count"] += 1
                    results[name] = "unhealthy"

            except Exception:
                check_info["failure_count"] += 1
                results[name] = "error"

        overall_health = (
            "healthy" if all(r == "healthy" for r in results.values()) else "degraded"
        )

        return {
            "overall": overall_health,
            "timestamp": datetime.fromtimestamp(self.last_check).isoformat(),
            "checks": results,
        }

    def heal(self) -> Dict:
        """Attempt healing actions for unhealthy components."""
        healing_results = {}

        for name, check_info in self.health_checks.items():
            if check_info.get("failure_count", 0) > 2:  # 3 consecutive failures
                if name in self.healing_actions:
                    try:
                        self.healing_actions[name]()
                        healing_results[name] = "healing_attempted"
                    except Exception:
                        healing_results[name] = "healing_failed"

        return healing_results

    def get_stats(self) -> Dict:
        """Get resilience statistics."""
        return {
            "registered_health_checks": len(self.health_checks),
            "registered_healing_actions": len(self.healing_actions),
            "last_check": datetime.fromtimestamp(self.last_check).isoformat()
            if self.last_check
            else None,
            "health_status": {
                name: {
                    "last_result": info["last_result"],
                    "failure_count": info["failure_count"],
                }
                for name, info in self.health_checks.items()
            },
        }
