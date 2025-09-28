"""Degradation Controller - Manages graceful degradation modes."""
from typing import Dict, Set, List, Optional, Any
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class DegradationController:
    """Manages graceful degradation modes for services."""
    
    def __init__(self):
        # Track which services are in which degradation modes
        self.active_degradations: Dict[str, Set[str]] = {}  # service_id -> set of mode_ids
        self.degradation_history: List[Dict] = []
        self.mode_configs: Dict[str, Dict] = {}  # mode_id -> configuration
        
        # Default degradation modes
        self._initialize_default_modes()
    
    def _initialize_default_modes(self):
        """Initialize default degradation modes."""
        self.mode_configs.update({
            "lite_explanations": {
                "mode_id": "lite_explanations",
                "description": "Reduce explanation complexity to improve latency",
                "triggers": ["high_latency"],
                "actions": ["disable_explain", "reduce_batch"],
                "impact_level": "low",
                "auto_exit_conditions": ["latency_ok"]
            },
            "cached_only": {
                "mode_id": "cached_only", 
                "description": "Use only cached responses, disable real-time processing",
                "triggers": ["dependency_down", "high_error"],
                "actions": ["use_cache", "shed_load"],
                "impact_level": "medium",
                "auto_exit_conditions": ["dependency_ok", "error_rate_ok"]
            },
            "minimal_features": {
                "mode_id": "minimal_features",
                "description": "Disable non-essential features to preserve core functionality",
                "triggers": ["high_error", "low_budget"],
                "actions": ["disable_features", "reduce_quality"],
                "impact_level": "medium", 
                "auto_exit_conditions": ["error_rate_ok", "budget_recovered"]
            },
            "emergency_mode": {
                "mode_id": "emergency_mode",
                "description": "Minimal functionality only, maximum preservation",
                "triggers": ["system_overload", "critical_failure"],
                "actions": ["disable_all_non_essential", "emergency_responses_only"],
                "impact_level": "high",
                "auto_exit_conditions": ["system_stable", "manual_override"]
            }
        })
    
    def enter_mode(self, service_id: str, mode_id: str, reason: str = "manual") -> Dict[str, Any]:
        """Enter a degradation mode for a service."""
        if service_id not in self.active_degradations:
            self.active_degradations[service_id] = set()
        
        if mode_id in self.active_degradations[service_id]:
            logger.info(f"Service {service_id} already in degradation mode {mode_id}")
            return {"already_active": True, "mode_id": mode_id}
        
        # Add to active degradations
        self.active_degradations[service_id].add(mode_id)
        
        # Record in history
        entry = {
            "service_id": service_id,
            "mode_id": mode_id,
            "action": "enter",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.mode_configs.get(mode_id, {})
        }
        self.degradation_history.append(entry)
        
        # Execute degradation actions
        actions_executed = self._execute_degradation_actions(service_id, mode_id, "enter")
        
        logger.warning(f"Service {service_id} entered degradation mode {mode_id} (reason: {reason})")
        
        return {
            "service_id": service_id,
            "mode_id": mode_id,
            "entered_at": entry["timestamp"],
            "actions_executed": actions_executed,
            "reason": reason
        }
    
    def exit_mode(self, service_id: str, mode_id: str, reason: str = "manual") -> Dict[str, Any]:
        """Exit a degradation mode for a service."""
        if service_id not in self.active_degradations:
            return {"error": f"No active degradations for service {service_id}"}
        
        if mode_id not in self.active_degradations[service_id]:
            return {"error": f"Service {service_id} not in degradation mode {mode_id}"}
        
        # Remove from active degradations
        self.active_degradations[service_id].remove(mode_id)
        
        # Clean up empty sets
        if not self.active_degradations[service_id]:
            del self.active_degradations[service_id]
        
        # Record in history
        entry = {
            "service_id": service_id,
            "mode_id": mode_id,
            "action": "exit",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.degradation_history.append(entry)
        
        # Execute recovery actions
        actions_executed = self._execute_degradation_actions(service_id, mode_id, "exit")
        
        logger.info(f"Service {service_id} exited degradation mode {mode_id} (reason: {reason})")
        
        return {
            "service_id": service_id,
            "mode_id": mode_id,
            "exited_at": entry["timestamp"],
            "actions_executed": actions_executed,
            "reason": reason
        }
    
    def exit_all_modes(self, service_id: str, reason: str = "recovery") -> Dict[str, Any]:
        """Exit all degradation modes for a service."""
        if service_id not in self.active_degradations:
            return {"error": f"No active degradations for service {service_id}"}
        
        modes_to_exit = list(self.active_degradations[service_id])
        results = []
        
        for mode_id in modes_to_exit:
            result = self.exit_mode(service_id, mode_id, reason)
            results.append(result)
        
        return {
            "service_id": service_id,
            "exited_modes": modes_to_exit,
            "results": results,
            "reason": reason
        }
    
    def is_in_mode(self, service_id: str, mode_id: str) -> bool:
        """Check if service is in specific degradation mode."""
        return (service_id in self.active_degradations and 
                mode_id in self.active_degradations[service_id])
    
    def get_active_modes(self, service_id: str) -> Set[str]:
        """Get all active degradation modes for a service."""
        return self.active_degradations.get(service_id, set()).copy()
    
    def get_all_active_degradations(self) -> Dict[str, Set[str]]:
        """Get all active degradations across all services."""
        return {k: v.copy() for k, v in self.active_degradations.items()}
    
    def add_mode_config(self, mode_config: Dict) -> None:
        """Add or update a degradation mode configuration."""
        mode_id = mode_config["mode_id"]
        self.mode_configs[mode_id] = mode_config
        logger.info(f"Added/updated degradation mode config: {mode_id}")
    
    def get_mode_config(self, mode_id: str) -> Optional[Dict]:
        """Get configuration for a degradation mode."""
        return self.mode_configs.get(mode_id)
    
    def should_auto_enter_mode(self, service_id: str, trigger: str) -> List[str]:
        """Get degradation modes that should be auto-entered for a trigger."""
        matching_modes = []
        
        for mode_id, config in self.mode_configs.items():
            # Skip if already in this mode
            if self.is_in_mode(service_id, mode_id):
                continue
            
            # Check if trigger matches
            if trigger in config.get("triggers", []):
                matching_modes.append(mode_id)
        
        # Sort by impact level (low impact modes first)
        impact_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        matching_modes.sort(key=lambda m: impact_order.get(
            self.mode_configs[m].get("impact_level", "medium"), 1
        ))
        
        return matching_modes
    
    def should_auto_exit_mode(self, service_id: str, condition: str) -> List[str]:
        """Get degradation modes that should be auto-exited for a condition."""
        matching_modes = []
        
        if service_id not in self.active_degradations:
            return matching_modes
        
        for mode_id in self.active_degradations[service_id]:
            config = self.mode_configs.get(mode_id, {})
            exit_conditions = config.get("auto_exit_conditions", [])
            
            if condition in exit_conditions:
                matching_modes.append(mode_id)
        
        return matching_modes
    
    def _execute_degradation_actions(self, service_id: str, mode_id: str, direction: str) -> List[str]:
        """Execute actions for entering or exiting degradation mode."""
        config = self.mode_configs.get(mode_id, {})
        actions = config.get("actions", [])
        
        executed_actions = []
        
        for action in actions:
            try:
                if direction == "enter":
                    result = self._apply_degradation_action(service_id, action)
                else:
                    result = self._revert_degradation_action(service_id, action)
                
                executed_actions.append(f"{action}: {result}")
                logger.debug(f"Executed {direction} action '{action}' for {service_id}: {result}")
                
            except Exception as e:
                logger.error(f"Failed to execute {direction} action '{action}' for {service_id}: {e}")
                executed_actions.append(f"{action}: failed - {str(e)}")
        
        return executed_actions
    
    def _apply_degradation_action(self, service_id: str, action: str) -> str:
        """Apply a specific degradation action."""
        # This would integrate with actual service controls in production
        if action == "disable_explain":
            return "Disabled explanation generation"
        elif action == "reduce_batch":
            return "Reduced batch size for processing"
        elif action == "use_cache":
            return "Switched to cached responses only"
        elif action == "shed_load":
            return "Enabled load shedding"
        elif action == "disable_features":
            return "Disabled non-essential features"
        elif action == "reduce_quality":
            return "Reduced output quality for speed"
        elif action == "disable_all_non_essential":
            return "Disabled all non-essential functionality"
        elif action == "emergency_responses_only":
            return "Limited to emergency response patterns"
        else:
            return f"Applied custom action: {action}"
    
    def _revert_degradation_action(self, service_id: str, action: str) -> str:
        """Revert a specific degradation action."""
        # This would integrate with actual service controls in production
        if action == "disable_explain":
            return "Re-enabled explanation generation"
        elif action == "reduce_batch":
            return "Restored normal batch size"
        elif action == "use_cache":
            return "Re-enabled real-time processing"
        elif action == "shed_load":
            return "Disabled load shedding"
        elif action == "disable_features":
            return "Re-enabled all features"
        elif action == "reduce_quality":
            return "Restored normal output quality"
        elif action == "disable_all_non_essential":
            return "Re-enabled all functionality"
        elif action == "emergency_responses_only":
            return "Restored full response capabilities"
        else:
            return f"Reverted custom action: {action}"
    
    def get_degradation_history(self, service_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get degradation history, optionally filtered by service."""
        history = self.degradation_history
        
        if service_id:
            history = [entry for entry in history if entry["service_id"] == service_id]
        
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get degradation controller statistics."""
        total_services_degraded = len(self.active_degradations)
        total_active_modes = sum(len(modes) for modes in self.active_degradations.values())
        
        mode_usage = {}
        for modes in self.active_degradations.values():
            for mode in modes:
                mode_usage[mode] = mode_usage.get(mode, 0) + 1
        
        return {
            "total_services_degraded": total_services_degraded,
            "total_active_modes": total_active_modes,
            "mode_usage": mode_usage,
            "available_modes": list(self.mode_configs.keys()),
            "degradation_events": len(self.degradation_history)
        }


# Convenience functions
def enter_mode(service_id: str, mode_id: str) -> None:
    """Global function to enter degradation mode."""
    controller = DegradationController()
    controller.enter_mode(service_id, mode_id)

def exit_mode(service_id: str, mode_id: str) -> None:
    """Global function to exit degradation mode."""
    controller = DegradationController()
    controller.exit_mode(service_id, mode_id)