"""Resilience Policy Manager - Manages runtime resilience policies."""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResiliencePolicyManager:
    """Manages resilience policies for services."""
    
    def __init__(self):
        self.policies: Dict[str, Dict] = {}
        
    def set_policy(self, service_id: str, policy: Dict) -> None:
        """Set resilience policy for a service."""
        # Validate policy structure
        self._validate_resilience_policy(policy)
        
        self.policies[service_id] = {
            **policy,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Resilience policy set for service {service_id}")
    
    def get_policy(self, service_id: str) -> Optional[Dict]:
        """Get resilience policy for a service."""
        return self.policies.get(service_id)
    
    def get_all_services(self) -> List[str]:
        """Get all services with resilience policies."""
        return list(self.policies.keys())
    
    def update_policy(self, service_id: str, updates: Dict) -> None:
        """Update resilience policy for a service."""
        if service_id not in self.policies:
            raise ValueError(f"No resilience policy found for service {service_id}")
        
        policy = self.policies[service_id].copy()
        policy.update(updates)
        policy["updated_at"] = datetime.utcnow().isoformat()
        
        # Validate updated policy
        self._validate_resilience_policy(policy)
        
        self.policies[service_id] = policy
        logger.info(f"Resilience policy updated for service {service_id}")
    
    def delete_policy(self, service_id: str) -> None:
        """Delete resilience policy for a service."""
        if service_id in self.policies:
            del self.policies[service_id]
            logger.info(f"Resilience policy deleted for service {service_id}")
    
    def _validate_resilience_policy(self, policy: Dict) -> None:
        """Validate resilience policy structure."""
        if "service_id" not in policy:
            raise ValueError("Missing required field: service_id")
        
        # Validate retries configuration
        if "retries" in policy:
            self._validate_retries(policy["retries"])
        
        # Validate circuit breaker configuration
        if "circuit_breaker" in policy:
            self._validate_circuit_breaker(policy["circuit_breaker"])
        
        # Validate rate limiting configuration
        if "rate_limit" in policy:
            self._validate_rate_limit(policy["rate_limit"])
        
        # Validate bulkhead configuration
        if "bulkhead" in policy:
            self._validate_bulkhead(policy["bulkhead"])
        
        # Validate degradation modes
        if "degradation_modes" in policy:
            self._validate_degradation_modes(policy["degradation_modes"])
    
    def _validate_retries(self, retries: Dict) -> None:
        """Validate retry configuration."""
        if not isinstance(retries, dict):
            raise ValueError("Retries configuration must be a dictionary")
        
        if "max" in retries and not isinstance(retries["max"], int):
            raise ValueError("Retry max must be an integer")
        
        if "backoff" in retries and retries["backoff"] not in ["exp", "lin"]:
            raise ValueError("Retry backoff must be 'exp' or 'lin'")
        
        numeric_fields = ["base_ms", "jitter_ms"]
        for field in numeric_fields:
            if field in retries and not isinstance(retries[field], (int, float)):
                raise ValueError(f"Retry {field} must be a number")
    
    def _validate_circuit_breaker(self, breaker: Dict) -> None:
        """Validate circuit breaker configuration."""
        if not isinstance(breaker, dict):
            raise ValueError("Circuit breaker configuration must be a dictionary")
        
        numeric_fields = [
            "failure_rate_threshold_pct",
            "request_volume_threshold", 
            "sleep_window_ms",
            "half_open_max_calls"
        ]
        
        for field in numeric_fields:
            if field in breaker and not isinstance(breaker[field], (int, float)):
                raise ValueError(f"Circuit breaker {field} must be a number")
        
        if "failure_rate_threshold_pct" in breaker:
            if not (0 <= breaker["failure_rate_threshold_pct"] <= 100):
                raise ValueError("Failure rate threshold must be between 0 and 100")
    
    def _validate_rate_limit(self, rate_limit: Dict) -> None:
        """Validate rate limiting configuration."""
        if not isinstance(rate_limit, dict):
            raise ValueError("Rate limit configuration must be a dictionary")
        
        numeric_fields = ["rps", "burst"]
        for field in numeric_fields:
            if field in rate_limit and not isinstance(rate_limit[field], (int, float)):
                raise ValueError(f"Rate limit {field} must be a number")
            
            if field in rate_limit and rate_limit[field] <= 0:
                raise ValueError(f"Rate limit {field} must be positive")
    
    def _validate_bulkhead(self, bulkhead: Dict) -> None:
        """Validate bulkhead configuration."""
        if not isinstance(bulkhead, dict):
            raise ValueError("Bulkhead configuration must be a dictionary")
        
        numeric_fields = ["max_concurrent", "queue_len"]
        for field in numeric_fields:
            if field in bulkhead and not isinstance(bulkhead[field], int):
                raise ValueError(f"Bulkhead {field} must be an integer")
            
            if field in bulkhead and bulkhead[field] <= 0:
                raise ValueError(f"Bulkhead {field} must be positive")
    
    def _validate_degradation_modes(self, modes: List) -> None:
        """Validate degradation modes configuration."""
        if not isinstance(modes, list):
            raise ValueError("Degradation modes must be a list")
        
        valid_triggers = ["high_latency", "high_error", "drift", "dependency_down", "low_budget"]
        
        for mode in modes:
            if not isinstance(mode, dict):
                raise ValueError("Each degradation mode must be a dictionary")
            
            if "mode_id" not in mode:
                raise ValueError("Degradation mode must have mode_id")
            
            if "triggers" in mode:
                if not isinstance(mode["triggers"], list):
                    raise ValueError("Degradation triggers must be a list")
                
                for trigger in mode["triggers"]:
                    if trigger not in valid_triggers:
                        raise ValueError(f"Invalid trigger: {trigger}. Must be one of {valid_triggers}")
            
            if "actions" in mode and not isinstance(mode["actions"], list):
                raise ValueError("Degradation actions must be a list")
    
    def get_retry_config(self, service_id: str) -> Optional[Dict]:
        """Get retry configuration for a service."""
        policy = self.get_policy(service_id)
        return policy.get("retries") if policy else None
    
    def get_circuit_breaker_config(self, service_id: str) -> Optional[Dict]:
        """Get circuit breaker configuration for a service."""
        policy = self.get_policy(service_id)
        return policy.get("circuit_breaker") if policy else None
    
    def get_rate_limit_config(self, service_id: str) -> Optional[Dict]:
        """Get rate limiting configuration for a service."""
        policy = self.get_policy(service_id)
        return policy.get("rate_limit") if policy else None
    
    def get_bulkhead_config(self, service_id: str) -> Optional[Dict]:
        """Get bulkhead configuration for a service."""
        policy = self.get_policy(service_id)
        return policy.get("bulkhead") if policy else None
    
    def get_degradation_modes(self, service_id: str) -> List[Dict]:
        """Get degradation modes for a service."""
        policy = self.get_policy(service_id)
        return policy.get("degradation_modes", []) if policy else []
    
    def find_degradation_mode_for_trigger(self, service_id: str, trigger: str) -> Optional[Dict]:
        """Find the first degradation mode that matches a trigger."""
        modes = self.get_degradation_modes(service_id)
        
        for mode in modes:
            if trigger in mode.get("triggers", []):
                return mode
        
        return None
    
    def get_default_policy(self) -> Dict:
        """Get default resilience policy."""
        return {
            "service_id": "default",
            "retries": {
                "max": 2,
                "backoff": "exp",
                "base_ms": 50,
                "jitter_ms": 20
            },
            "circuit_breaker": {
                "failure_rate_threshold_pct": 25,
                "request_volume_threshold": 40,
                "sleep_window_ms": 6000,
                "half_open_max_calls": 10
            },
            "rate_limit": {
                "rps": 200,
                "burst": 400
            },
            "bulkhead": {
                "max_concurrent": 128,
                "queue_len": 512
            },
            "degradation_modes": [
                {
                    "mode_id": "lite_explanations",
                    "triggers": ["high_latency"],
                    "actions": ["disable_explain", "reduce_batch"]
                },
                {
                    "mode_id": "cached_only",
                    "triggers": ["dependency_down"],
                    "actions": ["use_cache", "shed_load"]
                }
            ]
        }
    
    def apply_defaults(self, service_id: str) -> None:
        """Apply default resilience policy to a service."""
        default_policy = self.get_default_policy()
        default_policy["service_id"] = service_id
        self.set_policy(service_id, default_policy)
    
    def export_policies(self) -> Dict[str, Any]:
        """Export all resilience policies."""
        return {
            "policies": self.policies,
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    
    def import_policies(self, data: Dict[str, Any]) -> None:
        """Import resilience policies."""
        if "policies" not in data:
            raise ValueError("Invalid import data: missing policies")
        
        for service_id, policy in data["policies"].items():
            try:
                self._validate_resilience_policy(policy)
                self.policies[service_id] = policy
                logger.info(f"Imported resilience policy for service {service_id}")
            except Exception as e:
                logger.error(f"Failed to import policy for {service_id}: {e}")
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resilience policy manager statistics."""
        total_modes = 0
        trigger_counts = {}
        
        for policy in self.policies.values():
            modes = policy.get("degradation_modes", [])
            total_modes += len(modes)
            
            for mode in modes:
                for trigger in mode.get("triggers", []):
                    trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        features_count = {}
        for policy in self.policies.values():
            for feature in ["retries", "circuit_breaker", "rate_limit", "bulkhead"]:
                if feature in policy:
                    features_count[feature] = features_count.get(feature, 0) + 1
        
        return {
            "total_services": len(self.policies),
            "total_degradation_modes": total_modes,
            "trigger_distribution": trigger_counts,
            "feature_usage": features_count,
            "average_modes_per_service": total_modes / len(self.policies) if self.policies else 0
        }