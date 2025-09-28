"""SLO Policy Manager - Manages Service Level Objectives and error budgets."""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SLOManager:
    """Manages SLO policies for services."""
    
    def __init__(self):
        self.policies: Dict[str, Dict] = {}
        
    def set_policy(self, service_id: str, policy: Dict) -> None:
        """Set SLO policy for a service."""
        # Validate policy structure
        self._validate_slo_policy(policy)
        
        self.policies[service_id] = {
            **policy,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"SLO policy set for service {service_id}")
    
    def get_policy(self, service_id: str) -> Optional[Dict]:
        """Get SLO policy for a service."""
        return self.policies.get(service_id)
    
    def get_all_services(self) -> List[str]:
        """Get all services with SLO policies."""
        return list(self.policies.keys())
    
    def update_policy(self, service_id: str, updates: Dict) -> None:
        """Update SLO policy for a service."""
        if service_id not in self.policies:
            raise ValueError(f"No SLO policy found for service {service_id}")
        
        policy = self.policies[service_id].copy()
        policy.update(updates)
        policy["updated_at"] = datetime.utcnow().isoformat()
        
        # Validate updated policy
        self._validate_slo_policy(policy)
        
        self.policies[service_id] = policy
        logger.info(f"SLO policy updated for service {service_id}")
    
    def delete_policy(self, service_id: str) -> None:
        """Delete SLO policy for a service."""
        if service_id in self.policies:
            del self.policies[service_id]
            logger.info(f"SLO policy deleted for service {service_id}")
    
    def _validate_slo_policy(self, policy: Dict) -> None:
        """Validate SLO policy structure."""
        required_fields = ["service_id", "slos", "error_budget_days"]
        
        for field in required_fields:
            if field not in policy:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate SLOs
        if not isinstance(policy["slos"], list) or not policy["slos"]:
            raise ValueError("SLOs must be a non-empty list")
        
        valid_slis = ["latency_p95_ms", "availability_pct", "error_rate_pct", "drift_psi"]
        
        for slo in policy["slos"]:
            if not isinstance(slo, dict):
                raise ValueError("Each SLO must be a dictionary")
            
            slo_required = ["sli", "objective", "window"]
            for field in slo_required:
                if field not in slo:
                    raise ValueError(f"Missing required SLO field: {field}")
            
            if slo["sli"] not in valid_slis:
                raise ValueError(f"Invalid SLI: {slo['sli']}. Must be one of {valid_slis}")
            
            if not isinstance(slo["objective"], (int, float)):
                raise ValueError("SLO objective must be a number")
            
            # Validate objective ranges based on SLI type
            if slo["sli"] == "availability_pct" and not (0 <= slo["objective"] <= 100):
                raise ValueError("Availability percentage must be between 0 and 100")
            
            if slo["sli"] == "error_rate_pct" and not (0 <= slo["objective"] <= 100):
                raise ValueError("Error rate percentage must be between 0 and 100")
        
        # Validate error budget
        if not isinstance(policy["error_budget_days"], (int, float)) or policy["error_budget_days"] <= 0:
            raise ValueError("Error budget days must be a positive number")
    
    def get_slo_by_sli(self, service_id: str, sli: str) -> Optional[Dict]:
        """Get specific SLO by SLI type for a service."""
        policy = self.get_policy(service_id)
        if not policy:
            return None
        
        for slo in policy["slos"]:
            if slo["sli"] == sli:
                return slo
        
        return None
    
    def is_slo_violated(self, service_id: str, sli: str, actual_value: float) -> bool:
        """Check if an SLO is violated."""
        slo = self.get_slo_by_sli(service_id, sli)
        if not slo:
            return False
        
        objective = slo["objective"]
        
        # For availability and success metrics, actual should be >= objective
        if sli in ["availability_pct"]:
            return actual_value < objective
        
        # For error rates and latency, actual should be <= objective
        if sli in ["error_rate_pct", "latency_p95_ms", "drift_psi"]:
            return actual_value > objective
        
        return False
    
    def get_violation_severity(self, service_id: str, sli: str, actual_value: float) -> Optional[str]:
        """Get violation severity level."""
        if not self.is_slo_violated(service_id, sli, actual_value):
            return None
        
        slo = self.get_slo_by_sli(service_id, sli)
        if not slo:
            return None
        
        objective = slo["objective"]
        
        if sli == "availability_pct":
            if actual_value < 95.0:
                return "critical"
            elif actual_value < 98.0:
                return "high"
            elif actual_value < 99.0:
                return "medium"
            else:
                return "low"
        
        elif sli == "error_rate_pct":
            deviation = actual_value - objective
            if deviation > 10.0:
                return "critical"
            elif deviation > 5.0:
                return "high"
            elif deviation > 2.0:
                return "medium"
            else:
                return "low"
        
        elif sli == "latency_p95_ms":
            deviation_pct = (actual_value - objective) / objective * 100
            if deviation_pct > 200:
                return "critical"
            elif deviation_pct > 100:
                return "high"
            elif deviation_pct > 50:
                return "medium"
            else:
                return "low"
        
        elif sli == "drift_psi":
            if actual_value > 0.5:
                return "critical"
            elif actual_value > 0.3:
                return "high"
            elif actual_value > 0.1:
                return "medium"
            else:
                return "low"
        
        return "low"
    
    def export_policies(self) -> Dict[str, Any]:
        """Export all SLO policies."""
        return {
            "policies": self.policies,
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    
    def import_policies(self, data: Dict[str, Any]) -> None:
        """Import SLO policies."""
        if "policies" not in data:
            raise ValueError("Invalid import data: missing policies")
        
        for service_id, policy in data["policies"].items():
            try:
                self._validate_slo_policy(policy)
                self.policies[service_id] = policy
                logger.info(f"Imported SLO policy for service {service_id}")
            except Exception as e:
                logger.error(f"Failed to import policy for {service_id}: {e}")
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SLO manager statistics."""
        total_slos = sum(len(policy["slos"]) for policy in self.policies.values())
        
        sli_counts = {}
        for policy in self.policies.values():
            for slo in policy["slos"]:
                sli = slo["sli"]
                sli_counts[sli] = sli_counts.get(sli, 0) + 1
        
        return {
            "total_services": len(self.policies),
            "total_slos": total_slos,
            "sli_distribution": sli_counts,
            "average_slos_per_service": total_slos / len(self.policies) if self.policies else 0
        }