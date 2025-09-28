"""SLI (Service Level Indicator) evaluation and monitoring."""

import time
import statistics
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
import logging

logger = logging.getLogger(__name__)


def evaluate_sli(samples: dict, slo: dict) -> dict:
    """
    Evaluate SLI against SLO thresholds.
    
    Args:
        samples: Raw metric samples (e.g., {"latency_ms": [120, 150, 180], "timestamp": [...]})
        slo: SLO configuration with thresholds
        
    Returns:
        Evaluation result with status and metrics
    """
    try:
        sli_type = slo["sli"]
        objective = slo["objective"]
        window = slo["window"]
        
        if sli_type == "latency_p95_ms":
            return _evaluate_latency_p95(samples, objective, window)
        elif sli_type == "availability_pct":
            return _evaluate_availability(samples, objective, window)
        elif sli_type == "error_rate_pct":
            return _evaluate_error_rate(samples, objective, window)
        elif sli_type == "drift_psi":
            return _evaluate_drift_psi(samples, objective, window)
        else:
            return {
                "sli": sli_type,
                "status": "error",
                "message": f"Unknown SLI type: {sli_type}"
            }
    
    except Exception as e:
        logger.error(f"Failed to evaluate SLI {slo.get('sli', 'unknown')}: {e}")
        return {
            "sli": slo.get("sli", "unknown"),
            "status": "error",
            "message": str(e)
        }


def _evaluate_latency_p95(samples: dict, objective: float, window: str) -> dict:
    """Evaluate P95 latency SLI."""
    latency_samples = samples.get("latency_ms", [])
    
    if not latency_samples:
        return {
            "sli": "latency_p95_ms",
            "status": "insufficient_data",
            "message": "No latency samples available"
        }
    
    # Calculate P95 latency
    p95_latency = statistics.quantiles(latency_samples, n=20)[18]  # 95th percentile
    
    # Check against objective (e.g., ≤ 800ms)
    meets_slo = p95_latency <= objective
    
    return {
        "sli": "latency_p95_ms",
        "status": "pass" if meets_slo else "fail",
        "current_value": p95_latency,
        "objective": objective,
        "window": window,
        "sample_count": len(latency_samples),
        "breach_margin": max(0, p95_latency - objective),
        "evaluated_at": iso_format()
    }


def _evaluate_availability(samples: dict, objective: float, window: str) -> dict:
    """Evaluate availability SLI."""
    success_count = samples.get("success_count", [0])
    total_count = samples.get("total_count", [0])
    
    # Handle both list and scalar inputs
    if isinstance(success_count, list):
        success_count = sum(success_count) if success_count else 0
    if isinstance(total_count, list):
        total_count = sum(total_count) if total_count else 0
    
    if total_count == 0:
        return {
            "sli": "availability_pct",
            "status": "insufficient_data",
            "message": "No availability samples available"
        }
    
    # Calculate availability percentage
    availability_pct = (success_count / total_count) * 100
    
    # Check against objective (e.g., ≥ 99.9%)
    meets_slo = availability_pct >= objective
    
    return {
        "sli": "availability_pct",
        "status": "pass" if meets_slo else "fail",
        "current_value": availability_pct,
        "objective": objective,
        "window": window,
        "success_count": success_count,
        "total_count": total_count,
        "breach_margin": max(0, objective - availability_pct),
        "evaluated_at": iso_format()
    }


def _evaluate_error_rate(samples: dict, objective: float, window: str) -> dict:
    """Evaluate error rate SLI."""
    error_count = samples.get("error_count", [0])
    total_count = samples.get("total_count", [0])
    
    # Handle both list and scalar inputs
    if isinstance(error_count, list):
        error_count = sum(error_count) if error_count else 0
    if isinstance(total_count, list):
        total_count = sum(total_count) if total_count else 0
    
    if total_count == 0:
        return {
            "sli": "error_rate_pct",
            "status": "insufficient_data",
            "message": "No error rate samples available"
        }
    
    # Calculate error rate percentage
    error_rate_pct = (error_count / total_count) * 100
    
    # Check against objective (e.g., ≤ 1.0%)
    meets_slo = error_rate_pct <= objective
    
    return {
        "sli": "error_rate_pct",
        "status": "pass" if meets_slo else "fail",
        "current_value": error_rate_pct,
        "objective": objective,
        "window": window,
        "error_count": error_count,
        "total_count": total_count,
        "breach_margin": max(0, error_rate_pct - objective),
        "evaluated_at": iso_format()
    }


def _evaluate_drift_psi(samples: dict, objective: float, window: str) -> dict:
    """Evaluate Population Stability Index (PSI) for drift detection."""
    expected_dist = samples.get("expected_distribution", [])
    actual_dist = samples.get("actual_distribution", [])
    
    if not expected_dist or not actual_dist:
        return {
            "sli": "drift_psi",
            "status": "insufficient_data",
            "message": "Insufficient distribution data for PSI calculation"
        }
    
    if len(expected_dist) != len(actual_dist):
        return {
            "sli": "drift_psi",
            "status": "error",
            "message": "Expected and actual distributions have different lengths"
        }
    
    # Calculate PSI
    psi = 0.0
    for expected, actual in zip(expected_dist, actual_dist):
        if expected > 0 and actual > 0:
            psi += (actual - expected) * (log(actual / expected) if actual > 0 and expected > 0 else 0)
    
    # Check against objective (e.g., ≤ 0.1)
    meets_slo = psi <= objective
    
    return {
        "sli": "drift_psi",
        "status": "pass" if meets_slo else "fail",
        "current_value": psi,
        "objective": objective,
        "window": window,
        "breach_margin": max(0, psi - objective),
        "evaluated_at": iso_format()
    }


def log(x):
    """Safe logarithm function."""
    import math
    return math.log(x) if x > 0 else 0


class SLIMonitor:
    """
    SLI monitoring and evaluation service.
    
    Continuously monitors service level indicators and evaluates them
    against SLO objectives to detect breaches and degradation.
    """
    
    def __init__(self):
        """Initialize SLI monitor."""
        self.slo_policies: Dict[str, Dict] = {}
        self.sample_buffers: Dict[str, List[Dict]] = {}
        self.evaluation_history: List[Dict] = []
        
        logger.debug("SLI monitor initialized")
    
    def register_slo(self, service_id: str, slo_policy: Dict):
        """Register SLO policy for monitoring."""
        self.slo_policies[service_id] = slo_policy
        self.sample_buffers[service_id] = []
        logger.info(f"Registered SLO policy for service {service_id}")
    
    def add_samples(self, service_id: str, samples: Dict[str, Any]):
        """Add metric samples for a service."""
        if service_id not in self.sample_buffers:
            self.sample_buffers[service_id] = []
        
        timestamp = utc_now()
        sample_entry = {
            "timestamp": timestamp.isoformat(),
            "samples": samples
        }
        
        self.sample_buffers[service_id].append(sample_entry)
        
        # Keep buffer size limited
        if len(self.sample_buffers[service_id]) > 1000:
            self.sample_buffers[service_id] = self.sample_buffers[service_id][-500:]
    
    def evaluate_service_slos(self, service_id: str) -> List[Dict]:
        """Evaluate all SLOs for a service."""
        if service_id not in self.slo_policies:
            return []
        
        policy = self.slo_policies[service_id]
        slos = policy.get("slos", [])
        samples = self._get_windowed_samples(service_id, "30d")  # Default window
        
        evaluations = []
        
        for slo in slos:
            result = evaluate_sli(samples, slo)
            result["service_id"] = service_id
            evaluations.append(result)
            
            # Store in history
            self.evaluation_history.append(result)
        
        # Trim history
        if len(self.evaluation_history) > 10000:
            self.evaluation_history = self.evaluation_history[-5000:]
        
        return evaluations
    
    def get_current_status(self, service_id: str) -> Dict:
        """Get current SLO status for a service."""
        evaluations = self.evaluate_service_slos(service_id)
        
        if not evaluations:
            return {
                "service_id": service_id,
                "status": "no_slos_configured",
                "evaluations": []
            }
        
        # Determine overall status
        statuses = [e["status"] for e in evaluations]
        
        if "error" in statuses:
            overall_status = "error"
        elif "insufficient_data" in statuses:
            overall_status = "insufficient_data"
        elif "fail" in statuses:
            overall_status = "breach"
        else:
            overall_status = "healthy"
        
        return {
            "service_id": service_id,
            "status": overall_status,
            "evaluations": evaluations,
            "evaluated_at": iso_format()
        }
    
    def get_breach_summary(self, service_id: Optional[str] = None, hours: int = 24) -> Dict:
        """Get summary of SLO breaches."""
        cutoff_time = utc_now() - timedelta(hours=hours)
        
        recent_evaluations = [
            e for e in self.evaluation_history
            if datetime.fromisoformat(e["evaluated_at"]) >= cutoff_time
        ]
        
        if service_id:
            recent_evaluations = [e for e in recent_evaluations if e.get("service_id") == service_id]
        
        breaches = [e for e in recent_evaluations if e["status"] == "fail"]
        
        # Group breaches by service and SLI
        breach_summary = {}
        for breach in breaches:
            svc_id = breach.get("service_id", "unknown")
            sli = breach["sli"]
            
            if svc_id not in breach_summary:
                breach_summary[svc_id] = {}
            
            if sli not in breach_summary[svc_id]:
                breach_summary[svc_id][sli] = {
                    "count": 0,
                    "max_breach_margin": 0,
                    "latest_breach": None
                }
            
            breach_summary[svc_id][sli]["count"] += 1
            breach_summary[svc_id][sli]["max_breach_margin"] = max(
                breach_summary[svc_id][sli]["max_breach_margin"],
                breach.get("breach_margin", 0)
            )
            breach_summary[svc_id][sli]["latest_breach"] = breach["evaluated_at"]
        
        return {
            "time_window_hours": hours,
            "total_breaches": len(breaches),
            "services_affected": len(breach_summary),
            "breach_summary": breach_summary,
            "generated_at": iso_format()
        }
    
    def _get_windowed_samples(self, service_id: str, window: str) -> Dict:
        """Get samples within a time window."""
        if service_id not in self.sample_buffers:
            return {}
        
        # Parse window (e.g., "7d", "30d", "1h")
        window_seconds = self._parse_window(window)
        cutoff_time = utc_now() - timedelta(seconds=window_seconds)
        
        recent_samples = [
            entry for entry in self.sample_buffers[service_id]
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]
        
        if not recent_samples:
            return {}
        
        # Aggregate samples
        aggregated = {}
        for entry in recent_samples:
            samples = entry["samples"]
            for key, value in samples.items():
                if key not in aggregated:
                    aggregated[key] = []
                
                if isinstance(value, list):
                    aggregated[key].extend(value)
                else:
                    aggregated[key].append(value)
        
        return aggregated
    
    def _parse_window(self, window: str) -> int:
        """Parse time window string to seconds."""
        if window.endswith("d"):
            return int(window[:-1]) * 24 * 3600
        elif window.endswith("h"):
            return int(window[:-1]) * 3600
        elif window.endswith("m"):
            return int(window[:-1]) * 60
        else:
            # Default to 30 days
            return 30 * 24 * 3600