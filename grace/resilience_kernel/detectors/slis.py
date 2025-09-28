"""SLI Evaluator - Service Level Indicator monitoring and evaluation."""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import statistics

logger = logging.getLogger(__name__)


class SLIEvaluator:
    """Evaluates Service Level Indicators against SLO targets."""
    
    def __init__(self):
        self.evaluation_history: Dict[str, List[Dict]] = {}
    
    def evaluate_sli(self, samples: Dict[str, List[float]], slo: Dict) -> Dict[str, Any]:
        """
        Evaluate SLI samples against SLO policy.
        
        Args:
            samples: Dictionary of SLI metrics with sample values
            slo: SLO policy containing objectives and thresholds
            
        Returns:
            Dictionary containing evaluation results, violations, and recommendations
        """
        service_id = slo.get("service_id", "unknown")
        evaluation_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        evaluation_result = {
            "evaluation_id": evaluation_id,
            "service_id": service_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sli_values": {},
            "slo_targets": {},
            "violations": [],
            "warnings": [],
            "compliance_status": "compliant",
            "recommendations": []
        }
        
        # Evaluate each SLO
        for slo_config in slo.get("slos", []):
            sli_type = slo_config["sli"]
            objective = slo_config["objective"]
            window = slo_config.get("window", "30d")
            
            if sli_type in samples and samples[sli_type]:
                # Calculate SLI value from samples
                sli_value = self._calculate_sli_value(sli_type, samples[sli_type])
                evaluation_result["sli_values"][sli_type] = sli_value
                evaluation_result["slo_targets"][sli_type] = objective
                
                # Check for violations
                violation = self._check_violation(sli_type, sli_value, objective, window)
                if violation:
                    evaluation_result["violations"].append(violation)
                    evaluation_result["compliance_status"] = "violated"
                
                # Check for warnings (approaching violation)
                warning = self._check_warning(sli_type, sli_value, objective)
                if warning:
                    evaluation_result["warnings"].append(warning)
                    if evaluation_result["compliance_status"] == "compliant":
                        evaluation_result["compliance_status"] = "at_risk"
                
                # Generate recommendations
                recommendations = self._generate_recommendations(sli_type, sli_value, objective)
                evaluation_result["recommendations"].extend(recommendations)
        
        # Store evaluation history
        if service_id not in self.evaluation_history:
            self.evaluation_history[service_id] = []
        
        self.evaluation_history[service_id].append(evaluation_result)
        
        # Keep only last 100 evaluations per service
        if len(self.evaluation_history[service_id]) > 100:
            self.evaluation_history[service_id] = self.evaluation_history[service_id][-100:]
        
        return evaluation_result
    
    def _calculate_sli_value(self, sli_type: str, samples: List[float]) -> float:
        """Calculate SLI value from samples based on SLI type."""
        if not samples:
            return 0.0
        
        if sli_type == "latency_p95_ms":
            # 95th percentile latency
            return self._percentile(samples, 95)
        
        elif sli_type == "latency_p99_ms":
            # 99th percentile latency
            return self._percentile(samples, 99)
        
        elif sli_type == "latency_avg_ms":
            # Average latency
            return statistics.mean(samples)
        
        elif sli_type in ["availability_pct", "error_rate_pct", "success_rate_pct"]:
            # For percentages, take the average
            return statistics.mean(samples)
        
        elif sli_type == "drift_psi":
            # Population Stability Index - typically single value
            return samples[-1] if samples else 0.0
        
        elif sli_type in ["throughput_rps", "queue_depth", "cpu_util_pct", "memory_util_pct"]:
            # For utilization metrics, take average
            return statistics.mean(samples)
        
        else:
            # Default to mean for unknown metrics
            logger.warning(f"Unknown SLI type {sli_type}, using mean")
            return statistics.mean(samples)
    
    def _percentile(self, samples: List[float], percentile: float) -> float:
        """Calculate percentile value from samples."""
        if not samples:
            return 0.0
        
        sorted_samples = sorted(samples)
        index = (percentile / 100.0) * (len(sorted_samples) - 1)
        
        if index.is_integer():
            return sorted_samples[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index >= len(sorted_samples):
                return sorted_samples[lower_index]
            
            weight = index - lower_index
            return sorted_samples[lower_index] * (1 - weight) + sorted_samples[upper_index] * weight
    
    def _check_violation(self, sli_type: str, sli_value: float, objective: float, window: str) -> Optional[Dict]:
        """Check if SLI value violates SLO objective."""
        violation_detected = False
        
        if sli_type in ["latency_p95_ms", "latency_p99_ms", "latency_avg_ms", "error_rate_pct", "drift_psi"]:
            # For these metrics, lower is better (value should be <= objective)
            violation_detected = sli_value > objective
        
        elif sli_type in ["availability_pct", "success_rate_pct"]:
            # For these metrics, higher is better (value should be >= objective)
            violation_detected = sli_value < objective
        
        elif sli_type in ["throughput_rps"]:
            # For throughput, we expect it to be >= objective
            violation_detected = sli_value < objective
        
        if violation_detected:
            severity = self._assess_violation_severity(sli_type, sli_value, objective)
            
            return {
                "sli": sli_type,
                "actual": sli_value,
                "target": objective,
                "window": window,
                "severity": severity,
                "detected_at": datetime.utcnow().isoformat(),
                "deviation": self._calculate_deviation(sli_type, sli_value, objective)
            }
        
        return None
    
    def _check_warning(self, sli_type: str, sli_value: float, objective: float) -> Optional[Dict]:
        """Check if SLI value is approaching violation (warning threshold)."""
        warning_threshold = self._get_warning_threshold(sli_type, objective)
        
        warning_detected = False
        
        if sli_type in ["latency_p95_ms", "latency_p99_ms", "latency_avg_ms", "error_rate_pct"]:
            # For these metrics, warn when approaching from below
            warning_detected = sli_value > warning_threshold
        
        elif sli_type in ["availability_pct", "success_rate_pct"]:
            # For these metrics, warn when dropping towards threshold
            warning_detected = sli_value < warning_threshold
        
        if warning_detected:
            return {
                "sli": sli_type,
                "actual": sli_value,
                "warning_threshold": warning_threshold,
                "target": objective,
                "message": f"{sli_type} approaching SLO violation threshold"
            }
        
        return None
    
    def _get_warning_threshold(self, sli_type: str, objective: float) -> float:
        """Get warning threshold for an SLI type (typically 80% of the way to violation)."""
        if sli_type in ["latency_p95_ms", "latency_p99_ms", "latency_avg_ms"]:
            # Warn at 80% of objective
            return objective * 0.8
        
        elif sli_type == "error_rate_pct":
            # Warn at 80% of error rate objective
            return objective * 0.8
        
        elif sli_type in ["availability_pct", "success_rate_pct"]:
            # Warn when availability drops to within 80% of violation
            # E.g., if target is 99.9%, warn at 99.92%
            margin = 100 - objective
            warning_margin = margin * 0.2
            return objective + warning_margin
        
        elif sli_type == "drift_psi":
            # Warn at 80% of drift threshold
            return objective * 0.8
        
        else:
            # Default warning threshold
            return objective * 0.8
    
    def _assess_violation_severity(self, sli_type: str, sli_value: float, objective: float) -> str:
        """Assess severity of SLO violation."""
        deviation_pct = abs(self._calculate_deviation(sli_type, sli_value, objective))
        
        if sli_type in ["availability_pct", "success_rate_pct"]:
            # Availability violations are more critical
            if deviation_pct > 5:  # More than 5% below target
                return "critical"
            elif deviation_pct > 2:
                return "high"
            elif deviation_pct > 1:
                return "medium"
            else:
                return "low"
        
        elif sli_type in ["latency_p95_ms", "latency_p99_ms", "latency_avg_ms"]:
            # Latency violations
            if deviation_pct > 200:  # More than 200% above target
                return "critical"
            elif deviation_pct > 100:
                return "high"
            elif deviation_pct > 50:
                return "medium"
            else:
                return "low"
        
        elif sli_type == "error_rate_pct":
            # Error rate violations
            if deviation_pct > 500:  # 5x the target error rate
                return "critical"
            elif deviation_pct > 200:
                return "high"
            elif deviation_pct > 100:
                return "medium"
            else:
                return "low"
        
        elif sli_type == "drift_psi":
            # Model drift violations
            if sli_value > 0.5:
                return "critical"
            elif sli_value > 0.3:
                return "high"
            elif sli_value > 0.1:
                return "medium"
            else:
                return "low"
        
        return "medium"
    
    def _calculate_deviation(self, sli_type: str, sli_value: float, objective: float) -> float:
        """Calculate percentage deviation from objective."""
        if objective == 0:
            return 0.0
        
        if sli_type in ["availability_pct", "success_rate_pct"]:
            # For availability, calculate how much below target
            return ((objective - sli_value) / objective) * 100
        else:
            # For other metrics, calculate how much above target
            return ((sli_value - objective) / objective) * 100
    
    def _generate_recommendations(self, sli_type: str, sli_value: float, objective: float) -> List[str]:
        """Generate recommendations based on SLI performance."""
        recommendations = []
        
        if sli_type in ["latency_p95_ms", "latency_p99_ms", "latency_avg_ms"]:
            if sli_value > objective:
                recommendations.extend([
                    "Consider enabling request caching",
                    "Review database query performance",
                    "Check for resource contention",
                    "Consider horizontal scaling",
                    "Enable circuit breakers for slow dependencies"
                ])
        
        elif sli_type == "error_rate_pct":
            if sli_value > objective:
                recommendations.extend([
                    "Review error logs for patterns",
                    "Check dependency health",
                    "Validate input data quality",
                    "Consider enabling retries with backoff",
                    "Review error handling strategies"
                ])
        
        elif sli_type in ["availability_pct", "success_rate_pct"]:
            if sli_value < objective:
                recommendations.extend([
                    "Check for infrastructure issues",
                    "Review deployment health",
                    "Enable health checks and load balancer integration",
                    "Consider implementing redundancy",
                    "Review incident response procedures"
                ])
        
        elif sli_type == "drift_psi":
            if sli_value > objective:
                recommendations.extend([
                    "Retrain model with recent data",
                    "Review data pipeline for changes",
                    "Consider model A/B testing",
                    "Implement gradual rollout strategies",
                    "Review feature engineering pipeline"
                ])
        
        return recommendations
    
    def get_evaluation_history(self, service_id: str, limit: int = 10) -> List[Dict]:
        """Get recent evaluation history for a service."""
        if service_id not in self.evaluation_history:
            return []
        
        return self.evaluation_history[service_id][-limit:]
    
    def get_sli_trends(self, service_id: str, sli_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get SLI trends over time period."""
        if service_id not in self.evaluation_history:
            return {"trend": "no_data", "values": []}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_evaluations = []
        
        for eval_result in self.evaluation_history[service_id]:
            eval_time = datetime.fromisoformat(eval_result["timestamp"].replace('Z', '+00:00'))
            if eval_time > cutoff_time:
                recent_evaluations.append(eval_result)
        
        if not recent_evaluations:
            return {"trend": "no_data", "values": []}
        
        values = []
        for evaluation in recent_evaluations:
            if sli_type in evaluation["sli_values"]:
                values.append({
                    "timestamp": evaluation["timestamp"],
                    "value": evaluation["sli_values"][sli_type]
                })
        
        if len(values) < 2:
            return {"trend": "insufficient_data", "values": values}
        
        # Calculate trend
        recent_values = [v["value"] for v in values[-5:]]  # Last 5 values
        older_values = [v["value"] for v in values[:5]]   # First 5 values
        
        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)
        
        if recent_avg > older_avg * 1.1:
            trend = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "values": values,
            "recent_average": recent_avg,
            "older_average": older_avg,
            "change_pct": ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SLI evaluator statistics."""
        total_evaluations = sum(len(evals) for evals in self.evaluation_history.values())
        
        violation_counts = {}
        warning_counts = {}
        
        for service_evals in self.evaluation_history.values():
            for evaluation in service_evals:
                for violation in evaluation.get("violations", []):
                    sli = violation["sli"]
                    violation_counts[sli] = violation_counts.get(sli, 0) + 1
                
                for warning in evaluation.get("warnings", []):
                    sli = warning["sli"]
                    warning_counts[sli] = warning_counts.get(sli, 0) + 1
        
        return {
            "total_services_monitored": len(self.evaluation_history),
            "total_evaluations": total_evaluations,
            "violation_counts_by_sli": violation_counts,
            "warning_counts_by_sli": warning_counts,
            "average_evaluations_per_service": total_evaluations / len(self.evaluation_history) if self.evaluation_history else 0
        }