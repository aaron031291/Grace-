"""Anomaly Detector - Detects anomalous behavior in system metrics."""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import statistics
from collections import deque

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in system metrics using statistical methods."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Number of data points to keep in sliding window
            sensitivity: Standard deviation multiplier for anomaly threshold
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_windows: Dict[str, deque] = {}
        self.anomaly_history: List[Dict] = []
    
    def add_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """
        Add metric value and check for anomalies.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Timestamp of measurement
            
        Returns:
            Anomaly detection result if anomaly detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Initialize window if needed
        if metric_name not in self.metric_windows:
            self.metric_windows[metric_name] = deque(maxlen=self.window_size)
        
        window = self.metric_windows[metric_name]
        
        # Check for anomaly before adding new value
        anomaly = None
        if len(window) >= 10:  # Need at least 10 points for meaningful statistics
            anomaly = self._detect_anomaly(metric_name, value, window, timestamp)
        
        # Add new value to window
        window.append({
            "value": value,
            "timestamp": timestamp
        })
        
        return anomaly
    
    def _detect_anomaly(self, metric_name: str, value: float, window: deque, timestamp: datetime) -> Optional[Dict]:
        """Detect if value is anomalous compared to historical data."""
        values = [point["value"] for point in window]
        
        # Calculate statistics
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Handle case where stdev is 0 (all values are identical)
        if stdev == 0:
            # If all historical values are identical and new value is different, it's anomalous
            if value != mean:
                anomaly = {
                    "metric_name": metric_name,
                    "value": value,
                    "expected_value": mean,
                    "deviation": abs(value - mean),
                    "deviation_type": "zero_variance",
                    "severity": "high" if abs(value - mean) > mean * 0.5 else "medium",
                    "timestamp": timestamp.isoformat(),
                    "anomaly_id": f"anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}_{metric_name}"
                }
                
                self.anomaly_history.append(anomaly)
                logger.warning(f"Anomaly detected in {metric_name}: {value} (expected: {mean})")
                return anomaly
            return None
        
        # Calculate z-score
        z_score = abs(value - mean) / stdev
        
        # Check if value is anomalous
        if z_score > self.sensitivity:
            severity = self._assess_anomaly_severity(z_score)
            
            anomaly = {
                "metric_name": metric_name,
                "value": value,
                "expected_value": mean,
                "standard_deviation": stdev,
                "z_score": z_score,
                "threshold": self.sensitivity,
                "deviation": abs(value - mean),
                "deviation_type": "statistical",
                "severity": severity,
                "timestamp": timestamp.isoformat(),
                "anomaly_id": f"anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}_{metric_name}"
            }
            
            self.anomaly_history.append(anomaly)
            logger.warning(f"Statistical anomaly detected in {metric_name}: {value} (z-score: {z_score:.2f})")
            return anomaly
        
        return None
    
    def _assess_anomaly_severity(self, z_score: float) -> str:
        """Assess severity of anomaly based on z-score."""
        if z_score > 5.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > self.sensitivity:
            return "medium"
        else:
            return "low"
    
    def bulk_add_metrics(self, metrics: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Add multiple metrics in bulk and return all detected anomalies.
        
        Args:
            metrics: Dict where key is metric name and value is list of 
                    {"value": float, "timestamp": datetime} dicts
                    
        Returns:
            Dict mapping metric names to lists of anomaly detections
        """
        all_anomalies = {}
        
        for metric_name, metric_data in metrics.items():
            anomalies = []
            
            for data_point in metric_data:
                anomaly = self.add_metric(
                    metric_name,
                    data_point["value"],
                    data_point.get("timestamp")
                )
                
                if anomaly:
                    anomalies.append(anomaly)
            
            if anomalies:
                all_anomalies[metric_name] = anomalies
        
        return all_anomalies
    
    def get_metric_stats(self, metric_name: str) -> Optional[Dict]:
        """Get statistics for a specific metric."""
        if metric_name not in self.metric_windows:
            return None
        
        window = self.metric_windows[metric_name]
        if not window:
            return None
        
        values = [point["value"] for point in window]
        
        return {
            "metric_name": metric_name,
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None,
            "window_size": self.window_size
        }
    
    def get_recent_anomalies(self, metric_name: Optional[str] = None, hours: int = 24) -> List[Dict]:
        """Get recent anomalies, optionally filtered by metric."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_anomalies = []
        for anomaly in self.anomaly_history:
            anomaly_time = datetime.fromisoformat(anomaly["timestamp"].replace('Z', '+00:00'))
            
            if anomaly_time > cutoff_time:
                if metric_name is None or anomaly["metric_name"] == metric_name:
                    filtered_anomalies.append(anomaly)
        
        return sorted(filtered_anomalies, key=lambda x: x["timestamp"], reverse=True)
    
    def clear_metric_history(self, metric_name: str) -> bool:
        """Clear history for a specific metric."""
        if metric_name in self.metric_windows:
            self.metric_windows[metric_name].clear()
            return True
        return False
    
    def clear_all_history(self):
        """Clear all metric history."""
        self.metric_windows.clear()
        self.anomaly_history.clear()
        logger.info("Cleared all anomaly detection history")
    
    def adjust_sensitivity(self, new_sensitivity: float):
        """Adjust anomaly detection sensitivity."""
        old_sensitivity = self.sensitivity
        self.sensitivity = new_sensitivity
        logger.info(f"Adjusted anomaly detection sensitivity from {old_sensitivity} to {new_sensitivity}")
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies in the last N hours."""
        recent_anomalies = self.get_recent_anomalies(hours=hours)
        
        if not recent_anomalies:
            return {
                "total_anomalies": 0,
                "by_metric": {},
                "by_severity": {},
                "time_period_hours": hours
            }
        
        # Group by metric
        by_metric = {}
        by_severity = {}
        
        for anomaly in recent_anomalies:
            metric = anomaly["metric_name"]
            severity = anomaly["severity"]
            
            by_metric[metric] = by_metric.get(metric, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_anomalies": len(recent_anomalies),
            "by_metric": by_metric,
            "by_severity": by_severity,
            "time_period_hours": hours,
            "most_affected_metric": max(by_metric.items(), key=lambda x: x[1])[0] if by_metric else None
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get anomaly detector statistics."""
        total_anomalies = len(self.anomaly_history)
        monitored_metrics = len(self.metric_windows)
        
        severity_counts = {}
        for anomaly in self.anomaly_history:
            severity = anomaly["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "monitored_metrics": monitored_metrics,
            "total_anomalies_detected": total_anomalies,
            "severity_distribution": severity_counts,
            "sensitivity_threshold": self.sensitivity,
            "window_size": self.window_size
        }