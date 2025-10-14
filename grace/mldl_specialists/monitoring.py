"""
Model Monitoring & Observability

Production-grade monitoring for ML/DL models:
- Latency and throughput tracking (p50, p95, p99)
- Input/output distribution drift detection
- Calibration error monitoring
- Trust score time-series
- Automated alerting and rollback triggers
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from scipy.stats import ks_2samp, entropy
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency statistics"""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    sample_count: int


@dataclass
class ThroughputMetrics:
    """Throughput statistics"""
    requests_per_second: float
    successful_requests: int
    failed_requests: int
    timeout_requests: int
    total_requests: int
    error_rate: float
    timeout_rate: float


@dataclass
class DriftMetrics:
    """Distribution drift metrics"""
    kl_divergence: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    is_drifted: bool = False
    drift_score: float = 0.0  # 0-1, higher = more drift


@dataclass
class MonitoringAlert:
    """Monitoring alert"""
    alert_id: str
    severity: str  # critical, high, medium, low
    alert_type: str
    message: str
    model_id: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    requires_action: bool = False
    suggested_actions: List[str] = field(default_factory=list)


class ModelMonitor:
    """
    Real-time monitoring for a single model.
    
    Tracks:
    - Latency (p50, p95, p99)
    - Throughput (req/s, error rate)
    - Distribution drift (input/output)
    - Calibration error
    - Trust score evolution
    """
    
    def __init__(
        self,
        model_id: str,
        window_size: int = 1000,  # Samples to keep in memory
        alert_callback: Optional[callable] = None
    ):
        self.model_id = model_id
        self.window_size = window_size
        self.alert_callback = alert_callback
        
        # Sliding windows
        self.latencies = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.ood_flags = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Input features (for drift detection)
        self.input_features = deque(maxlen=window_size)
        
        # Baseline distributions (set during training/validation)
        self.baseline_input_distribution: Optional[np.ndarray] = None
        self.baseline_output_distribution: Optional[np.ndarray] = None
        
        # Counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        
        # Alerts
        self.active_alerts: List[MonitoringAlert] = []
        
        # Thresholds
        self.thresholds = {
            'latency_p95_ms': 500.0,
            'error_rate': 0.05,
            'ood_rate': 0.2,
            'drift_score': 0.3,
            'calibration_error': 0.15
        }
    
    def record_inference(
        self,
        latency_ms: float,
        confidence: float,
        prediction: Any,
        ood_flag: bool = False,
        input_features: Optional[np.ndarray] = None,
        success: bool = True,
        timeout: bool = False
    ):
        """
        Record inference metrics.
        
        Args:
            latency_ms: Inference latency
            confidence: Model confidence
            prediction: Prediction value
            ood_flag: Whether sample was OOD
            input_features: Input feature vector
            success: Whether inference succeeded
            timeout: Whether inference timed out
        """
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.latencies.append(latency_ms)
            self.confidences.append(confidence)
            self.predictions.append(prediction)
            self.ood_flags.append(ood_flag)
            self.timestamps.append(datetime.now())
            
            if input_features is not None:
                self.input_features.append(input_features)
        else:
            self.failed_requests += 1
        
        if timeout:
            self.timeout_requests += 1
        
        # Check for alerts every 100 requests
        if self.total_requests % 100 == 0:
            self._check_alerts()
    
    def get_latency_metrics(self) -> LatencyMetrics:
        """Get current latency statistics"""
        if not self.latencies:
            return LatencyMetrics(
                p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
                mean_ms=0.0, std_ms=0.0, min_ms=0.0, max_ms=0.0,
                sample_count=0
            )
        
        latencies_array = np.array(self.latencies)
        
        return LatencyMetrics(
            p50_ms=float(np.percentile(latencies_array, 50)),
            p95_ms=float(np.percentile(latencies_array, 95)),
            p99_ms=float(np.percentile(latencies_array, 99)),
            mean_ms=float(np.mean(latencies_array)),
            std_ms=float(np.std(latencies_array)),
            min_ms=float(np.min(latencies_array)),
            max_ms=float(np.max(latencies_array)),
            sample_count=len(self.latencies)
        )
    
    def get_throughput_metrics(self) -> ThroughputMetrics:
        """Get current throughput statistics"""
        error_rate = (
            self.failed_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        timeout_rate = (
            self.timeout_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        
        # Calculate requests per second (last minute)
        if len(self.timestamps) >= 2:
            time_window = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            rps = len(self.timestamps) / time_window if time_window > 0 else 0.0
        else:
            rps = 0.0
        
        return ThroughputMetrics(
            requests_per_second=rps,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            timeout_requests=self.timeout_requests,
            total_requests=self.total_requests,
            error_rate=error_rate,
            timeout_rate=timeout_rate
        )
    
    def set_baseline_distribution(
        self,
        input_distribution: Optional[np.ndarray] = None,
        output_distribution: Optional[np.ndarray] = None
    ):
        """
        Set baseline distributions for drift detection.
        
        Args:
            input_distribution: Baseline input features (N, D)
            output_distribution: Baseline outputs (N,) or (N, C)
        """
        self.baseline_input_distribution = input_distribution
        self.baseline_output_distribution = output_distribution
        logger.info(f"Set baseline distributions for {self.model_id}")
    
    def detect_input_drift(self) -> DriftMetrics:
        """
        Detect input distribution drift.
        
        Returns:
            DriftMetrics with drift indicators
        """
        if self.baseline_input_distribution is None or not self.input_features:
            return DriftMetrics(is_drifted=False, drift_score=0.0)
        
        current_features = np.array(list(self.input_features))
        
        # KL divergence (for probability distributions)
        # Kolmogorov-Smirnov test (for feature distributions)
        
        drift_scores = []
        ks_statistics = []
        p_values = []
        
        # Test each feature dimension
        num_features = min(
            self.baseline_input_distribution.shape[1],
            current_features.shape[1]
        )
        
        for dim in range(num_features):
            baseline_dim = self.baseline_input_distribution[:, dim]
            current_dim = current_features[:, dim]
            
            # KS test
            ks_stat, p_value = ks_2samp(baseline_dim, current_dim)
            ks_statistics.append(ks_stat)
            p_values.append(p_value)
            
            # Drift score based on KS statistic
            drift_scores.append(ks_stat)
        
        # Aggregate drift score
        avg_drift_score = float(np.mean(drift_scores))
        max_ks_stat = float(np.max(ks_statistics))
        min_p_value = float(np.min(p_values))
        
        # Drift detected if:
        # 1. Average drift score > threshold OR
        # 2. Any feature has significant drift (p < 0.05)
        is_drifted = (
            avg_drift_score > self.thresholds['drift_score'] or
            min_p_value < 0.05
        )
        
        return DriftMetrics(
            kl_divergence=None,  # Not computed for features
            ks_statistic=max_ks_stat,
            ks_p_value=min_p_value,
            is_drifted=is_drifted,
            drift_score=avg_drift_score
        )
    
    def detect_output_drift(self) -> DriftMetrics:
        """
        Detect output distribution drift.
        
        Returns:
            DriftMetrics with drift indicators
        """
        if self.baseline_output_distribution is None or not self.predictions:
            return DriftMetrics(is_drifted=False, drift_score=0.0)
        
        # For classification: compare class distributions
        # For regression: KS test on prediction values
        
        current_predictions = np.array(list(self.predictions))
        
        # Check if discrete (classification) or continuous (regression)
        if self._is_discrete(current_predictions):
            # Classification: KL divergence
            baseline_dist = self._get_class_distribution(self.baseline_output_distribution)
            current_dist = self._get_class_distribution(current_predictions)
            
            kl_div = float(entropy(current_dist, baseline_dist))
            
            is_drifted = kl_div > self.thresholds['drift_score']
            
            return DriftMetrics(
                kl_divergence=kl_div,
                is_drifted=is_drifted,
                drift_score=kl_div
            )
        else:
            # Regression: KS test
            ks_stat, p_value = ks_2samp(
                self.baseline_output_distribution.flatten(),
                current_predictions.flatten()
            )
            
            is_drifted = p_value < 0.05
            
            return DriftMetrics(
                ks_statistic=float(ks_stat),
                ks_p_value=float(p_value),
                is_drifted=is_drifted,
                drift_score=float(ks_stat)
            )
    
    def _is_discrete(self, values: np.ndarray) -> bool:
        """Check if values are discrete (classification) or continuous (regression)"""
        unique_values = np.unique(values)
        return len(unique_values) < 100  # Heuristic
    
    def _get_class_distribution(self, values: np.ndarray) -> np.ndarray:
        """Get class probability distribution"""
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)
        return probs
    
    def calculate_calibration_error(
        self,
        ground_truth: Optional[List[Any]] = None,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            ground_truth: Ground truth labels (if available)
            n_bins: Number of bins
        
        Returns:
            ECE score
        """
        if not self.confidences or ground_truth is None:
            return 0.0
        
        confidences = np.array(list(self.confidences))
        predictions = np.array(list(self.predictions))
        ground_truth = np.array(ground_truth)
        
        # Ensure same length
        min_len = min(len(confidences), len(ground_truth))
        confidences = confidences[:min_len]
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
        
        # Compute ECE
        from ..uncertainty_ood import compute_calibration_error
        
        return compute_calibration_error(
            confidences=confidences,
            predictions=predictions,
            true_labels=ground_truth,
            n_bins=n_bins
        )
    
    def get_ood_rate(self) -> float:
        """Get current OOD detection rate"""
        if not self.ood_flags:
            return 0.0
        
        return sum(self.ood_flags) / len(self.ood_flags)
    
    def _check_alerts(self):
        """Check monitoring thresholds and generate alerts"""
        alerts = []
        
        # Check latency
        latency_metrics = self.get_latency_metrics()
        if latency_metrics.p95_ms > self.thresholds['latency_p95_ms']:
            alerts.append(MonitoringAlert(
                alert_id=f"{self.model_id}_latency_{datetime.now().timestamp()}",
                severity='high',
                alert_type='latency_degradation',
                message=f"p95 latency ({latency_metrics.p95_ms:.1f}ms) exceeds threshold ({self.thresholds['latency_p95_ms']:.1f}ms)",
                model_id=self.model_id,
                metric_value=latency_metrics.p95_ms,
                threshold=self.thresholds['latency_p95_ms'],
                requires_action=True,
                suggested_actions=['investigate_bottleneck', 'scale_resources']
            ))
        
        # Check error rate
        throughput_metrics = self.get_throughput_metrics()
        if throughput_metrics.error_rate > self.thresholds['error_rate']:
            alerts.append(MonitoringAlert(
                alert_id=f"{self.model_id}_errors_{datetime.now().timestamp()}",
                severity='critical',
                alert_type='high_error_rate',
                message=f"Error rate ({throughput_metrics.error_rate:.2%}) exceeds threshold ({self.thresholds['error_rate']:.2%})",
                model_id=self.model_id,
                metric_value=throughput_metrics.error_rate,
                threshold=self.thresholds['error_rate'],
                requires_action=True,
                suggested_actions=['rollback_model', 'investigate_errors']
            ))
        
        # Check OOD rate
        ood_rate = self.get_ood_rate()
        if ood_rate > self.thresholds['ood_rate']:
            alerts.append(MonitoringAlert(
                alert_id=f"{self.model_id}_ood_{datetime.now().timestamp()}",
                severity='medium',
                alert_type='high_ood_rate',
                message=f"OOD rate ({ood_rate:.2%}) exceeds threshold ({self.thresholds['ood_rate']:.2%})",
                model_id=self.model_id,
                metric_value=ood_rate,
                threshold=self.thresholds['ood_rate'],
                requires_action=False,
                suggested_actions=['check_input_distribution', 'retrain_model']
            ))
        
        # Check drift
        if self.input_features and self.baseline_input_distribution is not None:
            drift_metrics = self.detect_input_drift()
            if drift_metrics.is_drifted:
                alerts.append(MonitoringAlert(
                    alert_id=f"{self.model_id}_drift_{datetime.now().timestamp()}",
                    severity='high',
                    alert_type='input_drift',
                    message=f"Input distribution drift detected (score={drift_metrics.drift_score:.3f})",
                    model_id=self.model_id,
                    metric_value=drift_metrics.drift_score,
                    threshold=self.thresholds['drift_score'],
                    requires_action=True,
                    suggested_actions=['retrain_model', 'investigate_data_source']
                ))
        
        # Store and callback
        self.active_alerts.extend(alerts)
        
        if alerts and self.alert_callback:
            for alert in alerts:
                self.alert_callback(alert)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'model_id': self.model_id,
            'timestamp': datetime.now().isoformat(),
            'latency': self.get_latency_metrics().__dict__,
            'throughput': self.get_throughput_metrics().__dict__,
            'ood_rate': self.get_ood_rate(),
            'active_alerts': len(self.active_alerts),
            'window_size': len(self.latencies),
            'total_requests': self.total_requests
        }


class TrustScoreLedger:
    """
    Time-series tracking of model trust scores.
    
    Aggregates model performance over time and provides
    trust scores for governance decisions.
    """
    
    def __init__(self, window_days: int = 7):
        self.window_days = window_days
        self.trust_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def record_trust_score(
        self,
        model_id: str,
        trust_score: float,
        timestamp: Optional[datetime] = None
    ):
        """Record trust score at timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if model_id not in self.trust_history:
            self.trust_history[model_id] = []
        
        self.trust_history[model_id].append((timestamp, trust_score))
        
        # Clean old entries
        cutoff = datetime.now() - timedelta(days=self.window_days)
        self.trust_history[model_id] = [
            (ts, score) for ts, score in self.trust_history[model_id]
            if ts >= cutoff
        ]
    
    def get_current_trust_score(self, model_id: str) -> float:
        """Get current trust score (latest value)"""
        if model_id not in self.trust_history or not self.trust_history[model_id]:
            return 0.5  # Neutral
        
        return self.trust_history[model_id][-1][1]
    
    def get_trust_trend(
        self,
        model_id: str
    ) -> str:
        """Get trust score trend: improving, declining, stable"""
        if model_id not in self.trust_history or len(self.trust_history[model_id]) < 2:
            return "unknown"
        
        recent = self.trust_history[model_id][-5:]  # Last 5 scores
        scores = [score for _, score in recent]
        
        # Linear regression slope
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
