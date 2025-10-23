"""
Loop Drift Detection - Detects anomalies in loop decisions (Class 10)
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import logging

from grace.clarity.unified_output import GraceLoopOutput

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects drift in loop decisions by comparing with historical patterns
    
    Features:
    - Statistical drift detection
    - Decision consistency checking
    - Anomaly flagging
    - Trend analysis
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.3,
        window_size: int = 50,
        min_samples: int = 10
    ):
        """
        Initialize drift detector
        
        Args:
            drift_threshold: Threshold for flagging drift
            window_size: Size of sliding window for comparison
            min_samples: Minimum samples before drift detection
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        
        self.decision_history: List[Dict[str, Any]] = []
        self.drift_alerts: List[Dict[str, Any]] = []
        
        logger.info(f"DriftDetector initialized (threshold={drift_threshold})")
    
    def detect_drift(
        self,
        new_output: GraceLoopOutput,
        historical_outputs: Optional[List[GraceLoopOutput]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in new output compared to history
        
        Returns:
            Drift detection result with anomaly flags
        """
        if historical_outputs is None:
            historical_outputs = self.decision_history
        
        # Need minimum samples for drift detection
        if len(historical_outputs) < self.min_samples:
            return {
                "drift_detected": False,
                "confidence": 0.0,
                "reason": "Insufficient historical data",
                "requires_review": False
            }
        
        # Use recent window
        recent_history = historical_outputs[-self.window_size:]
        
        # Calculate drift metrics
        confidence_drift = self._detect_confidence_drift(new_output, recent_history)
        decision_drift = self._detect_decision_drift(new_output, recent_history)
        trust_drift = self._detect_trust_drift(new_output, recent_history)
        pattern_drift = self._detect_pattern_drift(new_output, recent_history)
        
        # Aggregate drift score
        drift_score = np.mean([
            confidence_drift,
            decision_drift,
            trust_drift,
            pattern_drift
        ])
        
        drift_detected = drift_score > self.drift_threshold
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "confidence_drift": float(confidence_drift),
            "decision_drift": float(decision_drift),
            "trust_drift": float(trust_drift),
            "pattern_drift": float(pattern_drift),
            "requires_review": drift_detected and drift_score > 0.5,
            "anomaly_details": self._get_anomaly_details(
                new_output, recent_history, drift_score
            )
        }
        
        # Store decision in history
        self.decision_history.append(new_output.to_dict())
        
        # Log drift alert if detected
        if drift_detected:
            self._log_drift_alert(new_output, result)
        
        logger.info(
            f"Drift detection: {'DRIFT' if drift_detected else 'OK'} "
            f"(score={drift_score:.3f})"
        )
        
        return result
    
    def _detect_confidence_drift(
        self,
        new_output: GraceLoopOutput,
        history: List[Dict[str, Any]]
    ) -> float:
        """Detect drift in confidence levels"""
        historical_confidences = [
            h["trust_metrics"]["consensus_confidence"]
            for h in history
            if "trust_metrics" in h
        ]
        
        if not historical_confidences:
            return 0.0
        
        hist_mean = np.mean(historical_confidences)
        hist_std = np.std(historical_confidences)
        
        new_confidence = new_output.trust_metrics.consensus_confidence
        
        # Z-score
        if hist_std == 0:
            return 0.0
        
        z_score = abs(new_confidence - hist_mean) / hist_std
        
        # Normalize to 0-1
        drift = min(1.0, z_score / 3.0)  # 3 sigma = max
        
        return drift
    
    def _detect_decision_drift(
        self,
        new_output: GraceLoopOutput,
        history: List[Dict[str, Any]]
    ) -> float:
        """Detect drift in decision patterns"""
        # Compare decision types
        historical_types = [h.get("decision_type") for h in history]
        
        new_type = new_output.decision_type
        
        if new_type not in historical_types:
            return 0.8  # High drift for new decision type
        
        # Calculate type frequency
        type_freq = historical_types.count(new_type) / len(historical_types)
        
        # Low frequency = potential drift
        drift = 1.0 - type_freq
        
        return drift
    
    def _detect_trust_drift(
        self,
        new_output: GraceLoopOutput,
        history: List[Dict[str, Any]]
    ) -> float:
        """Detect drift in trust scores"""
        historical_trust = [
            h["trust_metrics"]["overall_trust"]
            for h in history
            if "trust_metrics" in h
        ]
        
        if not historical_trust:
            return 0.0
        
        hist_mean = np.mean(historical_trust)
        new_trust = new_output.trust_metrics.overall_trust
        
        # Relative change
        drift = abs(new_trust - hist_mean) / max(0.1, hist_mean)
        
        return min(1.0, drift)
    
    def _detect_pattern_drift(
        self,
        new_output: GraceLoopOutput,
        history: List[Dict[str, Any]]
    ) -> float:
        """Detect drift in reasoning patterns"""
        # Compare reasoning chain lengths
        historical_lengths = [
            len(h.get("reasoning_chain", []))
            for h in history
        ]
        
        if not historical_lengths:
            return 0.0
        
        hist_mean = np.mean(historical_lengths)
        new_length = len(new_output.reasoning_chain)
        
        # Significant change in reasoning complexity
        if hist_mean == 0:
            return 0.0
        
        drift = abs(new_length - hist_mean) / hist_mean
        
        return min(1.0, drift)
    
    def _get_anomaly_details(
        self,
        new_output: GraceLoopOutput,
        history: List[Dict[str, Any]],
        drift_score: float
    ) -> List[str]:
        """Get detailed anomaly descriptions"""
        details = []
        
        if drift_score > 0.5:
            details.append(f"High drift score: {drift_score:.3f}")
        
        if new_output.trust_metrics.overall_trust < 0.5:
            details.append(f"Low trust score: {new_output.trust_metrics.overall_trust:.3f}")
        
        if not new_output.trust_metrics.governance_passed:
            details.append("Governance validation failed")
        
        if new_output.specialist_agreement and new_output.specialist_agreement < 0.5:
            details.append(f"Low specialist agreement: {new_output.specialist_agreement:.3f}")
        
        return details
    
    def _log_drift_alert(self, output: GraceLoopOutput, result: Dict[str, Any]):
        """Log drift alert"""
        alert = {
            "output_id": output.output_id,
            "loop_id": output.loop_id,
            "timestamp": output.timestamp,
            "drift_score": result["drift_score"],
            "requires_review": result["requires_review"],
            "details": result["anomaly_details"]
        }
        
        self.drift_alerts.append(alert)
        
        logger.warning(
            f"DRIFT ALERT: {output.output_id} - "
            f"Score: {result['drift_score']:.3f}, "
            f"Details: {', '.join(result['anomaly_details'])}"
        )
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics"""
        if not self.drift_alerts:
            return {
                "total_alerts": 0,
                "drift_rate": 0.0
            }
        
        return {
            "total_alerts": len(self.drift_alerts),
            "drift_rate": len(self.drift_alerts) / max(1, len(self.decision_history)),
            "recent_alerts": self.drift_alerts[-10:],
            "review_required": sum(1 for a in self.drift_alerts if a["requires_review"])
        }
