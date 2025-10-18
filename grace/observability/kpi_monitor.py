"""
KPI Trust Monitor with event publishing and governance integration
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrustThreshold(Enum):
    """Trust score thresholds"""
    CRITICAL = 0.3
    WARNING = 0.5
    ACCEPTABLE = 0.7
    GOOD = 0.85


@dataclass
class TrustEvent:
    """Trust-related event"""
    event_id: str
    event_type: str
    component: str
    entity_id: str
    current_score: float
    previous_score: float
    threshold: Optional[float]
    severity: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class KPITrustMonitor:
    """
    KPI Trust Monitor with event publishing and integration
    
    Features:
    - Trust score tracking
    - Threshold monitoring
    - Anomaly detection
    - Event publishing to event bus
    - Governance integration
    - AVN healing triggers
    """
    
    def __init__(
        self,
        event_publisher=None,
        governance_validator=None,
        avn_core=None,
        metrics=None
    ):
        """
        Initialize KPI trust monitor
        
        Args:
            event_publisher: Event bus for publishing trust events
            governance_validator: Governance validator for reviews
            avn_core: AVN core for triggering healing
            metrics: Prometheus metrics instance
        """
        self.event_publisher = event_publisher
        self.governance_validator = governance_validator
        self.avn_core = avn_core
        self.metrics = metrics
        
        self.trust_scores: Dict[str, Dict[str, float]] = {}  # component -> entity -> score
        self.score_history: Dict[str, List[float]] = {}  # entity -> [scores]
        self.event_callbacks: List[Callable] = []
        
        self.thresholds = {
            "critical": TrustThreshold.CRITICAL.value,
            "warning": TrustThreshold.WARNING.value,
            "acceptable": TrustThreshold.ACCEPTABLE.value,
            "good": TrustThreshold.GOOD.value
        }
        
        logger.info("KPITrustMonitor initialized with event publishing")
    
    def update_trust_score(
        self,
        component: str,
        entity_id: str,
        new_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update trust score and check thresholds
        
        Args:
            component: Component name
            entity_id: Entity identifier
            new_score: New trust score (0-1)
            metadata: Additional metadata
        """
        # Initialize if needed
        if component not in self.trust_scores:
            self.trust_scores[component] = {}
        
        # Get previous score
        previous_score = self.trust_scores[component].get(entity_id, 0.5)
        
        # Update score
        self.trust_scores[component][entity_id] = new_score
        
        # Update history
        history_key = f"{component}:{entity_id}"
        if history_key not in self.score_history:
            self.score_history[history_key] = []
        self.score_history[history_key].append(new_score)
        
        # Keep last 100 scores
        if len(self.score_history[history_key]) > 100:
            self.score_history[history_key] = self.score_history[history_key][-100:]
        
        # Update metrics
        if self.metrics:
            self.metrics.update_trust_score(
                component,
                entity_id,
                new_score,
                previous_score
            )
        
        # Check thresholds
        self._check_thresholds(
            component,
            entity_id,
            new_score,
            previous_score,
            metadata or {}
        )
        
        # Detect anomalies
        self._detect_anomalies(
            component,
            entity_id,
            new_score,
            previous_score,
            metadata or {}
        )
        
        logger.debug(
            f"Trust score updated: {component}:{entity_id} "
            f"{previous_score:.3f} -> {new_score:.3f}"
        )
    
    def _check_thresholds(
        self,
        component: str,
        entity_id: str,
        current_score: float,
        previous_score: float,
        metadata: Dict[str, Any]
    ):
        """Check if trust score crossed thresholds"""
        # Critical threshold breach
        if current_score < self.thresholds["critical"]:
            if previous_score >= self.thresholds["critical"]:
                self._emit_event(
                    event_type="threshold_breach",
                    component=component,
                    entity_id=entity_id,
                    current_score=current_score,
                    previous_score=previous_score,
                    threshold=self.thresholds["critical"],
                    severity="critical",
                    metadata=metadata
                )
                
                # Trigger AVN healing
                self._trigger_healing(component, entity_id, current_score, metadata)
        
        # Warning threshold breach
        elif current_score < self.thresholds["warning"]:
            if previous_score >= self.thresholds["warning"]:
                self._emit_event(
                    event_type="threshold_breach",
                    component=component,
                    entity_id=entity_id,
                    current_score=current_score,
                    previous_score=previous_score,
                    threshold=self.thresholds["warning"],
                    severity="warning",
                    metadata=metadata
                )
                
                # Trigger governance review
                self._trigger_governance_review(
                    component,
                    entity_id,
                    current_score,
                    metadata
                )
        
        # Recovery event (crossed back above threshold)
        elif current_score >= self.thresholds["acceptable"]:
            if previous_score < self.thresholds["acceptable"]:
                self._emit_event(
                    event_type="recovery",
                    component=component,
                    entity_id=entity_id,
                    current_score=current_score,
                    previous_score=previous_score,
                    threshold=self.thresholds["acceptable"],
                    severity="info",
                    metadata=metadata
                )
    
    def _detect_anomalies(
        self,
        component: str,
        entity_id: str,
        current_score: float,
        previous_score: float,
        metadata: Dict[str, Any]
    ):
        """Detect anomalies in trust score patterns"""
        history_key = f"{component}:{entity_id}"
        history = self.score_history.get(history_key, [])
        
        if len(history) < 10:
            return
        
        # Sudden drop detection
        drop = previous_score - current_score
        if drop > 0.2:  # 20% sudden drop
            self._emit_event(
                event_type="anomaly",
                component=component,
                entity_id=entity_id,
                current_score=current_score,
                previous_score=previous_score,
                threshold=None,
                severity="warning",
                metadata={
                    **metadata,
                    "anomaly_type": "sudden_drop",
                    "drop_amount": drop
                }
            )
        
        # Trend detection (declining over time)
        recent = history[-10:]
        if len(recent) >= 10:
            import numpy as np
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            
            if trend < -0.02:  # Declining trend
                self._emit_event(
                    event_type="anomaly",
                    component=component,
                    entity_id=entity_id,
                    current_score=current_score,
                    previous_score=previous_score,
                    threshold=None,
                    severity="warning",
                    metadata={
                        **metadata,
                        "anomaly_type": "declining_trend",
                        "trend_slope": trend
                    }
                )
    
    def _emit_event(
        self,
        event_type: str,
        component: str,
        entity_id: str,
        current_score: float,
        previous_score: float,
        threshold: Optional[float],
        severity: str,
        metadata: Dict[str, Any]
    ):
        """Emit trust event"""
        import uuid
        
        event = TrustEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            component=component,
            entity_id=entity_id,
            current_score=current_score,
            previous_score=previous_score,
            threshold=threshold,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata
        )
        
        # Publish to event bus
        if self.event_publisher:
            try:
                self.event_publisher.publish({
                    "type": f"TRUST.{event_type.upper()}",
                    "severity": severity,
                    "component": component,
                    "entity_id": entity_id,
                    "data": {
                        "current_score": current_score,
                        "previous_score": previous_score,
                        "threshold": threshold,
                        "metadata": metadata
                    },
                    "timestamp": event.timestamp
                })
                
                logger.info(
                    f"Trust event published: {event_type} for {component}:{entity_id} "
                    f"(score: {current_score:.3f})"
                )
            except Exception as e:
                logger.error(f"Failed to publish trust event: {e}")
        
        # Call registered callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def _trigger_governance_review(
        self,
        component: str,
        entity_id: str,
        score: float,
        metadata: Dict[str, Any]
    ):
        """Trigger governance review for low trust"""
        if not self.governance_validator:
            return
        
        logger.warning(
            f"Triggering governance review: {component}:{entity_id} "
            f"(score: {score:.3f})"
        )
        
        # Create review request
        review_data = {
            "type": "trust_review",
            "component": component,
            "entity_id": entity_id,
            "trust_score": score,
            "reason": "trust_score_below_warning_threshold",
            "metadata": metadata
        }
        
        # Publish review request event
        if self.event_publisher:
            self.event_publisher.publish({
                "type": "GOVERNANCE.REVIEW_REQUEST",
                "severity": "warning",
                "data": review_data
            })
    
    def _trigger_healing(
        self,
        component: str,
        entity_id: str,
        score: float,
        metadata: Dict[str, Any]
    ):
        """Trigger AVN healing for critical trust score"""
        if not self.avn_core:
            return
        
        logger.critical(
            f"Triggering AVN healing: {component}:{entity_id} "
            f"(score: {score:.3f})"
        )
        
        # Report degraded metrics to AVN
        try:
            self.avn_core.report_metrics(
                entity_id,
                {
                    "latency": 1000,  # High latency to trigger healing
                    "error_rate": 1.0 - score,  # Convert low trust to high error rate
                    "trust_score": score,
                    "source": "kpi_trust_monitor",
                    "metadata": metadata
                }
            )
        except Exception as e:
            logger.error(f"Failed to trigger AVN healing: {e}")
    
    def register_callback(self, callback: Callable[[TrustEvent], None]):
        """Register callback for trust events"""
        self.event_callbacks.append(callback)
    
    def get_trust_score(
        self,
        component: str,
        entity_id: str
    ) -> Optional[float]:
        """Get current trust score"""
        return self.trust_scores.get(component, {}).get(entity_id)
    
    def get_trust_trend(
        self,
        component: str,
        entity_id: str,
        window: int = 10
    ) -> Optional[float]:
        """Get trust score trend (slope)"""
        history_key = f"{component}:{entity_id}"
        history = self.score_history.get(history_key, [])
        
        if len(history) < window:
            return None
        
        recent = history[-window:]
        import numpy as np
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        return float(trend)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        total_entities = sum(len(entities) for entities in self.trust_scores.values())
        
        all_scores = []
        for component_scores in self.trust_scores.values():
            all_scores.extend(component_scores.values())
        
        if not all_scores:
            return {
                "total_entities": 0,
                "avg_trust_score": 0.0
            }
        
        import numpy as np
        
        return {
            "total_entities": total_entities,
            "total_components": len(self.trust_scores),
            "avg_trust_score": float(np.mean(all_scores)),
            "min_trust_score": float(np.min(all_scores)),
            "max_trust_score": float(np.max(all_scores)),
            "below_critical": sum(1 for s in all_scores if s < self.thresholds["critical"]),
            "below_warning": sum(1 for s in all_scores if s < self.thresholds["warning"]),
            "acceptable": sum(1 for s in all_scores if s >= self.thresholds["acceptable"])
        }
