"""
Enhanced AVN Core - Health monitoring and anomaly detection for Grace governance kernel.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics
import logging
import json
from enum import Enum

from ..core.contracts import Experience


logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DECISION_INCONSISTENCY = "decision_inconsistency"
    TRUST_EROSION = "trust_erosion"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"
    COMPONENT_FAILURE = "component_failure"
    SECURITY_BREACH = "security_breach"


class SeverityLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


@dataclass
class HealthMetric:
    """Health metric for a component or system."""
    component_id: str
    metric_name: str
    value: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    timestamp: datetime = datetime.now()
    
    def is_healthy(self) -> bool:
        """Check if metric is within healthy thresholds."""
        if self.threshold_min is not None and self.value < self.threshold_min:
            return False
        if self.threshold_max is not None and self.value > self.threshold_max:
            return False
        return True


@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""
    alert_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    component_id: str
    description: str
    metrics: Dict[str, float]
    confidence: float
    timestamp: datetime
    resolution_actions: List[str]
    auto_resolve: bool = False


class ComponentHealth:
    """Tracks health status for a single component."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.metrics: Dict[str, List[HealthMetric]] = {}
        self.last_heartbeat = datetime.now()
        self.status = "healthy"
        self.alert_count = 0
        self.performance_baseline = {}
        self.is_active = True
    
    def add_metric(self, metric: HealthMetric):
        """Add a health metric."""
        if metric.metric_name not in self.metrics:
            self.metrics[metric.metric_name] = []
        
        self.metrics[metric.metric_name].append(metric)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics[metric.metric_name]) > 1000:
            self.metrics[metric.metric_name] = self.metrics[metric.metric_name][-1000:]
        
        self.last_heartbeat = datetime.now()
    
    def get_latest_metric(self, metric_name: str) -> Optional[HealthMetric]:
        """Get the latest metric value."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]
    
    def get_metric_trend(self, metric_name: str, window_minutes: int = 10) -> Optional[float]:
        """Get trend (slope) for a metric over a time window."""
        if metric_name not in self.metrics:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics[metric_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return None
        
        # Simple linear trend calculation
        values = [m.value for m in recent_metrics]
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def calculate_health_score(self) -> float:
        """Calculate overall health score for this component."""
        if not self.metrics:
            return 0.5  # Neutral score with no data
        
        health_factors = []
        
        # Check each metric type
        for metric_name, metric_history in self.metrics.items():
            if not metric_history:
                continue
            
            latest_metric = metric_history[-1]
            
            # Health based on threshold compliance
            if latest_metric.is_healthy():
                health_factors.append(1.0)
            else:
                # Calculate how far from threshold
                if latest_metric.threshold_min is not None and latest_metric.value < latest_metric.threshold_min:
                    deviation = (latest_metric.threshold_min - latest_metric.value) / latest_metric.threshold_min
                    health_factors.append(max(0.0, 1.0 - deviation))
                elif latest_metric.threshold_max is not None and latest_metric.value > latest_metric.threshold_max:
                    deviation = (latest_metric.value - latest_metric.threshold_max) / latest_metric.threshold_max
                    health_factors.append(max(0.0, 1.0 - deviation))
                else:
                    health_factors.append(0.5)  # Unhealthy but unknown deviation
        
        # Check heartbeat freshness
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        if time_since_heartbeat > 300:  # 5 minutes
            health_factors.append(0.0)
        elif time_since_heartbeat > 60:  # 1 minute
            health_factors.append(0.5)
        else:
            health_factors.append(1.0)
        
        return statistics.mean(health_factors) if health_factors else 0.0


class EnhancedAVNCore:
    """
    Enhanced Anomaly, Vulnerability, and Notification (AVN) core system.
    Provides health monitoring, anomaly detection, and predictive alerts for governance failover.
    """
    
    def __init__(self, event_bus, memory_core=None):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.component_health: Dict[str, ComponentHealth] = {}
        self.anomaly_detectors: List[Callable] = []
        self.active_alerts: Dict[str, AnomalyAlert] = {}
        self.alert_history: List[AnomalyAlert] = []
        self.healing_actions: Dict[str, Callable] = {}
        
        # Configuration
        self.monitoring_interval = 30  # seconds
        self.anomaly_thresholds = self._initialize_anomaly_thresholds()
        self.predictive_window = 300  # seconds for predictive analysis
        
        # Start monitoring tasks
        # Do not auto-start monitoring tasks during construction; start explicitly with start()
        self._health_task = None
        self._anomaly_task = None
        self._predictive_task = None

    async def start(self) -> None:
        """Start background monitoring tasks."""
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._start_health_monitoring())
        if self._anomaly_task is None:
            self._anomaly_task = asyncio.create_task(self._start_anomaly_detection())
        if self._predictive_task is None:
            self._predictive_task = asyncio.create_task(self._start_predictive_analysis())

    async def stop(self) -> None:
        """Cancel and await background tasks started by the AVN core."""
        for tname in ("_health_task", "_anomaly_task", "_predictive_task"):
            task = getattr(self, tname, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def _initialize_anomaly_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize anomaly detection thresholds."""
        return {
            "decision_latency": {
                "warning_threshold": 2.0,    # seconds
                "error_threshold": 5.0,      # seconds
                "critical_threshold": 10.0   # seconds
            },
            "confidence_score": {
                "warning_threshold": 0.6,
                "error_threshold": 0.4,
                "critical_threshold": 0.2
            },
            "trust_score": {
                "warning_threshold": 0.7,
                "error_threshold": 0.5,
                "critical_threshold": 0.3
            },
            "memory_usage": {
                "warning_threshold": 0.8,    # 80%
                "error_threshold": 0.9,      # 90%
                "critical_threshold": 0.95   # 95%
            },
            "error_rate": {
                "warning_threshold": 0.05,   # 5%
                "error_threshold": 0.1,      # 10%
                "critical_threshold": 0.2    # 20%
            }
        }
    
    async def _start_health_monitoring(self):
        """Start continuous health monitoring."""
        while True:
            try:
                await self._collect_system_metrics()
                await self._update_component_health_scores()
                await self._check_component_heartbeats()
                
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _start_anomaly_detection(self):
        """Start anomaly detection process."""
        while True:
            try:
                await self._detect_performance_anomalies()
                await self._detect_decision_anomalies()
                await self._detect_trust_anomalies()
                
                await asyncio.sleep(60)  # Run anomaly detection every minute
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(60)
    
    async def _start_predictive_analysis(self):
        """Start predictive failure analysis."""
        while True:
            try:
                await self._analyze_failure_patterns()
                await self._predict_component_failures()
                
                await asyncio.sleep(300)  # Run predictive analysis every 5 minutes
            except Exception as e:
                logger.error(f"Predictive analysis error: {e}")
                await asyncio.sleep(300)
    
    def register_component(self, component_id: str, 
                         health_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        """Register a component for health monitoring."""
        if component_id not in self.component_health:
            self.component_health[component_id] = ComponentHealth(component_id)
        
        # Set custom thresholds if provided
        if health_thresholds:
            component = self.component_health[component_id]
            for metric_name, thresholds in health_thresholds.items():
                if metric_name not in component.performance_baseline:
                    component.performance_baseline[metric_name] = thresholds
        
        logger.info(f"Registered component for health monitoring: {component_id}")
    
    async def report_metric(self, component_id: str, metric_name: str, 
                          value: float, thresholds: Optional[Dict[str, float]] = None):
        """Report a health metric for a component."""
        if component_id not in self.component_health:
            self.register_component(component_id)
        
        # Create metric with thresholds
        threshold_min = None
        threshold_max = None
        
        if thresholds:
            threshold_min = thresholds.get("min")
            threshold_max = thresholds.get("max")
        elif metric_name in self.anomaly_thresholds:
            # Use default thresholds
            thresholds_config = self.anomaly_thresholds[metric_name]
            if "min" in thresholds_config:
                threshold_min = thresholds_config["min"]
            if "max" in thresholds_config:
                threshold_max = thresholds_config["max"]
        
        metric = HealthMetric(
            component_id=component_id,
            metric_name=metric_name,
            value=value,
            threshold_min=threshold_min,
            threshold_max=threshold_max
        )
        
        self.component_health[component_id].add_metric(metric)
        
        # Check for immediate anomalies
        await self._check_metric_anomaly(metric)
    
    async def _collect_system_metrics(self):
        """Collect system-wide health metrics."""
        import psutil
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent / 100.0
            disk_percent = psutil.disk_usage('/').percent / 100.0
            
            await self.report_metric("system", "cpu_usage", cpu_percent / 100.0, 
                                   {"max": 0.9})
            await self.report_metric("system", "memory_usage", memory_percent, 
                                   {"max": 0.9})
            await self.report_metric("system", "disk_usage", disk_percent, 
                                   {"max": 0.95})
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    async def _update_component_health_scores(self):
        """Update health scores for all components."""
        for component_id, component in self.component_health.items():
            health_score = component.calculate_health_score()
            
            await self.report_metric(component_id, "health_score", health_score,
                                   {"min": 0.7})
    
    async def _check_component_heartbeats(self):
        """Check for components that haven't sent heartbeats."""
        current_time = datetime.now()
        
        for component_id, component in self.component_health.items():
            time_since_heartbeat = (current_time - component.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > 300:  # 5 minutes
                await self._raise_alert(
                    AnomalyType.COMPONENT_FAILURE,
                    SeverityLevel.ERROR,
                    component_id,
                    f"Component {component_id} heartbeat timeout ({time_since_heartbeat:.1f}s)",
                    {"time_since_heartbeat": time_since_heartbeat},
                    0.9
                )
    
    async def _check_metric_anomaly(self, metric: HealthMetric):
        """Check if a metric indicates an anomaly."""
        if not metric.is_healthy():
            # Determine severity based on how far from threshold
            severity = SeverityLevel.WARNING
            confidence = 0.7
            
            if metric.threshold_min is not None and metric.value < metric.threshold_min:
                deviation_ratio = (metric.threshold_min - metric.value) / metric.threshold_min
                if deviation_ratio > 0.5:
                    severity = SeverityLevel.CRITICAL
                    confidence = 0.9
                elif deviation_ratio > 0.2:
                    severity = SeverityLevel.ERROR
                    confidence = 0.8
            
            elif metric.threshold_max is not None and metric.value > metric.threshold_max:
                deviation_ratio = (metric.value - metric.threshold_max) / metric.threshold_max
                if deviation_ratio > 0.5:
                    severity = SeverityLevel.CRITICAL
                    confidence = 0.9
                elif deviation_ratio > 0.2:
                    severity = SeverityLevel.ERROR
                    confidence = 0.8
            
            await self._raise_alert(
                AnomalyType.PERFORMANCE_DEGRADATION,
                severity,
                metric.component_id,
                f"Metric {metric.metric_name} outside healthy range: {metric.value}",
                {metric.metric_name: metric.value},
                confidence
            )
    
    async def _detect_performance_anomalies(self):
        """Detect performance-related anomalies."""
        for component_id, component in self.component_health.items():
            # Check for performance degradation trends
            for metric_name in ["decision_latency", "response_time", "throughput"]:
                trend = component.get_metric_trend(metric_name, window_minutes=10)
                
                if trend is not None:
                    if metric_name in ["decision_latency", "response_time"] and trend > 0.1:
                        # Latency increasing
                        await self._raise_alert(
                            AnomalyType.PERFORMANCE_DEGRADATION,
                            SeverityLevel.WARNING,
                            component_id,
                            f"Increasing {metric_name} trend detected",
                            {"trend": trend, "metric": metric_name},
                            0.7
                        )
                    elif metric_name == "throughput" and trend < -0.1:
                        # Throughput decreasing
                        await self._raise_alert(
                            AnomalyType.PERFORMANCE_DEGRADATION,
                            SeverityLevel.WARNING,
                            component_id,
                            f"Decreasing {metric_name} trend detected",
                            {"trend": trend, "metric": metric_name},
                            0.7
                        )
    
    async def _detect_decision_anomalies(self):
        """Detect governance decision-related anomalies."""
        for component_id, component in self.component_health.items():
            if "governance" not in component_id.lower():
                continue
            
            # Check confidence score trends
            confidence_metric = component.get_latest_metric("confidence_score")
            if confidence_metric and confidence_metric.value < 0.5:
                await self._raise_alert(
                    AnomalyType.DECISION_INCONSISTENCY,
                    SeverityLevel.WARNING,
                    component_id,
                    f"Low decision confidence: {confidence_metric.value:.3f}",
                    {"confidence": confidence_metric.value},
                    0.8
                )
            
            # Check for constitutional violations
            compliance_metric = component.get_latest_metric("constitutional_compliance")
            if compliance_metric and compliance_metric.value < 0.8:
                await self._raise_alert(
                    AnomalyType.CONSTITUTIONAL_VIOLATION,
                    SeverityLevel.ERROR,
                    component_id,
                    f"Low constitutional compliance: {compliance_metric.value:.3f}",
                    {"compliance": compliance_metric.value},
                    0.9
                )
    
    async def _detect_trust_anomalies(self):
        """Detect trust-related anomalies."""
        for component_id, component in self.component_health.items():
            # Check trust score degradation
            trust_metric = component.get_latest_metric("trust_score")
            if trust_metric and trust_metric.value < 0.6:
                
                # Check if this is a declining trend
                trust_trend = component.get_metric_trend("trust_score", window_minutes=30)
                if trust_trend is not None and trust_trend < -0.05:
                    await self._raise_alert(
                        AnomalyType.TRUST_EROSION,
                        SeverityLevel.ERROR,
                        component_id,
                        f"Trust erosion detected: score={trust_metric.value:.3f}, trend={trust_trend:.3f}",
                        {"trust_score": trust_metric.value, "trust_trend": trust_trend},
                        0.85
                    )
    
    async def _analyze_failure_patterns(self):
        """Analyze historical failure patterns for prediction."""
        if not self.alert_history:
            return
        
        # Look for recurring patterns
        recent_alerts = [
            alert for alert in self.alert_history[-100:]  # Last 100 alerts
            if (datetime.now() - alert.timestamp).days <= 7
        ]
        
        # Group by component and anomaly type
        pattern_counts = {}
        for alert in recent_alerts:
            key = (alert.component_id, alert.anomaly_type.value)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        # Identify components with high failure rates
        for (component_id, anomaly_type), count in pattern_counts.items():
            if count >= 5:  # 5+ alerts in a week
                await self._raise_alert(
                    AnomalyType.COMPONENT_FAILURE,
                    SeverityLevel.WARNING,
                    component_id,
                    f"High failure rate detected: {count} {anomaly_type} alerts in 7 days",
                    {"alert_count": count, "anomaly_type": anomaly_type},
                    0.8
                )
    
    async def _predict_component_failures(self):
        """Predict potential component failures."""
        for component_id, component in self.component_health.items():
            health_score = component.calculate_health_score()
            
            # Predict failure if health score is declining rapidly
            health_trend = component.get_metric_trend("health_score", window_minutes=60)
            
            if health_trend is not None and health_trend < -0.1 and health_score < 0.6:
                await self._raise_alert(
                    AnomalyType.COMPONENT_FAILURE,
                    SeverityLevel.WARNING,
                    component_id,
                    f"Predictive failure alert: declining health (score={health_score:.3f}, trend={health_trend:.3f})",
                    {"health_score": health_score, "health_trend": health_trend, "predictive": True},
                    0.7
                )
    
    async def _raise_alert(self, anomaly_type: AnomalyType, severity: SeverityLevel,
                         component_id: str, description: str, 
                         metrics: Dict[str, float], confidence: float):
        """Raise an anomaly alert."""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{component_id}_{anomaly_type.value}"
        
        # Check for duplicate alerts
        existing_alert = None
        for alert in self.active_alerts.values():
            if (alert.component_id == component_id and 
                alert.anomaly_type == anomaly_type and
                (datetime.now() - alert.timestamp).total_seconds() < 300):  # 5 minutes
                existing_alert = alert
                break
        
        if existing_alert:
            return  # Don't raise duplicate alerts
        
        # Determine resolution actions
        resolution_actions = self._get_resolution_actions(anomaly_type, component_id, metrics)
        
        alert = AnomalyAlert(
            alert_id=alert_id,
            anomaly_type=anomaly_type,
            severity=severity,
            component_id=component_id,
            description=description,
            metrics=metrics,
            confidence=confidence,
            timestamp=datetime.now(),
            resolution_actions=resolution_actions,
            auto_resolve=(severity in [SeverityLevel.INFO, SeverityLevel.WARNING])
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Publish alert event
        await self.event_bus.publish("ANOMALY_DETECTED", {
            "alert_id": alert_id,
            "type": anomaly_type.value,
            "severity": severity.value,
            "component_id": component_id,
            "description": description,
            "metrics": metrics,
            "confidence": confidence,
            "resolution_actions": resolution_actions
        })
        
        logger.warning(f"Raised {severity.value} alert for {component_id}: {description}")
        
        # Auto-healing for certain types
        if alert.auto_resolve:
            await self._attempt_auto_healing(alert)
    
    def _get_resolution_actions(self, anomaly_type: AnomalyType, 
                              component_id: str, metrics: Dict[str, float]) -> List[str]:
        """Get recommended resolution actions for an anomaly."""
        actions = []
        
        if anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION:
            actions.extend([
                "restart_component",
                "scale_resources",
                "reduce_load"
            ])
        elif anomaly_type == AnomalyType.COMPONENT_FAILURE:
            actions.extend([
                "restart_component",
                "failover_to_backup",
                "check_dependencies"
            ])
        elif anomaly_type == AnomalyType.TRUST_EROSION:
            actions.extend([
                "recalibrate_trust_scores",
                "audit_recent_decisions",
                "review_source_credibility"
            ])
        elif anomaly_type == AnomalyType.CONSTITUTIONAL_VIOLATION:
            actions.extend([
                "review_constitutional_compliance",
                "escalate_to_parliament",
                "rollback_recent_changes"
            ])
        
        return actions
    
    async def _attempt_auto_healing(self, alert: AnomalyAlert):
        """Attempt automatic healing for an alert."""
        for action in alert.resolution_actions:
            if action in self.healing_actions:
                try:
                    healing_func = self.healing_actions[action]
                    success = await healing_func(alert.component_id, alert.metrics)
                    
                    if success:
                        await self.resolve_alert(alert.alert_id, f"Auto-healed via {action}")
                        break
                        
                except Exception as e:
                    logger.error(f"Auto-healing action {action} failed: {e}")
    
    def register_healing_action(self, action_name: str, healing_func: Callable):
        """Register a healing action function."""
        self.healing_actions[action_name] = healing_func
        logger.info(f"Registered healing action: {action_name}")
    
    async def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            del self.active_alerts[alert_id]
            
            await self.event_bus.publish("ANOMALY_RESOLVED", {
                "alert_id": alert_id,
                "component_id": alert.component_id,
                "resolution_note": resolution_note,
                "resolved_at": datetime.now().isoformat()
            })
            
            logger.info(f"Resolved alert {alert_id}: {resolution_note}")
    
    def get_component_health_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific component."""
        if component_id not in self.component_health:
            return None
        
        component = self.component_health[component_id]
        latest_metrics = {}
        
        for metric_name, metric_history in component.metrics.items():
            if metric_history:
                latest_metric = metric_history[-1]
                latest_metrics[metric_name] = {
                    "value": latest_metric.value,
                    "is_healthy": latest_metric.is_healthy(),
                    "timestamp": latest_metric.timestamp.isoformat()
                }
        
        return {
            "component_id": component_id,
            "health_score": component.calculate_health_score(),
            "status": component.status,
            "last_heartbeat": component.last_heartbeat.isoformat(),
            "metrics": latest_metrics,
            "active_alerts": [
                alert.alert_id for alert in self.active_alerts.values()
                if alert.component_id == component_id
            ]
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.component_health:
            return {
                "overall_health": 0.0,
                "healthy_components": 0,
                "total_components": 0,
                "active_alerts": 0,
                "critical_alerts": 0
            }
        
        health_scores = [comp.calculate_health_score() for comp in self.component_health.values()]
        overall_health = statistics.mean(health_scores)
        healthy_components = sum(1 for score in health_scores if score >= 0.7)
        
        active_alerts = len(self.active_alerts)
        critical_alerts = sum(
            1 for alert in self.active_alerts.values() 
            if alert.severity in [SeverityLevel.CRITICAL, SeverityLevel.CATASTROPHIC]
        )
        
        return {
            "overall_health": overall_health,
            "healthy_components": healthy_components,
            "total_components": len(self.component_health),
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "alert_breakdown": {
                severity.value: sum(
                    1 for alert in self.active_alerts.values() 
                    if alert.severity == severity
                ) for severity in SeverityLevel
            }
        }
    
    async def trigger_failover(self, component_id: str, reason: str) -> bool:
        """Trigger failover for a component."""
        try:
            # Publish failover event
            await self.event_bus.publish("COMPONENT_FAILOVER", {
                "component_id": component_id,
                "reason": reason,
                "triggered_at": datetime.now().isoformat(),
                "triggered_by": "avn_core"
            })
            
            logger.warning(f"Triggered failover for {component_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to trigger failover for {component_id}: {e}")
            return False