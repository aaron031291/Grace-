"""
KPI Trust Monitor - Monitors system health, performance, and trust metrics.
Part of Phase 2: Core Spine Boot implementation.
"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

from ..config.environment import get_grace_config

logger = logging.getLogger(__name__)


@dataclass
class KPIMetric:
    """Individual KPI metric data structure."""
    name: str
    value: float
    timestamp: datetime
    component_id: str
    instance_id: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class TrustScore:
    """Trust score tracking for components and instances."""
    component_id: str
    instance_id: str
    score: float
    confidence: float
    last_updated: datetime
    interaction_count: int
    recent_performance: List[float]


class KPITrustMonitor:
    """
    Monitors KPI metrics and trust scores across Grace components.
    Provides real-time health monitoring and anomaly detection.
    """
    
    def __init__(self, event_publisher: Optional[Callable] = None):
        self.config = get_grace_config()
        self.event_publisher = event_publisher
        self.running = False
        
        # Metric storage
        self.metrics: Dict[str, List[KPIMetric]] = {}
        self.trust_scores: Dict[str, TrustScore] = {}
        
        # Monitoring configuration
        self.monitoring_interval = self.config["health_config"]["monitoring_interval"]
        self.anomaly_thresholds = self.config["health_config"]["anomaly_thresholds"]
        
        # Statistics tracking
        self.metric_history_limit = 1000
        self.trust_decay_rate = self.config["trust_config"]["decay_rate"]
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        logger.info("KPITrustMonitor initialized")
    
    async def start(self):
        """Start the KPI trust monitoring system."""
        if self.running:
            logger.warning("KPITrustMonitor already running")
            return
            
        self.running = True
        logger.info("Starting KPITrustMonitor...")
        
        # Start monitoring loops
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._trust_decay_loop())
        
        if self.event_publisher:
            await self.event_publisher("kpi_monitor_started", {
                "instance_id": self.config["environment_config"]["instance_id"],
                "monitoring_interval": self.monitoring_interval
            })
    
    async def stop(self):
        """Stop the KPI trust monitoring system."""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopped KPITrustMonitor")
        
        if self.event_publisher:
            await self.event_publisher("kpi_monitor_stopped", {
                "instance_id": self.config["environment_config"]["instance_id"]
            })
    
    async def record_metric(self, name: str, value: float, component_id: str,
                          threshold_warning: Optional[float] = None,
                          threshold_critical: Optional[float] = None,
                          tags: Optional[Dict[str, str]] = None):
        """Record a KPI metric."""
        metric = KPIMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            component_id=component_id,
            instance_id=self.config["environment_config"]["instance_id"],
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            tags=tags or {}
        )
        
        # Store metric
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # Limit history size
        if len(self.metrics[name]) > self.metric_history_limit:
            self.metrics[name] = self.metrics[name][-self.metric_history_limit:]
        
        # Check for anomalies
        await self._check_metric_anomalies(metric)
        
        # Publish metric event
        if self.event_publisher:
            await self.event_publisher("kpi_metric_recorded", asdict(metric))
        
        logger.debug(f"Recorded metric {name}={value} for {component_id}")
    
    async def update_trust_score(self, component_id: str, performance_score: float,
                               confidence: float = 1.0):
        """Update trust score for a component based on performance."""
        instance_id = self.config["environment_config"]["instance_id"]
        key = f"{instance_id}:{component_id}"
        
        if key not in self.trust_scores:
            self.trust_scores[key] = TrustScore(
                component_id=component_id,
                instance_id=instance_id,
                score=0.5,  # Start neutral
                confidence=confidence,
                last_updated=datetime.now(),
                interaction_count=0,
                recent_performance=[]
            )
        
        trust_score = self.trust_scores[key]
        
        # Update performance history
        trust_score.recent_performance.append(performance_score)
        if len(trust_score.recent_performance) > 100:  # Keep last 100
            trust_score.recent_performance = trust_score.recent_performance[-100:]
        
        # Calculate new trust score using exponential moving average
        alpha = 0.1  # Learning rate
        trust_score.score = (1 - alpha) * trust_score.score + alpha * performance_score
        trust_score.confidence = min(1.0, confidence)
        trust_score.last_updated = datetime.now()
        trust_score.interaction_count += 1
        
        # Publish trust update event
        if self.event_publisher:
            await self.event_publisher("trust_score_updated", {
                "component_id": component_id,
                "instance_id": instance_id,
                "new_score": trust_score.score,
                "confidence": trust_score.confidence,
                "interaction_count": trust_score.interaction_count
            })
        
        logger.debug(f"Updated trust score for {component_id}: {trust_score.score:.3f}")
    
    def get_trust_score(self, component_id: str) -> Optional[TrustScore]:
        """Get current trust score for a component."""
        instance_id = self.config["environment_config"]["instance_id"]
        key = f"{instance_id}:{component_id}"
        return self.trust_scores.get(key)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)  # Recent metrics only
        
        health_data = {
            "timestamp": now.isoformat(),
            "instance_id": self.config["environment_config"]["instance_id"],
            "status": "healthy",
            "metrics_summary": {},
            "trust_summary": {},
            "anomalies": [],
            "alerts": []
        }
        
        # Summarize recent metrics
        for name, metric_list in self.metrics.items():
            recent_metrics = [m for m in metric_list if m.timestamp > cutoff]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                health_data["metrics_summary"][name] = {
                    "count": len(values),
                    "latest": values[-1],
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Summarize trust scores
        for key, trust_score in self.trust_scores.items():
            health_data["trust_summary"][trust_score.component_id] = {
                "score": trust_score.score,
                "confidence": trust_score.confidence,
                "interactions": trust_score.interaction_count,
                "last_updated": trust_score.last_updated.isoformat()
            }
        
        # Check overall system status
        low_trust_components = [
            ts.component_id for ts in self.trust_scores.values() 
            if ts.score < self.config["trust_config"].get("critical_threshold", 0.3)
        ]
        
        if low_trust_components:
            health_data["status"] = "degraded"
            health_data["alerts"].append(f"Low trust components: {', '.join(low_trust_components)}")
        
        return health_data
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    async def _monitoring_loop(self):
        """Main monitoring loop for collecting system metrics."""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Generate health report
                health_report = self.get_system_health()
                
                # Publish health report
                if self.event_publisher:
                    await self.event_publisher("system_health_report", health_report)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _trust_decay_loop(self):
        """Apply trust score decay over time for inactive components."""
        while self.running:
            try:
                now = datetime.now()
                decay_threshold = timedelta(days=1)
                
                for trust_score in self.trust_scores.values():
                    if now - trust_score.last_updated > decay_threshold:
                        # Apply decay
                        trust_score.score *= (1 - self.trust_decay_rate)
                        logger.debug(f"Applied trust decay to {trust_score.component_id}")
                
                # Run daily
                await asyncio.sleep(86400)  # 24 hours
                
            except Exception as e:
                logger.error(f"Error in trust decay loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _collect_system_metrics(self):
        """Collect basic system metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("cpu_usage", cpu_percent, "system",
                                   threshold_warning=80.0, threshold_critical=95.0)
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_metric("memory_usage", memory.percent, "system",
                                   threshold_warning=80.0, threshold_critical=95.0)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_metric("disk_usage", disk_percent, "system",
                                   threshold_warning=80.0, threshold_critical=95.0)
            
        except ImportError:
            # psutil not available, record placeholder metrics
            await self.record_metric("cpu_usage", 0.0, "system", tags={"source": "placeholder"})
            await self.record_metric("memory_usage", 0.0, "system", tags={"source": "placeholder"})
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _check_metric_anomalies(self, metric: KPIMetric):
        """Check for metric anomalies and trigger alerts."""
        anomaly_detected = False
        severity = "info"
        
        # Check against configured thresholds
        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            anomaly_detected = True
            severity = "critical"
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            anomaly_detected = True
            severity = "warning"
        
        # Check against dynamic anomaly detection
        if metric.name in self.metrics and len(self.metrics[metric.name]) > 10:
            recent_values = [m.value for m in self.metrics[metric.name][-10:]]
            mean_val = statistics.mean(recent_values)
            stdev_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            # Simple anomaly detection: value is more than 3 standard deviations from mean
            if stdev_val > 0 and abs(metric.value - mean_val) > 3 * stdev_val:
                anomaly_detected = True
                if severity == "info":
                    severity = "warning"
        
        if anomaly_detected:
            alert_data = {
                "metric_name": metric.name,
                "value": metric.value,
                "component_id": metric.component_id,
                "severity": severity,
                "timestamp": metric.timestamp.isoformat(),
                "threshold_warning": metric.threshold_warning,
                "threshold_critical": metric.threshold_critical
            }
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert_data)
                except Exception as e:
                    logger.error(f"Error calling alert callback: {e}")
            
            # Publish alert event
            if self.event_publisher:
                await self.event_publisher("kpi_anomaly_detected", alert_data)
            
            logger.warning(f"Anomaly detected: {metric.name}={metric.value} ({severity})")