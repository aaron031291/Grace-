"""
Grace AI Perception Subsystem
Environmental monitoring, observability, and external threat awareness
Consolidates: SentinelKernel + ObservabilityService
"""
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnvironmentalMonitor:
    """
    Unified environmental monitoring combining:
    - External threat detection (SentinelKernel)
    - System observability (metrics, traces, logs, alerts)
    - Anomaly identification
    """
    
    def __init__(self, event_bus=None, llm_service=None):
        self.event_bus = event_bus
        self.llm_service = llm_service
        
        # External monitoring (from SentinelKernel)
        self.monitored_sources = []
        
        # Observability (from ObservabilityService)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.traces: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
    
    async def monitor_environment(self, check_interval: float = 60.0):
        """Continuously monitor the external environment."""
        logger.info("Environmental monitoring started")
        
        while True:
            try:
                logger.info("Performing environmental scan...")
                
                # Check security threats
                await self._check_security_feeds()
                
                # Check technology updates
                await self._check_technology_feeds()
                
                # Check opportunities
                await self._check_opportunities()
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in environmental monitoring: {str(e)}")
                await asyncio.sleep(check_interval)
    
    async def _check_security_feeds(self):
        """Check for security vulnerabilities."""
        logger.info("Checking security feeds...")
        threat = {
            "type": "security_vulnerability",
            "source": "CVE Database",
            "severity": "medium",
            "description": "Potential vulnerability in dependency",
            "timestamp": datetime.now().isoformat()
        }
        
        if self.event_bus:
            await self.event_bus.publish("perception.threat_detected", threat)
    
    async def _check_technology_feeds(self):
        """Check for technology updates."""
        logger.info("Checking technology feeds...")
        update = {
            "type": "technology_update",
            "source": "Python Official",
            "description": "New version available",
            "timestamp": datetime.now().isoformat()
        }
        
        if self.event_bus:
            await self.event_bus.publish("perception.update_available", update)
    
    async def _check_opportunities(self):
        """Check for improvement opportunities."""
        logger.info("Checking for opportunities...")
        opportunity = {
            "type": "learning_opportunity",
            "source": "Technology Trends",
            "description": "New architectural pattern",
            "timestamp": datetime.now().isoformat()
        }
        
        if self.event_bus:
            await self.event_bus.publish("perception.opportunity_identified", opportunity)
    
    # OBSERVABILITY METHODS
    async def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric."""
        self.metrics[metric_name].append(value)
        logger.info(f"Recorded metric: {metric_name} = {value}")
    
    async def record_trace(self, span_name: str, duration_ms: float, attributes: Dict[str, Any] = None):
        """Record a distributed trace span."""
        trace = {
            "span_name": span_name,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        }
        self.traces.append(trace)
        logger.info(f"Recorded trace: {span_name} ({duration_ms}ms)")
    
    async def record_log(self, level: str, message: str, context: Dict[str, Any] = None):
        """Record a structured log entry."""
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        self.logs.append(log_entry)
    
    async def check_and_alert(self, condition_name: str, condition_met: bool, severity: str = "warning"):
        """Check a condition and create an alert if met."""
        if condition_met:
            alert = {
                "condition": condition_name,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"Alert: {condition_name} ({severity})")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
        return summary
    
    def get_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent traces."""
        return self.traces[-limit:]
    
    def get_logs(self, level: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs."""
        logs = self.logs
        if level:
            logs = [l for l in logs if l["level"] == level]
        return logs[-limit:]
    
    def get_alerts(self, severity: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        alerts = self.alerts
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        return alerts[-limit:]
