"""
Grace AI Immune System - Threat Detection Module
Identifies anomalies, vulnerabilities, and security issues
"""
import logging
from typing import Dict, Any, List
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class DetectionType(Enum):
    """Types of threat detection."""
    ANOMALY = "anomaly"
    VULNERABILITY = "vulnerability"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY = "security"
    DATA_INTEGRITY = "data_integrity"

class ThreatDetector:
    """Detects threats and anomalies in Grace's operations."""
    
    def __init__(self, kpi_monitor=None, immutable_logger=None):
        self.kpi_monitor = kpi_monitor
        self.immutable_logger = immutable_logger
        self.detection_rules: Dict[str, callable] = {}
        self.anomalies_detected: List[Dict[str, Any]] = []
    
    def register_detection_rule(self, rule_name: str, rule_function: callable):
        """Register a custom detection rule."""
        self.detection_rules[rule_name] = rule_function
        logger.info(f"Registered detection rule: {rule_name}")
    
    async def check_kpi_anomalies(self) -> List[Dict[str, Any]]:
        """Check for KPI anomalies."""
        anomalies = []
        
        if not self.kpi_monitor:
            return anomalies
        
        kpis = self.kpi_monitor.get_all_kpis()
        
        # Check for degradation
        for kpi_name, value in kpis.items():
            if value < 90.0:  # Threshold
                anomaly = {
                    "type": DetectionType.PERFORMANCE_DEGRADATION.value,
                    "kpi": kpi_name,
                    "value": value,
                    "detected_at": datetime.now().isoformat()
                }
                anomalies.append(anomaly)
                self.anomalies_detected.append(anomaly)
        
        return anomalies
    
    async def check_trust_anomalies(self, trust_threshold: float = 30.0) -> List[Dict[str, Any]]:
        """Check for trust score anomalies."""
        anomalies = []
        
        if not self.kpi_monitor:
            return anomalies
        
        overall_trust = self.kpi_monitor.get_overall_trust()
        
        if overall_trust < trust_threshold:
            anomaly = {
                "type": DetectionType.ANOMALY.value,
                "category": "trust_degradation",
                "trust_score": overall_trust,
                "detected_at": datetime.now().isoformat()
            }
            anomalies.append(anomaly)
            self.anomalies_detected.append(anomaly)
        
        return anomalies
    
    async def run_detection_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all registered detection rules."""
        detections = []
        
        for rule_name, rule_func in self.detection_rules.items():
            try:
                if asyncio.iscoroutinefunction(rule_func):
                    result = await rule_func(context)
                else:
                    result = rule_func(context)
                
                if result:
                    detections.append({
                        "rule": rule_name,
                        "result": result,
                        "detected_at": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error running detection rule {rule_name}: {str(e)}")
        
        return detections
    
    def get_anomalies_summary(self) -> Dict[str, int]:
        """Get a summary of detected anomalies."""
        summary = {}
        for anomaly in self.anomalies_detected:
            anomaly_type = anomaly.get("type", "unknown")
            summary[anomaly_type] = summary.get(anomaly_type, 0) + 1
        
        return summary
