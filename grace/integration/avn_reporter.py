"""
AVN Self-Diagnostic Reporter - Reports component health to AVN system
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Self-diagnostic report entry"""
    component: str
    data: Dict[str, Any]
    timestamp: datetime
    severity: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AVNReporter:
    """
    AVN (Autonomous Validation Network) Self-Diagnostic Reporter
    Collects and reports component health diagnostics
    """
    
    def __init__(self):
        self.diagnostic_history: List[DiagnosticReport] = []
        self.component_status: Dict[str, str] = {}
        logger.info("AVNReporter initialized")
    
    def report_diagnostic(
        self,
        component: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        severity: str = "info"
    ):
        """Report diagnostic data from a component"""
        report = DiagnosticReport(
            component=component,
            data=data,
            timestamp=timestamp or datetime.now(),
            severity=severity
        )
        
        self.diagnostic_history.append(report)
        
        # Update component status
        if 'status' in data:
            self.component_status[component] = data['status']
        
        # Log based on severity
        if severity == "critical":
            logger.critical(f"AVN Diagnostic [{component}]: {data}")
        elif severity == "error":
            logger.error(f"AVN Diagnostic [{component}]: {data}")
        elif severity == "warning":
            logger.warning(f"AVN Diagnostic [{component}]: {data}")
        else:
            logger.info(f"AVN Diagnostic [{component}]: {data}")
    
    def get_component_health(self, component: str) -> Optional[str]:
        """Get current health status of a component"""
        return self.component_status.get(component)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.component_status:
            return {'status': 'unknown', 'components': 0}
        
        statuses = list(self.component_status.values())
        
        # Determine overall status
        if 'offline' in statuses or 'critical' in statuses:
            overall = 'critical'
        elif 'degraded' in statuses:
            overall = 'degraded'
        elif all(s == 'healthy' for s in statuses):
            overall = 'healthy'
        else:
            overall = 'unknown'
        
        return {
            'status': overall,
            'components': len(self.component_status),
            'component_statuses': self.component_status,
            'total_reports': len(self.diagnostic_history)
        }
