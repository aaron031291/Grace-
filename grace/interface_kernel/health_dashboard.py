"""
Enhanced Health Interface for Grace - System Health Dashboard
=============================================================

Provides comprehensive system health monitoring and management interface:
- Real-time system status monitoring
- Component health analysis
- Performance metrics visualization
- Health alerts and notifications
- System diagnostics and troubleshooting
- Interactive health dashboard

Usage:
    from grace.interface_kernel.health_dashboard import HealthDashboard

    dashboard = HealthDashboard()
    status = await dashboard.get_comprehensive_health()
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import Grace components for health checks
try:
    from ..core.event_bus import EventBus
    from ..core.memory_core import MemoryCore
    from ..governance.governance_engine import GovernanceEngine
    from ..governance.verification_engine import VerificationEngine

    GRACE_COMPONENTS_AVAILABLE = True
except ImportError:
    GRACE_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""

    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    component_name: str
    status: HealthStatus
    metrics: List[HealthMetric]
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    uptime_seconds: Optional[float] = None


@dataclass
class SystemAlert:
    """System health alert."""

    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False


class HealthDashboard:
    """Comprehensive health monitoring dashboard for Grace."""

    def __init__(self, check_interval_seconds: int = 30):
        self.check_interval = check_interval_seconds
        self.component_health = {}
        self.alerts = []
        self.metrics_history = {}
        self.start_time = datetime.now()
        self.monitoring_active = False
        self._monitoring_task = None

        # Health check configuration
        self.thresholds = {
            "cpu_usage_warning": 80.0,
            "cpu_usage_critical": 95.0,
            "memory_usage_warning": 85.0,
            "memory_usage_critical": 95.0,
            "response_time_warning": 1000.0,  # ms
            "response_time_critical": 5000.0,  # ms
            "disk_usage_warning": 85.0,
            "disk_usage_critical": 95.0,
        }

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all systems."""
        check_start = time.time()

        # System-level checks
        system_health = await self._check_system_health()

        # Grace component checks
        grace_health = await self._check_grace_components()

        # Service-level checks
        service_health = await self._check_services()

        # Aggregate results
        all_components = {**system_health, **grace_health, **service_health}

        # Update component health cache
        for name, health in all_components.items():
            self.component_health[name] = health

            # Store metrics in history
            if name not in self.metrics_history:
                self.metrics_history[name] = []

            self.metrics_history[name].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": health.status.value,
                    "metrics": [asdict(m) for m in health.metrics],
                }
            )

            # Limit history size
            if len(self.metrics_history[name]) > 100:
                self.metrics_history[name] = self.metrics_history[name][-100:]

        # Generate alerts for issues
        await self._generate_alerts(all_components)

        check_duration = (time.time() - check_start) * 1000

        return {
            "check_duration_ms": check_duration,
            "components_checked": len(all_components),
            "timestamp": datetime.now().isoformat(),
            "overall_status": self._calculate_overall_status(all_components).value,
        }

    async def _check_system_health(self) -> Dict[str, ComponentHealth]:
        """Check system-level health metrics."""
        components = {}

        try:
            # CPU Health
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._get_status_for_value(
                cpu_percent,
                self.thresholds["cpu_usage_warning"],
                self.thresholds["cpu_usage_critical"],
            )

            cpu_health = ComponentHealth(
                component_name="CPU",
                status=cpu_status,
                metrics=[
                    HealthMetric(
                        name="cpu_usage",
                        value=cpu_percent,
                        unit="%",
                        status=cpu_status,
                        threshold_warning=self.thresholds["cpu_usage_warning"],
                        threshold_critical=self.thresholds["cpu_usage_critical"],
                        description="Overall CPU utilization",
                    )
                ],
                last_check=datetime.now(),
            )
            components["CPU"] = cpu_health

            # Memory Health
            memory = psutil.virtual_memory()
            memory_status = self._get_status_for_value(
                memory.percent,
                self.thresholds["memory_usage_warning"],
                self.thresholds["memory_usage_critical"],
            )

            memory_health = ComponentHealth(
                component_name="Memory",
                status=memory_status,
                metrics=[
                    HealthMetric(
                        name="memory_usage",
                        value=memory.percent,
                        unit="%",
                        status=memory_status,
                        threshold_warning=self.thresholds["memory_usage_warning"],
                        threshold_critical=self.thresholds["memory_usage_critical"],
                        description="RAM utilization",
                    ),
                    HealthMetric(
                        name="memory_available",
                        value=memory.available / (1024**3),  # GB
                        unit="GB",
                        status=HealthStatus.HEALTHY,
                        description="Available RAM",
                    ),
                ],
                last_check=datetime.now(),
            )
            components["Memory"] = memory_health

            # Disk Health
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._get_status_for_value(
                disk_percent,
                self.thresholds["disk_usage_warning"],
                self.thresholds["disk_usage_critical"],
            )

            disk_health = ComponentHealth(
                component_name="Disk",
                status=disk_status,
                metrics=[
                    HealthMetric(
                        name="disk_usage",
                        value=disk_percent,
                        unit="%",
                        status=disk_status,
                        threshold_warning=self.thresholds["disk_usage_warning"],
                        threshold_critical=self.thresholds["disk_usage_critical"],
                        description="Root disk usage",
                    ),
                    HealthMetric(
                        name="disk_free",
                        value=disk.free / (1024**3),  # GB
                        unit="GB",
                        status=HealthStatus.HEALTHY,
                        description="Available disk space",
                    ),
                ],
                last_check=datetime.now(),
            )
            components["Disk"] = disk_health

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

            # Create error component
            components["System"] = ComponentHealth(
                component_name="System",
                status=HealthStatus.CRITICAL,
                metrics=[],
                last_check=datetime.now(),
                error_message=str(e),
            )

        return components

    async def _check_grace_components(self) -> Dict[str, ComponentHealth]:
        """Check Grace-specific component health."""
        components = {}

        if not GRACE_COMPONENTS_AVAILABLE:
            components["Grace_Components"] = ComponentHealth(
                component_name="Grace Components",
                status=HealthStatus.UNKNOWN,
                metrics=[],
                last_check=datetime.now(),
                error_message="Grace components not available for health checking",
            )
            return components

        # This would be enhanced to actually check Grace components
        # For now, simulate basic checks

        grace_components = [
            "EventBus",
            "MemoryCore",
            "GovernanceEngine",
            "VerificationEngine",
            "TrustCore",
            "Parliament",
        ]

        for component_name in grace_components:
            try:
                # Simulate component health check
                check_start = time.time()

                # Mock health check - would be actual component checks
                await asyncio.sleep(0.01)  # Simulate check time

                response_time = (time.time() - check_start) * 1000

                response_status = self._get_status_for_value(
                    response_time,
                    self.thresholds["response_time_warning"],
                    self.thresholds["response_time_critical"],
                )

                health = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.HEALTHY,  # Would be actual status
                    metrics=[
                        HealthMetric(
                            name="response_time",
                            value=response_time,
                            unit="ms",
                            status=response_status,
                            threshold_warning=self.thresholds["response_time_warning"],
                            threshold_critical=self.thresholds[
                                "response_time_critical"
                            ],
                            description="Component response time",
                        )
                    ],
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
                )

                components[component_name] = health

            except Exception as e:
                components[component_name] = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.CRITICAL,
                    metrics=[],
                    last_check=datetime.now(),
                    error_message=str(e),
                )

        return components

    async def _check_services(self) -> Dict[str, ComponentHealth]:
        """Check external services and dependencies."""
        components = {}

        # Network connectivity check
        try:
            import socket

            check_start = time.time()

            # Test basic connectivity (DNS resolution)
            socket.gethostbyname("google.com")

            response_time = (time.time() - check_start) * 1000

            components["Network"] = ComponentHealth(
                component_name="Network",
                status=HealthStatus.HEALTHY,
                metrics=[
                    HealthMetric(
                        name="dns_response_time",
                        value=response_time,
                        unit="ms",
                        status=HealthStatus.HEALTHY,
                        description="DNS resolution time",
                    )
                ],
                last_check=datetime.now(),
                response_time_ms=response_time,
            )

        except Exception as e:
            components["Network"] = ComponentHealth(
                component_name="Network",
                status=HealthStatus.CRITICAL,
                metrics=[],
                last_check=datetime.now(),
                error_message=f"Network connectivity failed: {str(e)}",
            )

        return components

    def _get_status_for_value(
        self, value: float, warning_threshold: float, critical_threshold: float
    ) -> HealthStatus:
        """Determine health status based on value and thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _calculate_overall_status(
        self, components: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """Calculate overall system status from component statuses."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [comp.status for comp in components.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def _generate_alerts(self, components: Dict[str, ComponentHealth]):
        """Generate alerts for unhealthy components."""
        current_time = datetime.now()

        for name, health in components.items():
            if health.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                # Check if we already have an active alert for this component
                existing_alert = next(
                    (
                        alert
                        for alert in self.alerts
                        if alert.component == name and not alert.resolved
                    ),
                    None,
                )

                if not existing_alert:
                    alert = SystemAlert(
                        alert_id=f"alert_{int(current_time.timestamp())}_{name}",
                        severity="critical"
                        if health.status == HealthStatus.CRITICAL
                        else "warning",
                        component=name,
                        message=health.error_message
                        or f"{name} status is {health.status.value}",
                        timestamp=current_time,
                    )
                    self.alerts.append(alert)

                    logger.warning(f"Health alert generated: {alert.message}")

            elif health.status == HealthStatus.HEALTHY:
                # Resolve any existing alerts for this component
                for alert in self.alerts:
                    if alert.component == name and not alert.resolved:
                        alert.resolved = True
                        logger.info(f"Health alert resolved for {name}")

    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        overall_status = self._calculate_overall_status(self.component_health)

        # Get recent alerts
        recent_alerts = [
            alert
            for alert in self.alerts
            if not alert.resolved
            and (datetime.now() - alert.timestamp).total_seconds() < 3600  # Last hour
        ]

        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "monitoring_active": self.monitoring_active,
            "components": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "response_time_ms": health.response_time_ms,
                    "error": health.error_message,
                    "metrics": [asdict(m) for m in health.metrics],
                }
                for name, health in self.component_health.items()
            },
            "active_alerts": [asdict(alert) for alert in recent_alerts],
            "total_alerts": len(self.alerts),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            },
        }

    def get_component_history(
        self, component_name: str, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get historical metrics for a component."""
        if component_name not in self.metrics_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_history = []
        for entry in self.metrics_history[component_name]:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= cutoff_time:
                filtered_history.append(entry)

        return filtered_history

    def get_alerts(
        self, resolved: Optional[bool] = None, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get system alerts with optional filtering."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = []
        for alert in self.alerts:
            if alert.timestamp >= cutoff_time:
                if resolved is None or alert.resolved == resolved:
                    filtered_alerts.append(asdict(alert))

        return sorted(filtered_alerts, key=lambda x: x["timestamp"], reverse=True)

    async def run_diagnostic(
        self, component_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run detailed diagnostic for a component or all components."""
        diagnostic_start = time.time()

        if component_name:
            # Run diagnostic for specific component
            result = await self._run_component_diagnostic(component_name)
        else:
            # Run full system diagnostic
            result = await self._run_full_diagnostic()

        diagnostic_duration = (time.time() - diagnostic_start) * 1000
        result["diagnostic_duration_ms"] = diagnostic_duration
        result["timestamp"] = datetime.now().isoformat()

        return result

    async def _run_component_diagnostic(self, component_name: str) -> Dict[str, Any]:
        """Run diagnostic for a specific component."""
        if component_name not in self.component_health:
            return {
                "component": component_name,
                "status": "not_found",
                "message": f"Component {component_name} not found in health monitoring",
            }

        health = self.component_health[component_name]

        # Detailed component analysis
        diagnostic = {
            "component": component_name,
            "current_status": health.status.value,
            "last_check": health.last_check.isoformat(),
            "metrics": [asdict(m) for m in health.metrics],
            "history": self.get_component_history(component_name, 6),  # Last 6 hours
            "recommendations": [],
        }

        # Generate recommendations based on status
        if health.status == HealthStatus.CRITICAL:
            diagnostic["recommendations"].append(
                "Immediate attention required - component is in critical state"
            )
        elif health.status == HealthStatus.DEGRADED:
            diagnostic["recommendations"].append(
                "Monitor closely - component performance is degraded"
            )

        # Analyze trends
        history = self.get_component_history(component_name, 24)
        if len(history) > 5:
            recent_statuses = [entry["status"] for entry in history[-5:]]
            if recent_statuses.count("degraded") >= 3:
                diagnostic["recommendations"].append(
                    "Persistent degradation detected - investigate underlying cause"
                )

        return diagnostic

    async def _run_full_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic."""
        # Force a fresh health check
        check_result = await self.perform_health_check()

        overall_status = self._calculate_overall_status(self.component_health)

        diagnostic = {
            "overall_status": overall_status.value,
            "components_count": len(self.component_health),
            "unhealthy_components": [
                name
                for name, health in self.component_health.items()
                if health.status
                in [
                    HealthStatus.CRITICAL,
                    HealthStatus.UNHEALTHY,
                    HealthStatus.DEGRADED,
                ]
            ],
            "active_alerts": len(
                [alert for alert in self.alerts if not alert.resolved]
            ),
            "system_recommendations": [],
            "component_summaries": {},
        }

        # Component summaries
        for name, health in self.component_health.items():
            diagnostic["component_summaries"][name] = {
                "status": health.status.value,
                "response_time_ms": health.response_time_ms,
                "error": health.error_message,
            }

        # System-level recommendations
        unhealthy_count = len(diagnostic["unhealthy_components"])
        if unhealthy_count > 0:
            diagnostic["system_recommendations"].append(
                f"{unhealthy_count} components require attention"
            )

        if overall_status == HealthStatus.CRITICAL:
            diagnostic["system_recommendations"].append(
                "System is in critical state - immediate intervention required"
            )

        return diagnostic
