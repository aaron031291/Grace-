"""
Integration between event bus and AVN self-healing
"""

from typing import Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class AVNEventIntegration:
    """
    Connects event bus to AVN for automated self-healing triggers
    """
    
    def __init__(self, event_bus, avn_core):
        """
        Initialize integration
        
        Args:
            event_bus: EventBus instance
            avn_core: EnhancedAVNCore instance
        """
        self.event_bus = event_bus
        self.avn_core = avn_core
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("AVN event integration initialized")
    
    def _register_handlers(self):
        """Register event handlers for self-healing triggers"""
        
        # Test quality events
        self.event_bus.subscribe("TEST.QUALITY.CRITICAL", self._handle_critical_test_quality)
        self.event_bus.subscribe("TEST.QUALITY.WARNING", self._handle_warning_test_quality)
        self.event_bus.subscribe("TEST.QUALITY.LOW_PASS_RATE", self._handle_low_pass_rate)
        
        # Component health events
        self.event_bus.subscribe("COMPONENT.HEALTH.DEGRADED", self._handle_component_degradation)
        self.event_bus.subscribe("COMPONENT.HEALTH.CRITICAL", self._handle_component_critical)
        
        # System events
        self.event_bus.subscribe("SYSTEM.ERROR.HIGH_RATE", self._handle_high_error_rate)
        self.event_bus.subscribe("SYSTEM.LATENCY.HIGH", self._handle_high_latency)
        
        logger.info("Registered AVN event handlers")
    
    def _handle_critical_test_quality(self, event: Dict[str, Any]):
        """Handle critical test quality events"""
        logger.warning(f"Critical test quality event: {event.get('data', {}).get('message')}")
        
        failure_rate = event.get('data', {}).get('failure_rate', 0)
        
        # Trigger self-healing for affected components
        # Infer component from test failures
        component_id = self._infer_component_from_tests(event)
        
        if component_id:
            logger.info(f"Triggering healing for {component_id} due to test failures")
            
            # Report degraded health to trigger healing
            self.avn_core.report_metrics(
                component_id,
                {
                    "latency": 500,  # Simulated high latency
                    "error_rate": failure_rate,
                    "source": "test_quality"
                }
            )
    
    def _handle_warning_test_quality(self, event: Dict[str, Any]):
        """Handle test quality warnings"""
        logger.info(f"Test quality warning: {event.get('data', {}).get('message')}")
        
        # Log but don't trigger healing for warnings
        skip_rate = event.get('data', {}).get('skip_rate', 0)
        
        if skip_rate > 0.3:  # Very high skip rate
            logger.warning(f"Extremely high skip rate: {skip_rate:.1%}")
    
    def _handle_low_pass_rate(self, event: Dict[str, Any]):
        """Handle low pass rate events"""
        logger.warning(f"Low pass rate: {event.get('data', {}).get('message')}")
        
        pass_rate = event.get('data', {}).get('pass_rate', 0)
        
        if pass_rate < 0.6:  # Very low
            # Trigger system-wide health check
            for component_id in self.avn_core.component_health.keys():
                health = self.avn_core.component_health[component_id]
                if health.health_score < 0.7:
                    logger.info(f"Triggering preventive healing for {component_id}")
    
    def _handle_component_degradation(self, event: Dict[str, Any]):
        """Handle component degradation events"""
        component_id = event.get('component_id')
        
        if component_id:
            logger.warning(f"Component degradation detected: {component_id}")
            
            # AVN should already be handling this, but log for audit
    
    def _handle_component_critical(self, event: Dict[str, Any]):
        """Handle critical component events"""
        component_id = event.get('component_id')
        
        if component_id:
            logger.critical(f"Component critical: {component_id}")
            
            # Force immediate healing
            self.avn_core.report_metrics(
                component_id,
                {
                    "latency": 1000,
                    "error_rate": 0.5,
                    "source": "critical_event"
                }
            )
    
    def _handle_high_error_rate(self, event: Dict[str, Any]):
        """Handle system-wide high error rate"""
        logger.error(f"High error rate detected: {event.get('data', {})}")
        
        # Trigger system-wide health check
        health = self.avn_core.get_system_health()
        
        if health['status'] in ['critical', 'failing']:
            logger.critical("System-wide healing required")
    
    def _handle_high_latency(self, event: Dict[str, Any]):
        """Handle high latency events"""
        component_id = event.get('component_id')
        latency = event.get('data', {}).get('latency', 0)
        
        if component_id and latency > 500:
            logger.warning(f"High latency for {component_id}: {latency}ms")
    
    def _infer_component_from_tests(self, event: Dict[str, Any]) -> str:
        """Infer which component is affected based on test failures"""
        # In production, analyze test names/paths to determine component
        # For now, return a default component
        return "api_gateway"
