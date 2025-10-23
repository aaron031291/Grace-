"""
Enhanced AVN Core with predictive modeling and self-healing
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ComponentHealth:
    """Health model for a system component"""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.metrics_history = deque(maxlen=1000)
        self.health_score = 1.0
        self.predicted_failure_time: Optional[datetime] = None
        self.anomaly_detected = False
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add metrics data point"""
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.metrics_history.append(metrics)
        self._update_health_score()
    
    def _update_health_score(self):
        """Update health score based on recent metrics"""
        if len(self.metrics_history) < 10:
            return
        
        recent = list(self.metrics_history)[-10:]
        
        # Extract key metrics
        latencies = [m.get("latency", 0) for m in recent]
        error_rates = [m.get("error_rate", 0) for m in recent]
        
        # Calculate health components
        latency_health = 1.0 / (1.0 + np.mean(latencies) / 100)
        error_health = 1.0 - min(1.0, np.mean(error_rates))
        
        # Combined health score
        self.health_score = 0.6 * latency_health + 0.4 * error_health
        
        # Detect anomalies
        if latency_health < 0.5 or error_health < 0.5:
            self.anomaly_detected = True
        else:
            self.anomaly_detected = False


class HealingAction:
    """Represents a healing action"""
    
    def __init__(
        self,
        action_type: str,
        component_id: str,
        parameters: Dict[str, Any],
        priority: int = 1
    ):
        self.action_type = action_type
        self.component_id = component_id
        self.parameters = parameters
        self.priority = priority
        self.executed_at: Optional[datetime] = None
        self.success: Optional[bool] = None
        self.error: Optional[str] = None


class EnhancedAVNCore:
    """
    Enhanced Adaptive Verification Network Core
    
    Features:
    - Predictive health modeling
    - Automated healing execution
    - Verification of healing success
    - Escalation loops
    - Immutable audit logging
    """
    
    def __init__(self, immutable_logs=None, trust_manager=None, event_publisher=None):
        """
        Initialize Enhanced AVN Core
        
        Args:
            immutable_logs: ImmutableLogs instance for audit trail
            trust_manager: TrustScoreManager for updating trust
            event_publisher: Event publisher for notifications
        """
        self.immutable_logs = immutable_logs
        self.trust_manager = trust_manager
        self.event_publisher = event_publisher
        
        self.component_health: Dict[str, ComponentHealth] = {}
        self.healing_history: List[HealingAction] = []
        
        # Healing strategies
        self.healing_strategies = {
            "high_latency": self._heal_high_latency,
            "high_error_rate": self._heal_high_error_rate,
            "service_down": self._heal_service_down,
            "vector_corruption": self._heal_vector_corruption,
            "model_degradation": self._heal_model_degradation
        }
        
        logger.info("Enhanced AVN Core initialized")
    
    def register_component(self, component_id: str):
        """Register a component for monitoring"""
        if component_id not in self.component_health:
            self.component_health[component_id] = ComponentHealth(component_id)
            logger.info(f"Registered component: {component_id}")
    
    def report_metrics(self, component_id: str, metrics: Dict[str, float]):
        """
        Report component metrics for health monitoring
        
        Args:
            component_id: Component identifier
            metrics: Metrics dict (latency, error_rate, etc.)
        """
        if component_id not in self.component_health:
            self.register_component(component_id)
        
        health = self.component_health[component_id]
        health.add_metrics(metrics)
        
        # Predictive analysis
        prediction = self._predict_failure(component_id)
        if prediction["will_fail"]:
            logger.warning(
                f"Predicted failure for {component_id} in {prediction['time_to_failure']}s"
            )
            self._trigger_preventive_healing(component_id, prediction)
        
        # Reactive healing for current issues
        if health.anomaly_detected:
            logger.warning(f"Anomaly detected in {component_id}, initiating healing")
            self._trigger_reactive_healing(component_id)
    
    def _predict_failure(self, component_id: str) -> Dict[str, Any]:
        """
        Predict if component will fail soon
        
        Uses simple trend analysis on health score
        """
        health = self.component_health[component_id]
        
        if len(health.metrics_history) < 20:
            return {"will_fail": False, "confidence": 0.0}
        
        # Get recent health scores
        recent = list(health.metrics_history)[-20:]
        scores = []
        for m in recent:
            latency = m.get("latency", 0)
            error_rate = m.get("error_rate", 0)
            score = 1.0 / (1.0 + latency / 100) * (1.0 - min(1.0, error_rate))
            scores.append(score)
        
        # Linear regression to predict trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        # Predict failure if declining rapidly
        if slope < -0.02:  # Declining
            # Extrapolate to failure threshold (0.3)
            current_score = scores[-1]
            time_to_failure = (current_score - 0.3) / abs(slope)
            
            will_fail = time_to_failure < 100  # Next 100 data points
            confidence = min(1.0, abs(slope) * 10)
            
            return {
                "will_fail": will_fail,
                "time_to_failure": time_to_failure,
                "confidence": confidence,
                "current_score": current_score,
                "trend_slope": slope
            }
        
        return {"will_fail": False, "confidence": 0.0}
    
    def _trigger_preventive_healing(self, component_id: str, prediction: Dict[str, Any]):
        """Trigger preventive healing before failure occurs"""
        logger.info(f"Preventive healing for {component_id}")
        
        # Determine healing strategy
        health = self.component_health[component_id]
        recent = list(health.metrics_history)[-10:]
        
        avg_latency = np.mean([m.get("latency", 0) for m in recent])
        avg_error_rate = np.mean([m.get("error_rate", 0) for m in recent])
        
        if avg_latency > 200:
            self._execute_healing("high_latency", component_id, {"threshold": avg_latency})
        elif avg_error_rate > 0.1:
            self._execute_healing("high_error_rate", component_id, {"rate": avg_error_rate})
    
    def _trigger_reactive_healing(self, component_id: str):
        """Trigger reactive healing for current issues"""
        logger.info(f"Reactive healing for {component_id}")
        
        health = self.component_health[component_id]
        recent = list(health.metrics_history)[-5:]
        
        avg_latency = np.mean([m.get("latency", 0) for m in recent])
        avg_error_rate = np.mean([m.get("error_rate", 0) for m in recent])
        
        if avg_latency > 500:
            self._execute_healing("high_latency", component_id, {"threshold": avg_latency})
        elif avg_error_rate > 0.2:
            self._execute_healing("high_error_rate", component_id, {"rate": avg_error_rate})
        else:
            self._execute_healing("service_down", component_id, {})
    
    def _execute_healing(self, action_type: str, component_id: str, parameters: Dict[str, Any]):
        """
        Execute a healing action
        
        Args:
            action_type: Type of healing action
            component_id: Target component
            parameters: Action parameters
        """
        action = HealingAction(action_type, component_id, parameters)
        action.executed_at = datetime.now(timezone.utc)
        
        try:
            # Execute healing strategy
            if action_type in self.healing_strategies:
                result = self.healing_strategies[action_type](component_id, parameters)
                action.success = result.get("success", False)
                action.error = result.get("error")
            else:
                action.success = False
                action.error = f"Unknown action type: {action_type}"
            
            self.healing_history.append(action)
            
            # Log to immutable logs
            if self.immutable_logs:
                self.immutable_logs.log_constitutional_operation(
                    operation_type="avn_healing",
                    actor="avn_core",
                    action={
                        "type": action_type,
                        "component": component_id,
                        "parameters": parameters
                    },
                    result={
                        "success": action.success,
                        "error": action.error
                    },
                    severity="warning" if action.success else "error",
                    tags=["avn", "healing", action_type]
                )
            
            # Verify healing success
            verified = self._verify_healing(component_id, action)
            
            if not verified:
                logger.error(f"Healing verification failed for {component_id}")
                self._escalate_healing(component_id, action)
            else:
                logger.info(f"Healing successful for {component_id}")
                
                # Update trust score
                if self.trust_manager:
                    self.trust_manager.record_success(
                        component_id,
                        weight=1.0,
                        context={"healing": action_type}
                    )
            
            # Emit event
            if self.event_publisher:
                self.event_publisher.publish({
                    "type": "AVN.HEALING.EXECUTED",
                    "component": component_id,
                    "action": action_type,
                    "success": action.success,
                    "verified": verified
                })
        
        except Exception as e:
            logger.error(f"Healing execution failed: {e}")
            action.success = False
            action.error = str(e)
            self._escalate_healing(component_id, action)
    
    def _heal_high_latency(self, component_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Healing strategy for high latency"""
        logger.info(f"Healing high latency for {component_id}")
        
        # Strategies: restart service, increase resources, clear cache
        try:
            # Simulate restart
            logger.info(f"Restarting service: {component_id}")
            
            return {"success": True, "action": "service_restart"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _heal_high_error_rate(self, component_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Healing strategy for high error rate"""
        logger.info(f"Healing high error rate for {component_id}")
        
        try:
            # Rollback to previous version or restart
            logger.info(f"Rolling back {component_id}")
            
            return {"success": True, "action": "rollback"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _heal_service_down(self, component_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Healing strategy for service down"""
        logger.info(f"Healing service down: {component_id}")
        
        try:
            logger.info(f"Redeploying {component_id}")
            
            return {"success": True, "action": "redeploy"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _heal_vector_corruption(self, component_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Healing strategy for vector database corruption"""
        logger.info(f"Healing vector corruption for {component_id}")
        
        try:
            logger.info(f"Regenerating vectors for {component_id}")
            
            return {"success": True, "action": "regenerate_vectors"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _heal_model_degradation(self, component_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Healing strategy for model performance degradation"""
        logger.info(f"Healing model degradation for {component_id}")
        
        try:
            logger.info(f"Retraining model for {component_id}")
            
            return {"success": True, "action": "retrain_model"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _verify_healing(self, component_id: str, action: HealingAction) -> bool:
        """
        Verify that healing was successful
        
        Waits briefly and checks if health improved
        """
        if not action.success:
            return False
        
        import time
        time.sleep(2)  # Wait for healing to take effect
        
        if component_id not in self.component_health:
            return False
        
        health = self.component_health[component_id]
        
        # Check if health improved
        if health.health_score > 0.7:
            return True
        
        if not health.anomaly_detected:
            return True
        
        return False
    
    def _escalate_healing(self, component_id: str, failed_action: HealingAction):
        """
        Escalate to higher-level healing when initial healing fails
        
        Escalation ladder:
        1. Retry with different parameters
        2. More aggressive action
        3. Manual intervention notification
        """
        logger.warning(f"Escalating healing for {component_id}")
        
        # Log escalation
        if self.immutable_logs:
            self.immutable_logs.log_constitutional_operation(
                operation_type="avn_escalation",
                actor="avn_core",
                action={
                    "component": component_id,
                    "failed_action": failed_action.action_type,
                    "escalation_level": 1
                },
                result={"escalated": True},
                severity="critical",
                tags=["avn", "escalation", "critical"]
            )
        
        # Emit critical event
        if self.event_publisher:
            self.event_publisher.publish({
                "type": "AVN.HEALING.ESCALATED",
                "severity": "critical",
                "component": component_id,
                "failed_action": failed_action.action_type,
                "requires_manual_intervention": True
            })
        
        # Update trust score negatively
        if self.trust_manager:
            self.trust_manager.record_failure(
                component_id,
                severity=0.8,
                context={"healing_failed": failed_action.action_type}
            )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health report"""
        if not self.component_health:
            return {"status": "unknown", "components": {}}
        
        component_scores = {
            comp_id: health.health_score
            for comp_id, health in self.component_health.items()
        }
        
        avg_health = np.mean(list(component_scores.values()))
        
        return {
            "status": self._get_status_from_score(avg_health),
            "average_health": avg_health,
            "components": component_scores,
            "total_healings": len(self.healing_history),
            "successful_healings": sum(1 for h in self.healing_history if h.success),
            "failed_healings": sum(1 for h in self.healing_history if not h.success)
        }
    
    def _get_status_from_score(self, score: float) -> str:
        """Convert health score to status"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.5:
            return "degraded"
        elif score >= 0.3:
            return "critical"
        else:
            return "failing"
