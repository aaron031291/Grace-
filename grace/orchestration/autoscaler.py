"""
Advanced autoscaling with trust scores and backlog management
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import asyncio
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_usage: float
    memory_usage: float
    backlog_size: int
    error_rate: float
    trust_score: float
    request_rate: float
    avg_latency: float


@dataclass
class ScalingDecision:
    """Scaling decision result"""
    should_scale: bool
    target_instances: int
    reason: str
    confidence: float


class AdvancedAutoscaler:
    """
    Advanced autoscaling with multi-factor decision making
    """
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        target_cpu: float = 0.7,
        target_memory: float = 0.8,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: int = 300  # seconds
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu = target_cpu
        self.target_memory = target_memory
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.current_instances = min_instances
        self.last_scale_time: Optional[datetime] = None
        self.scaling_history: List[ScalingDecision] = []
        
        logger.info(f"Autoscaler initialized: {min_instances}-{max_instances} instances")
    
    def evaluate_scaling(
        self,
        metrics: ScalingMetrics,
        current_instances: int
    ) -> ScalingDecision:
        """
        Evaluate whether to scale based on comprehensive metrics
        
        Factors:
        - CPU and memory usage
        - Service backlog
        - Error rates
        - Trust scores
        - Request rate and latency
        """
        self.current_instances = current_instances
        
        # Check cooldown
        if self._in_cooldown():
            return ScalingDecision(
                should_scale=False,
                target_instances=current_instances,
                reason="In cooldown period",
                confidence=1.0
            )
        
        # Calculate component scores
        resource_score = self._calculate_resource_score(metrics)
        backlog_score = self._calculate_backlog_score(metrics)
        reliability_score = self._calculate_reliability_score(metrics)
        performance_score = self._calculate_performance_score(metrics)
        
        # Weighted decision
        weights = {
            "resource": 0.3,
            "backlog": 0.3,
            "reliability": 0.2,
            "performance": 0.2
        }
        
        overall_pressure = (
            resource_score * weights["resource"] +
            backlog_score * weights["backlog"] +
            reliability_score * weights["reliability"] +
            performance_score * weights["performance"]
        )
        
        # Make scaling decision
        if overall_pressure > self.scale_up_threshold:
            target = min(current_instances + 1, self.max_instances)
            reason = f"High pressure ({overall_pressure:.2f}): "
            
            reasons = []
            if resource_score > 0.7:
                reasons.append(f"CPU/Mem ({resource_score:.2f})")
            if backlog_score > 0.7:
                reasons.append(f"Backlog ({backlog_score:.2f})")
            if reliability_score > 0.7:
                reasons.append(f"Errors ({reliability_score:.2f})")
            if performance_score > 0.7:
                reasons.append(f"Latency ({performance_score:.2f})")
            
            reason += ", ".join(reasons)
            
            decision = ScalingDecision(
                should_scale=target != current_instances,
                target_instances=target,
                reason=reason,
                confidence=overall_pressure
            )
        
        elif overall_pressure < self.scale_down_threshold:
            target = max(current_instances - 1, self.min_instances)
            reason = f"Low pressure ({overall_pressure:.2f}), can scale down"
            
            decision = ScalingDecision(
                should_scale=target != current_instances,
                target_instances=target,
                reason=reason,
                confidence=1.0 - overall_pressure
            )
        
        else:
            decision = ScalingDecision(
                should_scale=False,
                target_instances=current_instances,
                reason=f"Pressure normal ({overall_pressure:.2f})",
                confidence=0.5
            )
        
        # Record decision
        self.scaling_history.append(decision)
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
        
        if decision.should_scale:
            self.last_scale_time = datetime.now(timezone.utc)
            logger.info(f"Scaling decision: {decision.reason}")
        
        return decision
    
    def _in_cooldown(self) -> bool:
        """Check if in cooldown period"""
        if self.last_scale_time is None:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.last_scale_time).total_seconds()
        return elapsed < self.cooldown_period
    
    def _calculate_resource_score(self, metrics: ScalingMetrics) -> float:
        """Calculate resource pressure score (0-1)"""
        cpu_pressure = metrics.cpu_usage / self.target_cpu
        mem_pressure = metrics.memory_usage / self.target_memory
        
        return max(cpu_pressure, mem_pressure)
    
    def _calculate_backlog_score(self, metrics: ScalingMetrics) -> float:
        """Calculate backlog pressure score (0-1)"""
        # Assume backlog > 100 is high pressure
        threshold = 100
        return min(1.0, metrics.backlog_size / threshold)
    
    def _calculate_reliability_score(self, metrics: ScalingMetrics) -> float:
        """Calculate reliability pressure from errors and trust (0-1)"""
        # High error rate = high pressure
        error_pressure = min(1.0, metrics.error_rate / 0.1)  # 10% error = max
        
        # Low trust = high pressure
        trust_pressure = 1.0 - metrics.trust_score
        
        return max(error_pressure, trust_pressure)
    
    def _calculate_performance_score(self, metrics: ScalingMetrics) -> float:
        """Calculate performance pressure (0-1)"""
        # High request rate with high latency = pressure
        latency_threshold = 500  # ms
        latency_pressure = min(1.0, metrics.avg_latency / latency_threshold)
        
        # High request rate alone also creates pressure
        request_threshold = 1000  # requests/sec
        request_pressure = min(1.0, metrics.request_rate / request_threshold)
        
        return max(latency_pressure, request_pressure * 0.5)
    
    async def scale_up(
        self,
        target_instances: int,
        spawn_callback: Callable
    ) -> List[str]:
        """
        Scale up instances with health checks
        
        Returns list of new instance IDs
        """
        instances_to_add = target_instances - self.current_instances
        new_instances = []
        
        logger.info(f"Scaling up: adding {instances_to_add} instances")
        
        for i in range(instances_to_add):
            try:
                instance_id = await spawn_callback()
                
                # Health check new instance
                if await self._health_check_instance(instance_id):
                    new_instances.append(instance_id)
                    logger.info(f"New instance healthy: {instance_id}")
                else:
                    logger.error(f"New instance failed health check: {instance_id}")
                
            except Exception as e:
                logger.error(f"Error spawning instance: {e}")
        
        return new_instances
    
    async def scale_down(
        self,
        target_instances: int,
        retire_callback: Callable
    ) -> List[str]:
        """
        Scale down instances gracefully
        
        Returns list of retired instance IDs
        """
        instances_to_remove = self.current_instances - target_instances
        retired_instances = []
        
        logger.info(f"Scaling down: removing {instances_to_remove} instances")
        
        for i in range(instances_to_remove):
            try:
                # Graceful shutdown
                instance_id = await self._select_instance_to_retire()
                
                if instance_id:
                    await self._graceful_retirement(instance_id)
                    await retire_callback(instance_id)
                    retired_instances.append(instance_id)
                    logger.info(f"Instance retired: {instance_id}")
            
            except Exception as e:
                logger.error(f"Error retiring instance: {e}")
        
        return retired_instances
    
    async def _health_check_instance(self, instance_id: str, timeout: float = 30.0) -> bool:
        """Health check for newly spawned instance"""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                # Simulate health check (replace with real check)
                await asyncio.sleep(1)
                # In production: check if instance responds to health endpoint
                return True
            except:
                await asyncio.sleep(2)
        
        return False
    
    async def _select_instance_to_retire(self) -> Optional[str]:
        """Select instance for retirement (lowest load)"""
        # In production: query instances and select one with lowest load
        return f"instance_{self.current_instances}"
    
    async def _graceful_retirement(self, instance_id: str):
        """Gracefully retire instance (drain connections)"""
        logger.info(f"Draining connections for {instance_id}")
        await asyncio.sleep(5)  # Simulate drain period
        logger.info(f"Instance ready for retirement: {instance_id}")
