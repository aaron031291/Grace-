"""
Scheduler metrics instrumentation for Prometheus
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import time
from prometheus_client import Counter, Gauge, Histogram
import logging

logger = logging.getLogger(__name__)


class SchedulerMetrics:
    """
    Prometheus metrics for scheduler monitoring
    """
    
    def __init__(self):
        # Loop execution metrics
        self.loop_executions = Counter(
            'grace_scheduler_loop_executions_total',
            'Total number of loop executions',
            ['loop_id', 'status']  # status: success, failure, timeout
        )
        
        self.loop_duration = Histogram(
            'grace_scheduler_loop_duration_seconds',
            'Loop execution duration in seconds',
            ['loop_id'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.loop_queue_depth = Gauge(
            'grace_scheduler_loop_queue_depth',
            'Number of pending executions in loop queue',
            ['loop_id']
        )
        
        self.active_loops = Gauge(
            'grace_scheduler_active_loops',
            'Number of currently active loops'
        )
        
        # Scheduler health metrics
        self.scheduler_status = Gauge(
            'grace_scheduler_status',
            'Scheduler status (1=running, 0=stopped)',
            ['scheduler_id']
        )
        
        self.scheduler_uptime = Gauge(
            'grace_scheduler_uptime_seconds',
            'Scheduler uptime in seconds',
            ['scheduler_id']
        )
        
        # Policy metrics
        self.policy_evaluations = Counter(
            'grace_scheduler_policy_evaluations_total',
            'Total policy evaluations',
            ['policy_id', 'result']  # result: allowed, denied
        )
        
        # Error metrics
        self.scheduler_errors = Counter(
            'grace_scheduler_errors_total',
            'Total scheduler errors',
            ['error_type']
        )
        
        # Queue metrics
        self.global_queue_size = Gauge(
            'grace_scheduler_global_queue_size',
            'Total size of all queues'
        )
        
        # Snapshot metrics
        self.snapshot_operations = Counter(
            'grace_scheduler_snapshot_operations_total',
            'Total snapshot operations',
            ['operation', 'status']  # operation: create, restore
        )
        
        logger.info("Scheduler metrics initialized")
    
    def record_loop_execution(
        self,
        loop_id: str,
        duration: float,
        status: str  # success, failure, timeout
    ):
        """Record a loop execution"""
        self.loop_executions.labels(loop_id=loop_id, status=status).inc()
        self.loop_duration.labels(loop_id=loop_id).observe(duration)
    
    def update_queue_depth(self, loop_id: str, depth: int):
        """Update queue depth for a loop"""
        self.loop_queue_depth.labels(loop_id=loop_id).set(depth)
    
    def update_active_loops(self, count: int):
        """Update count of active loops"""
        self.active_loops.set(count)
    
    def update_scheduler_status(self, scheduler_id: str, running: bool):
        """Update scheduler status"""
        self.scheduler_status.labels(scheduler_id=scheduler_id).set(1 if running else 0)
    
    def update_uptime(self, scheduler_id: str, uptime_seconds: float):
        """Update scheduler uptime"""
        self.scheduler_uptime.labels(scheduler_id=scheduler_id).set(uptime_seconds)
    
    def record_policy_evaluation(self, policy_id: str, allowed: bool):
        """Record policy evaluation result"""
        result = "allowed" if allowed else "denied"
        self.policy_evaluations.labels(policy_id=policy_id, result=result).inc()
    
    def record_error(self, error_type: str):
        """Record scheduler error"""
        self.scheduler_errors.labels(error_type=error_type).inc()
    
    def update_global_queue_size(self, size: int):
        """Update global queue size"""
        self.global_queue_size.set(size)
    
    def record_snapshot_operation(self, operation: str, success: bool):
        """Record snapshot operation"""
        status = "success" if success else "failure"
        self.snapshot_operations.labels(operation=operation, status=status).inc()


# Global metrics instance
scheduler_metrics = SchedulerMetrics()
