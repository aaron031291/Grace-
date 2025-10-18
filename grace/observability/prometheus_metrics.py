"""
Prometheus metrics for Grace system
"""

from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest
)
import logging

logger = logging.getLogger(__name__)


class GraceMetrics:
    """
    Comprehensive Prometheus metrics for Grace system
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics
        
        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()
        
        # Loop execution metrics
        self.loop_executions = Counter(
            'grace_loop_executions_total',
            'Total loop executions',
            ['loop_id', 'status'],  # status: success, failure, timeout
            registry=self.registry
        )
        
        self.loop_duration = Histogram(
            'grace_loop_duration_seconds',
            'Loop execution duration',
            ['loop_id'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )
        
        # Decision metrics
        self.decisions_made = Counter(
            'grace_decisions_total',
            'Total decisions made',
            ['component', 'decision_type'],
            registry=self.registry
        )
        
        self.decision_confidence = Histogram(
            'grace_decision_confidence',
            'Decision confidence scores',
            ['component'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        # Trust score metrics
        self.trust_scores = Gauge(
            'grace_trust_score',
            'Current trust scores',
            ['component', 'entity_id'],
            registry=self.registry
        )
        
        self.trust_score_changes = Counter(
            'grace_trust_score_changes_total',
            'Trust score change events',
            ['component', 'direction'],  # direction: increase, decrease
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'grace_errors_total',
            'Total errors by component',
            ['component', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # Governance metrics
        self.governance_validations = Counter(
            'grace_governance_validations_total',
            'Governance validation attempts',
            ['result'],  # result: passed, failed
            registry=self.registry
        )
        
        self.governance_violations = Counter(
            'grace_governance_violations_total',
            'Governance violations',
            ['violation_type', 'severity'],
            registry=self.registry
        )
        
        # Consensus metrics
        self.consensus_operations = Counter(
            'grace_consensus_operations_total',
            'Consensus operations',
            ['algorithm', 'result'],
            registry=self.registry
        )
        
        self.consensus_agreement = Histogram(
            'grace_consensus_agreement_score',
            'Consensus agreement scores',
            ['algorithm'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        # Component health metrics
        self.component_health = Gauge(
            'grace_component_health_score',
            'Component health scores',
            ['component'],
            registry=self.registry
        )
        
        self.component_uptime = Gauge(
            'grace_component_uptime_seconds',
            'Component uptime',
            ['component'],
            registry=self.registry
        )
        
        # AVN healing metrics
        self.healing_attempts = Counter(
            'grace_healing_attempts_total',
            'Self-healing attempts',
            ['component', 'action_type', 'result'],
            registry=self.registry
        )
        
        self.healing_duration = Histogram(
            'grace_healing_duration_seconds',
            'Healing action duration',
            ['component', 'action_type'],
            registry=self.registry
        )
        
        # Vector search metrics
        self.vector_searches = Counter(
            'grace_vector_searches_total',
            'Vector search operations',
            ['store_type'],
            registry=self.registry
        )
        
        self.vector_search_latency = Histogram(
            'grace_vector_search_latency_seconds',
            'Vector search latency',
            ['store_type'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry
        )
        
        # Document metrics
        self.documents_indexed = Counter(
            'grace_documents_indexed_total',
            'Documents indexed',
            ['embedding_provider'],
            registry=self.registry
        )
        
        # Swarm metrics
        self.swarm_nodes = Gauge(
            'grace_swarm_nodes_active',
            'Active swarm nodes',
            registry=self.registry
        )
        
        self.swarm_messages = Counter(
            'grace_swarm_messages_total',
            'Swarm messages exchanged',
            ['message_type', 'direction'],  # direction: sent, received
            registry=self.registry
        )
        
        logger.info("Grace metrics initialized")
    
    def record_loop_execution(
        self,
        loop_id: str,
        duration: float,
        status: str
    ):
        """Record loop execution metrics"""
        self.loop_executions.labels(loop_id=loop_id, status=status).inc()
        self.loop_duration.labels(loop_id=loop_id).observe(duration)
    
    def record_decision(
        self,
        component: str,
        decision_type: str,
        confidence: float
    ):
        """Record decision metrics"""
        self.decisions_made.labels(
            component=component,
            decision_type=decision_type
        ).inc()
        self.decision_confidence.labels(component=component).observe(confidence)
    
    def update_trust_score(
        self,
        component: str,
        entity_id: str,
        score: float,
        previous_score: Optional[float] = None
    ):
        """Update trust score metrics"""
        self.trust_scores.labels(
            component=component,
            entity_id=entity_id
        ).set(score)
        
        if previous_score is not None:
            direction = "increase" if score > previous_score else "decrease"
            self.trust_score_changes.labels(
                component=component,
                direction=direction
            ).inc()
    
    def record_error(
        self,
        component: str,
        error_type: str,
        severity: str
    ):
        """Record error metrics"""
        self.errors.labels(
            component=component,
            error_type=error_type,
            severity=severity
        ).inc()
    
    def record_governance_validation(
        self,
        passed: bool,
        violations: Optional[list] = None
    ):
        """Record governance validation metrics"""
        result = "passed" if passed else "failed"
        self.governance_validations.labels(result=result).inc()
        
        if violations:
            for violation in violations:
                self.governance_violations.labels(
                    violation_type=violation.get('type', 'unknown'),
                    severity=violation.get('severity', 'unknown')
                ).inc()
    
    def record_consensus(
        self,
        algorithm: str,
        agreement_score: float,
        success: bool
    ):
        """Record consensus metrics"""
        result = "success" if success else "failure"
        self.consensus_operations.labels(
            algorithm=algorithm,
            result=result
        ).inc()
        self.consensus_agreement.labels(algorithm=algorithm).observe(agreement_score)
    
    def update_component_health(
        self,
        component: str,
        health_score: float,
        uptime_seconds: float
    ):
        """Update component health metrics"""
        self.component_health.labels(component=component).set(health_score)
        self.component_uptime.labels(component=component).set(uptime_seconds)
    
    def record_healing(
        self,
        component: str,
        action_type: str,
        duration: float,
        success: bool
    ):
        """Record healing metrics"""
        result = "success" if success else "failure"
        self.healing_attempts.labels(
            component=component,
            action_type=action_type,
            result=result
        ).inc()
        self.healing_duration.labels(
            component=component,
            action_type=action_type
        ).observe(duration)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


# Global metrics instance
metrics_registry = GraceMetrics()
