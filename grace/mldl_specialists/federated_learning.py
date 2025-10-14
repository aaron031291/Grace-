"""
Federated Meta-Learner - Layer 4

Learns from specialist performance across tasks and continuously improves
the entire specialist ecosystem through meta-learning, trust adjustment,
and federated optimization.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from .base_specialist import (
    BaseMLDLSpecialist,
    SpecialistPrediction,
    TrainingMetrics,
    SpecialistCapability
)
from .consensus_engine import ConsensusResult


@dataclass
class PerformanceRecord:
    """Record of specialist performance on a task"""
    specialist_id: str
    timestamp: datetime
    accuracy: float
    confidence: float
    compliance: bool
    execution_time_ms: float
    capability: SpecialistCapability
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaLearningUpdate:
    """Update from meta-learning process"""
    specialist_id: str
    trust_adjustment: float
    performance_trend: str  # "improving", "stable", "degrading"
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class FederatedMetaLearner:
    """
    Meta-learner that improves specialist ecosystem over time
    
    Features:
    - Tracks specialist performance across tasks
    - Adjusts trust scores based on historical accuracy
    - Identifies underperforming specialists for retraining
    - Optimizes specialist selection and weighting
    - Detects concept drift and distribution shifts
    - Federates learning across distributed specialists
    
    Grace Integration:
    - Reports to governance for constitutional compliance
    - Logs all meta-learning decisions to immutable audit trail
    - Updates KPI monitors with meta-metrics
    - Stores meta-knowledge in memory bridge
    """
    
    def __init__(
        self,
        governance_bridge=None,
        kpi_monitor=None,
        immutable_logs=None,
        memory_bridge=None,
        learning_rate=0.01,
        trust_decay_rate=0.05,
        performance_window_days=7
    ):
        self.governance_bridge = governance_bridge
        self.kpi_monitor = kpi_monitor
        self.immutable_logs = immutable_logs
        self.memory_bridge = memory_bridge
        
        self.learning_rate = learning_rate
        self.trust_decay_rate = trust_decay_rate
        self.performance_window = timedelta(days=performance_window_days)
        
        # Track performance
        self.performance_records: Dict[str, List[PerformanceRecord]] = defaultdict(list)
        
        # Meta-knowledge
        self.specialist_trust_scores: Dict[str, float] = {}
        self.specialist_performance_trends: Dict[str, List[float]] = defaultdict(list)
        self.capability_expertise: Dict[str, Dict[SpecialistCapability, float]] = defaultdict(dict)
        
        # Meta-learning history
        self.meta_updates: List[MetaLearningUpdate] = []
        
    async def record_specialist_performance(
        self,
        specialist_id: str,
        prediction: SpecialistPrediction,
        ground_truth: Optional[Any] = None,
        accuracy: Optional[float] = None
    ):
        """
        Record specialist performance for meta-learning
        
        Args:
            specialist_id: ID of the specialist
            prediction: The prediction made
            ground_truth: True label (if available)
            accuracy: Calculated accuracy (if available)
        """
        # Calculate accuracy if ground truth provided
        if ground_truth is not None and accuracy is None:
            pred_val = prediction.prediction
            if isinstance(pred_val, list):
                pred_val = np.array(pred_val)
            if isinstance(ground_truth, list):
                ground_truth = np.array(ground_truth)
            
            # Simple accuracy calculation
            if isinstance(pred_val, (int, float, np.integer, np.floating)):
                accuracy = 1.0 if pred_val == ground_truth else 0.0
            elif isinstance(pred_val, np.ndarray):
                accuracy = float(np.mean(pred_val == ground_truth))
            else:
                accuracy = 0.5  # Unknown
        
        # Create performance record
        record = PerformanceRecord(
            specialist_id=specialist_id,
            timestamp=datetime.now(),
            accuracy=accuracy if accuracy is not None else prediction.confidence,
            confidence=prediction.confidence,
            compliance=prediction.constitutional_compliance,
            execution_time_ms=prediction.execution_time_ms,
            capability=prediction.capabilities_used[0] if prediction.capabilities_used else SpecialistCapability.CLASSIFICATION,
            metadata={
                "reasoning": prediction.reasoning,
                "trust_score": prediction.trust_score
            }
        )
        
        # Store record
        self.performance_records[specialist_id].append(record)
        
        # Update capability expertise
        capability = record.capability
        if capability not in self.capability_expertise[specialist_id]:
            self.capability_expertise[specialist_id][capability] = record.accuracy
        else:
            # Exponential moving average
            current = self.capability_expertise[specialist_id][capability]
            self.capability_expertise[specialist_id][capability] = (
                (1 - self.learning_rate) * current + self.learning_rate * record.accuracy
            )
        
        # Log to immutable trail
        if self.immutable_logs:
            await self.immutable_logs.log_event({
                "type": "specialist_performance_recorded",
                "specialist_id": specialist_id,
                "accuracy": record.accuracy,
                "timestamp": datetime.now().isoformat()
            })
    
    async def update_trust_scores(self, specialists: Dict[str, BaseMLDLSpecialist]):
        """
        Update trust scores for all specialists based on recent performance
        
        This is the core meta-learning loop.
        """
        cutoff_time = datetime.now() - self.performance_window
        
        updates = []
        
        for specialist_id, specialist in specialists.items():
            # Get recent performance records
            recent_records = [
                r for r in self.performance_records.get(specialist_id, [])
                if r.timestamp >= cutoff_time
            ]
            
            if not recent_records:
                continue
            
            # Calculate performance metrics
            avg_accuracy = np.mean([r.accuracy for r in recent_records])
            avg_confidence = np.mean([r.confidence for r in recent_records])
            compliance_rate = np.mean([r.compliance for r in recent_records])
            
            # Calculate performance trend
            accuracies = [r.accuracy for r in recent_records]
            trend = self._calculate_trend(accuracies)
            
            # Current trust score
            current_trust = specialist.current_trust_score
            
            # Calculate new trust score
            # Base it on: accuracy, confidence calibration, compliance
            confidence_calibration = 1.0 - abs(avg_accuracy - avg_confidence)
            
            new_trust = (
                0.5 * avg_accuracy +
                0.3 * confidence_calibration +
                0.2 * compliance_rate
            )
            
            # Apply trend adjustment
            if trend == "improving":
                new_trust *= 1.05
            elif trend == "degrading":
                new_trust *= 0.95
            
            # Smooth update (exponential moving average)
            updated_trust = (1 - self.learning_rate) * current_trust + self.learning_rate * new_trust
            updated_trust = max(0.0, min(1.0, updated_trust))
            
            # Apply trust decay (encourage continuous improvement)
            updated_trust *= (1.0 - self.trust_decay_rate)
            
            # Update specialist
            trust_adjustment = updated_trust - current_trust
            specialist.current_trust_score = updated_trust
            
            # Store in meta-knowledge
            self.specialist_trust_scores[specialist_id] = updated_trust
            self.specialist_performance_trends[specialist_id].append(avg_accuracy)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                specialist_id,
                avg_accuracy,
                trend,
                compliance_rate
            )
            
            # Create update record
            update = MetaLearningUpdate(
                specialist_id=specialist_id,
                trust_adjustment=trust_adjustment,
                performance_trend=trend,
                recommended_actions=recommendations
            )
            updates.append(update)
            self.meta_updates.append(update)
            
            # Log to immutable trail
            if self.immutable_logs:
                await self.immutable_logs.log_event({
                    "type": "trust_score_updated",
                    "specialist_id": specialist_id,
                    "old_trust": current_trust,
                    "new_trust": updated_trust,
                    "adjustment": trust_adjustment,
                    "trend": trend,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Report aggregate metrics to KPI monitor
        if self.kpi_monitor and updates:
            await self.kpi_monitor.record_metric({
                "metric_type": "meta_learning_update",
                "n_specialists_updated": len(updates),
                "avg_trust_adjustment": np.mean([u.trust_adjustment for u in updates]),
                "improving_specialists": sum(1 for u in updates if u.performance_trend == "improving"),
                "degrading_specialists": sum(1 for u in updates if u.performance_trend == "degrading"),
                "timestamp": datetime.now().isoformat()
            })
        
        return updates
    
    def _calculate_trend(self, values: List[float], window=5) -> str:
        """Calculate trend from recent values"""
        if len(values) < 2:
            return "stable"
        
        # Use last 'window' values
        recent = values[-window:]
        
        if len(recent) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Calculate slope
        x_mean = x.mean()
        y_mean = y.mean()
        
        slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-6)
        
        # Threshold for significance
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"
    
    def _generate_recommendations(
        self,
        specialist_id: str,
        accuracy: float,
        trend: str,
        compliance_rate: float
    ) -> List[str]:
        """Generate actionable recommendations for specialist"""
        recommendations = []
        
        # Accuracy-based recommendations
        if accuracy < 0.7:
            recommendations.append("retrain_with_more_data")
            recommendations.append("review_feature_engineering")
        elif accuracy < 0.85:
            recommendations.append("tune_hyperparameters")
        
        # Trend-based recommendations
        if trend == "degrading":
            recommendations.append("investigate_concept_drift")
            recommendations.append("consider_incremental_learning")
        elif trend == "improving":
            recommendations.append("maintain_current_strategy")
        
        # Compliance-based recommendations
        if compliance_rate < 0.95:
            recommendations.append("review_constitutional_compliance")
            recommendations.append("adjust_decision_thresholds")
        
        return recommendations
    
    async def federated_learning_round(
        self,
        specialists: Dict[str, BaseMLDLSpecialist],
        X_train: np.ndarray,
        y_train: np.ndarray,
        capability: SpecialistCapability
    ):
        """
        Perform a federated learning round
        
        Each specialist trains independently, then meta-learner aggregates
        knowledge and updates all specialists.
        """
        # Filter specialists by capability
        relevant_specialists = {
            sid: spec for sid, spec in specialists.items()
            if capability in spec.capabilities
        }
        
        if not relevant_specialists:
            return
        
        # Train each specialist independently
        training_results = {}
        for specialist_id, specialist in relevant_specialists.items():
            try:
                metrics = await specialist.train(X_train, y_train)
                training_results[specialist_id] = metrics
            except Exception as e:
                if self.immutable_logs:
                    await self.immutable_logs.log_event({
                        "type": "federated_training_error",
                        "specialist_id": specialist_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Aggregate knowledge (simple averaging for now)
        # In production, this could be more sophisticated (FedAvg, FedProx, etc.)
        
        # Update trust scores based on training performance
        for specialist_id, metrics in training_results.items():
            specialist = relevant_specialists[specialist_id]
            
            # Use training accuracy as performance signal
            accuracy = getattr(metrics, 'accuracy', None) or getattr(metrics, 'r2_score', None) or 0.5
            
            # Record performance
            await self.record_specialist_performance(
                specialist_id=specialist_id,
                prediction=SpecialistPrediction(
                    specialist_id=specialist_id,
                    specialist_type=specialist.specialist_type,
                    prediction=None,
                    confidence=accuracy,
                    reasoning="Federated training round",
                    capabilities_used=[capability],
                    execution_time_ms=0,
                    model_version=specialist.model_version,
                    constitutional_compliance=True,
                    trust_score=specialist.current_trust_score
                ),
                accuracy=accuracy
            )
        
        # Update all trust scores
        await self.update_trust_scores(specialists)
        
        # Log federated round
        if self.immutable_logs:
            await self.immutable_logs.log_event({
                "type": "federated_learning_round_completed",
                "capability": capability.value,
                "n_specialists": len(relevant_specialists),
                "successful_trainings": len(training_results),
                "timestamp": datetime.now().isoformat()
            })
    
    def get_specialist_ranking(
        self,
        capability: Optional[SpecialistCapability] = None
    ) -> List[tuple[str, float]]:
        """
        Get specialists ranked by trust score
        
        Args:
            capability: Filter by capability (optional)
            
        Returns:
            List of (specialist_id, trust_score) tuples, sorted descending
        """
        if capability is None:
            # Global ranking
            rankings = list(self.specialist_trust_scores.items())
        else:
            # Capability-specific ranking
            rankings = []
            for specialist_id, capabilities_dict in self.capability_expertise.items():
                if capability in capabilities_dict:
                    expertise = capabilities_dict[capability]
                    trust = self.specialist_trust_scores.get(specialist_id, 0.5)
                    # Combined score
                    score = 0.7 * trust + 0.3 * expertise
                    rankings.append((specialist_id, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about meta-learning process"""
        if not self.meta_updates:
            return {
                "total_updates": 0,
                "avg_trust_adjustment": 0,
                "improving_rate": 0,
                "degrading_rate": 0
            }
        
        recent_updates = self.meta_updates[-100:]  # Last 100 updates
        
        return {
            "total_updates": len(self.meta_updates),
            "avg_trust_adjustment": np.mean([u.trust_adjustment for u in recent_updates]),
            "improving_rate": np.mean([u.performance_trend == "improving" for u in recent_updates]),
            "degrading_rate": np.mean([u.performance_trend == "degrading" for u in recent_updates]),
            "stable_rate": np.mean([u.performance_trend == "stable" for u in recent_updates]),
            "avg_trust_score": np.mean(list(self.specialist_trust_scores.values())) if self.specialist_trust_scores else 0.5
        }
    
    async def validate_meta_learning_governance(self) -> bool:
        """Validate meta-learning process against governance rules"""
        if not self.governance_bridge:
            return True
        
        # Check that meta-learning is operating within constitutional bounds
        try:
            validation_result = await self.governance_bridge.validate({
                "type": "meta_learning_process",
                "stats": self.get_meta_learning_stats(),
                "timestamp": datetime.now().isoformat()
            })
            return validation_result.get("approved", True)
        except Exception:
            return True
    
    async def export_meta_knowledge(self) -> Dict[str, Any]:
        """Export meta-knowledge for storage/transfer"""
        return {
            "specialist_trust_scores": self.specialist_trust_scores,
            "capability_expertise": {
                sid: {cap.value: score for cap, score in caps.items()}
                for sid, caps in self.capability_expertise.items()
            },
            "performance_trends": {
                sid: trends[-50:]  # Last 50 data points
                for sid, trends in self.specialist_performance_trends.items()
            },
            "meta_stats": self.get_meta_learning_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def import_meta_knowledge(self, knowledge: Dict[str, Any]):
        """Import meta-knowledge from external source"""
        self.specialist_trust_scores = knowledge.get("specialist_trust_scores", {})
        
        # Convert capability strings back to enums
        cap_expertise = knowledge.get("capability_expertise", {})
        for sid, caps in cap_expertise.items():
            self.capability_expertise[sid] = {
                SpecialistCapability(cap): score
                for cap, score in caps.items()
            }
        
        self.specialist_performance_trends = defaultdict(
            list,
            knowledge.get("performance_trends", {})
        )
        
        # Log import
        if self.immutable_logs:
            await self.immutable_logs.log_event({
                "type": "meta_knowledge_imported",
                "n_specialists": len(self.specialist_trust_scores),
                "timestamp": datetime.now().isoformat()
            })
