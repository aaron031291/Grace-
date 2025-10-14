"""
Base ML/DL Specialist - Foundation for all Grace ML/DL models

Every specialist:
1. Examines data through its unique lens
2. Complies with constitutional governance
3. Reports to KPI/trust monitoring
4. Logs to immutable audit trail
5. Participates in federated meta-learning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import asyncio
import numpy as np


class SpecialistCapability(Enum):
    """Capabilities each specialist can provide"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"
    RECOMMENDATION = "recommendation"
    GRAPH_ANALYSIS = "graph_analysis"


@dataclass
class SpecialistPrediction:
    """Individual specialist's prediction with confidence and reasoning"""
    specialist_id: str
    specialist_type: str
    prediction: Any
    confidence: float  # 0.0 to 1.0
    reasoning: str
    capabilities_used: List[SpecialistCapability]
    execution_time_ms: float
    model_version: str
    constitutional_compliance: float  # 0.0 to 1.0
    trust_score: float  # Current trust from KPI monitor
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingMetrics:
    """Training metrics for KPI/trust monitoring"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class BaseMLDLSpecialist(ABC):
    """
    Base class for all ML/DL specialists in Grace architecture.
    
    Architecture Integration:
        - Layer 1: Individual analysis (this class)
        - Layer 2: Feeds to consensus engine
        - Layer 3: Governance validation
        - Layer 4: Federated learning updates
    """
    
    def __init__(
        self,
        specialist_id: str,
        specialist_type: str,
        capabilities: List[SpecialistCapability],
        governance_bridge=None,
        kpi_monitor=None,
        immutable_logs=None,
        memory_bridge=None,
        **kwargs
    ):
        self.specialist_id = specialist_id
        self.specialist_type = specialist_type
        self.capabilities = capabilities
        self.model_version = kwargs.get("model_version", "1.0.0")
        
        # Grace architecture bridges
        self.governance_bridge = governance_bridge
        self.kpi_monitor = kpi_monitor
        self.immutable_logs = immutable_logs
        self.memory_bridge = memory_bridge
        
        # Model state
        self.is_trained = False
        self.training_history: List[TrainingMetrics] = []
        self.prediction_count = 0
        self.current_trust_score = 1.0
        
        # Constitutional compliance thresholds
        self.min_confidence_threshold = 0.7
        self.min_trust_threshold = 0.6
        
    @abstractmethod
    async def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        **kwargs
    ) -> TrainingMetrics:
        """
        Train the model on provided data.
        
        Must:
        1. Train the underlying model
        2. Log to immutable audit trail
        3. Report metrics to KPI monitor
        4. Check governance compliance
        """
        pass
    
    @abstractmethod
    async def predict(
        self,
        X: np.ndarray,
        **kwargs
    ) -> SpecialistPrediction:
        """
        Make prediction on new data.
        
        Must:
        1. Generate prediction with confidence
        2. Provide reasoning/explanation
        3. Check constitutional compliance
        4. Log to immutable audit trail
        5. Update trust metrics
        """
        pass
    
    async def validate_governance(self, data: Any, prediction: Any) -> float:
        """
        Validate prediction against constitutional governance.
        
        Returns:
            Constitutional compliance score (0.0 to 1.0)
        """
        if not self.governance_bridge:
            return 1.0  # No governance = full compliance (permissive)
        
        try:
            # Check governance rules
            validation_result = await self.governance_bridge.validate_decision(
                specialist_id=self.specialist_id,
                input_data=data,
                prediction=prediction,
                confidence=getattr(prediction, "confidence", 0.0)
            )
            
            return validation_result.get("compliance_score", 1.0)
            
        except Exception as e:
            # Log governance check failure
            if self.immutable_logs:
                await self.immutable_logs.append(
                    operation_type="governance_check_failed",
                    operation_data={
                        "specialist_id": self.specialist_id,
                        "error": str(e)
                    },
                    user_id="system"
                )
            return 0.5  # Partial compliance on error
    
    async def log_to_immutable_trail(
        self,
        operation_type: str,
        operation_data: Dict[str, Any]
    ):
        """Log operation to Grace's immutable audit trail"""
        if not self.immutable_logs:
            return
        
        try:
            await self.immutable_logs.append(
                operation_type=f"mldl_specialist_{operation_type}",
                operation_data={
                    "specialist_id": self.specialist_id,
                    "specialist_type": self.specialist_type,
                    "model_version": self.model_version,
                    **operation_data
                },
                user_id="system"
            )
        except Exception as e:
            # Silent fail on logging (don't break predictions)
            print(f"Warning: Failed to log to immutable trail: {e}")
    
    async def report_kpi_metrics(
        self,
        metrics: TrainingMetrics,
        operation: str = "training"
    ):
        """Report metrics to Grace KPI/trust monitoring system"""
        if not self.kpi_monitor:
            return
        
        try:
            # Report primary metric
            primary_metric = metrics.accuracy or metrics.f1_score or metrics.r2_score
            if primary_metric:
                await self.kpi_monitor.record_metric(
                    name=f"{self.specialist_id}_{operation}_performance",
                    value=primary_metric * 100,
                    component_id=self.specialist_id,
                    threshold_warning=70.0,
                    threshold_critical=50.0,
                    tags={
                        "specialist_type": self.specialist_type,
                        "operation": operation
                    }
                )
            
            # Update trust score based on performance
            trust_score_obj = self.kpi_monitor.get_trust_score(self.specialist_id)
            if trust_score_obj:
                self.current_trust_score = trust_score_obj.score
                
        except Exception as e:
            print(f"Warning: Failed to report KPI metrics: {e}")
    
    async def store_in_memory(
        self,
        key: str,
        data: Any,
        memory_type: str = "prediction_cache"
    ):
        """Store data in Grace's memory system"""
        if not self.memory_bridge:
            return
        
        try:
            await self.memory_bridge.store(
                key=f"{self.specialist_id}:{key}",
                value=data,
                memory_type=memory_type,
                metadata={
                    "specialist_id": self.specialist_id,
                    "specialist_type": self.specialist_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            print(f"Warning: Failed to store in memory: {e}")
    
    async def retrieve_from_memory(
        self,
        key: str,
        memory_type: str = "prediction_cache"
    ) -> Optional[Any]:
        """Retrieve data from Grace's memory system"""
        if not self.memory_bridge:
            return None
        
        try:
            return await self.memory_bridge.retrieve(
                key=f"{self.specialist_id}:{key}",
                memory_type=memory_type
            )
        except Exception:
            return None
    
    def get_capabilities(self) -> List[SpecialistCapability]:
        """Return specialist's capabilities"""
        return self.capabilities
    
    def get_trust_score(self) -> float:
        """Get current trust score from KPI monitor"""
        return self.current_trust_score
    
    def is_ready(self) -> bool:
        """Check if specialist is ready for predictions"""
        return self.is_trained and self.current_trust_score >= self.min_trust_threshold
