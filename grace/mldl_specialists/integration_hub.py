"""
ML/DL Integration Hub - Central orchestration for all ML/DL specialists

This module provides end-to-end integration of:
- All 7 deep learning neural network specialists
- All classical ML specialists (supervised + unsupervised)
- Operational intelligence (uncertainty, OOD, monitoring, active learning)
- Model registry & deployment management
- Governance integration (constitutional compliance, trust validation)
- Event mesh integration (TriggerMesh workflows)
- Intelligence kernel integration (planning, routing, evaluation)
- Ingress kernel integration (data capture, validation)
- Learning kernel integration (curriculum, training, evaluation)

Architecture:
    Grace Governance Kernel
           |
           v
    ML/DL Integration Hub ←→ Event Bus
           |
           ├─→ Classical ML (7 specialists)
           ├─→ Deep Learning (7 neural networks)
           ├─→ Operational Intelligence
           ├─→ Model Registry & Deployment
           └─→ Active Learning & Curriculum
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core infrastructure
from ..core import EventBus, MemoryCore

# Import all specialists
from .supervised_specialists import (
    DecisionTreeSpecialist,
    RandomForestSpecialist,
    GradientBoostingSpecialist,
    SVMSpecialist,
    LogisticRegressionSpecialist,
    KNNSpecialist,
    NaiveBayesSpecialist
)

from .unsupervised_specialists import (
    KMeansSpecialist,
    DBSCANSpecialist,
    HierarchicalClusteringSpecialist,
    PCASpecialist,
    IsolationForestSpecialist
)

# Import deep learning specialists
from .deep_learning import (
    ANNSpecialist,
    CNNSpecialist,
    RNNSpecialist,
    LSTMSpecialist,
    TransformerSpecialist,
    AutoencoderSpecialist,
    GANSpecialist,
    DeviceManager
)

# Operational intelligence
from .uncertainty_ood import UncertaintyQuantifier, OutOfDistributionDetector
from .model_registry import ModelRegistry, DeploymentStage, ModelRegistryEntry
from .active_learning import ActiveLearningCoordinator, QueryStrategy
from .monitoring import ModelMonitor, AlertSeverity

logger = logging.getLogger(__name__)


class SpecialistCategory(Enum):
    """Categories of ML/DL specialists"""
    SUPERVISED_CLASSICAL = "supervised_classical"
    UNSUPERVISED_CLASSICAL = "unsupervised_classical"
    DEEP_LEARNING = "deep_learning"
    OPERATIONAL = "operational"


@dataclass
class MLDLRequest:
    """Unified request format for ML/DL operations"""
    request_id: str
    operation: str  # train, predict, evaluate, deploy, rollback
    specialist_type: str  # decision_tree, lstm, cnn, etc.
    data: Dict[str, Any]
    
    # Optional fields
    model_id: Optional[str] = None
    version: Optional[str] = None
    governance_approved: bool = False
    constitutional_compliance: bool = False
    trust_score: Optional[float] = None
    
    # Deployment context
    deployment_stage: Optional[str] = None
    canary_percentage: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MLDLResponse:
    """Unified response format for ML/DL operations"""
    request_id: str
    success: bool
    specialist_type: str
    operation: str
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Model info
    model_id: Optional[str] = None
    version: Optional[str] = None
    
    # Quality metrics
    uncertainty: Optional[float] = None
    is_ood: bool = False
    confidence: Optional[float] = None
    
    # Governance
    governance_validated: bool = False
    constitutional_compliant: bool = False
    trust_score: Optional[float] = None
    
    # Performance
    latency_ms: Optional[float] = None
    
    # Metadata
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MLDLIntegrationHub:
    """
    Central integration hub for all ML/DL specialists.
    
    Provides:
    - Unified interface for all specialist types
    - Automatic routing to appropriate specialist
    - Operational intelligence integration (uncertainty, OOD, monitoring)
    - Model registry & lifecycle management
    - Governance validation integration
    - Active learning coordination
    - Event bus integration
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        memory_core: Optional[MemoryCore] = None,
        governance_kernel=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ML/DL Integration Hub.
        
        Args:
            event_bus: Event bus for system communication
            memory_core: Memory core for persistence
            governance_kernel: Governance kernel for validation
            config: Configuration dictionary
        """
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.governance_kernel = governance_kernel
        self.config = config or {}
        
        # Specialist registries
        self.classical_supervised: Dict[str, Any] = {}
        self.classical_unsupervised: Dict[str, Any] = {}
        self.deep_learning_specialists: Dict[str, Any] = {}
        
        # Operational intelligence
        self.uncertainty_quantifier: Optional[UncertaintyQuantifier] = None
        self.ood_detector: Optional[OutOfDistributionDetector] = None
        self.model_monitor: Optional[ModelMonitor] = None
        self.active_learning: Optional[ActiveLearningCoordinator] = None
        
        # Model registry
        self.model_registry: Optional[ModelRegistry] = None
        
        # Device management for deep learning
        self.device = DeviceManager.get_device()
        
        # State
        self.is_initialized = False
        self.is_running = False
        
        logger.info(f"ML/DL Integration Hub initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize all specialists and operational components"""
        if self.is_initialized:
            logger.warning("Integration hub already initialized")
            return
        
        logger.info("Initializing ML/DL Integration Hub...")
        
        try:
            # Initialize operational intelligence
            await self._initialize_operational_intelligence()
            
            # Initialize model registry
            await self._initialize_model_registry()
            
            # Initialize classical ML specialists
            await self._initialize_classical_specialists()
            
            # Initialize deep learning specialists
            await self._initialize_deep_learning_specialists()
            
            # Initialize active learning
            await self._initialize_active_learning()
            
            # Subscribe to events
            await self._subscribe_to_events()
            
            self.is_initialized = True
            logger.info("ML/DL Integration Hub initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize integration hub: {e}")
            raise
    
    async def _initialize_operational_intelligence(self):
        """Initialize uncertainty quantification, OOD detection, and monitoring"""
        logger.info("Initializing operational intelligence...")
        
        # Uncertainty quantification
        self.uncertainty_quantifier = UncertaintyQuantifier(
            method="ensemble"  # ensemble, monte_carlo_dropout, deep_ensemble
        )
        
        # OOD detection
        self.ood_detector = OutOfDistributionDetector(
            method="mahalanobis"  # mahalanobis, isolation_forest, autoencoder
        )
        
        # Model monitoring
        self.model_monitor = ModelMonitor(
            event_bus=self.event_bus,
            alert_config=self.config.get("monitoring", {})
        )
        
        logger.info("Operational intelligence initialized")
    
    async def _initialize_model_registry(self):
        """Initialize model registry for lifecycle management"""
        logger.info("Initializing model registry...")
        
        registry_path = self.config.get("registry_path", "./models/registry")
        self.model_registry = ModelRegistry(
            registry_path=registry_path,
            event_bus=self.event_bus
        )
        
        await self.model_registry.initialize()
        logger.info(f"Model registry initialized at {registry_path}")
    
    async def _initialize_classical_specialists(self):
        """Initialize all classical ML specialists"""
        logger.info("Initializing classical ML specialists...")
        
        # Supervised learning
        self.classical_supervised = {
            "decision_tree": DecisionTreeSpecialist(specialist_id="dt_001"),
            "random_forest": RandomForestSpecialist(specialist_id="rf_001"),
            "gradient_boosting": GradientBoostingSpecialist(specialist_id="gb_001"),
            "svm": SVMSpecialist(specialist_id="svm_001"),
            "logistic_regression": LogisticRegressionSpecialist(specialist_id="lr_001"),
            "knn": KNNSpecialist(specialist_id="knn_001"),
            "naive_bayes": NaiveBayesSpecialist(specialist_id="nb_001")
        }
        
        # Unsupervised learning
        self.classical_unsupervised = {
            "kmeans": KMeansSpecialist(specialist_id="km_001"),
            "dbscan": DBSCANSpecialist(specialist_id="db_001"),
            "hierarchical": HierarchicalClusteringSpecialist(specialist_id="hc_001"),
            "pca": PCASpecialist(specialist_id="pca_001"),
            "isolation_forest": IsolationForestSpecialist(specialist_id="if_001")
        }
        
        logger.info(f"Initialized {len(self.classical_supervised)} supervised + "
                   f"{len(self.classical_unsupervised)} unsupervised specialists")
    
    async def _initialize_deep_learning_specialists(self):
        """Initialize all deep learning neural network specialists"""
        logger.info("Initializing deep learning specialists...")
        
        # Initialize all 7 deep learning specialists
        self.deep_learning_specialists = {
            "ann": ANNSpecialist(
                specialist_id="ann_001",
                device=self.device
            ),
            "cnn": CNNSpecialist(
                specialist_id="cnn_001",
                device=self.device
            ),
            "rnn": RNNSpecialist(
                specialist_id="rnn_001",
                device=self.device
            ),
            "lstm": LSTMSpecialist(
                specialist_id="lstm_001",
                device=self.device,
                sequence_length=30,
                forecast_horizon=7
            ),
            "transformer": TransformerSpecialist(
                specialist_id="transformer_001",
                device=self.device,
                model_name="distilbert-base-uncased"
            ),
            "autoencoder": AutoencoderSpecialist(
                specialist_id="autoencoder_001",
                device=self.device,
                latent_dim=32
            ),
            "gan": GANSpecialist(
                specialist_id="gan_001",
                device=self.device,
                latent_dim=100
            )
        }
        
        logger.info(f"Initialized {len(self.deep_learning_specialists)} "
                   f"deep learning specialists on {self.device}")
    
    async def _initialize_active_learning(self):
        """Initialize active learning coordinator"""
        logger.info("Initializing active learning...")
        
        self.active_learning = ActiveLearningCoordinator(
            model_registry=self.model_registry,
            event_bus=self.event_bus,
            default_strategy=QueryStrategy.UNCERTAINTY_SAMPLING
        )
        
        await self.active_learning.initialize()
        logger.info("Active learning coordinator initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant event bus topics"""
        if not self.event_bus:
            return
        
        logger.info("Subscribing to event bus topics...")
        
        # Subscribe to ML/DL events
        await self.event_bus.subscribe("ml.training.request", self._handle_training_request)
        await self.event_bus.subscribe("ml.prediction.request", self._handle_prediction_request)
        await self.event_bus.subscribe("ml.deployment.request", self._handle_deployment_request)
        await self.event_bus.subscribe("ml.rollback.trigger", self._handle_rollback_request)
        
        # Subscribe to governance events
        await self.event_bus.subscribe("governance.model.validate", self._handle_governance_validation)
        
        # Subscribe to active learning events
        await self.event_bus.subscribe("ml.active_learning.query", self._handle_active_learning_query)
        
        logger.info("Event subscriptions complete")
    
    async def process_request(self, request: MLDLRequest) -> MLDLResponse:
        """
        Process a unified ML/DL request.
        
        This is the main entry point for all ML/DL operations.
        
        Args:
            request: MLDLRequest with operation details
            
        Returns:
            MLDLResponse with results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Route to appropriate handler
            if request.operation == "train":
                response = await self._handle_training(request)
            elif request.operation == "predict":
                response = await self._handle_prediction(request)
            elif request.operation == "evaluate":
                response = await self._handle_evaluation(request)
            elif request.operation == "deploy":
                response = await self._handle_deployment(request)
            elif request.operation == "rollback":
                response = await self._handle_rollback(request)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
            
            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            response.latency_ms = latency_ms
            
            # Emit metrics
            if self.model_monitor:
                await self.model_monitor.record_prediction(
                    model_id=response.model_id or "unknown",
                    latency_ms=latency_ms,
                    error=not response.success
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            
            return MLDLResponse(
                request_id=request.request_id,
                success=False,
                specialist_type=request.specialist_type,
                operation=request.operation,
                error=str(e)
            )
    
    async def _handle_training(self, request: MLDLRequest) -> MLDLResponse:
        """Handle model training request"""
        specialist_type = request.specialist_type
        specialist = self._get_specialist(specialist_type)
        
        if not specialist:
            raise ValueError(f"Unknown specialist type: {specialist_type}")
        
        # Extract training data
        X_train = request.data.get("X_train")
        y_train = request.data.get("y_train")
        X_val = request.data.get("X_val")
        y_val = request.data.get("y_val")
        
        # Training parameters
        params = request.data.get("params", {})
        
        # Train model
        if specialist_type in self.deep_learning_specialists:
            # Deep learning training
            training_result = await specialist.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001),
                verbose=params.get("verbose", False)
            )
        else:
            # Classical ML training
            training_result = await specialist.fit(
                X_train=X_train,
                y_train=y_train,
                **params
            )
        
        # Register model
        model_id = f"{specialist_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.model_registry:
            registry_entry = ModelRegistryEntry(
                model_id=model_id,
                name=specialist_type,
                version="1.0.0",
                artifact_path=f"./models/{model_id}",
                framework="pytorch" if specialist_type in self.deep_learning_specialists else "sklearn",
                model_type=specialist.get_capability() if hasattr(specialist, "get_capability") else "unknown",
                owner="integration_hub",
                team="ml_team",
                training_data_hash="",  # Would calculate hash
                training_dataset_size=len(X_train),
                training_timestamp=datetime.now(),
                evaluation_metrics=training_result.get("metrics", {}),
                deploy_status=DeploymentStage.DEVELOPMENT
            )
            
            await self.model_registry.register_model(registry_entry)
        
        return MLDLResponse(
            request_id=request.request_id,
            success=True,
            specialist_type=specialist_type,
            operation="train",
            model_id=model_id,
            version="1.0.0",
            results=training_result
        )
    
    async def _handle_prediction(self, request: MLDLRequest) -> MLDLResponse:
        """Handle prediction request with full operational intelligence"""
        specialist_type = request.specialist_type
        specialist = self._get_specialist(specialist_type)
        
        if not specialist:
            raise ValueError(f"Unknown specialist type: {specialist_type}")
        
        # Extract input data
        X = request.data.get("X")
        
        # Make prediction
        prediction = await specialist.predict(X)
        
        # Uncertainty quantification
        uncertainty = None
        if self.uncertainty_quantifier and specialist_type in self.deep_learning_specialists:
            uncertainty = await self.uncertainty_quantifier.quantify(
                model=specialist.model,
                X=X,
                method="monte_carlo_dropout"
            )
        
        # OOD detection
        is_ood = False
        if self.ood_detector:
            is_ood = await self.ood_detector.is_out_of_distribution(X)
        
        # Governance validation (if required)
        governance_validated = False
        if request.governance_approved and self.governance_kernel:
            governance_result = await self._validate_with_governance(request, prediction)
            governance_validated = governance_result.get("approved", False)
        
        return MLDLResponse(
            request_id=request.request_id,
            success=True,
            specialist_type=specialist_type,
            operation="predict",
            model_id=request.model_id,
            version=request.version,
            results={"prediction": prediction},
            uncertainty=uncertainty,
            is_ood=is_ood,
            governance_validated=governance_validated
        )
    
    async def _handle_evaluation(self, request: MLDLRequest) -> MLDLResponse:
        """Handle model evaluation request"""
        # Implementation for model evaluation
        pass
    
    async def _handle_deployment(self, request: MLDLRequest) -> MLDLResponse:
        """Handle model deployment request"""
        model_id = request.model_id
        deployment_stage = request.deployment_stage or "canary"
        
        if not self.model_registry:
            raise RuntimeError("Model registry not initialized")
        
        # Get model from registry
        model_entry = await self.model_registry.get_model(model_id)
        
        if not model_entry:
            raise ValueError(f"Model not found: {model_id}")
        
        # Governance validation
        if self.governance_kernel:
            governance_result = await self._validate_deployment_with_governance(
                model_entry,
                deployment_stage
            )
            
            if not governance_result.get("approved", False):
                return MLDLResponse(
                    request_id=request.request_id,
                    success=False,
                    specialist_type=request.specialist_type,
                    operation="deploy",
                    model_id=model_id,
                    error="Deployment rejected by governance"
                )
        
        # Update deployment stage
        await self.model_registry.update_deployment_stage(
            model_id=model_id,
            stage=DeploymentStage[deployment_stage.upper()],
            canary_percentage=request.canary_percentage or 10.0
        )
        
        return MLDLResponse(
            request_id=request.request_id,
            success=True,
            specialist_type=request.specialist_type,
            operation="deploy",
            model_id=model_id,
            results={"deployment_stage": deployment_stage}
        )
    
    async def _handle_rollback(self, request: MLDLRequest) -> MLDLResponse:
        """Handle model rollback request"""
        model_id = request.model_id
        
        if not self.model_registry:
            raise RuntimeError("Model registry not initialized")
        
        # Trigger rollback
        await self.model_registry.rollback_model(model_id)
        
        return MLDLResponse(
            request_id=request.request_id,
            success=True,
            specialist_type=request.specialist_type,
            operation="rollback",
            model_id=model_id,
            results={"status": "rolled_back"}
        )
    
    def _get_specialist(self, specialist_type: str):
        """Get specialist by type"""
        # Check classical supervised
        if specialist_type in self.classical_supervised:
            return self.classical_supervised[specialist_type]
        
        # Check classical unsupervised
        if specialist_type in self.classical_unsupervised:
            return self.classical_unsupervised[specialist_type]
        
        # Check deep learning
        if specialist_type in self.deep_learning_specialists:
            return self.deep_learning_specialists[specialist_type]
        
        return None
    
    async def _validate_with_governance(
        self,
        request: MLDLRequest,
        prediction: Any
    ) -> Dict[str, Any]:
        """Validate prediction with governance kernel"""
        if not self.governance_kernel:
            return {"approved": True, "reason": "No governance kernel"}
        
        # Create governance request
        governance_request = {
            "type": "ml_prediction",
            "model_id": request.model_id,
            "specialist_type": request.specialist_type,
            "prediction": prediction,
            "trust_score": request.trust_score,
            "constitutional_compliance": request.constitutional_compliance
        }
        
        # Validate with governance
        result = await self.governance_kernel.process_governance_request(
            decision_subject="ml_prediction",
            inputs=governance_request
        )
        
        return result
    
    async def _validate_deployment_with_governance(
        self,
        model_entry: ModelRegistryEntry,
        deployment_stage: str
    ) -> Dict[str, Any]:
        """Validate deployment with governance kernel"""
        if not self.governance_kernel:
            return {"approved": True, "reason": "No governance kernel"}
        
        # Create governance request
        governance_request = {
            "type": "model_deployment",
            "model_id": model_entry.model_id,
            "version": model_entry.version,
            "deployment_stage": deployment_stage,
            "evaluation_metrics": model_entry.evaluation_metrics,
            "constitutional_compliance": model_entry.constitutional_compliance,
            "bias_check_passed": model_entry.bias_check_passed
        }
        
        # Validate with governance
        result = await self.governance_kernel.process_governance_request(
            decision_subject="model_deployment",
            inputs=governance_request
        )
        
        return result
    
    # Event handlers
    
    async def _handle_training_request(self, event: Dict[str, Any]):
        """Handle training request from event bus"""
        request = MLDLRequest(**event["payload"])
        response = await self.process_request(request)
        
        # Publish response
        if self.event_bus:
            await self.event_bus.publish({
                "type": "ml.training.response",
                "payload": response.__dict__
            })
    
    async def _handle_prediction_request(self, event: Dict[str, Any]):
        """Handle prediction request from event bus"""
        request = MLDLRequest(**event["payload"])
        response = await self.process_request(request)
        
        # Publish response
        if self.event_bus:
            await self.event_bus.publish({
                "type": "ml.prediction.response",
                "payload": response.__dict__
            })
    
    async def _handle_deployment_request(self, event: Dict[str, Any]):
        """Handle deployment request from event bus"""
        request = MLDLRequest(**event["payload"])
        response = await self.process_request(request)
        
        # Publish response
        if self.event_bus:
            await self.event_bus.publish({
                "type": "ml.deployment.response",
                "payload": response.__dict__
            })
    
    async def _handle_rollback_request(self, event: Dict[str, Any]):
        """Handle rollback request from event bus"""
        request = MLDLRequest(**event["payload"])
        response = await self.process_request(request)
        
        # Publish response
        if self.event_bus:
            await self.event_bus.publish({
                "type": "ml.rollback.response",
                "payload": response.__dict__
            })
    
    async def _handle_governance_validation(self, event: Dict[str, Any]):
        """Handle governance validation request"""
        # Process governance validation
        pass
    
    async def _handle_active_learning_query(self, event: Dict[str, Any]):
        """Handle active learning query"""
        if not self.active_learning:
            return
        
        # Query active learning for most informative samples
        model_id = event["payload"].get("model_id")
        n_samples = event["payload"].get("n_samples", 10)
        
        samples = await self.active_learning.query_samples(
            model_id=model_id,
            n_samples=n_samples
        )
        
        # Publish result
        if self.event_bus:
            await self.event_bus.publish({
                "type": "ml.active_learning.response",
                "payload": {"samples": samples}
            })
    
    async def start(self):
        """Start the integration hub"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            logger.warning("Integration hub already running")
            return
        
        logger.info("Starting ML/DL Integration Hub...")
        
        # Start monitoring
        if self.model_monitor:
            await self.model_monitor.start()
        
        # Start active learning
        if self.active_learning:
            await self.active_learning.start()
        
        self.is_running = True
        logger.info("ML/DL Integration Hub started successfully")
    
    async def shutdown(self):
        """Shutdown the integration hub"""
        if not self.is_running:
            return
        
        logger.info("Shutting down ML/DL Integration Hub...")
        
        # Stop monitoring
        if self.model_monitor:
            await self.model_monitor.stop()
        
        # Stop active learning
        if self.active_learning:
            await self.active_learning.stop()
        
        self.is_running = False
        logger.info("ML/DL Integration Hub shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration hub status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "device": self.device,
            "specialists": {
                "classical_supervised": len(self.classical_supervised),
                "classical_unsupervised": len(self.classical_unsupervised),
                "deep_learning": len(self.deep_learning_specialists)
            },
            "operational_intelligence": {
                "uncertainty_quantifier": self.uncertainty_quantifier is not None,
                "ood_detector": self.ood_detector is not None,
                "model_monitor": self.model_monitor is not None,
                "active_learning": self.active_learning is not None
            },
            "model_registry": self.model_registry is not None,
            "governance_connected": self.governance_kernel is not None
        }
    
    def list_specialists(self) -> Dict[str, List[str]]:
        """List all available specialists"""
        return {
            "classical_supervised": list(self.classical_supervised.keys()),
            "classical_unsupervised": list(self.classical_unsupervised.keys()),
            "deep_learning": list(self.deep_learning_specialists.keys())
        }
