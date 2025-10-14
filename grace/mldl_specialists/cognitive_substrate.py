"""
ML/DL Cognitive Substrate - Grace's Intelligence Layer

The ML/DL component is no longer a standalone product but the cognitive substrate
of the Grace architecture. It transforms raw signals into structured intelligence.

Core Purposes:
1. Pattern Interpretation - Recognize patterns/anomalies in data streams
2. Signal Compression - Compress high-dimensional data into embeddings
3. Simulation & Forecasting - Predict outcomes and run what-if scenarios
4. Autonomous Learning - Detect failures and optimize system behavior
5. External Verification - Optional SaaS capability for 3rd parties

Integration Points:
- Data/Tables: Reads/writes KPIs, predictions, embeddings
- APIs: Serves ML endpoints (predict, validate, synthesize)
- Kernels: Cognitive subcomponents (Pattern Recognition, Forecasting)
- TriggerMesh: Invoked on new data or KPI threshold crossings
- Governance: Validates model reliability and bias in real-time
- UI: Displays ML-driven insights, suggestions, anomaly detections

Flow:
External/Internal Event → API Layer → Table Update → TriggerMesh Event →
ML/DL Invoked (pattern detection) → Result to Tables → Governance Validates →
Kernels Act (decision, execution, adaptation)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from datetime import datetime
import asyncio
import logging
import numpy as np

# Classical ML Specialists
try:
    from grace.mldl_specialists.supervised_specialists import (
        DecisionTreeSpecialist,
        RandomForestSpecialist,
        GradientBoostingSpecialist,
        SVMSpecialist
    )
    from grace.mldl_specialists.unsupervised_specialists import (
        KMeansSpecialist,
        DBSCANSpecialist,
        PCASpecialist,
        IsolationForestSpecialist
    )
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Classical ML specialists not found - some features may be limited")

# Deep Learning Specialists
try:
    from grace.mldl_specialists.deep_learning import (
        ANNSpecialist,
        CNNSpecialist,
        RNNSpecialist,
        LSTMSpecialist,
        TransformerSpecialist,
        AutoencoderSpecialist,
        GANSpecialist,
        DeviceManager
    )
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Deep Learning specialists not available - install PyTorch: pip install torch transformers")
    DEEP_LEARNING_AVAILABLE = False

# Operational Intelligence
try:
    from grace.mldl_specialists.uncertainty_ood import (
        UncertaintyQuantifier,
        OODDetector
    )
    from grace.mldl_specialists.model_registry import ModelRegistry
    from grace.mldl_specialists.active_learning import ActiveLearningManager
    from grace.mldl_specialists.monitoring import MLModelMonitor
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Operational Intelligence modules not found - some features may be limited")

logger = logging.getLogger(__name__)


class CognitiveFunction(Enum):
    """Core cognitive functions provided by ML/DL substrate"""
    PATTERN_INTERPRETATION = "pattern_interpretation"  # Detect patterns/anomalies
    SIGNAL_COMPRESSION = "signal_compression"  # Compress to embeddings
    SIMULATION_FORECASTING = "simulation_forecasting"  # Predict outcomes
    AUTONOMOUS_LEARNING = "autonomous_learning"  # Learn from failures
    EXTERNAL_VERIFICATION = "external_verification"  # SaaS capability
    DATA_ENRICHMENT = "data_enrichment"  # Enhance inbound data
    TRUST_SCORING = "trust_scoring"  # Score data reliability
    ANOMALY_DETECTION = "anomaly_detection"  # Find outliers
    OPTIMIZATION = "optimization"  # Improve system performance


class IntegrationPoint(Enum):
    """Where ML/DL integrates into Grace architecture"""
    DATA_TABLES = "data_tables"  # Read/write KPIs, predictions
    API_LAYER = "api_layer"  # Serve ML endpoints
    KERNEL_SUBSTRATE = "kernel_substrate"  # Inside specific kernels
    TRIGGERMESH = "triggermesh"  # Event-driven invocation
    GOVERNANCE = "governance"  # Validate outcomes
    USER_INTERFACE = "user_interface"  # Display insights


@dataclass
class CognitiveEvent:
    """Event processed by the cognitive substrate"""
    event_id: str
    event_type: str
    source: IntegrationPoint
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1 (highest) to 10 (lowest)
    requires_governance: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveResult:
    """Result from cognitive processing"""
    event_id: str
    cognitive_function: CognitiveFunction
    interpretation: Any
    confidence: float  # 0.0 to 1.0
    embeddings: Optional[List[float]] = None
    predictions: Optional[Dict[str, Any]] = None
    anomalies: Optional[List[Dict[str, Any]]] = None
    optimizations: Optional[List[str]] = None
    constitutional_compliance: float = 1.0
    trust_score: float = 1.0
    processing_time_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class CognitiveSubstrate:
    """
    ML/DL Cognitive Substrate - The computational conscience of Grace.
    
    Transforms raw signals into structured intelligence across all system layers.
    Integrates with kernels, governance, tables, APIs, and TriggerMesh.
    
    NEW: Integrates all classical ML and deep learning specialists with
    operational intelligence (monitoring, active learning, uncertainty, registry).
    """
    
    def __init__(
        self,
        kpi_monitor=None,
        governance_engine=None,
        event_publisher: Optional[Callable] = None,
        immutable_logs=None,
        model_registry=None,
        ml_monitor=None,
        active_learning_manager=None,
        enable_gpu: bool = True
    ):
        """Initialize cognitive substrate with Grace integration points."""
        self.kpi_monitor = kpi_monitor
        self.governance_engine = governance_engine
        self.event_publisher = event_publisher
        self.immutable_logs = immutable_logs
        
        # Operational Intelligence Integration
        self.model_registry = model_registry or (ModelRegistry() if 'ModelRegistry' in globals() else None)
        self.ml_monitor = ml_monitor or (MLModelMonitor() if 'MLModelMonitor' in globals() else None)
        self.active_learning = active_learning_manager or (ActiveLearningManager() if 'ActiveLearningManager' in globals() else None)
        
        # Uncertainty and OOD detection
        self.uncertainty_quantifier = UncertaintyQuantifier() if 'UncertaintyQuantifier' in globals() else None
        self.ood_detector = OODDetector() if 'OODDetector' in globals() else None
        
        # Device management for deep learning
        self.device = "cpu"
        if DEEP_LEARNING_AVAILABLE and enable_gpu:
            self.device = DeviceManager.get_device()
            logger.info(f"Deep Learning enabled with device: {self.device}")
        
        # Specialist registry by cognitive function
        self.specialists: Dict[CognitiveFunction, List[Any]] = {
            func: [] for func in CognitiveFunction
        }
        
        # Specialist instances by ID
        self.specialist_instances: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_events_processed': 0,
            'by_function': {func: 0 for func in CognitiveFunction},
            'by_source': {src: 0 for src in IntegrationPoint},
            'avg_processing_time_ms': 0.0,
            'avg_confidence': 0.0,
            'governance_pass_rate': 1.0,
            'deep_learning_enabled': DEEP_LEARNING_AVAILABLE,
            'gpu_available': self.device in ['cuda', 'mps']
        }
        
        # Learning loop state
        self.learning_enabled = True
        self.auto_optimization = True
        
        logger.info(f"ML/DL Cognitive Substrate initialized as Grace's intelligence layer (GPU: {self.device})")
    
    def register_specialist(
        self,
        specialist: Any,
        cognitive_functions: List[CognitiveFunction],
        specialist_id: Optional[str] = None
    ):
        """
        Register a specialist for specific cognitive functions.
        
        Args:
            specialist: ML/DL specialist instance
            cognitive_functions: List of cognitive functions this specialist handles
            specialist_id: Unique identifier (auto-generated if None)
        """
        if specialist_id is None:
            specialist_id = f"{specialist.__class__.__name__}_{len(self.specialist_instances)}"
        
        # Register by cognitive function
        for func in cognitive_functions:
            self.specialists[func].append(specialist)
            logger.info(f"Registered {specialist.__class__.__name__} for {func.value}")
        
        # Store instance
        self.specialist_instances[specialist_id] = specialist
        
        # Register with model registry
        if self.model_registry:
            try:
                asyncio.create_task(self.model_registry.register_model(
                    model_id=specialist_id,
                    model_type=specialist.__class__.__name__,
                    model=specialist,
                    metrics={},
                    metadata={'cognitive_functions': [f.value for f in cognitive_functions]}
                ))
            except Exception as e:
                logger.warning(f"Could not register with model registry: {e}")
    
    async def create_specialist(
        self,
        specialist_type: str,
        specialist_id: str,
        cognitive_functions: List[CognitiveFunction],
        **kwargs
    ) -> Any:
        """
        Create and register a new specialist.
        
        Args:
            specialist_type: Type of specialist (e.g., "LSTM", "RandomForest", "Transformer")
            specialist_id: Unique identifier
            cognitive_functions: List of cognitive functions
            **kwargs: Specialist-specific parameters
            
        Returns:
            Created specialist instance
        """
        specialist_map = {
            # Classical ML
            'DecisionTree': DecisionTreeSpecialist if 'DecisionTreeSpecialist' in globals() else None,
            'RandomForest': RandomForestSpecialist if 'RandomForestSpecialist' in globals() else None,
            'GradientBoosting': GradientBoostingSpecialist if 'GradientBoostingSpecialist' in globals() else None,
            'SVM': SVMSpecialist if 'SVMSpecialist' in globals() else None,
            'KMeans': KMeansSpecialist if 'KMeansSpecialist' in globals() else None,
            'DBSCAN': DBSCANSpecialist if 'DBSCANSpecialist' in globals() else None,
            'PCA': PCASpecialist if 'PCASpecialist' in globals() else None,
            'IsolationForest': IsolationForestSpecialist if 'IsolationForestSpecialist' in globals() else None,
        }
        
        # Add deep learning specialists if available
        if DEEP_LEARNING_AVAILABLE:
            specialist_map.update({
                'ANN': ANNSpecialist,
                'CNN': CNNSpecialist,
                'RNN': RNNSpecialist,
                'LSTM': LSTMSpecialist,
                'Transformer': TransformerSpecialist,
                'Autoencoder': AutoencoderSpecialist,
                'GAN': GANSpecialist
            })
        
        specialist_class = specialist_map.get(specialist_type)
        if specialist_class is None:
            raise ValueError(f"Unknown specialist type: {specialist_type}")
        
        # Create specialist
        specialist = specialist_class(specialist_id=specialist_id, **kwargs)
        
        # Register specialist
        self.register_specialist(specialist, cognitive_functions, specialist_id)
        
        logger.info(f"Created and registered specialist: {specialist_id} ({specialist_type})")
        return specialist
    
    async def train_specialist(
        self,
        specialist_id: str,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Train a specialist with monitoring and registry integration.
        
        Args:
            specialist_id: Specialist to train
            X_train: Training features
            y_train: Training labels (None for unsupervised)
            X_val: Validation features
            y_val: Validation labels
            **training_kwargs: Model-specific training parameters
            
        Returns:
            Training metrics
        """
        if specialist_id not in self.specialist_instances:
            raise ValueError(f"Specialist {specialist_id} not found")
        
        specialist = self.specialist_instances[specialist_id]
        
        # Start monitoring
        if self.ml_monitor:
            self.ml_monitor.log_training_start(specialist_id)
        
        try:
            # Train model
            if hasattr(specialist, 'fit'):
                # Check if it's async
                if asyncio.iscoroutinefunction(specialist.fit):
                    result = await specialist.fit(
                        X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        **training_kwargs
                    )
                else:
                    # Classical ML (synchronous)
                    result = await asyncio.to_thread(
                        specialist.fit,
                        X_train, y_train
                    )
            else:
                raise AttributeError(f"Specialist {specialist_id} has no fit method")
            
            # Update model registry
            if self.model_registry:
                await self.model_registry.update_model(
                    model_id=specialist_id,
                    metrics=result if isinstance(result, dict) else {},
                    metadata={
                        'device': self.device,
                        'train_samples': len(X_train) if hasattr(X_train, '__len__') else 'unknown',
                        'val_samples': len(X_val) if X_val is not None and hasattr(X_val, '__len__') else 0,
                        'last_trained': datetime.utcnow().isoformat()
                    }
                )
            
            # Log success
            if self.ml_monitor:
                self.ml_monitor.log_training_complete(specialist_id, result)
            
            logger.info(f"Training complete for {specialist_id}")
            return result
            
        except Exception as e:
            if self.ml_monitor:
                self.ml_monitor.log_training_error(specialist_id, str(e))
            logger.error(f"Training failed for {specialist_id}: {e}")
            raise
    
    async def predict_with_specialist(
        self,
        specialist_id: str,
        X,
        detect_ood: bool = True,
        calculate_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with a specific specialist, including uncertainty and OOD detection.
        
        Args:
            specialist_id: Specialist to use
            X: Input features
            detect_ood: Whether to detect out-of-distribution samples
            calculate_uncertainty: Whether to quantify uncertainty
            
        Returns:
            Prediction with uncertainty and OOD information
        """
        if specialist_id not in self.specialist_instances:
            raise ValueError(f"Specialist {specialist_id} not found")
        
        specialist = self.specialist_instances[specialist_id]
        
        # Make prediction
        if hasattr(specialist, 'predict_async'):
            prediction = await specialist.predict_async(X, context={})
        elif hasattr(specialist, 'predict'):
            prediction = await asyncio.to_thread(specialist.predict, X)
        else:
            raise AttributeError(f"Specialist {specialist_id} has no predict method")
        
        # Get confidence/probabilities
        confidence = None
        proba = None
        if hasattr(specialist, 'predict_proba'):
            proba = await asyncio.to_thread(specialist.predict_proba, X)
            if proba is not None and len(proba.shape) > 1:
                # Use numpy if available, otherwise Python max
                try:
                    import numpy as np
                    confidence = float(np.max(proba, axis=1).mean())
                except ImportError:
                    confidence = max([max(row) for row in proba]) / len(proba)
        
        # Uncertainty quantification
        uncertainty = None
        if calculate_uncertainty and self.uncertainty_quantifier and proba is not None:
            uncertainty = self.uncertainty_quantifier.calculate_entropy(proba)
        
        # OOD detection
        is_ood = False
        if detect_ood and self.ood_detector and proba is not None:
            ood_scores = self.ood_detector.detect_ood(proba)
            try:
                import numpy as np
                is_ood = float(ood_scores.mean()) > 0.5
            except ImportError:
                is_ood = sum(ood_scores) / len(ood_scores) > 0.5
        
        # Log prediction
        if self.ml_monitor:
            self.ml_monitor.log_prediction(
                model_id=specialist_id,
                prediction=prediction,
                confidence=confidence,
                uncertainty=uncertainty,
                is_ood=is_ood
            )
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'is_ood': is_ood,
            'specialist_id': specialist_id,
            'specialist_type': specialist.__class__.__name__,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def process_cognitive_event(
        self,
        event: CognitiveEvent
    ) -> CognitiveResult:
        """
        Process an event through the cognitive substrate.
        
        Flow:
        1. Route to appropriate specialists
        2. Execute cognitive processing
        3. Validate through governance
        4. Update KPIs and trust scores
        5. Log to immutable audit trail
        6. Publish result events
        """
        start_time = datetime.now()
        
        # Determine which cognitive function(s) are needed
        functions = self._determine_functions(event)
        
        results = []
        for func in functions:
            specialists = self.specialists.get(func, [])
            if not specialists:
                logger.warning(f"No specialists registered for {func.value}")
                continue
            
            # Execute specialists in parallel
            specialist_results = await asyncio.gather(*[
                self._execute_specialist(specialist, event, func)
                for specialist in specialists
            ])
            results.extend(specialist_results)
        
        # Synthesize results
        cognitive_result = await self._synthesize_results(
            event,
            results,
            functions
        )
        
        # Governance validation
        if event.requires_governance and self.governance_engine:
            cognitive_result.constitutional_compliance = await self._validate_governance(
                cognitive_result
            )
        
        # Update KPIs and trust
        if self.kpi_monitor:
            await self._update_kpis(cognitive_result)
        
        # Immutable logging
        if self.immutable_logs:
            await self._log_cognitive_event(event, cognitive_result)
        
        # Publish result events
        if self.event_publisher:
            await self._publish_result(cognitive_result)
        
        # Update metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        cognitive_result.processing_time_ms = processing_time
        self._update_metrics(event, cognitive_result)
        
        return cognitive_result
    
    def _determine_functions(self, event: CognitiveEvent) -> List[CognitiveFunction]:
        """Determine which cognitive functions to apply based on event."""
        # Smart routing based on event type and source
        function_map = {
            'data_validation': [CognitiveFunction.TRUST_SCORING, CognitiveFunction.ANOMALY_DETECTION],
            'pattern_detection': [CognitiveFunction.PATTERN_INTERPRETATION],
            'forecast_request': [CognitiveFunction.SIMULATION_FORECASTING],
            'optimization_request': [CognitiveFunction.OPTIMIZATION, CognitiveFunction.AUTONOMOUS_LEARNING],
            'data_ingestion': [CognitiveFunction.DATA_ENRICHMENT, CognitiveFunction.SIGNAL_COMPRESSION],
            'external_request': [CognitiveFunction.EXTERNAL_VERIFICATION],
            'kpi_threshold_crossed': [CognitiveFunction.PATTERN_INTERPRETATION, CognitiveFunction.SIMULATION_FORECASTING],
        }
        
        return function_map.get(event.event_type, [CognitiveFunction.PATTERN_INTERPRETATION])
    
    async def _execute_specialist(
        self,
        specialist: Any,
        event: CognitiveEvent,
        cognitive_function: CognitiveFunction
    ) -> Dict[str, Any]:
        """Execute a single specialist."""
        try:
            # Prepare data for specialist
            prepared_data = self._prepare_data_for_specialist(event.data, specialist)
            
            # Execute specialist (assume async predict method)
            if hasattr(specialist, 'predict_async'):
                result = await specialist.predict_async(prepared_data, event.metadata)
            elif hasattr(specialist, 'predict'):
                result = await asyncio.to_thread(specialist.predict, prepared_data, event.metadata)
            else:
                raise AttributeError(f"Specialist {specialist.__class__.__name__} has no predict method")
            
            return {
                'specialist': specialist.__class__.__name__,
                'function': cognitive_function,
                'result': result,
                'success': True
            }
        except Exception as e:
            logger.error(f"Specialist {specialist.__class__.__name__} failed: {e}")
            return {
                'specialist': specialist.__class__.__name__,
                'function': cognitive_function,
                'error': str(e),
                'success': False
            }
    
    def _prepare_data_for_specialist(self, data: Dict[str, Any], specialist: Any) -> Any:
        """Prepare event data for specialist consumption."""
        # Extract relevant features based on specialist type
        # This is where signal compression and feature engineering happens
        return data
    
    async def _synthesize_results(
        self,
        event: CognitiveEvent,
        specialist_results: List[Dict[str, Any]],
        functions: List[CognitiveFunction]
    ) -> CognitiveResult:
        """Synthesize multiple specialist results into unified cognitive result."""
        successful_results = [r for r in specialist_results if r.get('success')]
        
        if not successful_results:
            # No successful results - return low-confidence result
            return CognitiveResult(
                event_id=event.event_id,
                cognitive_function=functions[0] if functions else CognitiveFunction.PATTERN_INTERPRETATION,
                interpretation=None,
                confidence=0.0,
                models_used=[],
                metadata={'error': 'All specialists failed'}
            )
        
        # Extract interpretations and confidences
        interpretations = []
        confidences = []
        models_used = []
        
        for result in successful_results:
            specialist_result = result['result']
            if hasattr(specialist_result, 'prediction'):
                interpretations.append(specialist_result.prediction)
                confidences.append(specialist_result.confidence)
            if hasattr(specialist_result, 'specialist_id'):
                models_used.append(specialist_result.specialist_id)
        
        # Weighted average or voting mechanism
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Primary interpretation (could be voting, averaging, or most confident)
        primary_interpretation = interpretations[0] if interpretations else None
        
        return CognitiveResult(
            event_id=event.event_id,
            cognitive_function=functions[0] if functions else CognitiveFunction.PATTERN_INTERPRETATION,
            interpretation=primary_interpretation,
            confidence=avg_confidence,
            models_used=models_used,
            metadata={
                'specialist_count': len(successful_results),
                'functions_applied': [f.value for f in functions]
            }
        )
    
    async def _validate_governance(self, result: CognitiveResult) -> float:
        """Validate result through governance engine."""
        # Check constitutional compliance
        if hasattr(self.governance_engine, 'validate_decision'):
            validation = await self.governance_engine.validate_decision({
                'interpretation': result.interpretation,
                'confidence': result.confidence,
                'models': result.models_used
            })
            return validation.get('compliance_score', 1.0)
        return 1.0
    
    async def _update_kpis(self, result: CognitiveResult):
        """Update KPI monitor with cognitive result metrics."""
        if not self.kpi_monitor:
            return
        
        await self.kpi_monitor.record_metric(
            name='cognitive_confidence',
            value=result.confidence * 100,
            component_id='cognitive_substrate',
            threshold_warning=70.0,
            threshold_critical=50.0,
            tags={'function': result.cognitive_function.value}
        )
        
        await self.kpi_monitor.record_metric(
            name='cognitive_processing_time',
            value=result.processing_time_ms,
            component_id='cognitive_substrate',
            threshold_warning=1000.0,
            threshold_critical=5000.0,
            tags={'function': result.cognitive_function.value}
        )
    
    async def _log_cognitive_event(self, event: CognitiveEvent, result: CognitiveResult):
        """Log cognitive event to immutable audit trail."""
        if not self.immutable_logs:
            return
        
        await self.immutable_logs.append(
            operation_type='cognitive_processing',
            operation_data={
                'event_id': event.event_id,
                'event_type': event.event_type,
                'source': event.source.value,
                'function': result.cognitive_function.value,
                'confidence': result.confidence,
                'compliance': result.constitutional_compliance,
                'models': result.models_used
            },
            user_id='cognitive_substrate'
        )
    
    async def _publish_result(self, result: CognitiveResult):
        """Publish cognitive result event to TriggerMesh."""
        await self.event_publisher(
            'cognitive_result',
            {
                'event_id': result.event_id,
                'function': result.cognitive_function.value,
                'confidence': result.confidence,
                'interpretation': result.interpretation,
                'compliance': result.constitutional_compliance,
                'timestamp': result.timestamp.isoformat()
            }
        )
    
    def _update_metrics(self, event: CognitiveEvent, result: CognitiveResult):
        """Update internal performance metrics."""
        self.metrics['total_events_processed'] += 1
        self.metrics['by_function'][result.cognitive_function] += 1
        self.metrics['by_source'][event.source] += 1
        
        # Running averages
        n = self.metrics['total_events_processed']
        self.metrics['avg_processing_time_ms'] = (
            (self.metrics['avg_processing_time_ms'] * (n - 1) + result.processing_time_ms) / n
        )
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * (n - 1) + result.confidence) / n
        )
        self.metrics['governance_pass_rate'] = (
            (self.metrics['governance_pass_rate'] * (n - 1) + result.constitutional_compliance) / n
        )
    
    async def pattern_interpretation(
        self,
        data_stream: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> CognitiveResult:
        """
        Pattern Interpretation - Detect patterns and anomalies in data streams.
        
        Example: Detect trust score pattern shifts before failure.
        """
        event = CognitiveEvent(
            event_id=f"pattern_{datetime.now().timestamp()}",
            event_type='pattern_detection',
            source=IntegrationPoint.DATA_TABLES,
            data=data_stream,
            metadata=context or {}
        )
        return await self.process_cognitive_event(event)
    
    async def signal_compression(
        self,
        high_dim_data: Any,
        target_dimensions: int = 128
    ) -> CognitiveResult:
        """
        Signal Compression - Compress high-dimensional data to embeddings.
        
        Makes system telemetry tractable for real-time reasoning.
        """
        event = CognitiveEvent(
            event_id=f"compress_{datetime.now().timestamp()}",
            event_type='data_ingestion',
            source=IntegrationPoint.API_LAYER,
            data={'raw_data': high_dim_data, 'target_dim': target_dimensions}
        )
        return await self.process_cognitive_event(event)
    
    async def simulation_forecasting(
        self,
        current_state: Dict[str, Any],
        forecast_horizon: int = 5
    ) -> CognitiveResult:
        """
        Simulation & Forecasting - Predict outcomes and run what-if scenarios.
        
        Example: Forecast trust drifts or KPI trajectories.
        """
        event = CognitiveEvent(
            event_id=f"forecast_{datetime.now().timestamp()}",
            event_type='forecast_request',
            source=IntegrationPoint.KERNEL_SUBSTRATE,
            data={'state': current_state, 'horizon': forecast_horizon}
        )
        return await self.process_cognitive_event(event)
    
    async def autonomous_learning(
        self,
        failure_data: Dict[str, Any]
    ) -> CognitiveResult:
        """
        Autonomous Learning - Learn from failures and optimize.
        
        Detects failure patterns and proposes improvements.
        """
        event = CognitiveEvent(
            event_id=f"learn_{datetime.now().timestamp()}",
            event_type='optimization_request',
            source=IntegrationPoint.KERNEL_SUBSTRATE,
            data=failure_data,
            priority=1  # High priority
        )
        return await self.process_cognitive_event(event)
    
    async def external_verification(
        self,
        external_data: Dict[str, Any],
        verification_type: str = 'fraud_detection'
    ) -> CognitiveResult:
        """
        External Verification - SaaS capability for 3rd parties.
        
        Validates external data and returns trust score.
        """
        event = CognitiveEvent(
            event_id=f"external_{datetime.now().timestamp()}",
            event_type='external_request',
            source=IntegrationPoint.API_LAYER,
            data={'data': external_data, 'type': verification_type},
            requires_governance=True
        )
        return await self.process_cognitive_event(event)
    
    async def active_learning_query(
        self,
        specialist_id: str,
        X_pool,
        n_samples: int = 10,
        strategy: str = "uncertainty"
    ):
        """
        Query samples for active learning.
        
        Args:
            specialist_id: Specialist to use
            X_pool: Pool of unlabeled samples
            n_samples: Number of samples to query
            strategy: Sampling strategy ("uncertainty", "diversity")
            
        Returns:
            Indices and samples to label
        """
        if specialist_id not in self.specialist_instances:
            raise ValueError(f"Specialist {specialist_id} not found")
        
        specialist = self.specialist_instances[specialist_id]
        
        if not self.active_learning:
            logger.warning("Active learning not available")
            return None, None
        
        # Query samples
        indices = await self.active_learning.query_samples(
            model=specialist,
            X_pool=X_pool,
            n_samples=n_samples,
            strategy=strategy
        )
        
        return indices, X_pool[indices] if indices is not None else None
    
    async def ensemble_predict(
        self,
        specialist_ids: List[str],
        X,
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Ensemble prediction from multiple specialists.
        
        Args:
            specialist_ids: List of specialists to use
            X: Input features
            weights: Voting weights (equal if None)
            
        Returns:
            Ensemble prediction with aggregated confidence
        """
        if weights is None:
            weights = [1.0 / len(specialist_ids)] * len(specialist_ids)
        
        predictions = []
        confidences = []
        
        for specialist_id, weight in zip(specialist_ids, weights):
            result = await self.predict_with_specialist(
                specialist_id, X,
                detect_ood=True,
                calculate_uncertainty=True
            )
            predictions.append(result['prediction'])
            if result['confidence'] is not None:
                confidences.append(result['confidence'] * weight)
        
        # Aggregate predictions
        # Simple averaging for now - could be voting for classification
        try:
            import numpy as np
            if isinstance(predictions[0], (list, np.ndarray)):
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
        except ImportError:
            # Fallback without numpy
            ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
        
        return {
            'prediction': ensemble_pred,
            'confidence': sum(confidences) / len(confidences) if confidences else None,
            'ensemble_size': len(specialist_ids),
            'specialist_ids': specialist_ids,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def list_specialists(self) -> List[Dict[str, Any]]:
        """List all registered specialists with their cognitive functions."""
        specialist_info = []
        
        for specialist_id, specialist in self.specialist_instances.items():
            # Find cognitive functions for this specialist
            functions = []
            for func, specialists in self.specialists.items():
                if specialist in specialists:
                    functions.append(func.value)
            
            info = {
                'specialist_id': specialist_id,
                'specialist_type': specialist.__class__.__name__,
                'cognitive_functions': functions,
                'device': self.device if DEEP_LEARNING_AVAILABLE else 'cpu'
            }
            
            # Add model-specific info if available
            if hasattr(specialist, 'get_model_summary'):
                try:
                    info.update(specialist.get_model_summary())
                except Exception:
                    pass
            
            specialist_info.append(info)
        
        return specialist_info
    
    def get_specialist(self, specialist_id: str) -> Any:
        """Get a specialist instance by ID."""
        return self.specialist_instances.get(specialist_id)
    
    async def cleanup(self):
        """Cleanup resources including GPU memory."""
        if DEEP_LEARNING_AVAILABLE and self.device in ["cuda", "mps"]:
            DeviceManager.clear_cache()
        
        logger.info("Cognitive Substrate cleanup complete")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cognitive substrate performance metrics."""
        return {
            **self.metrics,
            'by_function': {k.value: v for k, v in self.metrics['by_function'].items()},
            'by_source': {k.value: v for k, v in self.metrics['by_source'].items()},
            'specialists_registered': len(self.specialist_instances),
            'deep_learning_available': DEEP_LEARNING_AVAILABLE,
            'device': self.device
        }


# Global instance
_cognitive_substrate: Optional[CognitiveSubstrate] = None


def get_cognitive_substrate(**kwargs) -> CognitiveSubstrate:
    """Get or create global cognitive substrate instance."""
    global _cognitive_substrate
    if _cognitive_substrate is None:
        _cognitive_substrate = CognitiveSubstrate(**kwargs)
    return _cognitive_substrate
