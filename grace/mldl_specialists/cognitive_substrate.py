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
    """
    
    def __init__(
        self,
        kpi_monitor=None,
        governance_engine=None,
        event_publisher: Optional[Callable] = None,
        immutable_logs=None
    ):
        """Initialize cognitive substrate with Grace integration points."""
        self.kpi_monitor = kpi_monitor
        self.governance_engine = governance_engine
        self.event_publisher = event_publisher
        self.immutable_logs = immutable_logs
        
        # Specialist registry by cognitive function
        self.specialists: Dict[CognitiveFunction, List[Any]] = {
            func: [] for func in CognitiveFunction
        }
        
        # Performance metrics
        self.metrics = {
            'total_events_processed': 0,
            'by_function': {func: 0 for func in CognitiveFunction},
            'by_source': {src: 0 for src in IntegrationPoint},
            'avg_processing_time_ms': 0.0,
            'avg_confidence': 0.0,
            'governance_pass_rate': 1.0
        }
        
        # Learning loop state
        self.learning_enabled = True
        self.auto_optimization = True
        
        logger.info("ML/DL Cognitive Substrate initialized as Grace's intelligence layer")
    
    def register_specialist(
        self,
        specialist: Any,
        cognitive_functions: List[CognitiveFunction]
    ):
        """Register a specialist for specific cognitive functions."""
        for func in cognitive_functions:
            self.specialists[func].append(specialist)
            logger.info(f"Registered {specialist.__class__.__name__} for {func.value}")
    
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cognitive substrate performance metrics."""
        return {
            **self.metrics,
            'by_function': {k.value: v for k, v in self.metrics['by_function'].items()},
            'by_source': {k.value: v for k, v in self.metrics['by_source'].items()}
        }
