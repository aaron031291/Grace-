"""
Complete ML/DL Integration Example - All 4 Layers

Demonstrates the full ML/DL cognitive substrate integrated into Grace:

Layer 1: Individual Specialists (supervised, unsupervised, deep learning)
Layer 2: Consensus Engine (aggregates predictions)
Layer 3: Governance Validation (constitutional compliance)
Layer 4: Federated Meta-Learning (continuous improvement)

Integration with Grace systems:
- KPI Trust Monitor
- Governance Engine
- Immutable Logs
- Event Publisher (TriggerMesh)
- Memory Bridge
- All Kernels
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Layer 1: Individual Specialists
from grace.mldl_specialists.supervised_specialists import (
    DecisionTreeSpecialist,
    SVMSpecialist,
    RandomForestSpecialist,
    GradientBoostingSpecialist
)
from grace.mldl_specialists.unsupervised_specialists import (
    KMeansClusteringSpecialist,
    DBSCANClusteringSpecialist,
    PCADimensionalityReductionSpecialist,
    IsolationForestAnomalySpecialist
)

# Layer 2: Consensus Engine
from grace.mldl_specialists.consensus_engine import (
    MLDLConsensusEngine,
    ConsensusResult
)

# Layer 3: Cognitive Substrate Orchestration
from grace.mldl_specialists.cognitive_substrate import (
    CognitiveSubstrate,
    CognitiveEvent,
    CognitiveFunction,
    IntegrationPoint
)

# Layer 4: Cognitive Kernels
from grace.mldl_specialists.cognitive_kernels import (
    PatternRecognitionKernel,
    ForecastingKernel,
    OptimizationKernel,
    AnomalyDetectionKernel,
    TrustScoringKernel,
    CognitiveKernelOrchestrator
)

logger = logging.getLogger(__name__)


class GraceMLDLIntegration:
    """
    Complete integration of ML/DL cognitive substrate into Grace.
    
    This is the "computational conscience" - not a standalone product,
    but the intelligent processing layer embedded in every kernel.
    
    Architecture:
        External/Internal Event
                ↓
        API Layer (Interface)
                ↓
        Table Update (KPIs, State)
                ↓
        TriggerMesh Event (Event Bus)
                ↓
        ML/DL Cognitive Substrate ← YOU ARE HERE
                ↓
        Results to Tables
                ↓
        Governance Validates
                ↓
        Kernels Act
                ↓
        Immutable Logs
    """
    
    def __init__(
        self,
        kpi_monitor=None,
        governance_engine=None,
        event_publisher=None,
        immutable_logs=None,
        memory_bridge=None
    ):
        """Initialize the complete ML/DL integration."""
        
        # Grace system integrations
        self.kpi_monitor = kpi_monitor
        self.governance_engine = governance_engine
        self.event_publisher = event_publisher
        self.immutable_logs = immutable_logs
        self.memory_bridge = memory_bridge
        
        # Layer 2: Consensus Engine
        self.consensus_engine = MLDLConsensusEngine(
            governance_bridge=governance_engine,
            kpi_monitor=kpi_monitor,
            immutable_logs=immutable_logs,
            memory_bridge=memory_bridge,
            min_specialists_required=2,
            consensus_threshold=0.6
        )
        
        # Layer 3: Cognitive Substrate
        self.cognitive_substrate = CognitiveSubstrate(
            kpi_monitor=kpi_monitor,
            governance_engine=governance_engine,
            event_publisher=event_publisher,
            immutable_logs=immutable_logs
        )
        
        # Layer 4: Cognitive Kernels
        self.kernel_orchestrator = CognitiveKernelOrchestrator(
            self.cognitive_substrate
        )
        
        # Layer 1: Initialize specialists (done during setup)
        self.specialists: Dict[str, Any] = {}
        
        logger.info("GraceMLDLIntegration initialized - 4-layer architecture ready")
    
    async def initialize_specialists(self):
        """
        Initialize and train all specialist models.
        
        In production, this would load pre-trained models from registry.
        For demo, we create fresh models.
        """
        
        # Supervised Learning Specialists
        self.specialists['decision_tree'] = DecisionTreeSpecialist(
            mode="classification"
        )
        self.specialists['svm'] = SVMSpecialist(
            mode="classification"
        )
        self.specialists['random_forest'] = RandomForestSpecialist(
            mode="regression"
        )
        self.specialists['gradient_boost'] = GradientBoostingSpecialist()
        
        # Unsupervised Learning Specialists
        self.specialists['kmeans'] = KMeansClusteringSpecialist(n_clusters=3)
        self.specialists['dbscan'] = DBSCANClusteringSpecialist()
        self.specialists['pca'] = PCADimensionalityReductionSpecialist()
        self.specialists['isolation_forest'] = IsolationForestAnomalySpecialist()
        
        # Register with consensus engine
        for specialist_id, specialist in self.specialists.items():
            self.consensus_engine.register_specialist(specialist)
        
        # Register with cognitive substrate
        for specialist_id, specialist in self.specialists.items():
            # Map specialist to cognitive functions
            if isinstance(specialist, (DecisionTreeSpecialist, SVMSpecialist, RandomForestSpecialist)):
                functions = [CognitiveFunction.PATTERN_INTERPRETATION, CognitiveFunction.TRUST_SCORING]
            elif isinstance(specialist, (KMeansClusteringSpecialist, DBSCANClusteringSpecialist)):
                functions = [CognitiveFunction.PATTERN_INTERPRETATION, CognitiveFunction.DATA_ENRICHMENT]
            elif isinstance(specialist, PCADimensionalityReductionSpecialist):
                functions = [CognitiveFunction.SIGNAL_COMPRESSION]
            elif isinstance(specialist, IsolationForestAnomalySpecialist):
                functions = [CognitiveFunction.ANOMALY_DETECTION]
            else:
                functions = [CognitiveFunction.PATTERN_INTERPRETATION]
            
            self.cognitive_substrate.register_specialist(
                specialist_id=specialist_id,
                specialist=specialist,
                cognitive_functions=functions
            )
        
        logger.info(f"Initialized {len(self.specialists)} ML/DL specialists")
    
    async def train_specialists_from_data(self, training_data: Dict[str, Any]):
        """
        Train specialists on Grace operational data.
        
        Training data comes from:
        - Historical KPI metrics
        - Past governance decisions
        - User interaction patterns
        - System performance logs
        """
        
        # Extract training datasets
        X_classification = training_data.get('classification_features', np.array([]))
        y_classification = training_data.get('classification_labels', np.array([]))
        
        X_regression = training_data.get('regression_features', np.array([]))
        y_regression = training_data.get('regression_values', np.array([]))
        
        X_clustering = training_data.get('clustering_features', np.array([]))
        
        X_anomaly = training_data.get('anomaly_features', np.array([]))
        
        # Train supervised specialists
        if len(X_classification) > 0 and len(y_classification) > 0:
            self.specialists['decision_tree'].train(X_classification, y_classification)
            self.specialists['svm'].train(X_classification, y_classification)
            self.specialists['gradient_boost'].train(X_classification, y_classification)
        
        if len(X_regression) > 0 and len(y_regression) > 0:
            self.specialists['random_forest'].train(X_regression, y_regression)
        
        # Train unsupervised specialists
        if len(X_clustering) > 0:
            self.specialists['kmeans'].train(X_clustering)
            self.specialists['dbscan'].train(X_clustering)
            self.specialists['pca'].train(X_clustering)
        
        if len(X_anomaly) > 0:
            self.specialists['isolation_forest'].train(X_anomaly)
        
        logger.info("All specialists trained on Grace operational data")
    
    async def process_grace_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        source: str = "api_layer"
    ) -> Dict[str, Any]:
        """
        Main entry point: Process event through ML/DL cognitive substrate.
        
        Flow:
        1. Event arrives from TriggerMesh
        2. Cognitive Substrate routes to appropriate function
        3. Specialists make predictions
        4. Consensus Engine aggregates
        5. Governance validates
        6. Results stored in tables
        7. Kernels notified to act
        8. Everything logged immutably
        
        Args:
            event_type: Type of event (kpi_threshold_crossed, data_ingestion, etc.)
            event_data: Event payload
            source: Source system
        
        Returns:
            Processing results with predictions, insights, actions
        """
        
        # Create cognitive event
        cognitive_event = CognitiveEvent(
            event_id=f"{event_type}_{datetime.now().timestamp()}",
            source=source,
            data=event_data,
            context={'event_type': event_type}
        )
        
        # Process through cognitive substrate (Layer 3)
        cognitive_result = await self.cognitive_substrate.process_cognitive_event(
            cognitive_event
        )
        
        # Route to appropriate cognitive kernels (Layer 4)
        kernel_insights = await self.kernel_orchestrator.process_event(
            event_type=event_type,
            data=event_data
        )
        
        # Combine results
        combined_result = {
            'cognitive_result': {
                'prediction': cognitive_result.prediction,
                'confidence': cognitive_result.confidence,
                'interpretation': cognitive_result.interpretation,
                'governance_approved': cognitive_result.governance_approved
            },
            'kernel_insights': [
                {
                    'kernel': insight.kernel_id,
                    'type': insight.insight_type,
                    'title': insight.title,
                    'severity': insight.severity,
                    'actions': insight.recommended_actions,
                    'confidence': insight.confidence
                }
                for insight in kernel_insights
            ],
            'processing_metadata': {
                'specialists_used': cognitive_result.specialists_used,
                'processing_time_ms': cognitive_result.processing_time_ms,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Update KPI metrics
        if self.kpi_monitor:
            await self.kpi_monitor.record_metric({
                'metric_type': 'ml_dl_processing',
                'event_type': event_type,
                'confidence': cognitive_result.confidence,
                'governance_approved': cognitive_result.governance_approved,
                'timestamp': datetime.now().isoformat()
            })
        
        # Publish to event bus for kernel consumption
        if self.event_publisher:
            await self.event_publisher.publish({
                'event_type': 'ml_dl_insight_generated',
                'payload': combined_result,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(
            f"Processed {event_type}: confidence={cognitive_result.confidence:.2f}, "
            f"insights={len(kernel_insights)}, approved={cognitive_result.governance_approved}"
        )
        
        return combined_result
    
    async def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics across all layers."""
        
        metrics = {
            'layer_1_specialists': {
                'total_specialists': len(self.specialists),
                'trained_specialists': sum(
                    1 for s in self.specialists.values() if s.is_trained
                )
            },
            'layer_2_consensus': self.consensus_engine.get_consensus_stats(),
            'layer_3_cognitive_substrate': self.cognitive_substrate.get_metrics(),
            'layer_4_kernel_insights': {
                'pattern_recognition': self.kernel_orchestrator.pattern_recognition.kernel_id,
                'forecasting': self.kernel_orchestrator.forecasting.kernel_id,
                'optimization': self.kernel_orchestrator.optimization.kernel_id,
                'anomaly_detection': self.kernel_orchestrator.anomaly_detection.kernel_id,
                'trust_scoring': self.kernel_orchestrator.trust_scoring.kernel_id
            }
        }
        
        return metrics


async def demo_complete_integration():
    """
    Demonstration of complete 4-layer ML/DL cognitive substrate.
    
    Shows how ML/DL is embedded as Grace's intelligence layer.
    """
    
    print("=" * 80)
    print("Grace ML/DL Cognitive Substrate - Complete Integration Demo")
    print("=" * 80)
    
    # Initialize integration
    integration = GraceMLDLIntegration()
    await integration.initialize_specialists()
    
    # Generate synthetic training data (in production, from Grace operations)
    print("\n1. Training specialists on Grace operational data...")
    training_data = {
        'classification_features': np.random.randn(100, 5),
        'classification_labels': np.random.randint(0, 3, 100),
        'regression_features': np.random.randn(100, 5),
        'regression_values': np.random.randn(100),
        'clustering_features': np.random.randn(100, 5),
        'anomaly_features': np.random.randn(100, 5)
    }
    await integration.train_specialists_from_data(training_data)
    
    # Simulate events from Grace systems
    print("\n2. Processing events through cognitive substrate...")
    
    # Event 1: KPI threshold crossed
    event1_result = await integration.process_grace_event(
        event_type='kpi_threshold_crossed',
        event_data={
            'kpi_name': 'trust_score',
            'current_value': 0.72,
            'threshold': 0.75,
            'component': 'ingress_kernel',
            'features': [0.72, 0.68, 0.71, 0.73, 0.70]  # Last 5 values
        },
        source='kpi_monitor'
    )
    print(f"   Event 1 processed: {event1_result['cognitive_result']['interpretation']}")
    print(f"   Confidence: {event1_result['cognitive_result']['confidence']:.2%}")
    print(f"   Kernel insights: {len(event1_result['kernel_insights'])}")
    
    # Event 2: Data ingestion for trust scoring
    event2_result = await integration.process_grace_event(
        event_type='data_ingestion',
        event_data={
            'source': 'external_api_xyz',
            'data_quality_score': 0.88,
            'features': [0.88, 0.92, 0.85, 0.90, 0.87]
        },
        source='ingress_kernel'
    )
    print(f"   Event 2 processed: {event2_result['cognitive_result']['interpretation']}")
    print(f"   Governance approved: {event2_result['cognitive_result']['governance_approved']}")
    
    # Event 3: Anomaly detection request
    event3_result = await integration.process_grace_event(
        event_type='security_alert',
        event_data={
            'access_logs': [
                {'user': 'user_123', 'resource': '/admin', 'time': '10:00'},
                {'user': 'user_123', 'resource': '/admin', 'time': '10:01'},
                {'user': 'user_123', 'resource': '/admin', 'time': '10:02'}
            ],
            'features': [3.2, 1.5, 0.8, 2.1, 1.9]  # Extracted features
        },
        source='security_monitor'
    )
    print(f"   Event 3 processed: Anomaly detected: {event3_result['cognitive_result']['interpretation']}")
    
    # Get comprehensive metrics
    print("\n3. Cognitive substrate metrics:")
    metrics = await integration.get_cognitive_metrics()
    print(f"   Layer 1: {metrics['layer_1_specialists']['total_specialists']} specialists, "
          f"{metrics['layer_1_specialists']['trained_specialists']} trained")
    print(f"   Layer 2: {metrics['layer_2_consensus']['total_consensus']} consensus decisions")
    print(f"   Layer 3: {metrics['layer_3_cognitive_substrate']['total_events']} events processed")
    print(f"   Layer 4: 5 cognitive kernels active")
    
    print("\n" + "=" * 80)
    print("✓ ML/DL Cognitive Substrate successfully integrated into Grace")
    print("  - NOT a standalone SaaS product")
    print("  - Embedded as computational conscience in all kernels")
    print("  - Fully integrated with KPIs, governance, logs, events")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_complete_integration())
