"""
MLDL Kernel - Enhanced ML/DL system with next-generation specialists and cross-domain validation.

Provides enhanced ML/DL capabilities:
- Enhanced specialists with uncertainty quantification
- Cross-domain validators for governance oversight
- Hallucination detection and mitigation
- Graph neural networks, multimodal AI, federated learning
- Constitutional compliance and trust scoring
"""

# Enhanced MLDL components
try:
    from . import specialists
    from . import enhanced_governance_liaison

    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

# Core MLDL service
try:
    from .mldl_service import MLDLService

    # Model adapters
    from .adapters.base import BaseModelAdapter
    from .adapters.classic import (
        LogisticRegressionAdapter,
        LinearRegressionAdapter,
        SVMAdapter,
        KNNAdapter,
        DecisionTreeAdapter,
        XGBAdapter,
        NaiveBayesAdapter,
    )
    from .adapters.clustering import (
        KMeansAdapter,
        AgglomerativeClusteringAdapter,
        DBSCANAdapter,
        PCAAdapter,
    )

    # Training and evaluation
    from .training.job import TrainingJobRunner
    from .evaluation.metrics import evaluate, calibration, fairness

    # Registry and deployment
    from .registry.registry import ModelRegistry
    from .deployment.manager import DeploymentManager

    # Monitoring and snapshots
    from .monitoring.collector import MonitoringCollector
    from .snapshots.manager import SnapshotManager

    # Legacy quorum system (maintained for compatibility)
    from .quorum import (
        MLDLQuorum,
        SpecialistModel,
        SpecialistType,
        SpecialistOutput,
        QuorumConsensus,
    )

    # Bridges to other kernels
    from .bridges import (
        MLDLMeshBridge,
        MLDLGovernanceBridge,
        MLDLMLTBridge,
        MLDLIntelligenceBridge,
        MLDLMemoryBridge,
        MLDLIngressBridge,
    )

    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False

__all__ = []

# Add enhanced components if available
if ENHANCED_COMPONENTS_AVAILABLE:
    __all__.extend(["specialists", "enhanced_governance_liaison"])

if LEGACY_COMPONENTS_AVAILABLE:
    __all__.extend(
        [
            # Core service
            "MLDLService",
            # Model adapters
            "BaseModelAdapter",
            "LogisticRegressionAdapter",
            "LinearRegressionAdapter",
            "SVMAdapter",
            "KNNAdapter",
            "DecisionTreeAdapter",
            "XGBAdapter",
            "NaiveBayesAdapter",
            "KMeansAdapter",
            "AgglomerativeClusteringAdapter",
            "DBSCANAdapter",
            "PCAAdapter",
            # Training and evaluation
            "TrainingJobRunner",
            "evaluate",
            "calibration",
            "fairness",
            # Registry and deployment
            "ModelRegistry",
            "DeploymentManager",
            # Monitoring and snapshots
            "MonitoringCollector",
            "SnapshotManager",
            # Legacy quorum system
            "MLDLQuorum",
            "SpecialistModel",
            "SpecialistType",
            "SpecialistOutput",
            "QuorumConsensus",
            # Bridges
            "MLDLMeshBridge",
            "MLDLGovernanceBridge",
            "MLDLMLTBridge",
            "MLDLIntelligenceBridge",
            "MLDLMemoryBridge",
            "MLDLIngressBridge",
        ]
    )

"""
Grace MLDL - Machine Learning and Deep Learning specialists
"""

from .quorum_aggregator import (
    MLDLQuorumAggregator,
    SpecialistOutput,
    QuorumResult,
    ConsensusMethod
)
from .uncertainty import UncertaintyEstimator

__all__ = [
    'MLDLQuorumAggregator',
    'SpecialistOutput',
    'QuorumResult',
    'ConsensusMethod',
    'UncertaintyEstimator'
]
