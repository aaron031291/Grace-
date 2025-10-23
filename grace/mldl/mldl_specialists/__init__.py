"""
ML/DL Specialists Module

Individual specialists implementing classic ML and DL models,
integrated with Grace's governance, KPI, audit, and memory systems.
"""

# Base classes
from .base_specialist import (
    BaseMLDLSpecialist,
    SpecialistCapability,
    SpecialistPrediction,
    TrainingMetrics
)

# Supervised Learning Specialists
from .supervised import (
    DecisionTreeSpecialist,
    SVMSpecialist,
    RandomForestSpecialist,
    GradientBoostingSpecialist,
    NaiveBayesSpecialist
)

# Unsupervised Learning Specialists
from .unsupervised import (
    KMeansSpecialist,
    DBSCANSpecialist,
    PCASpecialist,
    AutoencoderSpecialist
)

# Orchestration
from .consensus_engine import MLDLConsensusEngine, ConsensusResult
from .federated_learning import FederatedMetaLearner, PerformanceRecord, MetaLearningUpdate


__all__ = [
    # Base
    "BaseMLDLSpecialist",
    "SpecialistCapability",
    "SpecialistPrediction",
    "TrainingMetrics",
    
    # Supervised
    "DecisionTreeSpecialist",
    "SVMSpecialist",
    "RandomForestSpecialist",
    "GradientBoostingSpecialist",
    "NaiveBayesSpecialist",
    
    # Unsupervised
    "KMeansSpecialist",
    "DBSCANSpecialist",
    "PCASpecialist",
    "AutoencoderSpecialist",
    
    # Orchestration
    "MLDLConsensusEngine",
    "ConsensusResult",
    "FederatedMetaLearner",
    "PerformanceRecord",
    "MetaLearningUpdate",
]

from .base_specialist import BaseMLDLSpecialist, SpecialistCapability
from .supervised import (
    DecisionTreeSpecialist,
    SVMSpecialist,
    RandomForestSpecialist,
    GradientBoostingSpecialist,
    NaiveBayesSpecialist,
)
from .unsupervised import (
    KMeansSpecialist,
    DBSCANSpecialist,
    PCASpecialist,
    AutoencoderSpecialist,
)
from .deep_learning import (
    ANNSpecialist,
    CNNSpecialist,
    RNNSpecialist,
    LSTMSpecialist,
    GANSpecialist,
    TransformerSpecialist,
)
from .reinforcement import (
    QLearningSpecialist,
    DQNSpecialist,
    PPOSpecialist,
)
from .ensemble import (
    StackingEnsembleSpecialist,
    BaggingSpecialist,
    BoostingSpecialist,
)
from .specialized import (
    AnomalyDetectionSpecialist,
    TimeSeriesSpecialist,
    NLPSpecialist,
    GraphNeuralNetworkSpecialist,
    RecommenderSpecialist,
)
from .consensus_engine import MLDLConsensusEngine
from .federated_learning import FederatedMetaLearner

__all__ = [
    "BaseMLDLSpecialist",
    "SpecialistCapability",
    # Supervised
    "DecisionTreeSpecialist",
    "SVMSpecialist",
    "RandomForestSpecialist",
    "GradientBoostingSpecialist",
    "NaiveBayesSpecialist",
    # Unsupervised
    "KMeansSpecialist",
    "DBSCANSpecialist",
    "PCASpecialist",
    "AutoencoderSpecialist",
    # Deep Learning
    "ANNSpecialist",
    "CNNSpecialist",
    "RNNSpecialist",
    "LSTMSpecialist",
    "GANSpecialist",
    "TransformerSpecialist",
    # Reinforcement
    "QLearningSpecialist",
    "DQNSpecialist",
    "PPOSpecialist",
    # Ensemble
    "StackingEnsembleSpecialist",
    "BaggingSpecialist",
    "BoostingSpecialist",
    # Specialized
    "AnomalyDetectionSpecialist",
    "TimeSeriesSpecialist",
    "NLPSpecialist",
    "GraphNeuralNetworkSpecialist",
    "RecommenderSpecialist",
    # Orchestration
    "MLDLConsensusEngine",
    "FederatedMetaLearner",
]
