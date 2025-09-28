"""
MLDL Kernel - Complete model lifecycle management system for Grace.

Provides unified ML/DL/RL model lifecycle: define → train → evaluate → register → deploy → monitor → rollback.
Includes adapters for classic ML, deep learning, reinforcement learning, clustering, and dimensionality reduction.
Features HPO, calibration, fairness evaluation, drift monitoring, model registry, and deployment management.
"""

# Core MLDL service
from .mldl_service import MLDLService

# Model adapters
from .adapters.base import BaseModelAdapter
from .adapters.classic import (
    LogisticRegressionAdapter, LinearRegressionAdapter, SVMAdapter,
    KNNAdapter, DecisionTreeAdapter, XGBAdapter, NaiveBayesAdapter
)
from .adapters.clustering import KMeansAdapter, AgglomerativeClusteringAdapter, DBSCANAdapter, PCAAdapter

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
from .quorum import MLDLQuorum, SpecialistModel, SpecialistType, SpecialistOutput, QuorumConsensus

# Bridges to other kernels
from .bridges import (
    MLDLMeshBridge, MLDLGovernanceBridge, MLDLMLTBridge,
    MLDLIntelligenceBridge, MLDLMemoryBridge, MLDLIngressBridge
)

__all__ = [
    # Core service
    'MLDLService',
    
    # Model adapters
    'BaseModelAdapter',
    'LogisticRegressionAdapter', 'LinearRegressionAdapter', 'SVMAdapter',
    'KNNAdapter', 'DecisionTreeAdapter', 'XGBAdapter', 'NaiveBayesAdapter',
    'KMeansAdapter', 'AgglomerativeClusteringAdapter', 'DBSCANAdapter', 'PCAAdapter',
    
    # Training and evaluation
    'TrainingJobRunner',
    'evaluate', 'calibration', 'fairness',
    
    # Registry and deployment
    'ModelRegistry', 'DeploymentManager',
    
    # Monitoring and snapshots
    'MonitoringCollector', 'SnapshotManager',
    
    # Legacy quorum system
    'MLDLQuorum', 'SpecialistModel', 'SpecialistType', 'SpecialistOutput', 'QuorumConsensus',
    
    # Bridges
    'MLDLMeshBridge', 'MLDLGovernanceBridge', 'MLDLMLTBridge',
    'MLDLIntelligenceBridge', 'MLDLMemoryBridge', 'MLDLIngressBridge'
]