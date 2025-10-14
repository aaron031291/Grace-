"""
Model Registry & Lifecycle Management

Centralized registry for all ML/DL models with:
- Version control and provenance tracking
- Deployment status management (sandbox → canary → production)
- Performance metrics tracking
- Automated rollback triggers
- Model card generation
- PyTorch/Deep Learning model support with GPU metrics
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import yaml
import json
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Check for PyTorch availability
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some features limited")


class DeploymentStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    SANDBOX = "sandbox"
    CANARY = "canary"
    STAGED = "staged"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLBACK = "rollback"


@dataclass
class ModelRegistryEntry:
    """Entry in the model registry"""
    # Identity
    model_id: str
    name: str
    version: str
    
    # Artifacts
    artifact_path: str
    framework: str  # sklearn, pytorch, tensorflow, etc.
    model_type: str  # classification, regression, clustering, etc.
    
    # Ownership
    owner: str
    team: str
    
    # Training provenance
    training_data_hash: str
    training_dataset_size: int
    training_timestamp: datetime
    training_duration_minutes: Optional[float] = None
    git_commit_hash: Optional[str] = None
    
    # Evaluation metrics
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_error: Optional[float] = None
    
    # Deployment
    deploy_status: DeploymentStage = DeploymentStage.DEVELOPMENT
    deployed_at: Optional[datetime] = None
    canary_percentage: float = 0.0  # 0-100
    
    # Performance
    expected_latency_p50_ms: Optional[float] = None
    expected_latency_p95_ms: Optional[float] = None
    expected_throughput_rps: Optional[float] = None
    
    # Governance
    model_card_path: Optional[str] = None
    signature_hash: Optional[str] = None
    constitutional_compliance: bool = False
    bias_check_passed: bool = False
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Runtime metrics (populated during serving)
    runtime_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time"""
    model_id: str
    version: str
    timestamp: datetime
    
    # Latency metrics
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    # Throughput
    requests_per_second: float
    
    # Quality metrics
    accuracy: Optional[float] = None
    calibration_error: Optional[float] = None
    
    # Operational
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Distribution drift
    input_drift_score: Optional[float] = None  # KL divergence
    output_drift_score: Optional[float] = None
    
    # OOD detection
    ood_rate: float = 0.0  # Percentage of OOD samples
    
    # Sample size
    num_requests: int = 0
    
    # GPU metrics (for deep learning models)
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    device: Optional[str] = None  # cuda, mps, cpu


class ModelRegistry:
    """
    Centralized registry for ML/DL models.
    
    Stores in YAML file: ml/registry/models.yaml
    """
    
    def __init__(self, registry_path: str = "ml/registry/models.yaml"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelRegistryEntry] = {}
        self.performance_history: Dict[str, List[ModelPerformanceSnapshot]] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from YAML file"""
        if not self.registry_path.exists():
            logger.info(f"Registry file not found, creating new: {self.registry_path}")
            self._save_registry()
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            models_data = data.get('models', [])
            for model_dict in models_data:
                # Convert string timestamps back to datetime
                if 'training_timestamp' in model_dict:
                    model_dict['training_timestamp'] = datetime.fromisoformat(model_dict['training_timestamp'])
                if 'deployed_at' in model_dict and model_dict['deployed_at']:
                    model_dict['deployed_at'] = datetime.fromisoformat(model_dict['deployed_at'])
                if 'created_at' in model_dict:
                    model_dict['created_at'] = datetime.fromisoformat(model_dict['created_at'])
                if 'updated_at' in model_dict:
                    model_dict['updated_at'] = datetime.fromisoformat(model_dict['updated_at'])
                
                # Convert deploy_status back to enum
                if 'deploy_status' in model_dict:
                    model_dict['deploy_status'] = DeploymentStage(model_dict['deploy_status'])
                
                entry = ModelRegistryEntry(**model_dict)
                self.models[entry.model_id] = entry
            
            logger.info(f"Loaded {len(self.models)} models from registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.models = {}
    
    def _save_registry(self):
        """Save registry to YAML file"""
        models_list = []
        for entry in self.models.values():
            entry_dict = asdict(entry)
            # Convert datetime to ISO format string
            entry_dict['training_timestamp'] = entry.training_timestamp.isoformat()
            if entry.deployed_at:
                entry_dict['deployed_at'] = entry.deployed_at.isoformat()
            entry_dict['created_at'] = entry.created_at.isoformat()
            entry_dict['updated_at'] = entry.updated_at.isoformat()
            # Convert enum to string
            entry_dict['deploy_status'] = entry.deploy_status.value
            
            models_list.append(entry_dict)
        
        data = {'models': models_list}
        
        with open(self.registry_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved {len(self.models)} models to registry")
    
    def register_model(self, entry: ModelRegistryEntry) -> bool:
        """
        Register a new model or update existing.
        
        Args:
            entry: Model registry entry
        
        Returns:
            True if successful
        """
        entry.updated_at = datetime.now()
        self.models[entry.model_id] = entry
        self._save_registry()
        
        logger.info(f"Registered model: {entry.model_id} v{entry.version}")
        return True
    
    async def register_pytorch_model(
        self,
        model_id: str,
        model: Any,
        metrics: Dict[str, float],
        metadata: Dict[str, Any],
        checkpoint_path: Optional[str] = None
    ) -> bool:
        """
        Register a PyTorch deep learning model.
        
        Args:
            model_id: Unique model identifier
            model: PyTorch model instance or specialist
            metrics: Training/evaluation metrics
            metadata: Additional metadata (device, train_samples, etc.)
            checkpoint_path: Path to saved checkpoint
            
        Returns:
            True if successful
        """
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available - registering as generic model")
        
        # Extract model info
        model_type = type(model).__name__
        framework = "pytorch" if PYTORCH_AVAILABLE else "unknown"
        
        # GPU metrics if available
        device = metadata.get('device', 'cpu')
        gpu_memory_mb = None
        gpu_util = None
        
        if PYTORCH_AVAILABLE and device in ['cuda', 'mps']:
            try:
                from grace.mldl_specialists.deep_learning import DeviceManager
                mem_info = DeviceManager.get_memory_usage()
                gpu_memory_mb = mem_info.get('allocated_mb', 0.0)
                gpu_util = mem_info.get('utilization_percent')
            except Exception as e:
                logger.warning(f"Could not get GPU metrics: {e}")
        
        # Create registry entry
        entry = ModelRegistryEntry(
            model_id=model_id,
            name=model_id,
            version=metadata.get('version', '1.0'),
            artifact_path=checkpoint_path or f"models/{model_id}.pt",
            framework=framework,
            model_type=model_type,
            owner=metadata.get('owner', 'system'),
            team=metadata.get('team', 'ml'),
            training_data_hash=metadata.get('training_data_hash', ''),
            training_dataset_size=metadata.get('train_samples', 0),
            training_timestamp=datetime.fromisoformat(metadata['last_trained']) if 'last_trained' in metadata else datetime.now(),
            training_duration_minutes=metadata.get('training_duration_minutes'),
            evaluation_metrics=metrics,
            deploy_status=DeploymentStage.DEVELOPMENT,
            tags=metadata.get('tags', ['pytorch', 'deep_learning']),
            description=metadata.get('description', f'{model_type} deep learning model'),
            runtime_metrics={
                'device': device,
                'gpu_memory_mb': gpu_memory_mb,
                'gpu_utilization_percent': gpu_util
            }
        )
        
        return self.register_model(entry)
    
    async def update_model(
        self,
        model_id: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing model's metrics and metadata.
        
        Args:
            model_id: Model identifier
            metrics: New metrics to add/update
            metadata: New metadata to add/update
            
        Returns:
            True if successful
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return False
        
        entry = self.models[model_id]
        
        if metrics:
            entry.evaluation_metrics.update(metrics)
        
        if metadata:
            entry.runtime_metrics.update(metadata)
            
            # Update specific fields if present
            if 'last_trained' in metadata:
                entry.training_timestamp = datetime.fromisoformat(metadata['last_trained'])
            if 'device' in metadata:
                entry.runtime_metrics['device'] = metadata['device']
        
        entry.updated_at = datetime.now()
        self._save_registry()
        
        logger.info(f"Updated model: {model_id}")
        return True
    
    def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Get model entry by ID"""
        return self.models.get(model_id)
    
    def list_models(
        self,
        deploy_status: Optional[DeploymentStage] = None,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelRegistryEntry]:
        """
        List models with optional filtering.
        
        Args:
            deploy_status: Filter by deployment status
            framework: Filter by framework
            tags: Filter by tags (any match)
        
        Returns:
            List of matching models
        """
        models = list(self.models.values())
        
        if deploy_status:
            models = [m for m in models if m.deploy_status == deploy_status]
        
        if framework:
            models = [m for m in models if m.framework == framework]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        return models
    
    def update_deployment_status(
        self,
        model_id: str,
        new_status: DeploymentStage,
        canary_percentage: float = 0.0
    ) -> bool:
        """
        Update model deployment status.
        
        Args:
            model_id: Model identifier
            new_status: New deployment stage
            canary_percentage: Percentage for canary deployment
        
        Returns:
            True if successful
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return False
        
        entry = self.models[model_id]
        old_status = entry.deploy_status
        
        entry.deploy_status = new_status
        entry.canary_percentage = canary_percentage
        entry.updated_at = datetime.now()
        
        if new_status in [DeploymentStage.CANARY, DeploymentStage.PRODUCTION]:
            entry.deployed_at = datetime.now()
        
        self._save_registry()
        
        logger.info(
            f"Updated {model_id} deployment: {old_status.value} → {new_status.value}"
        )
        return True
    
    def record_performance_snapshot(
        self,
        snapshot: ModelPerformanceSnapshot
    ):
        """Record performance metrics snapshot"""
        model_id = snapshot.model_id
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append(snapshot)
        
        # Update runtime metrics in registry entry
        if model_id in self.models:
            entry = self.models[model_id]
            entry.runtime_metrics = {
                'last_snapshot': snapshot.timestamp.isoformat(),
                'latency_p95_ms': snapshot.latency_p95_ms,
                'error_rate': snapshot.error_rate,
                'ood_rate': snapshot.ood_rate
            }
            self._save_registry()
    
    def check_rollback_triggers(
        self,
        model_id: str,
        window_minutes: int = 10
    ) -> tuple[bool, List[str]]:
        """
        Check if model should be rolled back based on performance.
        
        Args:
            model_id: Model to check
            window_minutes: Time window to analyze
        
        Returns:
            (should_rollback, reasons)
        """
        if model_id not in self.models:
            return False, []
        
        if model_id not in self.performance_history:
            return False, []
        
        entry = self.models[model_id]
        recent_snapshots = self._get_recent_snapshots(model_id, window_minutes)
        
        if not recent_snapshots:
            return False, []
        
        reasons = []
        should_rollback = False
        
        # Check error rate
        avg_error_rate = sum(s.error_rate for s in recent_snapshots) / len(recent_snapshots)
        if avg_error_rate > 0.05:  # 5% error rate
            reasons.append(f"High error rate: {avg_error_rate:.2%}")
            should_rollback = True
        
        # Check latency degradation
        if entry.expected_latency_p95_ms:
            avg_latency_p95 = sum(s.latency_p95_ms for s in recent_snapshots) / len(recent_snapshots)
            if avg_latency_p95 > entry.expected_latency_p95_ms * 1.5:
                reasons.append(
                    f"Latency degradation: {avg_latency_p95:.1f}ms "
                    f"(expected {entry.expected_latency_p95_ms:.1f}ms)"
                )
                should_rollback = True
        
        # Check OOD rate
        avg_ood_rate = sum(s.ood_rate for s in recent_snapshots) / len(recent_snapshots)
        if avg_ood_rate > 0.2:  # 20% OOD samples
            reasons.append(f"High OOD rate: {avg_ood_rate:.2%}")
            should_rollback = True
        
        # Check input drift
        recent_drift = [s.input_drift_score for s in recent_snapshots if s.input_drift_score]
        if recent_drift:
            avg_drift = sum(recent_drift) / len(recent_drift)
            if avg_drift > 0.3:  # Significant drift
                reasons.append(f"Input distribution drift: {avg_drift:.3f}")
                should_rollback = True
        
        return should_rollback, reasons
    
    def _get_recent_snapshots(
        self,
        model_id: str,
        window_minutes: int
    ) -> List[ModelPerformanceSnapshot]:
        """Get performance snapshots within time window"""
        if model_id not in self.performance_history:
            return []
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        return [
            s for s in self.performance_history[model_id]
            if s.timestamp >= cutoff
        ]
    
    def generate_model_card(
        self,
        model_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate model card documentation.
        
        Args:
            model_id: Model to document
            output_path: Where to save (default: docs/model_cards/{model_id}.md)
        
        Returns:
            Path to generated model card
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        entry = self.models[model_id]
        
        if output_path is None:
            output_path = f"docs/model_cards/{model_id}.md"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown content
        content = self._generate_model_card_content(entry)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        # Update registry with model card path
        entry.model_card_path = str(output_file)
        self._save_registry()
        
        logger.info(f"Generated model card: {output_file}")
        return str(output_file)
    
    def _generate_model_card_content(self, entry: ModelRegistryEntry) -> str:
        """Generate model card markdown content"""
        return f"""# Model Card: {entry.name}

## Model Details

- **Model ID**: {entry.model_id}
- **Version**: {entry.version}
- **Type**: {entry.model_type}
- **Framework**: {entry.framework}
- **Owner**: {entry.owner} ({entry.team})
- **Status**: {entry.deploy_status.value}

## Description

{entry.description or "No description provided"}

## Training Data

- **Dataset Hash**: `{entry.training_data_hash}`
- **Dataset Size**: {entry.training_dataset_size:,} samples
- **Trained**: {entry.training_timestamp.isoformat()}
- **Training Duration**: {entry.training_duration_minutes:.1f} minutes
- **Git Commit**: {entry.git_commit_hash or "N/A"}

## Evaluation Metrics

{self._format_metrics(entry.evaluation_metrics)}

- **Calibration Error (ECE)**: {entry.calibration_error:.4f if entry.calibration_error else "N/A"}

## Performance Characteristics

- **Expected Latency (p50)**: {entry.expected_latency_p50_ms:.1f} ms
- **Expected Latency (p95)**: {entry.expected_latency_p95_ms:.1f} ms
- **Expected Throughput**: {entry.expected_throughput_rps:.0f} req/s

## Governance

- **Constitutional Compliance**: {"✅ Passed" if entry.constitutional_compliance else "❌ Not verified"}
- **Bias Check**: {"✅ Passed" if entry.bias_check_passed else "❌ Not verified"}
- **Signature Hash**: `{entry.signature_hash or "N/A"}`

## Deployment

- **Deployed At**: {entry.deployed_at.isoformat() if entry.deployed_at else "Not deployed"}
- **Canary Percentage**: {entry.canary_percentage}%

## Tags

{', '.join(f"`{tag}`" for tag in entry.tags)}

## Artifacts

- **Model Path**: `{entry.artifact_path}`
- **Model Card**: `{entry.model_card_path or "This file"}`

---

*Generated: {datetime.now().isoformat()}*
"""
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics dictionary as markdown list"""
        if not metrics:
            return "No metrics recorded"
        
        lines = []
        for key, value in metrics.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}")
        return '\n'.join(lines)


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
