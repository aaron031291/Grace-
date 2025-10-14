"""
Standard Model Interface for Grace ML/DL Specialists

All models must implement this interface for consistent interaction with:
- TriggerMesh workflows
- API Layer
- Governance Engine
- Trust Ledger
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


class ModelStatus(Enum):
    """Model deployment status"""
    SANDBOX = "sandbox"
    CANARY = "canary"
    STAGED = "staged"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


@dataclass
class ModelMetadata:
    """Standard metadata for all models"""
    name: str
    version: str
    framework: str  # sklearn, pytorch, tensorflow, etc.
    owner: str
    artifact_path: str
    training_data_hash: str
    deploy_status: ModelStatus
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Governance
    model_card_url: Optional[str] = None
    dataset_provenance: Optional[str] = None
    signature: Optional[str] = None  # GPG/HMAC signature
    
    # Performance
    expected_latency_p95_ms: Optional[float] = None
    expected_throughput_rps: Optional[float] = None
    memory_mb: Optional[float] = None
    
    # Trust
    calibration_score: Optional[float] = None
    trust_score: float = 0.8


@dataclass
class PredictionResult:
    """Standard prediction output with uncertainty and provenance"""
    prediction: Any
    confidence: float  # 0.0-1.0
    model_id: str
    model_version: str
    
    # Uncertainty & OOD
    uncertainty: Optional[float] = None
    ood_flag: bool = False
    ood_score: Optional[float] = None
    
    # Performance
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Provenance
    input_hash: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Governance
    governance_approved: bool = False
    trust_score: float = 0.0
    
    # Explainability
    explanation: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Model health check result"""
    healthy: bool
    model_id: str
    model_version: str
    
    # Resource usage
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Performance
    recent_latency_p95_ms: Optional[float] = None
    recent_error_rate: Optional[float] = None
    
    # Calibration
    recent_calibration_error: Optional[float] = None
    
    # Deployment
    uptime_seconds: float = 0.0
    last_inference_time: Optional[datetime] = None
    inference_count: int = 0
    
    # Issues
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ModelInterface(ABC):
    """
    Standard interface all Grace models must implement.
    
    Ensures:
    - Consistent predict() API for TriggerMesh/API layer
    - Explainability via explain()
    - Health monitoring via health()
    - Metadata for governance via metadata()
    """
    
    def __init__(self, model_metadata: ModelMetadata):
        self.model_metadata = model_metadata
        self._inference_count = 0
        self._last_inference_time: Optional[datetime] = None
        self._recent_latencies: List[float] = []
        self._recent_errors: int = 0
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Make prediction on input data.
        
        Must return PredictionResult with:
        - prediction: actual prediction value
        - confidence: 0.0-1.0 confidence score
        - ood_flag: True if input is out-of-distribution
        - uncertainty: optional uncertainty estimate
        
        Args:
            input_data: Input features as dict
        
        Returns:
            PredictionResult with prediction and metadata
        """
        pass
    
    @abstractmethod
    async def explain(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain prediction for given input.
        
        Should return:
        - feature_importance: dict of feature -> importance score
        - explanation: human-readable explanation
        - evidence: supporting evidence (attention, activations, etc.)
        
        Args:
            input_data: Input features as dict
        
        Returns:
            Explanation dictionary
        """
        pass
    
    def metadata(self) -> ModelMetadata:
        """
        Get model metadata.
        
        Returns:
            ModelMetadata object with all model information
        """
        return self.model_metadata
    
    @abstractmethod
    async def health(self) -> HealthStatus:
        """
        Check model health status.
        
        Should include:
        - Resource usage (CPU, memory, GPU)
        - Recent performance metrics
        - Calibration status
        - Warnings/errors
        
        Returns:
            HealthStatus object
        """
        pass
    
    # Utility methods
    
    def compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Compute deterministic hash of input for caching/provenance."""
        # Sort keys for determinism
        sorted_data = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def track_inference(self, latency_ms: float, error: bool = False):
        """Track inference metrics for health monitoring."""
        self._inference_count += 1
        self._last_inference_time = datetime.now()
        self._recent_latencies.append(latency_ms)
        
        # Keep only last 100 latencies
        if len(self._recent_latencies) > 100:
            self._recent_latencies.pop(0)
        
        if error:
            self._recent_errors += 1
    
    def get_recent_latency_p95(self) -> Optional[float]:
        """Get p95 latency from recent inferences."""
        if not self._recent_latencies:
            return None
        
        sorted_latencies = sorted(self._recent_latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[p95_idx]
    
    def get_recent_error_rate(self) -> float:
        """Get recent error rate."""
        if self._inference_count == 0:
            return 0.0
        
        # Error rate over last 100 inferences
        recent_count = min(self._inference_count, 100)
        return self._recent_errors / recent_count
    
    @abstractmethod
    async def detect_ood(self, input_data: Dict[str, Any]) -> tuple[bool, float]:
        """
        Detect if input is out-of-distribution.
        
        Implementations should use:
        - Distance-in-embedding (Mahalanobis)
        - Predictive entropy
        - Ensemble disagreement
        - Softmax calibration
        
        Args:
            input_data: Input features
        
        Returns:
            (ood_flag, ood_score) where ood_score is 0.0-1.0
        """
        pass
    
    @abstractmethod
    async def calibrate(self, validation_data: List[tuple[Dict[str, Any], Any]]) -> float:
        """
        Calibrate model confidence on validation data.
        
        Compute calibration error (ECE) and optionally apply:
        - Temperature scaling
        - Platt scaling
        - Isotonic regression
        
        Args:
            validation_data: List of (input, ground_truth) tuples
        
        Returns:
            Calibration error (ECE)
        """
        pass


class ModelWrapper(ModelInterface):
    """
    Base wrapper for existing models to implement ModelInterface.
    
    Provides default implementations for:
    - Input hashing
    - Inference tracking
    - Basic health checks
    """
    
    def __init__(self, model_metadata: ModelMetadata, base_model: Any):
        super().__init__(model_metadata)
        self.base_model = base_model
        self._start_time = datetime.now()
    
    async def health(self) -> HealthStatus:
        """Default health check implementation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        warnings = []
        errors = []
        
        # Check latency SLO
        recent_p95 = self.get_recent_latency_p95()
        if recent_p95 and self.model_metadata.expected_latency_p95_ms:
            if recent_p95 > self.model_metadata.expected_latency_p95_ms * 1.5:
                warnings.append(f"Latency p95 {recent_p95:.1f}ms exceeds SLO by 50%")
        
        # Check error rate
        error_rate = self.get_recent_error_rate()
        if error_rate > 0.05:
            errors.append(f"Error rate {error_rate:.1%} exceeds 5% threshold")
        
        return HealthStatus(
            healthy=(len(errors) == 0),
            model_id=self.model_metadata.name,
            model_version=self.model_metadata.version,
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            recent_latency_p95_ms=recent_p95,
            recent_error_rate=error_rate,
            uptime_seconds=uptime,
            last_inference_time=self._last_inference_time,
            inference_count=self._inference_count,
            warnings=warnings,
            errors=errors
        )
    
    async def detect_ood(self, input_data: Dict[str, Any]) -> tuple[bool, float]:
        """
        Default OOD detection using simple heuristics.
        
        Override for model-specific OOD detection.
        """
        # Default: always in-distribution
        # Subclasses should implement proper OOD detection
        return False, 0.0
    
    async def calibrate(self, validation_data: List[tuple[Dict[str, Any], Any]]) -> float:
        """
        Default calibration - compute ECE.
        
        Override to apply calibration methods.
        """
        # Default: assume well-calibrated
        # Subclasses should compute actual ECE
        return 0.0
