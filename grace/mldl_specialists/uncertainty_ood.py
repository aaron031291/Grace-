"""
Uncertainty Quantification & Out-of-Distribution Detection

Provides production-grade uncertainty awareness:
- Confidence calibration (temperature scaling, Platt scaling)
- OOD detection (Mahalanobis distance, entropy, embedding distance)
- Uncertainty-aware decision routing
- Human-in-the-loop escalation triggers
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty in predictions"""
    ALEATORIC = "aleatoric"  # Irreducible (data noise)
    EPISTEMIC = "epistemic"  # Reducible (model uncertainty)
    TOTAL = "total"


class OODMethod(Enum):
    """Out-of-distribution detection methods"""
    MAHALANOBIS = "mahalanobis"
    SOFTMAX_ENTROPY = "softmax_entropy"
    PREDICTIVE_ENTROPY = "predictive_entropy"
    EMBEDDING_DISTANCE = "embedding_distance"
    ENERGY_SCORE = "energy_score"
    ENSEMBLE_VARIANCE = "ensemble_variance"


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification result"""
    confidence: float  # Original model confidence
    calibrated_confidence: float  # After calibration
    uncertainty_score: float  # 0.0 (certain) to 1.0 (uncertain)
    uncertainty_type: UncertaintyType
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OODDetectionResult:
    """Out-of-distribution detection result"""
    is_ood: bool
    ood_score: float  # Higher = more out-of-distribution
    threshold: float
    method: OODMethod
    requires_review: bool  # Escalate to human?
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TemperatureScaling:
    """
    Temperature scaling for confidence calibration.
    
    Learns optimal temperature T from validation set to minimize
    calibration error (ECE).
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._calibrated = False
    
    def calibrate(
        self,
        logits: np.ndarray,
        true_labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Learn optimal temperature from validation data.
        
        Args:
            logits: Raw model outputs before softmax (N, C)
            true_labels: Ground truth labels (N,)
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        from scipy.optimize import minimize
        
        def nll_loss(temp):
            """Negative log likelihood with temperature"""
            scaled_logits = logits / temp
            probs = self._softmax(scaled_logits)
            # Cross-entropy loss
            log_probs = np.log(probs[np.arange(len(true_labels)), true_labels] + 1e-10)
            return -np.mean(log_probs)
        
        # Optimize temperature
        result = minimize(
            nll_loss,
            x0=np.array([1.0]),
            method='BFGS',
            options={'maxiter': max_iter}
        )
        
        self.temperature = float(result.x[0])
        self._calibrated = True
        
        logger.info(f"Temperature scaling calibrated: T={self.temperature:.3f}")
    
    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits"""
        return self._softmax(logits / self.temperature)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class MahalanobisOOD:
    """
    Mahalanobis distance-based OOD detection.
    
    Computes distance from test sample to training distribution
    in feature space.
    """
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit Mahalanobis detector on training embeddings.
        
        Args:
            embeddings: Training set embeddings (N, D)
        """
        self.mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings, rowvar=False)
        
        # Add regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        
        try:
            self.cov_inv = np.linalg.inv(cov)
            self._fitted = True
        except np.linalg.LinAlgError:
            logger.error("Covariance matrix not invertible")
            self._fitted = False
    
    def detect(
        self,
        embedding: np.ndarray,
        return_distance: bool = True
    ) -> OODDetectionResult:
        """
        Detect if sample is out-of-distribution.
        
        Args:
            embedding: Test sample embedding (D,)
            return_distance: Whether to return raw distance
        
        Returns:
            OODDetectionResult
        """
        if not self._fitted:
            return OODDetectionResult(
                is_ood=False,
                ood_score=0.0,
                threshold=self.threshold,
                method=OODMethod.MAHALANOBIS,
                requires_review=False,
                metadata={'error': 'Detector not fitted'}
            )
        
        # Compute Mahalanobis distance
        diff = embedding - self.mean
        distance = np.sqrt(mahalanobis(diff, self.cov_inv, VI=None))
        
        is_ood = distance > self.threshold
        requires_review = distance > (self.threshold * 1.5)  # High severity
        
        return OODDetectionResult(
            is_ood=is_ood,
            ood_score=float(distance),
            threshold=self.threshold,
            method=OODMethod.MAHALANOBIS,
            requires_review=requires_review,
            metadata={'raw_distance': float(distance)}
        )


class EntropyOOD:
    """
    Entropy-based OOD detection for classification.
    
    High entropy = uncertain prediction = potential OOD
    """
    
    def __init__(
        self,
        threshold: Optional[float] = None,
        num_classes: Optional[int] = None
    ):
        self.threshold = threshold
        self.num_classes = num_classes
        
        # If threshold not specified, use 80% of max entropy
        if threshold is None and num_classes is not None:
            max_entropy = np.log(num_classes)
            self.threshold = 0.8 * max_entropy
    
    def detect(
        self,
        probabilities: np.ndarray,
        use_predictive: bool = False
    ) -> OODDetectionResult:
        """
        Detect OOD based on prediction entropy.
        
        Args:
            probabilities: Class probabilities (C,) or ensemble predictions (M, C)
            use_predictive: Use predictive entropy (for ensembles)
        
        Returns:
            OODDetectionResult
        """
        if use_predictive and len(probabilities.shape) == 2:
            # Predictive entropy for ensemble
            mean_probs = np.mean(probabilities, axis=0)
            ent = entropy(mean_probs)
            method = OODMethod.PREDICTIVE_ENTROPY
        else:
            # Standard softmax entropy
            ent = entropy(probabilities)
            method = OODMethod.SOFTMAX_ENTROPY
        
        if self.threshold is None:
            # Auto-threshold at 80% of max entropy
            max_ent = np.log(len(probabilities) if len(probabilities.shape) == 1 else probabilities.shape[1])
            self.threshold = 0.8 * max_ent
        
        is_ood = ent > self.threshold
        requires_review = ent > (self.threshold * 1.2)
        
        return OODDetectionResult(
            is_ood=is_ood,
            ood_score=float(ent),
            threshold=self.threshold,
            method=method,
            requires_review=requires_review,
            metadata={'entropy': float(ent)}
        )


class EnsembleVarianceOOD:
    """
    Ensemble variance for epistemic uncertainty estimation.
    
    High variance across ensemble predictions = high uncertainty
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def detect(
        self,
        ensemble_predictions: np.ndarray
    ) -> OODDetectionResult:
        """
        Detect OOD based on ensemble prediction variance.
        
        Args:
            ensemble_predictions: Predictions from ensemble members (M, ...) 
        
        Returns:
            OODDetectionResult
        """
        variance = np.var(ensemble_predictions, axis=0)
        mean_variance = float(np.mean(variance))
        
        is_ood = mean_variance > self.threshold
        requires_review = mean_variance > (self.threshold * 2.0)
        
        return OODDetectionResult(
            is_ood=is_ood,
            ood_score=mean_variance,
            threshold=self.threshold,
            method=OODMethod.ENSEMBLE_VARIANCE,
            requires_review=requires_review,
            metadata={
                'variance': mean_variance,
                'num_models': len(ensemble_predictions)
            }
        )


class UncertaintyAwareRouter:
    """
    Routes predictions based on uncertainty and OOD detection.
    
    Decision flow:
    - High confidence + in-distribution → Auto-approve
    - Medium confidence + in-distribution → Auto-approve with monitoring
    - Low confidence OR OOD → Escalate to human review
    - Very low confidence + OOD → Reject or fallback
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.60,
        ood_detector: Optional[MahalanobisOOD] = None
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.ood_detector = ood_detector
    
    def route_decision(
        self,
        confidence: float,
        ood_result: Optional[OODDetectionResult] = None,
        embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Route prediction based on confidence and OOD status.
        
        Args:
            confidence: Model confidence
            ood_result: OOD detection result (optional)
            embedding: Sample embedding for OOD detection (optional)
        
        Returns:
            Routing decision with action and reasoning
        """
        # Run OOD detection if not provided
        if ood_result is None and embedding is not None and self.ood_detector:
            ood_result = self.ood_detector.detect(embedding)
        
        # Decision logic
        is_ood = ood_result.is_ood if ood_result else False
        requires_review = ood_result.requires_review if ood_result else False
        
        if is_ood and requires_review:
            return {
                'action': 'reject',
                'reason': 'High OOD score - sample very different from training',
                'requires_human_review': True,
                'confidence': confidence,
                'ood_score': ood_result.ood_score if ood_result else None
            }
        
        if is_ood or confidence < self.low_confidence_threshold:
            return {
                'action': 'escalate_to_human',
                'reason': 'Low confidence or OOD detected',
                'requires_human_review': True,
                'confidence': confidence,
                'ood_score': ood_result.ood_score if ood_result else None
            }
        
        if confidence < self.high_confidence_threshold:
            return {
                'action': 'approve_with_monitoring',
                'reason': 'Medium confidence - monitor closely',
                'requires_human_review': False,
                'confidence': confidence,
                'monitor': True
            }
        
        # High confidence + in-distribution
        return {
            'action': 'approve',
            'reason': 'High confidence prediction',
            'requires_human_review': False,
            'confidence': confidence,
            'ood_score': ood_result.ood_score if ood_result else None
        }


def compute_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    
    Args:
        confidences: Predicted confidences (N,)
        predictions: Predicted labels (N,)
        true_labels: True labels (N,)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = np.mean(predictions[in_bin] == true_labels[in_bin])
            # Average confidence in this bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # Add weighted calibration error
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def uncertainty_sampling_priority(
    confidence: float,
    ood_score: float,
    diversity_score: float = 0.5
) -> float:
    """
    Calculate priority score for active learning sample selection.
    
    Higher score = higher priority for human labeling.
    
    Args:
        confidence: Model confidence (lower = more uncertain)
        ood_score: OOD detection score (higher = more unusual)
        diversity_score: Sample diversity (higher = more diverse)
    
    Returns:
        Priority score (0-1, higher = more important)
    """
    # Uncertainty component (inverse confidence)
    uncertainty_component = 1.0 - confidence
    
    # Normalize OOD score (assume typical range 0-10)
    ood_component = min(ood_score / 10.0, 1.0)
    
    # Weighted combination
    priority = (
        0.5 * uncertainty_component +
        0.3 * ood_component +
        0.2 * diversity_score
    )
    
    return min(priority, 1.0)
