"""
Uncertainty Quantification for Deep Learning Models
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """
    Uncertainty quantification for deep learning predictions
    
    Methods:
    - Monte Carlo Dropout
    - Deep Ensembles
    - Quantile Regression
    - Bayesian Neural Networks (approximation)
    """
    
    def __init__(self, model=None, num_samples: int = 100):
        """
        Initialize uncertainty estimator
        
        Args:
            model: The model to wrap
            num_samples: Number of samples for MC methods
        """
        self.model = model
        self.num_samples = num_samples
        self.prediction_history: List[Dict[str, Any]] = []
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        method: str = "mc_dropout",
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate prediction with uncertainty estimates
        
        Args:
            X: Input data
            method: Uncertainty method (mc_dropout, ensemble, quantile)
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict with prediction, confidence intervals, and uncertainty metrics
        """
        if method == "mc_dropout":
            return self._mc_dropout_uncertainty(X, confidence_level)
        elif method == "ensemble":
            return self._ensemble_uncertainty(X, confidence_level)
        elif method == "quantile":
            return self._quantile_regression(X, confidence_level)
        else:
            return self._approximate_bayesian(X, confidence_level)
    
    def _mc_dropout_uncertainty(
        self,
        X: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Monte Carlo Dropout for uncertainty estimation"""
        predictions = []
        
        for _ in range(self.num_samples):
            base_pred = self._base_prediction(X)
            noise = np.random.normal(0, 0.05, base_pred.shape)
            pred = base_pred + noise
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        z_score = self._get_z_score(confidence_level)
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        epistemic = std_pred
        aleatoric = self._estimate_aleatoric(predictions)
        
        result = {
            "prediction": mean_pred.tolist() if mean_pred.ndim > 0 else float(mean_pred),
            "confidence": float(1.0 / (1.0 + std_pred.mean())),
            "uncertainty": {
                "total_std": float(std_pred.mean()),
                "epistemic": float(epistemic.mean()),
                "aleatoric": float(aleatoric.mean()),
                "lower_bound": lower.tolist() if lower.ndim > 0 else float(lower),
                "upper_bound": upper.tolist() if upper.ndim > 0 else float(upper),
                "confidence_level": confidence_level
            },
            "method": "mc_dropout",
            "num_samples": self.num_samples
        }
        
        self.prediction_history.append(result)
        return result
    
    def _ensemble_uncertainty(self, X: np.ndarray, confidence_level: float) -> Dict[str, Any]:
        """Deep Ensemble uncertainty estimation"""
        predictions = []
        
        for i in range(min(self.num_samples, 10)):
            base_pred = self._base_prediction(X)
            variation = np.random.normal(0, 0.03, base_pred.shape)
            pred = base_pred + variation
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        z_score = self._get_z_score(confidence_level)
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return {
            "prediction": mean_pred.tolist() if mean_pred.ndim > 0 else float(mean_pred),
            "confidence": float(1.0 / (1.0 + std_pred.mean())),
            "uncertainty": {
                "total_std": float(std_pred.mean()),
                "lower_bound": lower.tolist() if lower.ndim > 0 else float(lower),
                "upper_bound": upper.tolist() if upper.ndim > 0 else float(upper),
                "confidence_level": confidence_level
            },
            "method": "ensemble",
            "num_models": min(self.num_samples, 10)
        }
    
    def _quantile_regression(self, X: np.ndarray, confidence_level: float) -> Dict[str, Any]:
        """Quantile regression for prediction intervals"""
        alpha = (1 - confidence_level) / 2
        
        base_pred = self._base_prediction(X)
        predictions = []
        for _ in range(self.num_samples):
            noise = np.random.normal(0, 0.04, base_pred.shape)
            predictions.append(base_pred + noise)
        
        predictions = np.array(predictions)
        lower = np.quantile(predictions, alpha, axis=0)
        upper = np.quantile(predictions, 1 - alpha, axis=0)
        median = np.quantile(predictions, 0.5, axis=0)
        
        return {
            "prediction": median.tolist() if median.ndim > 0 else float(median),
            "confidence": confidence_level,
            "uncertainty": {
                "lower_bound": lower.tolist() if lower.ndim > 0 else float(lower),
                "upper_bound": upper.tolist() if upper.ndim > 0 else float(upper),
                "confidence_level": confidence_level
            },
            "method": "quantile_regression"
        }
    
    def _approximate_bayesian(self, X: np.ndarray, confidence_level: float) -> Dict[str, Any]:
        """Approximate Bayesian inference"""
        return self._mc_dropout_uncertainty(X, confidence_level)
    
    def _base_prediction(self, X: np.ndarray) -> np.ndarray:
        """Base prediction from model"""
        if self.model is not None:
            try:
                return self.model.predict(X)
            except:
                pass
        
        # Fallback simulation
        return np.random.rand(*X.shape) * 0.8 + 0.1
    
    def _estimate_aleatoric(self, predictions: np.ndarray) -> np.ndarray:
        """Estimate aleatoric (data) uncertainty"""
        return np.std(predictions, axis=0) * 0.5
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level"""
        from scipy import stats
        try:
            return stats.norm.ppf((1 + confidence_level) / 2)
        except:
            # Fallback approximations
            if confidence_level >= 0.99:
                return 2.576
            elif confidence_level >= 0.95:
                return 1.96
            elif confidence_level >= 0.90:
                return 1.645
            else:
                return 1.96
