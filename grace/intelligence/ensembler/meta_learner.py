"""
Meta Ensembler - Stacking/blending/voting with uncertainty quantification.

Handles:
1. Ensemble methods: stacking, blending, voting
2. Uncertainty estimation: calibration, variance, entropy
3. Meta-model training and prediction
4. Dynamic ensemble weight adjustment
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
import json
import math

logger = logging.getLogger(__name__)


class MetaEnsembler:
    """Meta-learning ensemble system with uncertainty quantification."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Ensemble models (mock implementations)
        self.meta_models = {
            "linear": {"weights": None, "intercept": None},
            "logistic": {"weights": None, "intercept": None},
            "neural": {"layers": [], "trained": False}
        }
        
        # Historical performance tracking
        self.performance_history: List[Dict] = []
        self.calibration_data: List[Tuple] = []
        
        logger.info("Meta Ensembler initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default ensembler configuration."""
        return {
            "type": "stack",
            "meta_model": "lr-meta@1.0.2", 
            "uncertainty_methods": ["calibration", "variance", "entropy"],
            "min_samples_for_training": 100,
            "calibration_method": "platt",
            "ensemble_weights": "adaptive"
        }
    
    def fit(self, reports: List[Dict], y: Optional[List] = None):
        """
        Fit meta-ensembler on specialist reports.
        
        Args:
            reports: List of SpecialistReport objects
            y: Optional ground truth labels for supervised training
        """
        try:
            if not reports:
                logger.warning("No reports provided for fitting")
                return
            
            # Extract predictions from reports
            predictions_matrix = self._extract_predictions(reports)
            
            if predictions_matrix is None or predictions_matrix.size == 0:
                logger.warning("No valid predictions found in reports")
                return
            
            # Fit meta-models based on ensemble type
            ensemble_type = self.config.get("type", "stack")
            
            if ensemble_type == "stack" and y is not None:
                self._fit_stacking_model(predictions_matrix, y)
            elif ensemble_type == "blend":
                self._fit_blending_weights(predictions_matrix, y)
            else:  # voting
                self._fit_voting_weights(predictions_matrix, reports)
            
            # Update performance tracking
            self._update_performance_history(reports, predictions_matrix)
            
            logger.info(f"Meta-ensembler fitted with {len(reports)} specialist reports")
            
        except Exception as e:
            logger.error(f"Error fitting meta-ensembler: {e}")
    
    def predict(self, model_outputs: List[Dict]) -> Dict[str, Any]:
        """
        Generate ensemble prediction from model outputs.
        
        Args:
            model_outputs: List of model prediction dictionaries
            
        Returns:
            Dictionary with prediction, probabilities, and uncertainties
        """
        try:
            if not model_outputs:
                return {"error": "No model outputs provided"}
            
            # Extract predictions
            predictions = []
            probabilities = []
            confidences = []
            
            for output in model_outputs:
                if "prediction" in output:
                    predictions.append(output["prediction"])
                if "probabilities" in output:
                    probabilities.append(output["probabilities"])
                if "confidence" in output:
                    confidences.append(output["confidence"])
            
            if not predictions:
                return {"error": "No predictions found in outputs"}
            
            # Generate ensemble prediction
            ensemble_type = self.config.get("type", "stack")
            
            if ensemble_type == "stack":
                result = self._predict_stacking(predictions, probabilities)
            elif ensemble_type == "blend":
                result = self._predict_blending(predictions, probabilities)
            else:  # voting
                result = self._predict_voting(predictions, probabilities)
            
            # Add uncertainty quantification
            uncertainties = self.uncertainty(model_outputs)
            result["uncertainties"] = uncertainties
            
            # Add metadata
            result["meta_info"] = {
                "ensemble_type": ensemble_type,
                "num_models": len(model_outputs),
                "timestamp": iso_format()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {"error": str(e)}
    
    def uncertainty(self, model_outputs: List[Dict]) -> Dict[str, float]:
        """
        Calculate uncertainty metrics from model outputs.
        
        Args:
            model_outputs: List of model prediction dictionaries
            
        Returns:
            Dictionary with uncertainty measures
        """
        try:
            uncertainties = {}
            
            # Extract predictions and probabilities
            predictions = [out.get("prediction") for out in model_outputs if "prediction" in out]
            probabilities = [out.get("probabilities") for out in model_outputs if "probabilities" in out]
            confidences = [out.get("confidence") for out in model_outputs if "confidence" in out]
            
            # Variance-based uncertainty
            if predictions and len(predictions) > 1:
                if all(isinstance(p, (int, float)) for p in predictions):
                    pred_list = [float(p) for p in predictions]
                    mean_pred = sum(pred_list) / len(pred_list)
                    variance = sum((x - mean_pred)**2 for x in pred_list) / len(pred_list)
                    std_dev = variance ** 0.5
                    uncertainties["variance"] = float(variance)
                    uncertainties["std_dev"] = float(std_dev)
            
            # Entropy-based uncertainty (for classification)
            if probabilities:
                avg_probs = self._average_probabilities(probabilities)
                if avg_probs is not None:
                    # Calculate entropy
                    entropy = -sum(p * math.log(p + 1e-15) for p in avg_probs if p > 0)
                    uncertainties["entropy"] = float(entropy)
                    
                    # Calculate max probability (inverse uncertainty measure)
                    uncertainties["max_probability"] = float(max(avg_probs))
            
            # Agreement-based uncertainty
            if predictions and len(predictions) > 1:
                # For classification: disagreement rate
                if all(isinstance(p, str) for p in predictions):
                    unique_preds = set(predictions)
                    agreement = predictions.count(max(set(predictions), key=predictions.count)) / len(predictions)
                    uncertainties["agreement"] = float(agreement)
                    uncertainties["disagreement"] = float(1.0 - agreement)
            
            # Confidence-based uncertainty
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                std_confidence = ((sum((x - avg_confidence)**2 for x in confidences) / len(confidences)) ** 0.5) if len(confidences) > 1 else 0.0
                uncertainties["mean_confidence"] = float(avg_confidence)
                uncertainties["confidence_std"] = float(std_confidence)
            
            # Calibration uncertainty (if historical data available)
            if self.calibration_data:
                uncertainties["calibration_error"] = self._estimate_calibration_error()
            
            return uncertainties
            
        except Exception as e:
            logger.error(f"Error calculating uncertainties: {e}")
            return {"error": str(e)}
    
    def _extract_predictions(self, reports: List[Dict]) -> Optional[List[float]]:
        """Extract predictions matrix from specialist reports."""
        predictions = []
        
        for report in reports:
            if "candidates" in report:
                # Extract predictions from first candidate
                candidates = report["candidates"]
                if candidates and "metrics" in candidates[0]:
                    metrics = candidates[0]["metrics"]
                    if "accuracy" in metrics:
                        predictions.append(metrics["accuracy"])
                    elif "rmse" in metrics:
                        predictions.append(1.0 / (1.0 + metrics["rmse"]))  # Convert to score
        
        if predictions:
            return [float(p) for p in predictions if isinstance(p, (int, float))]
        return None
    
    def _fit_stacking_model(self, X: List[float], y: List[float]):
        """Fit stacking meta-model (simplified linear regression)."""
        try:
            if not X or not y or len(X) != len(y):
                return
            
            # Simple linear regression implementation
            n = len(X)
            sum_x = sum(X)
            sum_y = sum(y)
            sum_xy = sum(x * y for x, y in zip(X, y))
            sum_x2 = sum(x * x for x in X)
            
            # Calculate slope and intercept
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                # Use simple average if no variance
                self.meta_models["linear"]["weights"] = [1.0]
                self.meta_models["linear"]["intercept"] = sum_y / n
                return
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            self.meta_models["linear"]["intercept"] = intercept
            self.meta_models["linear"]["weights"] = [slope]
            
            logger.info("Stacking model fitted")
            
        except Exception as e:
            logger.error(f"Error fitting stacking model: {e}")
    
    def _fit_blending_weights(self, X: List[float], y: Optional[List[float]]):
        """Fit blending weights based on historical performance."""
        # Simplified: equal weights for now
        num_models = len(X) if isinstance(X, list) else 1
        weights = [1.0 / num_models] * num_models
        
        self.meta_models["linear"]["weights"] = weights
        logger.info(f"Blending weights fitted: {weights}")
    
    def _fit_voting_weights(self, X: List[float], reports: List[Dict]):
        """Fit voting weights based on specialist confidence."""
        weights = []
        
        for report in reports:
            # Use specialist confidence as weight
            confidence = report.get("confidence", 0.5)
            weights.append(confidence)
        
        if weights:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)
            self.meta_models["linear"]["weights"] = weights
        
        logger.info(f"Voting weights fitted: {weights}")
    
    def _predict_stacking(self, predictions: List, probabilities: List) -> Dict:
        """Generate stacking prediction."""
        if self.meta_models["linear"]["weights"] is None:
            return self._predict_voting(predictions, probabilities)
        
        # Convert predictions to features
        features = [float(p) if isinstance(p, (int, float)) else 0.5 for p in predictions]
        
        # Apply linear meta-model
        weights = self.meta_models["linear"]["weights"]
        intercept = self.meta_models["linear"]["intercept"] or 0.0
        
        if weights and len(weights) == len(features):
            ensemble_score = intercept + sum(f * w for f, w in zip(features, weights))
        else:
            ensemble_score = intercept + (sum(features) / len(features) if features else 0.5)
        
        # Convert score to prediction
        if all(isinstance(p, str) for p in predictions if p):
            # Classification: use score as confidence for majority vote
            prediction = max(set(predictions), key=predictions.count)
            confidence = float(max(0.0, min(1.0, ensemble_score)))  # Clamp to [0, 1]
        else:
            # Regression: use score directly
            prediction = float(ensemble_score)
            confidence = 0.8  # Default confidence
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "ensemble_score": float(ensemble_score)
        }
    
    def _predict_blending(self, predictions: List, probabilities: List) -> Dict:
        """Generate blending prediction."""
        weights = self.meta_models["linear"]["weights"]
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        
        # Weighted average for regression, weighted voting for classification
        if all(isinstance(p, (int, float)) for p in predictions):
            # Regression: weighted average
            total_weight = sum(weights[:len(predictions)])
            if total_weight > 0:
                weighted_pred = sum(p * w for p, w in zip(predictions, weights[:len(predictions)])) / total_weight
            else:
                weighted_pred = sum(predictions) / len(predictions)
            return {
                "prediction": float(weighted_pred),
                "confidence": 0.75
            }
        else:
            # Classification: weighted voting
            vote_counts = {}
            for i, pred in enumerate(predictions):
                weight = weights[i] if i < len(weights) else 1.0 / len(predictions)
                vote_counts[pred] = vote_counts.get(pred, 0) + weight
            
            best_pred = max(vote_counts, key=vote_counts.get)
            confidence = vote_counts[best_pred] / sum(vote_counts.values())
            
            return {
                "prediction": best_pred,
                "confidence": float(confidence),
                "vote_distribution": vote_counts
            }
    
    def _predict_voting(self, predictions: List, probabilities: List) -> Dict:
        """Generate voting prediction."""
        if not predictions:
            return {"error": "No predictions to vote on"}
        
        # Simple majority voting for classification
        if all(isinstance(p, str) for p in predictions):
            vote_counts = {}
            for pred in predictions:
                vote_counts[pred] = vote_counts.get(pred, 0) + 1
            
            best_pred = max(vote_counts, key=vote_counts.get)
            confidence = vote_counts[best_pred] / len(predictions)
            
            return {
                "prediction": best_pred,
                "confidence": float(confidence),
                "vote_counts": vote_counts
            }
        
        # Average for regression
        else:
            numeric_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
            if numeric_preds:
                avg_pred = np.mean(numeric_preds)
                return {
                    "prediction": float(avg_pred),
                    "confidence": 0.7,
                    "std_dev": float(np.std(numeric_preds)) if len(numeric_preds) > 1 else 0.0
                }
        
        return {"error": "Unable to generate voting prediction"}
    
    def _average_probabilities(self, probabilities: List) -> Optional[List[float]]:
        """Calculate average probability distribution."""
        try:
            valid_probs = []
            for probs in probabilities:
                if isinstance(probs, list) and len(probs) > 0:
                    valid_probs.append(probs)
            
            if valid_probs:
                # Ensure all have same length
                min_len = min(len(p) for p in valid_probs)
                truncated_probs = [p[:min_len] for p in valid_probs]
                
                avg_probs = [sum(probs[i] for probs in truncated_probs) / len(truncated_probs) for i in range(min_len)]
                # Normalize to ensure they sum to 1
                total = sum(avg_probs)
                return [p / total for p in avg_probs] if total > 0 else avg_probs
            
        except Exception as e:
            logger.error(f"Error averaging probabilities: {e}")
        
        return None
    
    def _estimate_calibration_error(self) -> float:
        """Estimate calibration error from historical data."""
        if len(self.calibration_data) < 10:
            return 0.1  # Default moderate uncertainty
        
        # Calculate expected calibration error (simplified)
        errors = []
        for predicted_prob, actual_outcome in self.calibration_data[-100:]:  # Last 100 samples
            error = abs(predicted_prob - actual_outcome)
            errors.append(error)
        
        return float(sum(errors) / len(errors))
    
    def _update_performance_history(self, reports: List[Dict], predictions: List[float]):
        """Update performance tracking history."""
        history_entry = {
            "timestamp": iso_format(),
            "num_specialists": len(reports),
            "prediction_variance": ((sum((p - sum(predictions)/len(predictions))**2 for p in predictions) / len(predictions)) if predictions and len(predictions) > 1 else 0.0),
            "ensemble_type": self.config.get("type", "stack")
        }
        
        self.performance_history.append(history_entry)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_state(self) -> Dict[str, Any]:
        """Get ensembler state for snapshots."""
        return {
            "config": self.config,
            "meta_models": self.meta_models,
            "performance_history": self.performance_history[-50:],  # Last 50 entries
            "version": "1.0.0"
        }
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """Load ensembler state from snapshot."""
        try:
            self.config = state.get("config", self.config)
            self.meta_models = state.get("meta_models", self.meta_models)
            self.performance_history = state.get("performance_history", [])
            
            logger.info("Meta-ensembler state loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensembler state: {e}")
            return False