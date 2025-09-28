"""
Inference Engine - Online service with canary/shadow deployment and A/B routes.

Handles:
1. Single model and ensemble inference execution
2. Canary deployments with gradual traffic ramp-up  
3. Shadow deployments for comparison and validation
4. A/B testing routes with traffic splitting
5. Policy gating based on uncertainty thresholds
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Executes inference with canary/shadow deployment capabilities."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Model registry (mock implementations)
        self.model_registry = {
            "xgb@1.3.2": MockModel("xgboost", latency=50),
            "random_forest@2.0.1": MockModel("random_forest", latency=30),
            "neural_net@2.1.0": MockModel("neural_network", latency=200),
            "bert-base@1.2.0": MockModel("transformer", latency=400),
            "resnet50@1.1.0": MockModel("cnn", latency=300),
            "lstm_model@1.0.5": MockModel("lstm", latency=150)
        }
        
        # Deployment tracking
        self.canary_deployments: Dict[str, Dict] = {}
        self.shadow_deployments: Dict[str, Dict] = {}
        self.ab_routes: Dict[str, Dict] = {}
        
        # Performance tracking
        self.inference_metrics: List[Dict] = []
        
        logger.info("Inference Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default inference engine configuration."""
        return {
            "timeout_ms": 800,
            "batch_max": 64,
            "retry_attempts": 2,
            "canary_auto_promote_threshold": 0.95,
            "shadow_agreement_threshold": 0.90,
            "policy_enforcement": True
        }
    
    def execute(self, plan: Dict[str, Any], task_req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference plan and return results.
        
        Args:
            plan: Execution plan with route and policy
            task_req: Original task request
            
        Returns:
            InferenceResult dictionary
        """
        try:
            start_time = datetime.now()
            
            # Extract plan details
            route = plan.get("route", {})
            policy = plan.get("policy", {})
            models = route.get("models", [])
            ensemble_type = route.get("ensemble", "none")
            
            # Check canary deployment
            canary_pct = route.get("canary_pct", 0)
            use_canary = canary_pct > 0 and random.randint(1, 100) <= canary_pct
            
            # Execute primary inference
            primary_results = self._execute_models(models, task_req, plan)
            
            # Apply ensemble if needed
            if ensemble_type != "none" and len(primary_results) > 1:
                ensemble_result = self._apply_ensemble(primary_results, ensemble_type)
            else:
                ensemble_result = primary_results[0] if primary_results else {"error": "No model results"}
            
            # Shadow deployment execution (parallel)
            shadow_result = None
            if route.get("shadow", False):
                shadow_result = self._execute_shadow(models, task_req, plan)
            
            # Policy gating
            policy_result = self._apply_policy_gates(ensemble_result, policy)
            if not policy_result["approved"]:
                return self._create_rejection_result(plan, policy_result["reason"])
            
            # Create final result
            result = self._create_inference_result(
                plan, task_req, ensemble_result, primary_results, shadow_result, start_time
            )
            
            # Track metrics
            self._track_inference_metrics(result, plan, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            return self._create_error_result(plan, str(e))
    
    def _execute_models(self, models: List[str], task_req: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict]:
        """Execute inference on specified models."""
        results = []
        input_data = task_req.get("input", {})
        
        for model_key in models:
            try:
                model = self.model_registry.get(model_key)
                if not model:
                    logger.warning(f"Model {model_key} not found in registry")
                    continue
                
                # Execute model inference
                prediction_result = model.predict(input_data)
                
                # Add model metadata
                prediction_result["model_key"] = model_key
                prediction_result["inference_time"] = datetime.now().isoformat()
                
                results.append(prediction_result)
                
            except Exception as e:
                logger.error(f"Model {model_key} inference failed: {e}")
                results.append({
                    "model_key": model_key,
                    "error": str(e),
                    "prediction": None
                })
        
        return results
    
    def _apply_ensemble(self, model_results: List[Dict], ensemble_type: str) -> Dict[str, Any]:
        """Apply ensemble method to combine model results."""
        valid_results = [r for r in model_results if "prediction" in r and r["prediction"] is not None]
        
        if not valid_results:
            return {"error": "No valid model predictions for ensembling"}
        
        if ensemble_type == "vote":
            return self._majority_vote(valid_results)
        elif ensemble_type == "blend":
            return self._blend_predictions(valid_results)
        elif ensemble_type == "stack":
            return self._stack_predictions(valid_results)
        else:
            return valid_results[0]  # Default to first result
    
    def _majority_vote(self, results: List[Dict]) -> Dict[str, Any]:
        """Apply majority voting ensemble."""
        predictions = [r["prediction"] for r in results]
        
        if all(isinstance(p, str) for p in predictions):
            # Classification: count votes
            vote_counts = {}
            for pred in predictions:
                vote_counts[pred] = vote_counts.get(pred, 0) + 1
            
            best_pred = max(vote_counts, key=vote_counts.get)
            confidence = vote_counts[best_pred] / len(predictions)
            
            return {
                "prediction": best_pred,
                "confidence": confidence,
                "method": "majority_vote",
                "vote_distribution": vote_counts
            }
        if all(isinstance(p, (int, float)) for p in predictions):
            # Regression: median
            numeric_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
            if numeric_preds:
                numeric_preds.sort()
                n = len(numeric_preds)
                median_pred = (numeric_preds[n//2 - 1] + numeric_preds[n//2]) / 2 if n % 2 == 0 else numeric_preds[n//2]
                std_dev = ((sum((x - median_pred)**2 for x in numeric_preds) / n) ** 0.5) if n > 1 else 0.0
                return {
                    "prediction": float(median_pred),
                    "confidence": 0.75,
                    "method": "median_vote",
                    "std_dev": float(std_dev)
                }
        
        return {"error": "Unable to apply majority vote"}
    
    def _blend_predictions(self, results: List[Dict]) -> Dict[str, Any]:
        """Apply blended averaging ensemble."""
        predictions = [r["prediction"] for r in results]
        confidences = [r.get("confidence", 0.5) for r in results]
        
        # Weight by confidence
        total_weight = sum(confidences)
        if total_weight == 0:
            weights = [1/len(results)] * len(results)
        else:
            weights = [c/total_weight for c in confidences]
        
        if all(isinstance(p, (int, float)) for p in predictions):
            # Weighted average for regression
            weighted_pred = sum(w * p for w, p in zip(weights, predictions))
            avg_confidence = sum(w * c for w, c in zip(weights, confidences))
            
            return {
                "prediction": float(weighted_pred),
                "confidence": float(avg_confidence),
                "method": "blend",
                "weights": weights
            }
        else:
            # Weighted voting for classification
            vote_weights = {}
            for pred, weight in zip(predictions, weights):
                vote_weights[pred] = vote_weights.get(pred, 0) + weight
            
            best_pred = max(vote_weights, key=vote_weights.get)
            confidence = vote_weights[best_pred]
            
            return {
                "prediction": best_pred,
                "confidence": float(confidence),
                "method": "weighted_vote",
                "vote_weights": vote_weights
            }
    
    def _stack_predictions(self, results: List[Dict]) -> Dict[str, Any]:
        """Apply stacking ensemble (simplified meta-learning)."""
        predictions = [r["prediction"] for r in results]
        
        # Simplified stacking: weighted combination based on historical performance
        # In real implementation, this would use trained meta-model
        
        # Mock meta-model weights (normally learned from validation data)
        model_weights = {
            "xgb@1.3.2": 0.35,
            "random_forest@2.0.1": 0.25, 
            "neural_net@2.1.0": 0.30,
            "bert-base@1.2.0": 0.40,
            "resnet50@1.1.0": 0.35,
            "lstm_model@1.0.5": 0.30
        }
        
        weights = []
        for result in results:
            model_key = result.get("model_key", "")
            weight = model_weights.get(model_key, 0.25)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        else:
            weights = [1/len(results)] * len(results)
        
        if all(isinstance(p, (int, float)) for p in predictions):
            # Weighted combination for regression
            stacked_pred = sum(w * p for w, p in zip(weights, predictions))
            
            # Estimate confidence based on agreement
            variance = sum(w * (p - stacked_pred)**2 for w, p in zip(weights, predictions))
            confidence = max(0.5, 1.0 - variance)
            
            return {
                "prediction": float(stacked_pred),
                "confidence": float(confidence),
                "method": "stacking",
                "meta_weights": weights,
                "variance": float(variance)
            }
        else:
            # Meta-voting for classification
            vote_scores = {}
            for pred, weight in zip(predictions, weights):
                vote_scores[pred] = vote_scores.get(pred, 0) + weight
            
            best_pred = max(vote_scores, key=vote_scores.get)
            confidence = vote_scores[best_pred]
            
            return {
                "prediction": best_pred,
                "confidence": float(confidence),
                "method": "meta_voting",
                "vote_scores": vote_scores
            }
    
    def _execute_shadow(self, models: List[str], task_req: Dict[str, Any], plan: Dict[str, Any]) -> Optional[Dict]:
        """Execute shadow deployment for comparison."""
        try:
            # For demonstration, use a different model or variant
            shadow_models = [f"{model}_shadow" if "_shadow" not in model else model for model in models[:1]]
            
            # Mock shadow execution (in reality, this would be actual alternative models)
            shadow_results = []
            for model_key in shadow_models:
                # Create mock shadow result
                shadow_result = {
                    "model_key": model_key,
                    "prediction": "shadow_prediction",
                    "confidence": 0.72,
                    "shadow": True
                }
                shadow_results.append(shadow_result)
            
            return {
                "shadow_results": shadow_results,
                "shadow_timestamp": datetime.now().isoformat(),
                "shadow_agreement": 0.85  # Mock agreement score
            }
            
        except Exception as e:
            logger.error(f"Shadow execution failed: {e}")
            return None
    
    def _apply_policy_gates(self, result: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, bool]:
        """Apply policy gates to check if result meets requirements."""
        if not self.config.get("policy_enforcement", True):
            return {"approved": True, "reason": None}
        
        try:
            # Check confidence threshold
            min_confidence = policy.get("min_confidence", 0.5)
            confidence = result.get("confidence", 0.0)
            
            if confidence < min_confidence:
                return {
                    "approved": False,
                    "reason": f"Confidence {confidence:.3f} below threshold {min_confidence}"
                }
            
            # Check calibration (mock implementation)
            min_calibration = policy.get("min_calibration", 0.9)
            # In real implementation, this would check actual calibration
            estimated_calibration = confidence * 0.95  # Mock calibration
            
            if estimated_calibration < min_calibration:
                return {
                    "approved": False,
                    "reason": f"Calibration {estimated_calibration:.3f} below threshold {min_calibration}"
                }
            
            # Check for errors
            if "error" in result:
                return {
                    "approved": False,
                    "reason": f"Inference error: {result['error']}"
                }
            
            return {"approved": True, "reason": None}
            
        except Exception as e:
            logger.error(f"Policy gate evaluation failed: {e}")
            return {
                "approved": False, 
                "reason": f"Policy evaluation error: {str(e)}"
            }
    
    def _create_inference_result(self, plan: Dict, task_req: Dict, ensemble_result: Dict, 
                               model_results: List[Dict], shadow_result: Optional[Dict], 
                               start_time: datetime) -> Dict[str, Any]:
        """Create final InferenceResult object."""
        end_time = datetime.now()
        total_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract outputs
        outputs = {
            "y_hat": ensemble_result.get("prediction"),
            "proba": ensemble_result.get("probabilities"),
            "confidence": ensemble_result.get("confidence", 0.5)
        }
        
        # Runtime metrics
        metrics = {
            "ensemble_method": ensemble_result.get("method", "single"),
            "num_models": len(model_results),
            "success_rate": len([r for r in model_results if "error" not in r]) / max(1, len(model_results)),
            "calibration_est": outputs["confidence"] * 0.95,  # Mock calibration
            "drift_score": 0.02  # Mock drift score
        }
        
        # Add shadow metrics if available
        if shadow_result:
            metrics["shadow_agreement"] = shadow_result.get("shadow_agreement", 0.0)
        
        # Uncertainty metrics
        uncertainties = {
            "prediction_variance": ensemble_result.get("variance", 0.0),
            "confidence_uncertainty": 1.0 - outputs["confidence"],
            "method_uncertainty": 0.1  # Mock method uncertainty
        }
        
        # Lineage
        lineage = {
            "plan_id": plan.get("plan_id"),
            "models": [r.get("model_key") for r in model_results],
            "ensemble": ensemble_result.get("method", "none"),
            "feature_view": "default_v1"
        }
        
        # Governance
        governance = {
            "approved": True,
            "policy_version": "1.0.0",
            "redactions": []
        }
        
        # Timing breakdown
        timing = {
            "total_ms": total_ms,
            "per_stage": {
                "routing": 5,
                "inference": total_ms - 20,
                "ensemble": 10,
                "policy_check": 5
            }
        }
        
        return {
            "req_id": task_req.get("req_id"),
            "outputs": outputs,
            "metrics": metrics,
            "uncertainties": uncertainties,
            "lineage": lineage,
            "governance": governance,
            "timing": timing,
            "shadow_result": shadow_result
        }
    
    def _create_rejection_result(self, plan: Dict, reason: str) -> Dict[str, Any]:
        """Create rejection result for policy violations."""
        return {
            "req_id": plan.get("req_id"),
            "status": "rejected",
            "reason": reason,
            "outputs": None,
            "governance": {
                "approved": False,
                "policy_version": "1.0.0",
                "rejection_reason": reason
            },
            "timing": {"total_ms": 10}
        }
    
    def _create_error_result(self, plan: Dict, error: str) -> Dict[str, Any]:
        """Create error result for inference failures."""
        return {
            "req_id": plan.get("req_id"),
            "status": "error",
            "error": error,
            "outputs": None,
            "timing": {"total_ms": 5}
        }
    
    def _track_inference_metrics(self, result: Dict, plan: Dict, start_time: datetime):
        """Track inference metrics for monitoring."""
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "plan_id": plan.get("plan_id"),
            "latency_ms": result["timing"]["total_ms"],
            "success": "error" not in result,
            "confidence": result.get("outputs", {}).get("confidence", 0.0),
            "num_models": len(plan.get("route", {}).get("models", [])),
            "ensemble_type": plan.get("route", {}).get("ensemble", "none")
        }
        
        self.inference_metrics.append(metrics_entry)
        
        # Keep only recent metrics
        if len(self.inference_metrics) > 10000:
            self.inference_metrics = self.inference_metrics[-5000:]
    
    def get_metrics(self, since: Optional[str] = None, segment: Optional[str] = None) -> Dict[str, Any]:
        """Get inference metrics for monitoring."""
        recent_metrics = self.inference_metrics[-1000:]  # Last 1000 inferences
        
        if not recent_metrics:
            return {"message": "No metrics available"}
        
        # Calculate summary statistics
        latencies = [m["latency_ms"] for m in recent_metrics]
        confidences = [m["confidence"] for m in recent_metrics]
        success_rate = sum(m["success"] for m in recent_metrics) / len(recent_metrics)
        
        return {
            "total_inferences": len(recent_metrics),
            "success_rate": success_rate,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "avg_confidence": sum(confidences) / len(confidences),
            "ensemble_distribution": self._get_ensemble_distribution(recent_metrics)
        }
    
    def _get_ensemble_distribution(self, metrics: List[Dict]) -> Dict[str, int]:
        """Get distribution of ensemble methods used."""
        distribution = {}
        for metric in metrics:
            ensemble_type = metric.get("ensemble_type", "none")
            distribution[ensemble_type] = distribution.get(ensemble_type, 0) + 1
        return distribution


class MockModel:
    """Mock model implementation for testing."""
    
    def __init__(self, model_type: str, latency: int = 100):
        self.model_type = model_type
        self.latency = latency
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock prediction implementation."""
        # Simulate processing time
        import time
        time.sleep(self.latency / 1000.0)
        
        # Generate mock predictions based on model type
        if self.model_type in ["xgboost", "random_forest"]:
            # Classification
            classes = ["approved", "rejected", "pending"]
            prediction = random.choice(classes)
            confidence = random.uniform(0.6, 0.95)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": [0.1, 0.8, 0.1] if prediction == "rejected" else [0.7, 0.2, 0.1]
            }
        
        elif self.model_type == "neural_network":
            # Regression
            prediction = random.uniform(0.0, 1.0)
            confidence = random.uniform(0.7, 0.9)
            
            return {
                "prediction": prediction,
                "confidence": confidence
            }
        
        else:
            # Default mock prediction
            return {
                "prediction": "default_output",
                "confidence": 0.75
            }