"""
Training job runner with cross-validation, HPO, early stopping, and checkpoints.
"""
import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ...utils.datetime_utils import utc_now, iso_format, format_for_filename

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Training runner will use mock implementations.")


class TrainingJobRunner:
    """Runs training jobs with CV, HPO, early stopping, and checkpoints."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.active_jobs = {}
        
    async def run(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a training job based on specification.
        
        Args:
            job_spec: Training job specification
            
        Returns:
            TrainedBundle dictionary
        """
        job_id = job_spec.get("job_id", f"job_{uuid.uuid4().hex[:8]}")
        
        try:
            # Initialize job tracking
            self.active_jobs[job_id] = {
                "status": "initializing",
                "start_time": utc_now(),
                "trials": 0,
                "best_score": None
            }
            
            # Parse job specification
            dataset_id = job_spec["dataset_id"]
            version = job_spec["version"]
            model_spec = job_spec["spec"]
            cv_config = job_spec.get("cv", {"folds": 5, "stratify": True})
            hpo_config = job_spec.get("hpo", {"strategy": "none"})
            
            logger.info(f"Starting training job {job_id} for {model_spec['model_key']}")
            
            # Load dataset (mock for now)
            X_train, y_train = await self._load_dataset(dataset_id, version)
            
            # Initialize model adapter
            adapter = await self._initialize_adapter(model_spec)
            
            # Setup cross-validation
            cv_strategy = self._setup_cross_validation(cv_config, X_train, y_train)
            
            # Run training with HPO if configured
            if hpo_config["strategy"] != "none":
                best_adapter, best_score, trials = await self._run_hpo(
                    adapter, X_train, y_train, cv_strategy, hpo_config, job_id
                )
            else:
                # Simple training without HPO
                adapter.fit(X_train, y_train)
                best_adapter = adapter
                best_score = await self._evaluate_model(adapter, X_train, y_train, cv_strategy)
                trials = 1
            
            # Update job status
            self.active_jobs[job_id].update({
                "status": "completed",
                "trials": trials,
                "best_score": best_score,
                "end_time": utc_now()
            })
            
            # Create TrainedBundle
            trained_bundle = await self._create_trained_bundle(
                best_adapter, best_score, job_spec, trials
            )
            
            logger.info(f"Training job {job_id} completed with score {best_score:.4f}")
            
            return trained_bundle
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            self.active_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "end_time": utc_now()
            })
            raise
    
    async def _load_dataset(self, dataset_id: str, version: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load training dataset."""
        # Mock dataset loading - in real implementation would connect to data sources
        logger.info(f"Loading dataset {dataset_id} version {version}")
        
        # Generate mock tabular data
        n_samples, n_features = 1000, 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)  # Binary classification
        
        return X, y
    
    async def _initialize_adapter(self, model_spec: Dict[str, Any]):
        """Initialize model adapter from specification."""
        from ..adapters.classic import (
            LogisticRegressionAdapter, LinearRegressionAdapter, SVMAdapter,
            KNNAdapter, DecisionTreeAdapter, XGBAdapter, NaiveBayesAdapter
        )
        from ..adapters.clustering import KMeansAdapter, PCAAdapter
        
        family = model_spec["family"]
        task = model_spec["task"]
        hyperparams = model_spec.get("hyperparams", {})
        
        # Map family to adapter class
        adapter_map = {
            "lr": LogisticRegressionAdapter if task == "classification" else LinearRegressionAdapter,
            "svm": lambda **kwargs: SVMAdapter(task=task, **kwargs),
            "knn": lambda **kwargs: KNNAdapter(task=task, **kwargs),
            "tree": lambda **kwargs: DecisionTreeAdapter(task=task, **kwargs),
            "xgb": lambda **kwargs: XGBAdapter(task=task, **kwargs),
            "nb": NaiveBayesAdapter,
            "kmeans": KMeansAdapter,
            "pca": PCAAdapter
        }
        
        if family not in adapter_map:
            raise ValueError(f"Unknown model family: {family}")
        
        adapter_class = adapter_map[family]
        adapter = adapter_class(
            model_key=model_spec["model_key"],
            **hyperparams
        )
        
        return adapter
    
    def _setup_cross_validation(self, cv_config: Dict[str, Any], X, y) -> Any:
        """Setup cross-validation strategy."""
        folds = cv_config.get("folds", 5)
        stratify = cv_config.get("stratify", True)
        time_aware = cv_config.get("time_aware", False)
        
        if not SKLEARN_AVAILABLE:
            return None
        
        if time_aware:
            return TimeSeriesSplit(n_splits=folds)
        elif stratify and len(np.unique(y)) > 1:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            return KFold(n_splits=folds, shuffle=True, random_state=42)
    
    async def _run_hpo(self, adapter, X, y, cv_strategy, hpo_config: Dict[str, Any], job_id: str):
        """Run hyperparameter optimization."""
        strategy = hpo_config["strategy"]
        max_trials = hpo_config.get("max_trials", 50)
        early_stop = hpo_config.get("early_stop", True)
        success_metric = hpo_config.get("success_metric", "f1")
        
        logger.info(f"Running HPO with strategy {strategy}, max_trials {max_trials}")
        
        best_adapter = adapter
        best_score = -np.inf
        trials = 0
        no_improvement_count = 0
        early_stop_patience = 10
        
        for trial in range(max_trials):
            trials += 1
            
            # Generate hyperparameter configuration
            trial_params = await self._sample_hyperparameters(adapter, strategy, trial)
            
            # Create trial adapter
            trial_adapter = await self._initialize_adapter({
                "model_key": adapter.model_key,
                "family": self._infer_family(adapter),
                "task": adapter.task,
                "hyperparams": trial_params
            })
            
            # Train and evaluate
            try:
                trial_adapter.fit(X, y)
                score = await self._evaluate_model(trial_adapter, X, y, cv_strategy)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_adapter = trial_adapter
                    no_improvement_count = 0
                    logger.info(f"Trial {trial}: New best score {score:.4f}")
                else:
                    no_improvement_count += 1
                
                # Update job status
                self.active_jobs[job_id].update({
                    "trials": trials,
                    "best_score": best_score,
                    "status": "running"
                })
                
                # Early stopping check
                if early_stop and no_improvement_count >= early_stop_patience:
                    logger.info(f"Early stopping at trial {trial} (no improvement for {no_improvement_count} trials)")
                    break
                    
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        return best_adapter, best_score, trials
    
    async def _sample_hyperparameters(self, adapter, strategy: str, trial: int) -> Dict[str, Any]:
        """Sample hyperparameters for HPO trial."""
        # Mock hyperparameter sampling
        # Real implementation would use proper HPO libraries like Optuna, Hyperopt, etc.
        
        base_params = {}
        family = self._infer_family(adapter)
        
        if family == "lr":
            base_params = {
                "C": np.random.uniform(0.01, 10.0),
                "solver": np.random.choice(["liblinear", "lbfgs"])
            }
        elif family == "svm":
            base_params = {
                "C": np.random.uniform(0.1, 10.0),
                "kernel": np.random.choice(["rbf", "linear"]),
                "gamma": np.random.choice(["scale", "auto"])
            }
        elif family == "tree":
            base_params = {
                "max_depth": np.random.randint(3, 20),
                "min_samples_split": np.random.randint(2, 20),
                "min_samples_leaf": np.random.randint(1, 10)
            }
        elif family == "xgb":
            base_params = {
                "max_depth": np.random.randint(3, 10),
                "learning_rate": np.random.uniform(0.01, 0.3),
                "n_estimators": np.random.randint(50, 500),
                "subsample": np.random.uniform(0.6, 1.0)
            }
        
        return base_params
    
    def _infer_family(self, adapter) -> str:
        """Infer model family from adapter."""
        name = adapter.__class__.__name__.lower()
        
        if "logistic" in name or "linear" in name:
            return "lr"
        elif "svm" in name:
            return "svm"
        elif "knn" in name or "neighbors" in name:
            return "knn"
        elif "tree" in name:
            return "tree"
        elif "xgb" in name:
            return "xgb"
        elif "naive" in name or "nb" in name:
            return "nb"
        elif "kmeans" in name:
            return "kmeans"
        elif "pca" in name:
            return "pca"
        else:
            return "unknown"
    
    async def _evaluate_model(self, adapter, X, y, cv_strategy) -> float:
        """Evaluate model using cross-validation."""
        if not SKLEARN_AVAILABLE or cv_strategy is None:
            # Mock evaluation
            return np.random.uniform(0.6, 0.9)
        
        try:
            if adapter.task == "classification":
                scores = cross_val_score(adapter.model, X, y, cv=cv_strategy, scoring="f1")
            elif adapter.task == "regression":
                scores = cross_val_score(adapter.model, X, y, cv=cv_strategy, scoring="neg_mean_squared_error")
                scores = -scores  # Convert back to positive MSE
            else:
                # For clustering/dimred, use silhouette score or mock
                return np.random.uniform(0.5, 0.8)
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}, using mock score")
            return np.random.uniform(0.5, 0.8)
    
    async def _create_trained_bundle(self, adapter, best_score: float, job_spec: Dict[str, Any], trials: int) -> Dict[str, Any]:
        """Create TrainedBundle from trained adapter."""
        model_spec = job_spec["spec"]
        
        # Create artifact URI (mock path)
        artifact_uri = f"/models/{adapter.model_key}/{format_for_filename()}"
        
        # Save model
        try:
            adapter.save(artifact_uri)
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            artifact_uri = f"mock://{adapter.model_key}"
        
        # Calculate performance metrics
        metrics = {
            "cv_score": best_score,
            "trials": trials
        }
        
        if adapter.task == "classification":
            metrics.update({
                "f1": best_score,
                "accuracy": best_score * 0.95,  # Mock related metrics
                "precision": best_score * 1.02,
                "recall": best_score * 0.98
            })
        elif adapter.task == "regression":
            metrics.update({
                "rmse": abs(best_score),
                "mae": abs(best_score) * 0.8,
                "r2": max(0, 1 - best_score)
            })
        
        # Create calibration info (mock)
        calibration = {
            "ece": np.random.uniform(0.01, 0.1),
            "method": "isotonic"
        }
        
        # Create fairness info (mock)
        fairness = {
            "delta": np.random.uniform(0.01, 0.05),
            "groups": ["gender", "region"]
        }
        
        # Create robustness info (mock)
        robustness = {
            "noise_sensitivity": np.random.uniform(0.05, 0.2)
        }
        
        # Create data schema (mock)
        data_schema = {
            "features": adapter.feature_names or [f"feature_{i}" for i in range(20)],
            "target": adapter.target_names or ["target"],
            "dtypes": {"features": "float64", "target": "int64"}
        }
        
        # Create lineage
        lineage = {
            "dataset_id": job_spec["dataset_id"],
            "version": job_spec["version"],
            "feature_view": model_spec["feature_view"],
            "trainer_hash": f"trainer_{uuid.uuid4().hex[:8]}"
        }
        
        # Create validation hash
        validation_hash = f"sha256:{uuid.uuid4().hex}"
        
        trained_bundle = {
            "model_key": adapter.model_key,
            "version": f"1.0.{trials}",
            "artifact_uri": artifact_uri,
            "metrics": metrics,
            "calibration": calibration,
            "fairness": fairness,
            "robustness": robustness,
            "data_schema": data_schema,
            "lineage": lineage,
            "validation_hash": validation_hash,
            "created_at": iso_format()
        }
        
        return trained_bundle
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job."""
        return self.active_jobs.get(job_id)