"""
Base adapter for MLDL models - provides consistent interface for all ML/DL models.
"""
import pickle
import joblib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """Base class for all model adapters in MLDL kernel."""
    
    def __init__(self, model_key: str, task: str, name: str, version: str = "1.0.0"):
        self.model_key = model_key
        self.task = task  # classification, regression, clustering, dimred, rl
        self.name = name
        self.version = version
        self.model = None
        self.feature_names = None
        self.target_names = None
        self.fitted = False
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "training_params": {},
            "performance_metrics": {}
        }
    
    @abstractmethod
    def fit(self, X, y=None, **kwargs) -> 'BaseModelAdapter':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training targets (optional for unsupervised)
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs) -> Union[np.ndarray, List]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X, **kwargs) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classification tasks).
        
        Args:
            X: Input features
            **kwargs: Additional parameters
            
        Returns:
            Probability predictions or None if not supported
        """
        return None
    
    def save(self, uri: str) -> str:
        """
        Save model to specified URI.
        
        Args:
            uri: Save location URI
            
        Returns:
            Actual save path/URI
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted model")
        
        try:
            # Save model using joblib (handles sklearn models well)
            model_path = f"{uri}/model.pkl"
            joblib.dump(self.model, model_path)
            
            # Save metadata
            metadata_path = f"{uri}/metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save adapter info
            adapter_info = {
                "model_key": self.model_key,
                "task": self.task,
                "name": self.name,
                "version": self.version,
                "adapter_class": self.__class__.__name__,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "fitted": self.fitted
            }
            
            info_path = f"{uri}/adapter_info.json"
            with open(info_path, 'w') as f:
                json.dump(adapter_info, f, indent=2)
            
            logger.info(f"Model {self.model_key} saved to {uri}")
            return uri
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_key}: {e}")
            raise
    
    @classmethod
    def load(cls, uri: str) -> 'BaseModelAdapter':
        """
        Load model from URI.
        
        Args:
            uri: Load location URI
            
        Returns:
            Loaded model adapter instance
        """
        try:
            # Load adapter info
            info_path = f"{uri}/adapter_info.json"
            with open(info_path, 'r') as f:
                adapter_info = json.load(f)
            
            # Create adapter instance
            instance = cls(
                model_key=adapter_info["model_key"],
                task=adapter_info["task"],
                name=adapter_info["name"],
                version=adapter_info["version"]
            )
            
            # Load model
            model_path = f"{uri}/model.pkl"
            instance.model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = f"{uri}/metadata.json"
            with open(metadata_path, 'r') as f:
                instance.metadata = json.load(f)
            
            # Restore properties
            instance.feature_names = adapter_info.get("feature_names")
            instance.target_names = adapter_info.get("target_names")
            instance.fitted = adapter_info.get("fitted", False)
            
            logger.info(f"Model {instance.model_key} loaded from {uri}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load model from {uri}: {e}")
            raise
    
    def explain(self, X, method: str = "auto") -> Dict[str, Any]:
        """
        Generate model explanations.
        
        Args:
            X: Input data to explain
            method: Explanation method ('auto', 'shap', 'lime', 'permutation')
            
        Returns:
            Explanation results
        """
        # Basic explanation placeholder
        # Real implementations would use SHAP, LIME, etc.
        explanations = {
            "method": method,
            "model_key": self.model_key,
            "explained_samples": len(X) if hasattr(X, '__len__') else 1,
            "feature_importance": self._get_feature_importance(),
            "timestamp": datetime.now().isoformat()
        }
        
        return explanations
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.fitted or not hasattr(self.model, 'feature_importances_'):
            return None
        
        if self.feature_names and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            **self.metadata,
            "model_key": self.model_key,
            "task": self.task,
            "name": self.name,
            "version": self.version,
            "fitted": self.fitted,
            "feature_names": self.feature_names,
            "target_names": self.target_names
        }
    
    def update_metadata(self, **kwargs):
        """Update model metadata."""
        self.metadata.update(kwargs)
        self.metadata["updated_at"] = datetime.now().isoformat()
    
    def validate_input(self, X) -> bool:
        """Validate input data format."""
        try:
            if hasattr(X, 'shape'):
                return len(X.shape) >= 1
            elif hasattr(X, '__len__'):
                return len(X) > 0
            else:
                return False
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            "model_key": self.model_key,
            "task": self.task,
            "name": self.name,
            "version": self.version,
            "adapter_class": self.__class__.__name__,
            "fitted": self.fitted,
            "feature_count": len(self.feature_names) if self.feature_names else None,
            "target_count": len(self.target_names) if self.target_names else None,
            "model_type": type(self.model).__name__ if self.model else None,
            "supports_probability": hasattr(self, 'predict_proba') and callable(getattr(self, 'predict_proba')),
            "supports_explanation": True,  # All adapters support basic explanation
            "metadata": self.metadata
        }
        
        # Add model-specific parameters if available
        if hasattr(self.model, 'get_params'):
            info["model_params"] = self.model.get_params()
        
        return info