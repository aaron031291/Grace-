"""
Supervised Learning Specialists - Layer 1

Individual ML models for supervised learning tasks:
- Decision Tree Specialist
- SVM Specialist  
- Random Forest Specialist
- Gradient Boosting Specialist
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging

from .base_specialist import BaseSpecialist, SpecialistCapability, SpecialistPrediction

logger = logging.getLogger(__name__)


class DecisionTreeSpecialist(BaseSpecialist):
    """
    Decision Tree Specialist - Interpretable tree-based classification/regression.
    
    Use cases:
    - Trust score classification (high/medium/low)
    - KPI threshold prediction
    - Rule-based governance decisions
    """
    
    def __init__(
        self,
        specialist_id: str = "decision_tree_specialist",
        mode: str = "classification",  # or "regression"
        max_depth: int = 10
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.REGRESSION,
            SpecialistCapability.TRUST_SCORING
        ]
        super().__init__(specialist_id, capabilities)
        
        self.mode = mode
        if mode == "classification":
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        self.is_trained = False
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using decision tree."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            # Extract features
            features = self._extract_features(input_data)
            X = np.array(features).reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            # Get confidence (probability for classification, std for regression)
            if self.mode == "classification" and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            else:
                # For regression, use inverse normalized error as confidence
                confidence = 0.75  # Default
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                confidence=confidence,
                metadata={
                    'mode': self.mode,
                    'feature_count': len(features)
                }
            )
            
        except Exception as e:
            logger.error(f"DecisionTreeSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the decision tree model."""
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"{self.specialist_id} trained with {len(X)} samples")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        features = []
        
        # Handle different input formats
        if 'features' in input_data:
            features = input_data['features']
        elif 'values' in input_data:
            features = input_data['values']
        else:
            # Extract all numeric values
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
        
        return features


class SVMSpecialist(BaseSpecialist):
    """
    SVM Specialist - Support Vector Machine for classification/regression.
    
    Use cases:
    - High-dimensional pattern recognition
    - Anomaly detection (one-class SVM)
    - Binary classification tasks
    """
    
    def __init__(
        self,
        specialist_id: str = "svm_specialist",
        mode: str = "classification",
        kernel: str = "rbf"
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.REGRESSION,
            SpecialistCapability.ANOMALY_DETECTION
        ]
        super().__init__(specialist_id, capabilities)
        
        self.mode = mode
        if mode == "classification":
            self.model = SVC(kernel=kernel, probability=True, random_state=42)
        else:
            self.model = SVR(kernel=kernel)
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using SVM."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            # Extract and scale features
            features = self._extract_features(input_data)
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            
            # Get confidence
            if self.mode == "classification":
                proba = self.model.predict_proba(X_scaled)[0]
                confidence = float(np.max(proba))
            else:
                confidence = 0.75  # Default for regression
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                confidence=confidence,
                metadata={
                    'mode': self.mode,
                    'kernel': self.model.kernel
                }
            )
            
        except Exception as e:
            logger.error(f"SVMSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the SVM model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info(f"{self.specialist_id} trained with {len(X)} samples")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        features = []
        
        if 'features' in input_data:
            features = input_data['features']
        elif 'values' in input_data:
            features = input_data['values']
        else:
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
        
        return features


class RandomForestSpecialist(BaseSpecialist):
    """
    Random Forest Specialist - Ensemble of decision trees.
    
    Use cases:
    - Robust predictions with confidence intervals
    - Feature importance analysis
    - KPI forecasting
    - Trust score aggregation
    """
    
    def __init__(
        self,
        specialist_id: str = "random_forest_specialist",
        mode: str = "classification",
        n_estimators: int = 100
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.REGRESSION,
            SpecialistCapability.FORECASTING,
            SpecialistCapability.TRUST_SCORING
        ]
        super().__init__(specialist_id, capabilities)
        
        self.mode = mode
        if mode == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
        
        self.is_trained = False
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using random forest."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            # Extract features
            features = self._extract_features(input_data)
            X = np.array(features).reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            # Get confidence and feature importance
            if self.mode == "classification":
                proba = self.model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            else:
                # For regression, use tree variance as confidence proxy
                tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
                variance = np.var(tree_predictions)
                confidence = max(0.0, 1.0 - min(variance / 10.0, 1.0))  # Normalize
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_.tolist()
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                confidence=confidence,
                metadata={
                    'mode': self.mode,
                    'n_estimators': self.model.n_estimators,
                    'feature_importance': feature_importance
                }
            )
            
        except Exception as e:
            logger.error(f"RandomForestSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the random forest model."""
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"{self.specialist_id} trained with {len(X)} samples")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        features = []
        
        if 'features' in input_data:
            features = input_data['features']
        elif 'values' in input_data:
            features = input_data['values']
        else:
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
        
        return features


class GradientBoostingSpecialist(BaseSpecialist):
    """
    Gradient Boosting Specialist - Sequential ensemble with boosting.
    
    Use cases:
    - High-accuracy predictions
    - Ranking and scoring tasks
    - Complex pattern recognition
    """
    
    def __init__(
        self,
        specialist_id: str = "gradient_boosting_specialist",
        n_estimators: int = 100,
        learning_rate: float = 0.1
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.REGRESSION,
            SpecialistCapability.FORECASTING
        ]
        super().__init__(specialist_id, capabilities)
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        self.is_trained = False
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using gradient boosting."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            features = self._extract_features(input_data)
            X = np.array(features).reshape(1, -1)
            
            prediction = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                confidence=confidence,
                metadata={
                    'n_estimators': self.model.n_estimators,
                    'learning_rate': self.model.learning_rate
                }
            )
            
        except Exception as e:
            logger.error(f"GradientBoostingSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the gradient boosting model."""
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"{self.specialist_id} trained with {len(X)} samples")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        features = []
        
        if 'features' in input_data:
            features = input_data['features']
        elif 'values' in input_data:
            features = input_data['values']
        else:
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
        
        return features
