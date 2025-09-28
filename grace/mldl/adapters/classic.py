"""
Classic ML model adapters for tabular data.
"""
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Classic ML adapters will use mock implementations.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LogisticRegressionAdapter(BaseModelAdapter):
    """Logistic Regression adapter."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", "lr_classifier"),
            task="classification",
            name="LogisticRegression",
            version=kwargs.get("version", "1.0.0")
        )
        
        if SKLEARN_AVAILABLE:
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version']}
            )
        else:
            self.model = MockModel("logistic_regression")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("Logistic Regression requires target labels")
        
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        # Store feature names if provided
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if SKLEARN_AVAILABLE:
            return self.model.predict_proba(X)
        else:
            # Mock probability prediction
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.random.rand(n_samples, 2)


class LinearRegressionAdapter(BaseModelAdapter):
    """Linear Regression adapter."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", "lr_regressor"),
            task="regression",
            name="LinearRegression",
            version=kwargs.get("version", "1.0.0")
        )
        
        if SKLEARN_AVAILABLE:
            self.model = LinearRegression(
                **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version']}
            )
        else:
            self.model = MockModel("linear_regression")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("Linear Regression requires target values")
        
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)


class SVMAdapter(BaseModelAdapter):
    """Support Vector Machine adapter."""
    
    def __init__(self, task="classification", **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", f"svm_{task}"),
            task=task,
            name=f"SVM_{task.capitalize()}",
            version=kwargs.get("version", "1.0.0")
        )
        
        if SKLEARN_AVAILABLE:
            if task == "classification":
                self.model = SVC(
                    probability=True,
                    random_state=42,
                    **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version', 'task']}
                )
            else:  # regression
                self.model = SVR(
                    **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version', 'task']}
                )
        else:
            self.model = MockModel(f"svm_{task}")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("SVM requires target values")
        
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task == "classification" and SKLEARN_AVAILABLE:
            return self.model.predict_proba(X)
        elif not SKLEARN_AVAILABLE and self.task == "classification":
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.random.rand(n_samples, 2)
        
        return None


class KNNAdapter(BaseModelAdapter):
    """K-Nearest Neighbors adapter."""
    
    def __init__(self, task="classification", **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", f"knn_{task}"),
            task=task,
            name=f"KNN_{task.capitalize()}",
            version=kwargs.get("version", "1.0.0")
        )
        
        if SKLEARN_AVAILABLE:
            if task == "classification":
                self.model = KNeighborsClassifier(
                    n_neighbors=kwargs.get("n_neighbors", 5),
                    **{k: v for k, v in kwargs.items() 
                       if k not in ['model_key', 'version', 'task', 'n_neighbors']}
                )
            else:  # regression
                self.model = KNeighborsRegressor(
                    n_neighbors=kwargs.get("n_neighbors", 5),
                    **{k: v for k, v in kwargs.items() 
                       if k not in ['model_key', 'version', 'task', 'n_neighbors']}
                )
        else:
            self.model = MockModel(f"knn_{task}")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("KNN requires target values")
        
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task == "classification" and SKLEARN_AVAILABLE:
            return self.model.predict_proba(X)
        elif not SKLEARN_AVAILABLE and self.task == "classification":
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.random.rand(n_samples, 2)
        
        return None


class DecisionTreeAdapter(BaseModelAdapter):
    """Decision Tree adapter."""
    
    def __init__(self, task="classification", **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", f"tree_{task}"),
            task=task,
            name=f"DecisionTree_{task.capitalize()}",
            version=kwargs.get("version", "1.0.0")
        )
        
        if SKLEARN_AVAILABLE:
            if task == "classification":
                self.model = DecisionTreeClassifier(
                    random_state=42,
                    **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version', 'task']}
                )
            else:  # regression
                self.model = DecisionTreeRegressor(
                    random_state=42,
                    **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version', 'task']}
                )
        else:
            self.model = MockModel(f"tree_{task}")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("Decision Tree requires target values")
        
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task == "classification" and SKLEARN_AVAILABLE:
            return self.model.predict_proba(X)
        elif not SKLEARN_AVAILABLE and self.task == "classification":
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.random.rand(n_samples, 2)
        
        return None


class XGBAdapter(BaseModelAdapter):
    """XGBoost adapter."""
    
    def __init__(self, task="classification", **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", f"xgb_{task}"),
            task=task,
            name=f"XGBoost_{task.capitalize()}",
            version=kwargs.get("version", "1.0.0")
        )
        
        if XGBOOST_AVAILABLE:
            if task == "classification":
                self.model = xgb.XGBClassifier(
                    random_state=42,
                    **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version', 'task']}
                )
            else:  # regression
                self.model = xgb.XGBRegressor(
                    random_state=42,
                    **{k: v for k, v in kwargs.items() if k not in ['model_key', 'version', 'task']}
                )
        else:
            self.model = MockModel(f"xgb_{task}")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("XGBoost requires target values")
        
        self.model.fit(X, y, **kwargs)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task == "classification" and XGBOOST_AVAILABLE:
            return self.model.predict_proba(X)
        elif not XGBOOST_AVAILABLE and self.task == "classification":
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.random.rand(n_samples, 2)
        
        return None


class NaiveBayesAdapter(BaseModelAdapter):
    """Naive Bayes adapter."""
    
    def __init__(self, variant="gaussian", **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", f"nb_{variant}"),
            task="classification",
            name=f"NaiveBayes_{variant.capitalize()}",
            version=kwargs.get("version", "1.0.0")
        )
        
        if SKLEARN_AVAILABLE:
            if variant == "gaussian":
                self.model = GaussianNB(
                    **{k: v for k, v in kwargs.items() 
                       if k not in ['model_key', 'version', 'variant']}
                )
            elif variant == "multinomial":
                self.model = MultinomialNB(
                    **{k: v for k, v in kwargs.items() 
                       if k not in ['model_key', 'version', 'variant']}
                )
            else:
                raise ValueError(f"Unknown Naive Bayes variant: {variant}")
        else:
            self.model = MockModel(f"nb_{variant}")
    
    def fit(self, X, y=None, **kwargs):
        if y is None:
            raise ValueError("Naive Bayes requires target labels")
        
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if SKLEARN_AVAILABLE:
            return self.model.predict_proba(X)
        else:
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.random.rand(n_samples, 2)


class MockModel:
    """Mock model for when dependencies are not available."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.fitted = False
    
    def fit(self, X, y=None, **kwargs):
        self.fitted = True
        logger.info(f"Mock {self.model_type} model fitted (no-op)")
        return self
    
    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(X) if hasattr(X, '__len__') else 1
        
        # Generate mock predictions based on model type
        if "classification" in self.model_type or "nb_" in self.model_type:
            return np.random.choice([0, 1], size=n_samples)
        else:  # regression
            return np.random.randn(n_samples)
    
    def predict_proba(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(X) if hasattr(X, '__len__') else 1
        probs = np.random.rand(n_samples, 2)
        # Normalize to sum to 1
        return probs / probs.sum(axis=1, keepdims=True)
    
    def get_params(self):
        return {"model_type": self.model_type, "mock": True}