"""
Clustering and dimensionality reduction model adapters.
"""

import numpy as np
from typing import Optional
import logging

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "Scikit-learn not available. Clustering/DimRed adapters will use mock implementations."
    )


class KMeansAdapter(BaseModelAdapter):
    """K-Means clustering adapter."""

    def __init__(self, **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", "kmeans_clustering"),
            task="clustering",
            name="KMeans",
            version=kwargs.get("version", "1.0.0"),
        )

        if SKLEARN_AVAILABLE:
            self.model = KMeans(
                n_clusters=kwargs.get("n_clusters", 8),
                random_state=42,
                n_init=10,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model_key", "version", "n_clusters"]
                },
            )
        else:
            self.model = MockClusteringModel("kmeans", kwargs.get("n_clusters", 8))

    def fit(self, X, y=None, **kwargs):
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs

        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        # Store cluster centers if available
        if SKLEARN_AVAILABLE and hasattr(self.model, "cluster_centers_"):
            self.metadata["cluster_centers"] = self.model.cluster_centers_.tolist()
            self.metadata["inertia"] = self.model.inertia_

        return self

    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def fit_predict(self, X, **kwargs):
        """Fit model and predict clusters in one step."""
        self.fit(X, **kwargs)
        return self.predict(X)

    def get_cluster_centers(self):
        """Get cluster centers if available."""
        if (
            self.fitted
            and SKLEARN_AVAILABLE
            and hasattr(self.model, "cluster_centers_")
        ):
            return self.model.cluster_centers_
        return None


class AgglomerativeClusteringAdapter(BaseModelAdapter):
    """Agglomerative (Hierarchical) clustering adapter."""

    def __init__(self, **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", "agglo_clustering"),
            task="clustering",
            name="AgglomerativeClustering",
            version=kwargs.get("version", "1.0.0"),
        )

        if SKLEARN_AVAILABLE:
            self.model = AgglomerativeClustering(
                n_clusters=kwargs.get("n_clusters", 8),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model_key", "version", "n_clusters"]
                },
            )
        else:
            self.model = MockClusteringModel(
                "agglomerative", kwargs.get("n_clusters", 8)
            )

    def fit(self, X, y=None, **kwargs):
        # Agglomerative clustering doesn't have separate fit/predict
        self.labels_ = self.model.fit_predict(X)
        self.fitted = True
        self.metadata["training_params"] = kwargs

        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        return self

    def predict(self, X, **kwargs):
        # Agglomerative clustering doesn't support prediction on new data
        # Return labels from fitting
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        if hasattr(self, "labels_"):
            return self.labels_
        else:
            logger.warning(
                "Agglomerative clustering doesn't support prediction on new data"
            )
            return self.model.fit_predict(X)

    def fit_predict(self, X, **kwargs):
        """Fit model and predict clusters in one step."""
        self.fit(X, **kwargs)
        return self.labels_


class DBSCANAdapter(BaseModelAdapter):
    """DBSCAN clustering adapter."""

    def __init__(self, **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", "dbscan_clustering"),
            task="clustering",
            name="DBSCAN",
            version=kwargs.get("version", "1.0.0"),
        )

        if SKLEARN_AVAILABLE:
            self.model = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model_key", "version", "eps", "min_samples"]
                },
            )
        else:
            self.model = MockClusteringModel("dbscan")

    def fit(self, X, y=None, **kwargs):
        # DBSCAN doesn't have separate fit/predict
        self.labels_ = self.model.fit_predict(X)
        self.fitted = True
        self.metadata["training_params"] = kwargs

        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        # Store DBSCAN-specific metrics
        if SKLEARN_AVAILABLE:
            n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            n_noise = list(self.labels_).count(-1)
            self.metadata["n_clusters"] = n_clusters
            self.metadata["n_noise_points"] = n_noise

        return self

    def predict(self, X, **kwargs):
        # DBSCAN doesn't support prediction on new data
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        if hasattr(self, "labels_"):
            return self.labels_
        else:
            logger.warning("DBSCAN doesn't support prediction on new data")
            return self.model.fit_predict(X)

    def fit_predict(self, X, **kwargs):
        """Fit model and predict clusters in one step."""
        self.fit(X, **kwargs)
        return self.labels_


class PCAAdapter(BaseModelAdapter):
    """Principal Component Analysis adapter."""

    def __init__(self, **kwargs):
        super().__init__(
            model_key=kwargs.get("model_key", "pca_dimred"),
            task="dimred",
            name="PCA",
            version=kwargs.get("version", "1.0.0"),
        )

        if SKLEARN_AVAILABLE:
            self.model = PCA(
                n_components=kwargs.get("n_components", None),
                random_state=42,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model_key", "version", "n_components"]
                },
            )
        else:
            self.model = MockDimRedModel("pca", kwargs.get("n_components"))

    def fit(self, X, y=None, **kwargs):
        self.model.fit(X, y)
        self.fitted = True
        self.metadata["training_params"] = kwargs

        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        # Store PCA-specific metrics
        if SKLEARN_AVAILABLE:
            self.metadata["explained_variance_ratio"] = (
                self.model.explained_variance_ratio_.tolist()
            )
            self.metadata["explained_variance"] = (
                self.model.explained_variance_.tolist()
            )
            self.metadata["n_components"] = self.model.n_components_
            self.metadata["cumulative_variance"] = np.cumsum(
                self.model.explained_variance_ratio_
            ).tolist()

        return self

    def predict(self, X, **kwargs):
        """Transform data to reduced dimensions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.transform(X, **kwargs)

    def transform(self, X, **kwargs):
        """Transform data to reduced dimensions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before transformation")

        return self.model.transform(X)

    def inverse_transform(self, X_transformed, **kwargs):
        """Transform data back to original dimensions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before inverse transformation")

        if SKLEARN_AVAILABLE:
            return self.model.inverse_transform(X_transformed)
        else:
            # Mock inverse transform
            original_shape = (
                X_transformed.shape[0],
                len(self.feature_names)
                if self.feature_names
                else X_transformed.shape[1] * 2,
            )
            return np.random.randn(*original_shape)

    def fit_transform(self, X, y=None, **kwargs):
        """Fit model and transform data in one step."""
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def get_explained_variance_ratio(self):
        """Get explained variance ratio if available."""
        if (
            self.fitted
            and SKLEARN_AVAILABLE
            and hasattr(self.model, "explained_variance_ratio_")
        ):
            return self.model.explained_variance_ratio_
        return None

    def get_components(self):
        """Get principal components if available."""
        if self.fitted and SKLEARN_AVAILABLE and hasattr(self.model, "components_"):
            return self.model.components_
        return None


class MockClusteringModel:
    """Mock clustering model for when dependencies are not available."""

    def __init__(self, model_type: str, n_clusters: int = 8):
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.fitted = False

    def fit(self, X, y=None, **kwargs):
        self.fitted = True
        logger.info(f"Mock {self.model_type} clustering model fitted (no-op)")
        return self

    def predict(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        n_samples = len(X) if hasattr(X, "__len__") else 1

        if self.model_type == "dbscan":
            # DBSCAN can have noise points (-1)
            clusters = list(range(self.n_clusters)) + [-1]
            return np.random.choice(clusters, size=n_samples)
        else:
            return np.random.choice(range(self.n_clusters), size=n_samples)

    def fit_predict(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.predict(X)


class MockDimRedModel:
    """Mock dimensionality reduction model for when dependencies are not available."""

    def __init__(self, model_type: str, n_components: Optional[int] = None):
        self.model_type = model_type
        self.n_components = n_components
        self.fitted = False

    def fit(self, X, y=None, **kwargs):
        self.fitted = True

        # Infer n_components if not specified
        if self.n_components is None:
            if hasattr(X, "shape"):
                self.n_components = min(X.shape[1], 10)  # Default to min(features, 10)
            else:
                self.n_components = 5

        logger.info(
            f"Mock {self.model_type} dimensionality reduction model fitted (no-op)"
        )
        return self

    def transform(self, X, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before transformation")

        n_samples = len(X) if hasattr(X, "__len__") else 1
        return np.random.randn(n_samples, self.n_components)

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def inverse_transform(self, X_transformed, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before inverse transformation")

        # Estimate original dimensions (mock)
        n_samples = X_transformed.shape[0]
        original_features = X_transformed.shape[1] * 2  # Mock expansion
        return np.random.randn(n_samples, original_features)
