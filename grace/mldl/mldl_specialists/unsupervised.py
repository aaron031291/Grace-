"""
Unsupervised Learning Specialists

Clustering, dimensionality reduction, and representation learning
without labeled data.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .base_specialist import (
    BaseMLDLSpecialist,
    SpecialistCapability,
    SpecialistPrediction,
    TrainingMetrics
)


class KMeansSpecialist(BaseMLDLSpecialist):
    """
    K-Means Clustering Specialist - Partitioning into K clusters
    
    Strengths:
        - Simple and fast
        - Scales well to large datasets
        - Guaranteed convergence
    
    Grace Integration:
        - Cluster quality metrics tracked
        - Centroid positions auditable
        - Cluster assignments governable
    """
    
    def __init__(self, n_clusters=3, **kwargs):
        super().__init__(
            specialist_id="kmeans_specialist",
            specialist_type="KMeans",
            capabilities=[SpecialistCapability.CLUSTERING],
            **kwargs
        )
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        self.cluster_centers_ = None
    
    async def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None, **kwargs) -> TrainingMetrics:
        """Train K-Means clustering model"""
        start_time = datetime.now()
        
        # Fit the model
        self.model.fit(X_train)
        self.is_trained = True
        self.cluster_centers_ = self.model.cluster_centers_
        
        # Calculate clustering quality metrics
        labels = self.model.labels_
        
        # Silhouette score (-1 to 1, higher is better)
        silhouette = silhouette_score(X_train, labels) if len(np.unique(labels)) > 1 else 0.0
        
        # Davies-Bouldin Index (lower is better, 0 is best)
        db_index = davies_bouldin_score(X_train, labels) if len(np.unique(labels)) > 1 else 0.0
        
        # Calinski-Harabasz Index (higher is better)
        ch_index = calinski_harabasz_score(X_train, labels) if len(np.unique(labels)) > 1 else 0.0
        
        # Inertia (sum of squared distances to nearest cluster center)
        inertia = self.model.inertia_
        
        metrics = TrainingMetrics(
            custom_metrics={
                "silhouette_score": float(silhouette),
                "davies_bouldin_index": float(db_index),
                "calinski_harabasz_index": float(ch_index),
                "inertia": float(inertia),
                "n_clusters": self.n_clusters,
                "n_iterations": self.model.n_iter_
            }
        )
        
        self.training_history.append(metrics)
        
        # Log to immutable trail
        await self.log_to_immutable_trail(
            operation_type="training",
            operation_data={
                "samples_trained": len(X_train),
                "n_clusters": self.n_clusters,
                "metrics": metrics.__dict__,
                "cluster_sizes": np.bincount(labels).tolist()
            }
        )
        
        # Report to KPI monitor
        await self.report_kpi_metrics(metrics, operation="training")
        
        return metrics
    
    async def predict(self, X: np.ndarray, **kwargs) -> SpecialistPrediction:
        """Assign samples to clusters"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = datetime.now()
        
        # Predict cluster assignments
        cluster_labels = self.model.predict(X)
        
        # Calculate distances to cluster centers for confidence
        distances = self.model.transform(X)
        min_distances = np.min(distances, axis=1)
        
        # Confidence based on proximity to cluster center
        # Closer to center = higher confidence
        max_dist = np.max(distances)
        confidence = float((1.0 - (min_distances.mean() / (max_dist + 1e-6))))
        confidence = max(0.5, min(1.0, confidence))
        
        reasoning = f"K-Means with {self.n_clusters} clusters, "
        reasoning += f"average distance to center: {min_distances.mean():.3f}"
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Validate governance
        compliance = await self.validate_governance(X, cluster_labels)
        
        result = SpecialistPrediction(
            specialist_id=self.specialist_id,
            specialist_type=self.specialist_type,
            prediction=cluster_labels.tolist(),
            confidence=confidence,
            reasoning=reasoning,
            capabilities_used=[SpecialistCapability.CLUSTERING],
            execution_time_ms=execution_time,
            model_version=self.model_version,
            constitutional_compliance=compliance,
            trust_score=self.current_trust_score,
            metadata={
                "n_clusters": self.n_clusters,
                "cluster_centers": self.cluster_centers_.tolist(),
                "distances_to_centers": min_distances.tolist()
            }
        )
        
        await self.log_to_immutable_trail(
            operation_type="prediction",
            operation_data={
                "samples": X.shape[0],
                "confidence": confidence,
                "compliance": compliance,
                "cluster_distribution": np.bincount(cluster_labels).tolist()
            }
        )
        
        self.prediction_count += 1
        return result


class DBSCANSpecialist(BaseMLDLSpecialist):
    """
    DBSCAN Specialist - Density-Based Spatial Clustering
    
    Strengths:
        - Finds arbitrarily shaped clusters
        - Robust to outliers
        - No need to specify number of clusters
    
    Grace Integration:
        - Outlier detection tracked
        - Density parameters logged
        - Cluster discovery auditable
    """
    
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        super().__init__(
            specialist_id="dbscan_specialist",
            specialist_type="DBSCAN",
            capabilities=[
                SpecialistCapability.CLUSTERING,
                SpecialistCapability.ANOMALY_DETECTION
            ],
            **kwargs
        )
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.n_clusters_found_ = 0
        self.n_outliers_ = 0
    
    async def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None, **kwargs) -> TrainingMetrics:
        """Train DBSCAN clustering model"""
        start_time = datetime.now()
        
        # Fit the model
        labels = self.model.fit_predict(X_train)
        self.is_trained = True
        
        # Count clusters (excluding noise points labeled as -1)
        self.n_clusters_found_ = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_outliers_ = np.sum(labels == -1)
        
        # Calculate clustering quality metrics (only for non-noise points)
        non_noise_mask = labels != -1
        
        if self.n_clusters_found_ > 1 and non_noise_mask.sum() > 0:
            X_clustered = X_train[non_noise_mask]
            labels_clustered = labels[non_noise_mask]
            
            silhouette = silhouette_score(X_clustered, labels_clustered)
            db_index = davies_bouldin_score(X_clustered, labels_clustered)
            ch_index = calinski_harabasz_score(X_clustered, labels_clustered)
        else:
            silhouette = 0.0
            db_index = 0.0
            ch_index = 0.0
        
        metrics = TrainingMetrics(
            custom_metrics={
                "silhouette_score": float(silhouette),
                "davies_bouldin_index": float(db_index),
                "calinski_harabasz_index": float(ch_index),
                "n_clusters_found": self.n_clusters_found_,
                "n_outliers": self.n_outliers_,
                "outlier_ratio": float(self.n_outliers_ / len(X_train)),
                "eps": self.eps,
                "min_samples": self.min_samples
            }
        )
        
        self.training_history.append(metrics)
        
        # Store labels for reference
        self.labels_ = labels
        
        await self.log_to_immutable_trail(
            operation_type="training",
            operation_data={
                "samples_trained": len(X_train),
                "n_clusters_found": self.n_clusters_found_,
                "n_outliers": self.n_outliers_,
                "metrics": metrics.__dict__
            }
        )
        
        await self.report_kpi_metrics(metrics, operation="training")
        
        return metrics
    
    async def predict(self, X: np.ndarray, **kwargs) -> SpecialistPrediction:
        """Predict cluster assignments (note: DBSCAN doesn't natively support prediction)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = datetime.now()
        
        # DBSCAN doesn't have a predict method, so we use fit_predict
        # This is a limitation - in production, you'd use a nearest neighbor approach
        cluster_labels = self.model.fit_predict(X)
        
        n_outliers = np.sum(cluster_labels == -1)
        outlier_ratio = n_outliers / len(X)
        
        # Confidence based on outlier ratio
        confidence = max(0.5, 1.0 - outlier_ratio)
        
        reasoning = f"DBSCAN found {self.n_clusters_found_} clusters, "
        reasoning += f"{n_outliers} outliers ({outlier_ratio:.1%})"
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        compliance = await self.validate_governance(X, cluster_labels)
        
        result = SpecialistPrediction(
            specialist_id=self.specialist_id,
            specialist_type=self.specialist_type,
            prediction=cluster_labels.tolist(),
            confidence=confidence,
            reasoning=reasoning,
            capabilities_used=[SpecialistCapability.CLUSTERING, SpecialistCapability.ANOMALY_DETECTION],
            execution_time_ms=execution_time,
            model_version=self.model_version,
            constitutional_compliance=compliance,
            trust_score=self.current_trust_score,
            metadata={
                "n_clusters": self.n_clusters_found_,
                "n_outliers": int(n_outliers),
                "outlier_ratio": float(outlier_ratio),
                "eps": self.eps,
                "min_samples": self.min_samples
            }
        )
        
        await self.log_to_immutable_trail(
            operation_type="prediction",
            operation_data={
                "samples": X.shape[0],
                "confidence": confidence,
                "compliance": compliance,
                "n_outliers": int(n_outliers)
            }
        )
        
        self.prediction_count += 1
        return result


class PCASpecialist(BaseMLDLSpecialist):
    """
    PCA Specialist - Principal Component Analysis for dimensionality reduction
    
    Strengths:
        - Reduces dimensionality while preserving variance
        - Removes correlated features
        - Useful for visualization
    
    Grace Integration:
        - Variance explained tracked
        - Component loadings auditable
        - Transformation reversible for transparency
    """
    
    def __init__(self, n_components=None, **kwargs):
        super().__init__(
            specialist_id="pca_specialist",
            specialist_type="PCA",
            capabilities=[SpecialistCapability.DIMENSIONALITY_REDUCTION],
            **kwargs
        )
        self.n_components = n_components or 0.95  # Default: preserve 95% variance
        self.model = PCA(n_components=self.n_components, random_state=42)
        self.explained_variance_ratio_ = None
    
    async def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None, **kwargs) -> TrainingMetrics:
        """Train PCA model"""
        start_time = datetime.now()
        
        # Fit the model
        self.model.fit(X_train)
        self.is_trained = True
        
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        
        # Calculate metrics
        total_variance_explained = self.explained_variance_ratio_.sum()
        n_components_selected = len(self.explained_variance_ratio_)
        
        metrics = TrainingMetrics(
            custom_metrics={
                "n_components": n_components_selected,
                "variance_explained": float(total_variance_explained),
                "original_dimensions": X_train.shape[1],
                "reduced_dimensions": n_components_selected,
                "compression_ratio": float(n_components_selected / X_train.shape[1]),
                "singular_values": self.model.singular_values_.tolist()
            }
        )
        
        self.training_history.append(metrics)
        
        await self.log_to_immutable_trail(
            operation_type="training",
            operation_data={
                "samples_trained": len(X_train),
                "n_components": n_components_selected,
                "variance_explained": float(total_variance_explained),
                "metrics": metrics.__dict__
            }
        )
        
        await self.report_kpi_metrics(metrics, operation="training")
        
        return metrics
    
    async def predict(self, X: np.ndarray, **kwargs) -> SpecialistPrediction:
        """Transform data to reduced dimensions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = datetime.now()
        
        # Transform to lower dimensions
        X_transformed = self.model.transform(X)
        
        # Can also inverse transform to check reconstruction quality
        X_reconstructed = self.model.inverse_transform(X_transformed)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        
        # Confidence based on variance preserved and reconstruction quality
        variance_explained = self.explained_variance_ratio_.sum()
        confidence = float(variance_explained * (1.0 - min(reconstruction_error, 0.5)))
        confidence = max(0.5, min(1.0, confidence))
        
        reasoning = f"PCA reduced from {X.shape[1]} to {X_transformed.shape[1]} dimensions, "
        reasoning += f"preserving {variance_explained:.1%} variance"
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        compliance = await self.validate_governance(X, X_transformed)
        
        result = SpecialistPrediction(
            specialist_id=self.specialist_id,
            specialist_type=self.specialist_type,
            prediction=X_transformed.tolist(),
            confidence=confidence,
            reasoning=reasoning,
            capabilities_used=[SpecialistCapability.DIMENSIONALITY_REDUCTION],
            execution_time_ms=execution_time,
            model_version=self.model_version,
            constitutional_compliance=compliance,
            trust_score=self.current_trust_score,
            metadata={
                "n_components": len(self.explained_variance_ratio_),
                "variance_explained": float(variance_explained),
                "reconstruction_error": float(reconstruction_error),
                "explained_variance_ratio": self.explained_variance_ratio_.tolist()
            }
        )
        
        await self.log_to_immutable_trail(
            operation_type="prediction",
            operation_data={
                "samples": X.shape[0],
                "confidence": confidence,
                "compliance": compliance,
                "dimensions_before": X.shape[1],
                "dimensions_after": X_transformed.shape[1]
            }
        )
        
        self.prediction_count += 1
        return result


class AutoencoderSpecialist(BaseMLDLSpecialist):
    """
    Autoencoder Specialist - Neural network for representation learning
    
    Strengths:
        - Non-linear dimensionality reduction
        - Learns complex representations
        - Can detect anomalies via reconstruction error
    
    Grace Integration:
        - Encoder/decoder weights auditable
        - Reconstruction quality tracked
        - Training convergence monitored
    
    Note: This is a simple implementation. For production, consider using
    PyTorch or TensorFlow for more sophisticated architectures.
    """
    
    def __init__(self, encoding_dim=32, **kwargs):
        super().__init__(
            specialist_id="autoencoder_specialist",
            specialist_type="Autoencoder",
            capabilities=[
                SpecialistCapability.DIMENSIONALITY_REDUCTION,
                SpecialistCapability.ANOMALY_DETECTION,
                SpecialistCapability.FEATURE_EXTRACTION
            ],
            **kwargs
        )
        self.encoding_dim = encoding_dim
        
        # Simple autoencoder using numpy (for production, use PyTorch/TF)
        self.encoder_weights = None
        self.encoder_bias = None
        self.decoder_weights = None
        self.decoder_bias = None
        
        self.input_dim = None
        self.reconstruction_threshold = None
    
    def _sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _encode(self, X):
        """Encode input to latent representation"""
        return self._sigmoid(np.dot(X, self.encoder_weights) + self.encoder_bias)
    
    def _decode(self, encoded):
        """Decode latent representation back to input space"""
        return self._sigmoid(np.dot(encoded, self.decoder_weights) + self.decoder_bias)
    
    async def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None, **kwargs) -> TrainingMetrics:
        """Train autoencoder"""
        start_time = datetime.now()
        
        self.input_dim = X_train.shape[1]
        
        # Initialize weights
        np.random.seed(42)
        self.encoder_weights = np.random.randn(self.input_dim, self.encoding_dim) * 0.01
        self.encoder_bias = np.zeros(self.encoding_dim)
        self.decoder_weights = np.random.randn(self.encoding_dim, self.input_dim) * 0.01
        self.decoder_bias = np.zeros(self.input_dim)
        
        # Simple gradient descent training
        learning_rate = kwargs.get("learning_rate", 0.01)
        epochs = kwargs.get("epochs", 100)
        
        for epoch in range(epochs):
            # Forward pass
            encoded = self._encode(X_train)
            decoded = self._decode(encoded)
            
            # Calculate reconstruction error
            error = X_train - decoded
            loss = np.mean(error ** 2)
            
            # Backward pass (simplified)
            # Update decoder
            decoder_grad = -2 * np.dot(encoded.T, error * decoded * (1 - decoded)) / len(X_train)
            self.decoder_weights -= learning_rate * decoder_grad
            self.decoder_bias -= learning_rate * np.mean(error * decoded * (1 - decoded), axis=0)
            
            # Update encoder
            decoder_error = np.dot(error * decoded * (1 - decoded), self.decoder_weights.T)
            encoder_grad = -2 * np.dot(X_train.T, decoder_error * encoded * (1 - encoded)) / len(X_train)
            self.encoder_weights -= learning_rate * encoder_grad
            self.encoder_bias -= learning_rate * np.mean(decoder_error * encoded * (1 - encoded), axis=0)
        
        self.is_trained = True
        
        # Calculate final reconstruction error
        encoded = self._encode(X_train)
        decoded = self._decode(encoded)
        reconstruction_error = np.mean((X_train - decoded) ** 2, axis=1)
        self.reconstruction_threshold = np.percentile(reconstruction_error, 95)
        
        metrics = TrainingMetrics(
            custom_metrics={
                "encoding_dim": self.encoding_dim,
                "input_dim": self.input_dim,
                "final_loss": float(np.mean(reconstruction_error)),
                "reconstruction_threshold": float(self.reconstruction_threshold),
                "compression_ratio": float(self.encoding_dim / self.input_dim)
            }
        )
        
        self.training_history.append(metrics)
        
        await self.log_to_immutable_trail(
            operation_type="training",
            operation_data={
                "samples_trained": len(X_train),
                "encoding_dim": self.encoding_dim,
                "metrics": metrics.__dict__
            }
        )
        
        await self.report_kpi_metrics(metrics, operation="training")
        
        return metrics
    
    async def predict(self, X: np.ndarray, **kwargs) -> SpecialistPrediction:
        """Encode data and optionally detect anomalies"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = datetime.now()
        
        # Encode
        encoded = self._encode(X)
        
        # Decode for reconstruction error (anomaly detection)
        decoded = self._decode(encoded)
        reconstruction_errors = np.mean((X - decoded) ** 2, axis=1)
        
        # Detect anomalies
        is_anomaly = reconstruction_errors > self.reconstruction_threshold
        n_anomalies = np.sum(is_anomaly)
        
        # Confidence based on reconstruction quality
        avg_error = np.mean(reconstruction_errors)
        confidence = max(0.5, 1.0 - min(avg_error / (self.reconstruction_threshold + 1e-6), 0.5))
        
        reasoning = f"Autoencoder: {self.input_dim}→{self.encoding_dim} dimensions, "
        reasoning += f"{n_anomalies} anomalies detected"
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        compliance = await self.validate_governance(X, encoded)
        
        result = SpecialistPrediction(
            specialist_id=self.specialist_id,
            specialist_type=self.specialist_type,
            prediction=encoded.tolist(),
            confidence=confidence,
            reasoning=reasoning,
            capabilities_used=[
                SpecialistCapability.DIMENSIONALITY_REDUCTION,
                SpecialistCapability.ANOMALY_DETECTION
            ],
            execution_time_ms=execution_time,
            model_version=self.model_version,
            constitutional_compliance=compliance,
            trust_score=self.current_trust_score,
            metadata={
                "encoding_dim": self.encoding_dim,
                "reconstruction_errors": reconstruction_errors.tolist(),
                "anomalies": is_anomaly.tolist(),
                "n_anomalies": int(n_anomalies)
            }
        )
        
        await self.log_to_immutable_trail(
            operation_type="prediction",
            operation_data={
                "samples": X.shape[0],
                "confidence": confidence,
                "compliance": compliance,
                "n_anomalies": int(n_anomalies)
            }
        )
        
        self.prediction_count += 1
        return result
