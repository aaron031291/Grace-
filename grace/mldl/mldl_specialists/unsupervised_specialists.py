"""
Unsupervised Learning Specialists - Layer 1

Individual ML models for unsupervised learning tasks:
- K-Means Clustering Specialist
- DBSCAN Clustering Specialist
- PCA Dimensionality Reduction Specialist
- Isolation Forest Anomaly Detection Specialist
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

from .base_specialist import BaseSpecialist, SpecialistCapability, SpecialistPrediction

logger = logging.getLogger(__name__)


class KMeansClusteringSpecialist(BaseSpecialist):
    """
    K-Means Clustering Specialist - Unsupervised pattern grouping.
    
    Use cases:
    - User behavior segmentation
    - KPI pattern clustering
    - Component grouping by similarity
    - Trust score bucketing
    """
    
    def __init__(
        self,
        specialist_id: str = "kmeans_clustering_specialist",
        n_clusters: int = 5
    ):
        capabilities = [
            SpecialistCapability.CLUSTERING,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities)
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cluster_labels: Optional[Dict[int, str]] = None
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Predict cluster assignment for input data."""
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
            
            # Predict cluster
            cluster = self.model.predict(X_scaled)[0]
            
            # Calculate confidence based on distance to cluster center
            distances = self.model.transform(X_scaled)[0]
            min_distance = distances[cluster]
            max_distance = np.max(distances)
            confidence = 1.0 - (min_distance / (max_distance + 1e-10))
            
            # Get cluster label if available
            cluster_label = self.cluster_labels.get(cluster, f"cluster_{cluster}") if self.cluster_labels else f"cluster_{cluster}"
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=cluster_label,
                confidence=float(confidence),
                metadata={
                    'cluster_id': int(cluster),
                    'distance_to_center': float(min_distance),
                    'n_clusters': self.model.n_clusters
                }
            )
            
        except Exception as e:
            logger.error(f"KMeansClusteringSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, cluster_labels: Optional[Dict[int, str]] = None):
        """Train the K-means model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.cluster_labels = cluster_labels
        self.is_trained = True
        logger.info(f"{self.specialist_id} trained with {len(X)} samples into {self.model.n_clusters} clusters")
    
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


class DBSCANClusteringSpecialist(BaseSpecialist):
    """
    DBSCAN Clustering Specialist - Density-based clustering with outlier detection.
    
    Use cases:
    - Anomaly detection (outliers = noise)
    - Arbitrary-shaped cluster discovery
    - Event pattern grouping
    """
    
    def __init__(
        self,
        specialist_id: str = "dbscan_clustering_specialist",
        eps: float = 0.5,
        min_samples: int = 5
    ):
        capabilities = [
            SpecialistCapability.CLUSTERING,
            SpecialistCapability.ANOMALY_DETECTION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities)
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cluster_centers: Optional[np.ndarray] = None
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Predict if input is anomaly or belongs to a cluster."""
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
            
            # Find nearest cluster center
            if self.cluster_centers is not None and len(self.cluster_centers) > 0:
                distances = np.linalg.norm(self.cluster_centers - X_scaled, axis=1)
                nearest_cluster = int(np.argmin(distances))
                min_distance = float(distances[nearest_cluster])
                
                # Determine if outlier based on distance threshold (eps)
                is_outlier = min_distance > self.model.eps
                
                if is_outlier:
                    prediction_value = "outlier"
                    confidence = min(1.0, min_distance / self.model.eps)
                else:
                    prediction_value = f"cluster_{nearest_cluster}"
                    confidence = max(0.0, 1.0 - min_distance / self.model.eps)
            else:
                prediction_value = "outlier"
                confidence = 0.5
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=prediction_value,
                confidence=confidence,
                metadata={
                    'is_outlier': is_outlier if self.cluster_centers is not None else True,
                    'eps': self.model.eps,
                    'min_samples': self.model.min_samples
                }
            )
            
        except Exception as e:
            logger.error(f"DBSCANClusteringSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray):
        """Train the DBSCAN model."""
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate cluster centers (excluding noise points labeled -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if unique_labels:
            centers = []
            for label in unique_labels:
                cluster_points = X_scaled[labels == label]
                center = np.mean(cluster_points, axis=0)
                centers.append(center)
            self.cluster_centers = np.array(centers)
        
        self.is_trained = True
        n_clusters = len(unique_labels)
        n_noise = list(labels).count(-1)
        logger.info(f"{self.specialist_id} trained: {n_clusters} clusters, {n_noise} noise points")
    
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


class PCADimensionalityReductionSpecialist(BaseSpecialist):
    """
    PCA Dimensionality Reduction Specialist - Feature compression and extraction.
    
    Use cases:
    - High-dimensional data compression
    - Feature extraction for visualization
    - Noise reduction
    - Signal compression (per user spec)
    """
    
    def __init__(
        self,
        specialist_id: str = "pca_specialist",
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ):
        capabilities = [
            SpecialistCapability.DIMENSIONALITY_REDUCTION,
            SpecialistCapability.SIGNAL_COMPRESSION
        ]
        super().__init__(specialist_id, capabilities)
        
        # If n_components not specified, preserve 95% variance
        if n_components is None:
            self.model = PCA(n_components=variance_threshold, random_state=42)
        else:
            self.model = PCA(n_components=n_components, random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Transform input data to reduced dimensionality."""
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
            
            # Transform to reduced dimensions
            X_reduced = self.model.transform(X_scaled)[0]
            
            # Calculate confidence based on explained variance
            explained_variance_ratio = float(np.sum(self.model.explained_variance_ratio_))
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=X_reduced.tolist(),
                confidence=explained_variance_ratio,
                metadata={
                    'original_dims': len(features),
                    'reduced_dims': len(X_reduced),
                    'explained_variance': explained_variance_ratio,
                    'compression_ratio': len(features) / len(X_reduced)
                }
            )
            
        except Exception as e:
            logger.error(f"PCADimensionalityReductionSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray):
        """Train the PCA model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        
        explained_variance = np.sum(self.model.explained_variance_ratio_)
        logger.info(
            f"{self.specialist_id} trained: {X.shape[1]} -> {self.model.n_components_} dims, "
            f"{explained_variance*100:.1f}% variance retained"
        )
    
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


class IsolationForestAnomalySpecialist(BaseSpecialist):
    """
    Isolation Forest Anomaly Detection Specialist - Tree-based outlier detection.
    
    Use cases:
    - Security threat detection
    - Data quality anomaly detection
    - System behavior anomaly detection
    - Fraud detection
    """
    
    def __init__(
        self,
        specialist_id: str = "isolation_forest_specialist",
        contamination: float = 0.1,
        n_estimators: int = 100
    ):
        capabilities = [
            SpecialistCapability.ANOMALY_DETECTION,
            SpecialistCapability.TRUST_SCORING
        ]
        super().__init__(specialist_id, capabilities)
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Predict if input is an anomaly."""
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
            
            # Predict: 1 = normal, -1 = anomaly
            prediction = self.model.predict(X_scaled)[0]
            is_anomaly = (prediction == -1)
            
            # Get anomaly score (lower = more anomalous)
            anomaly_score = self.model.score_samples(X_scaled)[0]
            
            # Convert to confidence (higher = more confident in anomaly detection)
            # Normalize anomaly score to 0-1 range
            confidence = 1.0 / (1.0 + np.exp(anomaly_score))  # Sigmoid normalization
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value="anomaly" if is_anomaly else "normal",
                confidence=float(confidence),
                metadata={
                    'is_anomaly': is_anomaly,
                    'anomaly_score': float(anomaly_score),
                    'contamination': self.model.contamination,
                    'n_estimators': self.model.n_estimators
                }
            )
            
        except Exception as e:
            logger.error(f"IsolationForestAnomalySpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray):
        """Train the Isolation Forest model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info(f"{self.specialist_id} trained with {len(X)} samples, contamination={self.model.contamination}")
    
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
