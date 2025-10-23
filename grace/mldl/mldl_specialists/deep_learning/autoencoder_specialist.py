"""
Autoencoder Specialist - Unsupervised Representation Learning

Use cases:
- Anomaly detection (reconstruction error)
- Dimensionality reduction
- Feature learning
- Denoising
- Data compression
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .base_deep_specialist import BaseDeepLearningSpecialist
from ..base_specialist import SpecialistCapability

logger = logging.getLogger(__name__)


class AutoencoderNetwork(nn.Module):
    """Autoencoder architecture"""
    
    def __init__(
        self,
        input_size: int,
        latent_dim: int = 32,
        hidden_sizes: List[int] = [128, 64]
    ):
        super(AutoencoderNetwork, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Latent layer
        encoder_layers.append(nn.Linear(prev_size, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_size = latent_dim
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass"""
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def encode(self, x):
        """Encode to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)


class AutoencoderSpecialist(BaseDeepLearningSpecialist):
    """
    Autoencoder Specialist for unsupervised learning.
    
    Use cases:
    - Anomaly detection via reconstruction error
    - Non-linear dimensionality reduction
    - Feature learning and extraction
    - Data denoising
    - Compression for storage/transmission
    
    Features:
    - Configurable latent dimension
    - Anomaly detection threshold tuning
    - Latent space visualization
    - Denoising capability
    """
    
    def __init__(
        self,
        specialist_id: str = "autoencoder_specialist",
        latent_dim: int = 32,
        hidden_sizes: List[int] = [128, 64],
        anomaly_threshold: float = 0.05,
        device: Optional[str] = None
    ):
        capabilities = [
            SpecialistCapability.ANOMALY_DETECTION,
            SpecialistCapability.DIMENSIONALITY_REDUCTION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes
        self.anomaly_threshold = anomaly_threshold
        self.input_size: Optional[int] = None
        
        # For anomaly detection
        self.reconstruction_errors: List[float] = []
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build autoencoder model"""
        if len(input_shape) == 1:
            self.input_size = input_shape[0]
        else:
            self.input_size = int(np.prod(input_shape))
        
        self.latent_dim = kwargs.get('latent_dim', self.latent_dim)
        
        model = AutoencoderNetwork(
            input_size=self.input_size,
            latent_dim=self.latent_dim,
            hidden_sizes=self.hidden_sizes
        )
        
        # Use MSE loss for reconstruction
        self.criterion = nn.MSELoss()
        
        logger.info(f"Built Autoencoder: input_size={self.input_size}, "
                   f"latent_dim={self.latent_dim}")
        
        return model
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to latent space"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_tensor)
        
        return latent.cpu().numpy()
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct data"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
        
        return reconstructed.cpu().numpy()
    
    def detect_anomaly(self, X: np.ndarray) -> Tuple[bool, float]:
        """
        Detect anomaly based on reconstruction error
        
        Args:
            X: Input data
        
        Returns:
            is_anomaly, reconstruction_error
        """
        # Reconstruct
        X_reconstructed = self.reconstruct(X.reshape(1, -1))
        
        # Calculate reconstruction error
        error = np.mean((X - X_reconstructed.flatten()) ** 2)
        
        # Compare to threshold
        is_anomaly = error > self.anomaly_threshold
        
        return is_anomaly, float(error)
    
    def calibrate_threshold(self, X_normal: np.ndarray, percentile: float = 95):
        """
        Calibrate anomaly threshold using normal data
        
        Args:
            X_normal: Normal (non-anomalous) data
            percentile: Percentile for threshold (default 95th)
        """
        # Calculate reconstruction errors
        X_reconstructed = self.reconstruct(X_normal)
        errors = np.mean((X_normal - X_reconstructed) ** 2, axis=1)
        
        # Set threshold at percentile
        self.anomaly_threshold = float(np.percentile(errors, percentile))
        
        logger.info(f"Calibrated anomaly threshold: {self.anomaly_threshold:.6f} "
                   f"(at {percentile}th percentile)")
    
    def _process_output(self, output) -> Tuple[Any, float]:
        """Process autoencoder output"""
        reconstructed, latent = output
        
        # Return latent representation
        latent_np = latent.cpu().numpy().flatten()
        confidence = 0.8
        
        return latent_np.tolist(), confidence
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> 'SpecialistPrediction':
        """
        Make autoencoder prediction
        
        Expected input_data:
        {
            'features': array,
            'task': 'encode' | 'reconstruct' | 'detect_anomaly'
        }
        """
        if not self.is_trained:
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            features = np.array(self._extract_features(input_data))
            task = input_data.get('task', 'encode')
            
            if task == 'encode':
                # Encode to latent space
                latent = self.encode(features.reshape(1, -1))
                prediction_value = latent.flatten().tolist()
                confidence = 1.0
                metadata = {
                    'task': 'encode',
                    'latent_dim': self.latent_dim
                }
            
            elif task == 'reconstruct':
                # Reconstruct input
                reconstructed = self.reconstruct(features.reshape(1, -1))
                prediction_value = reconstructed.flatten().tolist()
                
                # Calculate reconstruction error
                error = np.mean((features - reconstructed.flatten()) ** 2)
                confidence = max(0.0, 1.0 - error)
                metadata = {
                    'task': 'reconstruct',
                    'reconstruction_error': float(error)
                }
            
            else:  # detect_anomaly
                is_anomaly, error = self.detect_anomaly(features)
                prediction_value = "anomaly" if is_anomaly else "normal"
                confidence = min(1.0, error / self.anomaly_threshold) if is_anomaly else max(0.0, 1.0 - error / self.anomaly_threshold)
                metadata = {
                    'task': 'anomaly_detection',
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': error,
                    'threshold': self.anomaly_threshold
                }
            
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=prediction_value,
                confidence=float(confidence),
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Autoencoder prediction failed: {e}")
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
