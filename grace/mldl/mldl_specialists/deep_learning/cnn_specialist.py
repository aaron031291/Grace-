"""
CNN Specialist - Convolutional Neural Network for Image/Document Processing

Use cases:
- Document image classification
- OCR enhancement
- Visual governance artifact processing
- Diagram/flowchart analysis
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .base_deep_specialist import BaseDeepLearningSpecialist
from ..base_specialist import SpecialistCapability

logger = logging.getLogger(__name__)


class CNNNetwork(nn.Module):
    """Convolutional Neural Network architecture"""
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28
    ):
        super(CNNNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate size after convolutions and pooling
        # After 3 pooling layers: image_size / 2^3
        final_size = image_size // 8
        fc_input_size = 128 * final_size * final_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNSpecialist(BaseDeepLearningSpecialist):
    """
    CNN Specialist for image and document processing.
    
    Use cases:
    - Document image classification (contracts, policies, forms)
    - OCR preprocessing and enhancement
    - Diagram/flowchart analysis
    - Visual governance artifact processing
    - Scanned document quality assessment
    
    Features:
    - Handles grayscale and RGB images
    - Automatic image normalization
    - Data augmentation support
    - Transfer learning ready
    """
    
    def __init__(
        self,
        specialist_id: str = "cnn_specialist",
        num_classes: int = 10,
        input_channels: int = 1,
        image_size: int = 28,
        device: Optional[str] = None
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """
        Build CNN model
        
        Args:
            input_shape: (channels, height, width) or (height, width)
            **kwargs: Additional parameters (num_classes, etc.)
        
        Returns:
            CNN network
        """
        # Parse input shape
        if len(input_shape) == 3:
            self.input_channels, height, width = input_shape
            self.image_size = height
        elif len(input_shape) == 2:
            height, width = input_shape
            self.image_size = height
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")
        
        # Override with kwargs if provided
        self.num_classes = kwargs.get('num_classes', self.num_classes)
        
        model = CNNNetwork(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            image_size=self.image_size
        )
        
        # Use CrossEntropyLoss for classification
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Built CNN model: channels={self.input_channels}, "
                   f"image_size={self.image_size}, num_classes={self.num_classes}")
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN
        
        Args:
            image: Image array (H, W) or (H, W, C)
        
        Returns:
            Preprocessed image (C, H, W)
        """
        # Convert to float
        image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Handle different input formats
        if len(image.shape) == 2:
            # Grayscale (H, W) -> (1, H, W)
            image = image[np.newaxis, :, :]
        elif len(image.shape) == 3:
            # RGB (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))
        
        # Resize if needed
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            # Simple resize using nearest neighbor (for demonstration)
            # In production, use PIL or cv2 for better quality
            logger.warning(f"Image size mismatch: {image.shape[1:]} vs {self.image_size}")
        
        return image
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        """
        Train CNN on image data
        
        Args:
            X_train: Training images of shape (n_samples, H, W) or (n_samples, C, H, W)
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print training progress
        """
        # Preprocess images
        if len(X_train.shape) == 3:
            # Add channel dimension: (N, H, W) -> (N, C, H, W)
            X_train = X_train[:, np.newaxis, :, :]
        
        if X_val is not None and len(X_val.shape) == 3:
            X_val = X_val[:, np.newaxis, :, :]
        
        # Normalize
        X_train = X_train.astype(np.float32) / 255.0 if X_train.max() > 1.0 else X_train.astype(np.float32)
        if X_val is not None:
            X_val = X_val.astype(np.float32) / 255.0 if X_val.max() > 1.0 else X_val.astype(np.float32)
        
        logger.info(f"Training CNN: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
        
        # Call parent fit method
        super().fit(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose
        )
    
    def predict_image(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict class for a single image
        
        Args:
            image: Image array
        
        Returns:
            predicted_class, confidence, class_probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        # Preprocess
        image_processed = self.preprocess_image(image)
        
        # Convert to tensor
        X = torch.FloatTensor(image_processed).unsqueeze(0)
        X = self.to_device(X)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            probabilities = F.softmax(output, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        class_probs = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, class_probs
    
    def _process_output(self, output: torch.Tensor) -> Tuple[Any, float]:
        """Process CNN output into prediction and confidence"""
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> 'SpecialistPrediction':
        """
        Make image classification prediction
        
        Expected input_data:
        {
            'image': numpy array of image,
            'return_probabilities': bool (optional)
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
            # Extract image
            if 'image' in input_data:
                image = np.array(input_data['image'])
            else:
                raise ValueError("No 'image' in input_data")
            
            # Predict
            predicted_class, confidence, class_probs = self.predict_image(image)
            
            # Build metadata
            metadata = {
                'model_type': 'CNN',
                'num_classes': self.num_classes,
                'image_size': self.image_size,
                'predicted_class': int(predicted_class),
                'confidence': float(confidence)
            }
            
            if input_data.get('return_probabilities', False):
                metadata['class_probabilities'] = class_probs.tolist()
            
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=int(predicted_class),
                confidence=float(confidence),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
