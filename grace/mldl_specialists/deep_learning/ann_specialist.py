"""
ANN Specialist - Artificial Neural Network (Feedforward MLP)

Use cases:
- General function approximation
- Classification and regression tasks
- Feature learning
- Non-linear pattern recognition
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


class ANNNetwork(nn.Module):
    """Feedforward Artificial Neural Network (Multi-Layer Perceptron)"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.3,
        task_type: str = 'classification'
    ):
        super(ANNNetwork, self).__init__()
        
        self.task_type = task_type
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ANNSpecialist(BaseDeepLearningSpecialist):
    """
    ANN Specialist - Feedforward neural network for general ML tasks.
    
    Use cases:
    - General function approximation
    - Classification (binary/multi-class)
    - Regression
    - Non-linear feature learning
    - Trust score prediction
    - Decision scoring
    
    Features:
    - Configurable architecture (number of layers, hidden units)
    - Batch normalization and dropout
    - Supports both classification and regression
    - Fast training on tabular data
    """
    
    def __init__(
        self,
        specialist_id: str = "ann_specialist",
        hidden_sizes: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        task_type: str = 'classification',  # or 'regression'
        num_classes: int = 2,
        device: Optional[str] = None
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.REGRESSION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.task_type = task_type
        self.num_classes = num_classes
        self.input_size: Optional[int] = None
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build ANN model"""
        if len(input_shape) == 1:
            self.input_size = input_shape[0]
        else:
            self.input_size = int(np.prod(input_shape))
        
        # Override with kwargs
        self.task_type = kwargs.get('task_type', self.task_type)
        self.num_classes = kwargs.get('num_classes', self.num_classes)
        
        # Output size
        if self.task_type == 'classification':
            output_size = self.num_classes
        else:
            output_size = 1  # Regression
        
        model = ANNNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=output_size,
            dropout=self.dropout,
            task_type=self.task_type
        )
        
        # Loss function
        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        logger.info(f"Built ANN model: input_size={self.input_size}, "
                   f"hidden_sizes={self.hidden_sizes}, output_size={output_size}")
        
        return model
    
    def _process_output(self, output: torch.Tensor) -> Tuple[Any, float]:
        """Process ANN output"""
        if self.task_type == 'classification':
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            return predicted_class, confidence
        else:
            # Regression
            prediction = output.item()
            confidence = 0.8  # Default for regression
            return prediction, confidence
