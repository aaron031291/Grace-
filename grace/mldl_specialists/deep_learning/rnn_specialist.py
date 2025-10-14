"""
RNN Specialist - Recurrent Neural Network for Sequences

Use cases:
- Basic sequence processing
- Short-term temporal pattern recognition
- Text generation
- Sequential decision making
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


class RNNNetwork(nn.Module):
    """Vanilla RNN architecture"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(RNNNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass"""
        # RNN forward
        rnn_out, h_n = self.rnn(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        
        # Output
        output = self.fc(last_hidden)
        return output


class RNNSpecialist(BaseDeepLearningSpecialist):
    """
    RNN Specialist for basic sequence processing.
    
    Note: For most use cases, LSTM is preferred over vanilla RNN
    due to vanishing gradient issues. Use this for:
    - Short sequences (< 20 timesteps)
    - Simple temporal patterns
    - When computational efficiency is critical
    
    For long-term dependencies, use LSTMSpecialist instead.
    """
    
    def __init__(
        self,
        specialist_id: str = "rnn_specialist",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 10,
        device: Optional[str] = None
    ):
        capabilities = [
            SpecialistCapability.PATTERN_RECOGNITION,
            SpecialistCapability.REGRESSION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.input_size: Optional[int] = None
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build RNN model"""
        if len(input_shape) == 2:
            self.input_size = input_shape[1]
        else:
            self.input_size = input_shape[0]
        
        output_size = kwargs.get('output_size', 1)
        
        model = RNNNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size,
            dropout=self.dropout
        )
        
        self.criterion = nn.MSELoss()
        
        logger.info(f"Built RNN model: input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}")
        
        return model
    
    def _process_output(self, output: 'torch.Tensor') -> Tuple[Any, float]:
        """Process RNN output"""
        prediction = output.cpu().numpy().flatten()[0]
        confidence = 0.75
        return float(prediction), confidence
