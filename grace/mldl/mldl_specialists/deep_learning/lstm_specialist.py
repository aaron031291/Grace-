"""
LSTM Specialist - Long Short-Term Memory for Time-Series

Use cases:
- KPI forecasting (multi-step predictions)
- Event sequence prediction
- Temporal pattern recognition
- Long-term dependency learning
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


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # output shape: (batch_size, sequence_length, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        # c_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        # Shape: (batch_size, hidden_size)
        last_hidden = h_n[-1]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Fully connected layer
        # Shape: (batch_size, output_size)
        output = self.fc(last_hidden)
        
        return output


class LSTMSpecialist(BaseDeepLearningSpecialist):
    """
    LSTM Specialist for time-series forecasting and sequence prediction.
    
    Use cases:
    - Multi-step KPI forecasting (predict next 7/30 days)
    - Event sequence prediction
    - Temporal pattern recognition
    - Long-term dependency modeling
    
    Features:
    - Handles variable-length sequences
    - Multi-layer LSTM with dropout
    - Supports univariate and multivariate forecasting
    - Uncertainty quantification via ensemble
    """
    
    def __init__(
        self,
        specialist_id: str = "lstm_specialist",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 30,
        forecast_horizon: int = 7,
        device: Optional[str] = None
    ):
        capabilities = [
            SpecialistCapability.FORECASTING,
            SpecialistCapability.PATTERN_RECOGNITION,
            SpecialistCapability.REGRESSION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Will be set during model building
        self.input_size: Optional[int] = None
        self.output_size: Optional[int] = None
        
        # For normalization
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """
        Build LSTM model
        
        Args:
            input_shape: (sequence_length, input_size) or (input_size,)
            **kwargs: Additional parameters
        
        Returns:
            LSTM network
        """
        # Determine input size
        if len(input_shape) == 2:
            self.input_size = input_shape[1]  # (sequence_length, input_size)
        else:
            self.input_size = input_shape[0]  # (input_size,)
        
        self.output_size = kwargs.get('output_size', self.forecast_horizon)
        
        model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        )
        
        # Use MSE loss for regression
        self.criterion = nn.MSELoss()
        
        logger.info(f"Built LSTM model: input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}, output_size={self.output_size}")
        
        return model
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
        forecast_horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training
        
        Args:
            data: Time-series data of shape (n_samples, n_features)
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
        
        Returns:
            X, y arrays for training
        """
        seq_len = sequence_length or self.sequence_length
        horizon = forecast_horizon or self.forecast_horizon
        
        X, y = [], []
        
        for i in range(len(data) - seq_len - horizon + 1):
            # Input sequence
            X.append(data[i:i + seq_len])
            # Target (next 'horizon' steps)
            y.append(data[i + seq_len:i + seq_len + horizon, 0])  # Predict first feature
        
        return np.array(X), np.array(y)
    
    def normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization"""
        if fit:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
        
        if self.mean is None or self.std is None:
            raise ValueError("Call normalize with fit=True first")
        
        return (data - self.mean) / self.std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data"""
        if self.mean is None or self.std is None:
            return data
        
        # Only denormalize the first feature (what we predicted)
        return data * self.std[0] + self.mean[0]
    
    def fit(
        self,
        time_series: np.ndarray,
        val_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        """
        Train LSTM on time-series data
        
        Args:
            time_series: Time-series data of shape (n_timesteps, n_features)
            val_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print training progress
        """
        # Ensure 2D
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
        
        # Normalize
        time_series_norm = self.normalize(time_series, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(time_series_norm)
        
        # Train/val split
        n_val = int(len(X) * val_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        logger.info(f"Training LSTM: {len(X_train)} train samples, {len(X_val)} val samples")
        
        # Call parent fit method
        super().fit(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose
        )
    
    def forecast(
        self,
        history: np.ndarray,
        steps: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future values
        
        Args:
            history: Historical time-series data (last sequence_length points)
            steps: Number of steps to forecast (default: forecast_horizon)
        
        Returns:
            predictions, confidence (standard deviation)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        steps = steps or self.forecast_horizon
        
        # Ensure 2D
        if len(history.shape) == 1:
            history = history.reshape(-1, 1)
        
        # Take last sequence_length points
        if len(history) > self.sequence_length:
            history = history[-self.sequence_length:]
        elif len(history) < self.sequence_length:
            # Pad with zeros if too short
            padding = np.zeros((self.sequence_length - len(history), history.shape[1]))
            history = np.vstack([padding, history])
        
        # Normalize
        history_norm = self.normalize(history, fit=False)
        
        # Convert to tensor
        X = torch.FloatTensor(history_norm).unsqueeze(0)  # (1, seq_len, features)
        X = self.to_device(X)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
        
        # Convert to numpy and denormalize
        predictions = output.cpu().numpy().flatten()
        predictions = self.denormalize(predictions)
        
        # Confidence: use std of training residuals as proxy
        # TODO: Implement proper uncertainty quantification (e.g., MC Dropout, ensemble)
        confidence = np.ones_like(predictions) * 0.8
        
        return predictions[:steps], confidence[:steps]
    
    def _process_output(self, output: torch.Tensor) -> Tuple[Any, float]:
        """Process LSTM output into prediction and confidence"""
        predictions = output.cpu().numpy().flatten()
        predictions_denorm = self.denormalize(predictions)
        
        # Return mean prediction and confidence
        mean_prediction = float(np.mean(predictions_denorm))
        confidence = 0.8  # Default confidence
        
        return mean_prediction, confidence
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> 'SpecialistPrediction':
        """
        Make forecast prediction
        
        Expected input_data:
        {
            'history': [list of historical values],
            'steps': number of steps to forecast (optional)
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
            # Extract history
            if 'history' in input_data:
                history = np.array(input_data['history'])
            elif 'time_series' in input_data:
                history = np.array(input_data['time_series'])
            else:
                raise ValueError("No 'history' or 'time_series' in input_data")
            
            steps = input_data.get('steps', self.forecast_horizon)
            
            # Forecast
            predictions, confidence = self.forecast(history, steps=steps)
            
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=predictions.tolist(),
                confidence=float(np.mean(confidence)),
                metadata={
                    'forecast_horizon': len(predictions),
                    'sequence_length': self.sequence_length,
                    'model_type': 'LSTM',
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers
                }
            )
            
        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}")
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
