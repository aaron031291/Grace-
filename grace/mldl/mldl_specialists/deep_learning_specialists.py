"""
Deep Learning Specialists - Neural Network Models

Individual deep learning models for advanced AI tasks:
- Artificial Neural Network (ANN/MLP) Specialist
- Convolutional Neural Network (CNN) Specialist
- Recurrent Neural Network (RNN) Specialist
- Long Short-Term Memory (LSTM) Specialist
- Generative Adversarial Network (GAN) Specialist
- Autoencoder Specialist
- Transformer Specialist
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass

from .base_specialist import BaseSpecialist, SpecialistCapability, SpecialistPrediction

# Deep learning framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass

logger = logging.getLogger(__name__)


# ============================================================================
# Neural Network Architectures
# ============================================================================

class MLPNetwork(nn.Module):
    """Multi-Layer Perceptron (Feedforward Neural Network)"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNNNetwork(nn.Module):
    """Convolutional Neural Network for image/spatial data"""
    
    def __init__(self, input_channels: int, num_classes: int, input_size: int = 28):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after convolutions and pooling
        # input_size -> pool -> pool -> pool = input_size // 8
        conv_output_size = (input_size // 8) ** 2 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class RNNNetwork(nn.Module):
    """Recurrent Neural Network for sequence data"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_dim)
        out, hidden = self.rnn(x, hidden)
        
        # Take the last time step
        out = self.fc(out[:, -1, :])
        
        return out, hidden


class LSTMNetwork(nn.Module):
    """Long Short-Term Memory Network for long-term dependencies"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_dim)
        out, hidden = self.lstm(x, hidden)
        
        # Take the last time step
        out = self.fc(out[:, -1, :])
        
        return out, hidden


class Generator(nn.Module):
    """GAN Generator - creates synthetic data"""
    
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    """GAN Discriminator - distinguishes real from fake"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):
    """Autoencoder for dimensionality reduction and reconstruction"""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        # Decode
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class TransformerClassifier(nn.Module):
    """Transformer model for sequence classification"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        if PYTORCH_AVAILABLE:
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


# ============================================================================
# Deep Learning Specialists
# ============================================================================

class ANNSpecialist(BaseSpecialist):
    """
    Artificial Neural Network (MLP) Specialist.
    
    Use cases:
    - General-purpose function approximation
    - Non-linear classification and regression
    - Feature learning from tabular data
    - Trust score prediction with complex patterns
    """
    
    def __init__(
        self,
        specialist_id: str = "ann_specialist",
        input_dim: int = 10,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        task_type: str = "classification"  # or "regression"
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.REGRESSION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ANNSpecialist. Install with: pip install torch")
        
        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create network
        self.model = MLPNetwork(input_dim, hidden_dims, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function
        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss() if output_dim > 1 else nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using ANN."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            self.model.eval()
            
            # Extract features
            features = self._extract_features(input_data)
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(X)
                
                if self.task_type == "classification":
                    if self.output_dim > 1:
                        probs = F.softmax(output, dim=1)
                        prediction = torch.argmax(probs, dim=1).item()
                        confidence = probs.max().item()
                    else:
                        prob = torch.sigmoid(output).item()
                        prediction = 1 if prob > 0.5 else 0
                        confidence = prob if prediction == 1 else 1 - prob
                else:
                    prediction = output.item()
                    confidence = 0.8  # Default for regression
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction),
                confidence=float(confidence),
                metadata={
                    'task_type': self.task_type,
                    'model_type': 'ANN',
                    'device': str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"ANNSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the ANN model."""
        self.model.train()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate loss
                if self.task_type == "classification" and self.output_dim > 1:
                    loss = self.criterion(outputs, batch_y.long())
                else:
                    loss = self.criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        if 'features' in input_data:
            return input_data['features']
        elif 'values' in input_data:
            return input_data['values']
        else:
            features = []
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
            return features


class CNNSpecialist(BaseSpecialist):
    """
    Convolutional Neural Network Specialist.
    
    Use cases:
    - Image classification and object detection
    - Document image processing (scanned contracts, policies)
    - Diagram and flowchart analysis
    - Visual governance artifact processing
    """
    
    def __init__(
        self,
        specialist_id: str = "cnn_specialist",
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNNSpecialist. Install with: pip install torch")
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Create network
        self.model = CNNNetwork(input_channels, num_classes, input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using CNN."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            self.model.eval()
            
            # Extract image data
            # Expected format: (channels, height, width) or flat array
            if 'image' in input_data:
                image = np.array(input_data['image'])
            elif 'features' in input_data:
                image = np.array(input_data['features'])
            else:
                raise ValueError("No image data found in input")
            
            # Reshape if needed
            if len(image.shape) == 1:
                image = image.reshape(self.input_channels, self.input_size, self.input_size)
            elif len(image.shape) == 2:
                image = image.reshape(1, self.input_channels, self.input_size, self.input_size)
            
            X = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(X)
                probs = F.softmax(output, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs.max().item()
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=int(prediction),
                confidence=float(confidence),
                metadata={
                    'model_type': 'CNN',
                    'num_classes': self.num_classes,
                    'device': str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"CNNSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32):
        """Train the CNN model."""
        self.model.train()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")


class RNNSpecialist(BaseSpecialist):
    """
    Recurrent Neural Network Specialist.
    
    Use cases:
    - Sequential data processing
    - Short-term time-series prediction
    - Basic text sequence modeling
    - Event sequence analysis
    """
    
    def __init__(
        self,
        specialist_id: str = "rnn_specialist",
        input_dim: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2
    ):
        capabilities = [
            SpecialistCapability.FORECASTING,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNNSpecialist. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create network
        self.model = RNNNetwork(input_dim, hidden_dim, output_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using RNN."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            self.model.eval()
            
            # Extract sequence data
            if 'sequence' in input_data:
                sequence = np.array(input_data['sequence'])
            elif 'features' in input_data:
                sequence = np.array(input_data['features'])
            else:
                raise ValueError("No sequence data found")
            
            # Reshape to (batch, seq_len, input_dim)
            if len(sequence.shape) == 1:
                sequence = sequence.reshape(1, -1, self.input_dim)
            elif len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            X = torch.FloatTensor(sequence).to(self.device)
            
            # Predict
            with torch.no_grad():
                output, _ = self.model(X)
                prediction = output.cpu().numpy()[0]
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
                confidence=0.75,
                metadata={
                    'model_type': 'RNN',
                    'sequence_length': sequence.shape[1],
                    'device': str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"RNNSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the RNN model."""
        self.model.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs, _ = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")


class LSTMSpecialist(BaseSpecialist):
    """
    Long Short-Term Memory Specialist.
    
    Use cases:
    - Long-term KPI forecasting (7-day, 30-day predictions)
    - Time-series anomaly detection
    - Event sequence prediction
    - Speech recognition and text generation
    """
    
    def __init__(
        self,
        specialist_id: str = "lstm_specialist",
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2
    ):
        capabilities = [
            SpecialistCapability.FORECASTING,
            SpecialistCapability.PATTERN_RECOGNITION,
            SpecialistCapability.ANOMALY_DETECTION
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMSpecialist. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create network
        self.model = LSTMNetwork(input_dim, hidden_dim, output_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using LSTM."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            self.model.eval()
            
            # Extract sequence data
            if 'sequence' in input_data:
                sequence = np.array(input_data['sequence'])
            elif 'features' in input_data:
                sequence = np.array(input_data['features'])
            else:
                raise ValueError("No sequence data found")
            
            # Reshape to (batch, seq_len, input_dim)
            if len(sequence.shape) == 1:
                sequence = sequence.reshape(1, -1, self.input_dim)
            elif len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            X = torch.FloatTensor(sequence).to(self.device)
            
            # Predict
            with torch.no_grad():
                output, _ = self.model(X)
                prediction = output.cpu().numpy()[0]
            
            # Calculate confidence based on prediction stability
            # (in production, could use ensemble variance or Bayesian LSTM)
            confidence = 0.85
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
                confidence=confidence,
                metadata={
                    'model_type': 'LSTM',
                    'sequence_length': sequence.shape[1],
                    'device': str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"LSTMSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model."""
        self.model.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs, _ = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")


class GANSpecialist(BaseSpecialist):
    """
    Generative Adversarial Network Specialist.
    
    Use cases:
    - Synthetic data generation for privacy-preserving ML
    - Data augmentation for training
    - Anomaly detection (discriminator identifies outliers)
    - Realistic test data generation
    """
    
    def __init__(
        self,
        specialist_id: str = "gan_specialist",
        input_dim: int = 10,
        latent_dim: int = 64
    ):
        capabilities = [
            SpecialistCapability.PATTERN_RECOGNITION,
            SpecialistCapability.ANOMALY_DETECTION
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GANSpecialist. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Create networks
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss
        self.criterion = nn.BCELoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Generate synthetic data or detect anomalies."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            mode = input_data.get('mode', 'generate')  # 'generate' or 'discriminate'
            
            if mode == 'generate':
                # Generate synthetic data
                self.generator.eval()
                
                num_samples = input_data.get('num_samples', 1)
                z = torch.randn(num_samples, self.latent_dim).to(self.device)
                
                with torch.no_grad():
                    generated = self.generator(z)
                    samples = generated.cpu().numpy()
                
                return SpecialistPrediction(
                    specialist_id=self.specialist_id,
                    prediction_value=samples.tolist(),
                    confidence=0.8,
                    metadata={
                        'model_type': 'GAN',
                        'mode': 'generation',
                        'num_samples': num_samples
                    }
                )
            
            else:  # discriminate mode - anomaly detection
                self.discriminator.eval()
                
                features = self._extract_features(input_data)
                X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    score = self.discriminator(X).item()
                
                # Low score = likely fake/anomaly, high score = likely real
                is_anomaly = score < 0.5
                confidence = abs(score - 0.5) * 2  # Convert to 0-1 range
                
                return SpecialistPrediction(
                    specialist_id=self.specialist_id,
                    prediction_value="anomaly" if is_anomaly else "normal",
                    confidence=float(confidence),
                    metadata={
                        'model_type': 'GAN',
                        'mode': 'discrimination',
                        'discriminator_score': float(score)
                    }
                )
            
        except Exception as e:
            logger.error(f"GANSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the GAN model."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for real_data in dataloader:
                batch_size = real_data.size(0)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                d_real = self.discriminator(real_data)
                d_real_loss = self.criterion(d_real, real_labels)
                
                # Fake data
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data.detach())
                d_fake_loss = self.criterion(d_fake, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                g_output = self.discriminator(fake_data)
                g_loss = self.criterion(g_output, real_labels)  # Generator wants discriminator to think it's real
                
                g_loss.backward()
                self.g_optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        if 'features' in input_data:
            return input_data['features']
        elif 'values' in input_data:
            return input_data['values']
        else:
            features = []
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
            return features


class AutoencoderSpecialist(BaseSpecialist):
    """
    Autoencoder Specialist.
    
    Use cases:
    - Non-linear dimensionality reduction
    - Anomaly detection via reconstruction error
    - Feature learning and extraction
    - Data denoising and compression
    """
    
    def __init__(
        self,
        specialist_id: str = "autoencoder_specialist",
        input_dim: int = 10,
        latent_dim: int = 3
    ):
        capabilities = [
            SpecialistCapability.DIMENSIONALITY_REDUCTION,
            SpecialistCapability.ANOMALY_DETECTION,
            SpecialistCapability.SIGNAL_COMPRESSION
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AutoencoderSpecialist. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Create network
        self.model = Autoencoder(input_dim, latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Store mean reconstruction error for anomaly detection
        self.mean_reconstruction_error = None
        self.std_reconstruction_error = None
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Encode to latent space or detect anomalies."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            self.model.eval()
            
            mode = input_data.get('mode', 'encode')  # 'encode' or 'anomaly_detect'
            
            # Extract features
            features = self._extract_features(input_data)
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                reconstructed, latent = self.model(X)
                
                if mode == 'encode':
                    # Return latent representation
                    latent_vector = latent.cpu().numpy()[0]
                    
                    return SpecialistPrediction(
                        specialist_id=self.specialist_id,
                        prediction_value=latent_vector.tolist(),
                        confidence=0.85,
                        metadata={
                            'model_type': 'Autoencoder',
                            'mode': 'encoding',
                            'original_dim': self.input_dim,
                            'latent_dim': self.latent_dim
                        }
                    )
                
                else:  # anomaly detection mode
                    # Calculate reconstruction error
                    reconstruction_error = F.mse_loss(reconstructed, X).item()
                    
                    # Determine if anomaly
                    if self.mean_reconstruction_error is not None:
                        # Z-score based anomaly detection
                        z_score = (reconstruction_error - self.mean_reconstruction_error) / (self.std_reconstruction_error + 1e-10)
                        is_anomaly = z_score > 3.0  # 3 sigma threshold
                        confidence = min(abs(z_score) / 3.0, 1.0)
                    else:
                        is_anomaly = reconstruction_error > 0.1  # Default threshold
                        confidence = 0.5
                    
                    return SpecialistPrediction(
                        specialist_id=self.specialist_id,
                        prediction_value="anomaly" if is_anomaly else "normal",
                        confidence=float(confidence),
                        metadata={
                            'model_type': 'Autoencoder',
                            'mode': 'anomaly_detection',
                            'reconstruction_error': float(reconstruction_error),
                            'z_score': float(z_score) if self.mean_reconstruction_error else None
                        }
                    )
            
        except Exception as e:
            logger.error(f"AutoencoderSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the Autoencoder model."""
        self.model.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X in dataloader:
                self.optimizer.zero_grad()
                
                reconstructed, _ = self.model(batch_X)
                loss = self.criterion(reconstructed, batch_X)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Calculate reconstruction error statistics for anomaly detection
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch_X in dataloader:
                reconstructed, _ = self.model(batch_X)
                errors = F.mse_loss(reconstructed, batch_X, reduction='none').mean(dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        self.mean_reconstruction_error = np.mean(reconstruction_errors)
        self.std_reconstruction_error = np.std(reconstruction_errors)
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data."""
        if 'features' in input_data:
            return input_data['features']
        elif 'values' in input_data:
            return input_data['values']
        else:
            features = []
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
            return features


class TransformerSpecialist(BaseSpecialist):
    """
    Transformer Specialist - Attention-based sequence model.
    
    Use cases:
    - NLP tasks (text classification, sentiment analysis)
    - Policy document analysis
    - Governance compliance checking
    - Semantic similarity for audit logs
    - Time-series forecasting with attention
    """
    
    def __init__(
        self,
        specialist_id: str = "transformer_specialist",
        input_dim: int = 10,
        num_classes: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4
    ):
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.PATTERN_RECOGNITION,
            SpecialistCapability.FORECASTING
        ]
        super().__init__(specialist_id, capabilities)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerSpecialist. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Create network
        self.model = TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using Transformer."""
        if not self.is_trained:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            self.model.eval()
            
            # Extract sequence data
            if 'sequence' in input_data:
                sequence = np.array(input_data['sequence'])
            elif 'features' in input_data:
                sequence = np.array(input_data['features'])
            else:
                raise ValueError("No sequence data found")
            
            # Reshape to (batch, seq_len, input_dim)
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            X = torch.FloatTensor(sequence).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(X)
                probs = F.softmax(output, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs.max().item()
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=int(prediction),
                confidence=float(confidence),
                metadata={
                    'model_type': 'Transformer',
                    'num_classes': self.num_classes,
                    'sequence_length': sequence.shape[1],
                    'device': str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"TransformerSpecialist prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 16):
        """Train the Transformer model."""
        self.model.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"{self.specialist_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")
