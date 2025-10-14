"""
Base Deep Learning Specialist

Foundation for all PyTorch-based neural network specialists:
- GPU/CPU device management
- Common neural network utilities
- Training/inference abstractions
- Model checkpointing
- Integration with Grace operational intelligence
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import logging

# PyTorch imports (optional dependency)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..base_specialist import BaseSpecialist, SpecialistCapability, SpecialistPrediction

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages GPU/CPU device selection and memory"""
    
    @staticmethod
    def get_device() -> str:
        """Get best available device (CUDA, MPS, or CPU)"""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU")
        
        return device
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"device": "cpu", "allocated_mb": 0, "reserved_mb": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        return {
            "device": "cuda",
            "allocated_mb": allocated,
            "reserved_mb": reserved
        }


@dataclass
class TrainingMetrics:
    """Training metrics for deep learning models"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rate': self.learning_rate
        }


class BaseDeepLearningSpecialist(BaseSpecialist, ABC):
    """
    Base class for all deep learning specialists.
    
    Provides:
    - PyTorch model management
    - Device handling (GPU/CPU)
    - Training infrastructure
    - Checkpointing
    - Integration with Grace operational intelligence
    """
    
    def __init__(
        self,
        specialist_id: str,
        capabilities: List[SpecialistCapability],
        device: Optional[str] = None
    ):
        super().__init__(specialist_id, capabilities)
        
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch torchvision"
            )
        
        # Device management
        self.device = device or DeviceManager.get_device()
        logger.info(f"{specialist_id} initialized on device: {self.device}")
        
        # Model will be defined by subclasses
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        
        # Training state
        self.is_trained = False
        self.training_history: List[TrainingMetrics] = []
        self.best_val_loss: float = float('inf')
        
    @abstractmethod
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build the neural network architecture"""
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device"""
        return tensor.to(self.device)
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model checkpoint"""
        if self.model is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_history': [m.to_dict() for m in self.training_history],
            'specialist_id': self.specialist_id,
            'device': self.device,
            'is_trained': self.is_trained,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.is_trained = checkpoint.get('is_trained', False)
        
        # Restore training history
        history = checkpoint.get('training_history', [])
        self.training_history = [
            TrainingMetrics(**m) for m in history
        ]
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('metadata', {})
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> TrainingMetrics:
        """Train for one epoch"""
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise ValueError("Model, optimizer, or criterion not initialized")
        
        # Training phase
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            
            # Accuracy (for classification)
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                _, predicted = torch.max(outputs.data, 1)
                if len(targets.shape) == 1:  # Class indices
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else None
        
        # Validation phase
        val_loss = None
        val_accuracy = None
        
        if val_loader:
            self.model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = self.to_device(inputs)
                    targets = self.to_device(targets)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss_sum += loss.item()
                    
                    # Accuracy
                    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                        _, predicted = torch.max(outputs.data, 1)
                        if len(targets.shape) == 1:
                            val_total += targets.size(0)
                            val_correct += (predicted == targets).sum().item()
            
            val_loss = val_loss_sum / len(val_loader)
            val_accuracy = val_correct / val_total if val_total > 0 else None
            
            # Update best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
        
        # Create metrics
        epoch = len(self.training_history) + 1
        lr = self.optimizer.param_groups[0]['lr']
        
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=lr
        )
        
        self.training_history.append(metrics)
        
        return metrics
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        """Train the model"""
        # Create data loaders
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train) if y_train.dtype == np.float32 or y_train.dtype == np.float64 else torch.LongTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val) if y_val.dtype == np.float32 or y_val.dtype == np.float64 else torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model if not already built
        if self.model is None:
            input_shape = X_train.shape[1:]
            self.model = self.build_model(input_shape)
            self.model = self.model.to(self.device)
        
        # Initialize optimizer if not already initialized
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            
            if verbose:
                log_msg = f"Epoch {metrics.epoch}/{epochs} - Loss: {metrics.train_loss:.4f}"
                if metrics.train_accuracy:
                    log_msg += f" - Acc: {metrics.train_accuracy:.4f}"
                if metrics.val_loss:
                    log_msg += f" - Val Loss: {metrics.val_loss:.4f}"
                if metrics.val_accuracy:
                    log_msg += f" - Val Acc: {metrics.val_accuracy:.4f}"
                
                logger.info(log_msg)
        
        self.is_trained = True
        logger.info(f"{self.specialist_id} training completed")
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistPrediction:
        """Make prediction using neural network"""
        if not self.is_trained or self.model is None:
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained'}
            )
        
        try:
            # Extract features
            features = self._extract_features(input_data)
            
            # Convert to tensor
            X = torch.FloatTensor(np.array(features)).unsqueeze(0)
            X = self.to_device(X)
            
            # Inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
            
            # Process output
            prediction_value, confidence = self._process_output(output)
            
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=prediction_value,
                confidence=confidence,
                metadata={
                    'device': self.device,
                    'model_type': self.__class__.__name__,
                    'output_shape': list(output.shape)
                }
            )
            
        except Exception as e:
            logger.error(f"{self.specialist_id} prediction failed: {e}")
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    @abstractmethod
    def _process_output(self, output: torch.Tensor) -> Tuple[Any, float]:
        """Process model output into prediction and confidence"""
        pass
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from input data"""
        features = []
        
        if 'features' in input_data:
            features = input_data['features']
        elif 'values' in input_data:
            features = input_data['values']
        elif 'tensor' in input_data:
            features = input_data['tensor']
        else:
            for value in input_data.values():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, (list, np.ndarray)):
                    features.extend(value)
        
        return features
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"Model: {self.__class__.__name__}\n"
        summary += f"Total parameters: {total_params:,}\n"
        summary += f"Trainable parameters: {trainable_params:,}\n"
        summary += f"Device: {self.device}\n"
        summary += f"Trained: {self.is_trained}\n"
        
        if self.training_history:
            last_metrics = self.training_history[-1]
            summary += f"Last training loss: {last_metrics.train_loss:.4f}\n"
            if last_metrics.val_loss:
                summary += f"Last validation loss: {last_metrics.val_loss:.4f}\n"
        
        return summary
