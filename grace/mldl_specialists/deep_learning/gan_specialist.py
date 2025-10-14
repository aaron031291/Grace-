"""
GAN Specialist - Generative Adversarial Network

Use cases:
- Synthetic data generation
- Data augmentation
- Privacy-preserving synthetic datasets
- Anomaly detection (discriminator scores)
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


class Generator(nn.Module):
    """GAN Generator network"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_sizes: List[int]):
        super(Generator, self).__init__()
        
        layers = []
        prev_size = latent_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer with tanh activation
        layers.extend([
            nn.Linear(prev_size, output_dim),
            nn.Tanh()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        """Generate fake data from noise"""
        return self.network(z)


class Discriminator(nn.Module):
    """GAN Discriminator network"""
    
    def __init__(self, input_dim: int, hidden_sizes: List[int]):
        super(Discriminator, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer (probability real/fake)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Classify real vs fake"""
        return self.network(x)


class GANSpecialist(BaseDeepLearningSpecialist):
    """
    GAN Specialist for data generation.
    
    Use cases:
    - Synthetic data generation for testing
    - Data augmentation (increase training data)
    - Privacy-preserving synthetic datasets
    - Anomaly detection (low discriminator scores)
    - Simulate rare events
    
    Features:
    - Configurable generator/discriminator architecture
    - Training stability techniques
    - Synthetic data quality metrics
    - Conditional generation (future enhancement)
    """
    
    def __init__(
        self,
        specialist_id: str = "gan_specialist",
        latent_dim: int = 100,
        generator_hidden: List[int] = [256, 512, 256],
        discriminator_hidden: List[int] = [256, 128],
        device: Optional[str] = None
    ):
        capabilities = [
            SpecialistCapability.PATTERN_RECOGNITION,
            SpecialistCapability.ANOMALY_DETECTION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.latent_dim = latent_dim
        self.generator_hidden = generator_hidden
        self.discriminator_hidden = discriminator_hidden
        
        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None
        self.optimizer_G: Optional[torch.optim.Optimizer] = None
        self.optimizer_D: Optional[torch.optim.Optimizer] = None
        
        self.input_dim: Optional[int] = None
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build GAN (generator + discriminator)"""
        if len(input_shape) == 1:
            self.input_dim = input_shape[0]
        else:
            self.input_dim = int(np.prod(input_shape))
        
        # Build generator
        self.generator = Generator(
            latent_dim=self.latent_dim,
            output_dim=self.input_dim,
            hidden_sizes=self.generator_hidden
        ).to(self.device)
        
        # Build discriminator
        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            hidden_sizes=self.discriminator_hidden
        ).to(self.device)
        
        # Optimizers
        lr = kwargs.get('learning_rate', 0.0002)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss
        self.criterion = nn.BCELoss()
        
        logger.info(f"Built GAN: latent_dim={self.latent_dim}, "
                   f"input_dim={self.input_dim}")
        
        # Return generator as the main model
        return self.generator
    
    def train_gan_epoch(
        self,
        real_data: torch.Tensor,
        batch_size: int = 64
    ) -> Tuple[float, float]:
        """
        Train GAN for one epoch
        
        Returns:
            generator_loss, discriminator_loss
        """
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.optimizer_D.zero_grad()
        
        # Real data
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        
        # Try to fool discriminator
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)  # Want discriminator to think it's real
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item(), d_loss.item()
    
    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.0002,
        verbose: bool = True
    ):
        """
        Train GAN
        
        Args:
            X_train: Real data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print progress
        """
        # Build models if not built
        if self.generator is None or self.discriminator is None:
            input_shape = (X_train.shape[1],)
            self.build_model(input_shape, learning_rate=learning_rate)
        
        # Normalize data to [-1, 1]
        X_train = X_train.astype(np.float32)
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_train = X_train * 2 - 1
        
        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            n_batches = 0
            
            # Create batches
            n_samples = len(X_train)
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size:
                    continue
                
                batch_data = torch.FloatTensor(X_train[batch_indices]).to(self.device)
                
                g_loss, d_loss = self.train_gan_epoch(batch_data, batch_size)
                
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                n_batches += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_g_loss = epoch_g_loss / n_batches
                avg_d_loss = epoch_d_loss / n_batches
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"GAN training completed")
    
    def generate(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate synthetic data
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Generated data
        """
        if not self.is_trained or self.generator is None:
            raise ValueError("GAN not trained")
        
        # Sample noise
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        
        # Generate
        self.generator.eval()
        with torch.no_grad():
            fake_data = self.generator(z)
        
        return fake_data.cpu().numpy()
    
    def _process_output(self, output: 'torch.Tensor') -> Tuple[Any, float]:
        """Process GAN output"""
        generated = output.cpu().numpy().flatten()
        confidence = 0.75
        return generated.tolist(), confidence
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> 'SpecialistPrediction':
        """
        Generate synthetic data
        
        Expected input_data:
        {
            'n_samples': int (number of samples to generate)
        }
        """
        if not self.is_trained:
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'GAN not trained'}
            )
        
        try:
            n_samples = input_data.get('n_samples', 1)
            
            # Generate synthetic data
            synthetic_data = self.generate(n_samples)
            
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=synthetic_data.tolist(),
                confidence=0.75,
                metadata={
                    'task': 'generation',
                    'n_samples': n_samples,
                    'latent_dim': self.latent_dim
                }
            )
        
        except Exception as e:
            logger.error(f"GAN generation failed: {e}")
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
