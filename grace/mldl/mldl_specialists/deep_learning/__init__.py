"""
Deep Learning Specialists Module

Neural network specialists for advanced ML tasks:
- ANN (Artificial Neural Networks)
- CNN (Convolutional Neural Networks)
- RNN (Recurrent Neural Networks)
- LSTM (Long Short-Term Memory)
- Transformer (Attention-based models)
- Autoencoder (Unsupervised representation learning)
- GAN (Generative Adversarial Networks)
"""

from .base_deep_specialist import BaseDeepLearningSpecialist, DeviceManager
from .ann_specialist import ANNSpecialist
from .cnn_specialist import CNNSpecialist
from .rnn_specialist import RNNSpecialist
from .lstm_specialist import LSTMSpecialist
from .transformer_specialist import TransformerSpecialist
from .autoencoder_specialist import AutoencoderSpecialist
from .gan_specialist import GANSpecialist

__all__ = [
    'BaseDeepLearningSpecialist',
    'DeviceManager',
    'ANNSpecialist',
    'CNNSpecialist',
    'RNNSpecialist',
    'LSTMSpecialist',
    'TransformerSpecialist',
    'AutoencoderSpecialist',
    'GANSpecialist',
]
