# Deep Learning Specialists for Grace

**Complete neural network implementations for AI governance.**

---

## 🎯 Overview

This module provides **7 production-ready deep learning specialists** built on PyTorch:

1. **ANN** - Feedforward neural networks (classification/regression)
2. **CNN** - Convolutional networks (image/document processing)
3. **RNN** - Recurrent networks (short sequences)
4. **LSTM** - Long Short-Term Memory (KPI forecasting, long sequences)
5. **Transformer** - BERT/RoBERTa (policy analysis, NLP)
6. **Autoencoder** - Anomaly detection, dimensionality reduction
7. **GAN** - Generative Adversarial Networks (synthetic data)

---

## 📦 Installation

```bash
# Required: PyTorch
pip install torch torchvision

# For Transformers
pip install transformers

# GPU support (optional, CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Quick Start

### **Import Specialists**

```python
from grace.mldl_specialists.deep_learning import (
    ANNSpecialist,
    CNNSpecialist,
    RNNSpecialist,
    LSTMSpecialist,
    TransformerSpecialist,
    AutoencoderSpecialist,
    GANSpecialist
)
```

### **Example: LSTM Forecasting**

```python
import numpy as np

# Initialize LSTM
lstm = LSTMSpecialist(
    specialist_id="kpi_forecaster",
    hidden_size=64,
    sequence_length=30,
    forecast_horizon=7
)

# Generate time-series data
t = np.linspace(0, 100, 500)
time_series = np.sin(t) + 0.1 * np.random.randn(500)
time_series = time_series.reshape(-1, 1)

# Train
lstm.fit(time_series, epochs=50, verbose=True)

# Forecast next 7 steps
history = time_series[-30:]
predictions, confidence = lstm.forecast(history, steps=7)

print(f"Forecast: {predictions}")
```

### **Example: Transformer Text Classification**

```python
# Initialize Transformer
transformer = TransformerSpecialist(
    specialist_id="policy_classifier",
    model_name="distilbert-base-uncased",
    num_labels=2
)

# Get embeddings for semantic similarity
policy_text = "All user data must be encrypted at rest"
embeddings = transformer.get_embeddings(policy_text)

print(f"Embedding dimension: {len(embeddings)}")
```

### **Example: Autoencoder Anomaly Detection**

```python
# Initialize Autoencoder
autoencoder = AutoencoderSpecialist(
    specialist_id="anomaly_detector",
    latent_dim=10,
    hidden_sizes=[64, 32]
)

# Train on normal data
X_normal = np.random.randn(1000, 20)
autoencoder.fit(X_normal, epochs=30, verbose=True)

# Calibrate threshold
autoencoder.calibrate_threshold(X_normal, percentile=95)

# Detect anomalies
test_sample = np.random.randn(20)
is_anomaly, error = autoencoder.detect_anomaly(test_sample)

print(f"Anomaly: {is_anomaly}, Error: {error:.6f}")
```

---

## 📚 Complete Examples

Run the integration example:

```bash
python grace/mldl_specialists/deep_learning_integration_example.py
```

This demonstrates:
- ANN for trust score prediction
- LSTM for KPI forecasting
- CNN for document classification
- Autoencoder for anomaly detection
- GAN for synthetic data generation
- Transformer for policy analysis
- Integrated governance pipeline

---

## 🏗️ Architecture

### **Base Infrastructure**

All specialists inherit from `BaseDeepLearningSpecialist`:

```python
class BaseDeepLearningSpecialist(BaseSpecialist):
    - PyTorch model management
    - GPU/CPU device handling
    - Training infrastructure (epochs, checkpoints)
    - Model loading/saving
    - Integration with Grace operational intelligence
```

### **Device Management**

Automatic GPU/CPU detection:

```python
from grace.mldl_specialists.deep_learning import DeviceManager

device = DeviceManager.get_device()
# → "cuda", "mps" (Apple Silicon), or "cpu"

memory = DeviceManager.get_memory_usage()
# → {"device": "cuda", "allocated_mb": 512, "reserved_mb": 1024}
```

---

## 🎯 Use Cases

| **Specialist** | **Use Cases** | **Input** | **Output** |
|---------------|---------------|-----------|------------|
| **ANN** | Trust scoring, general ML | Tabular features | Class/value |
| **CNN** | Document classification, OCR | Images (28x28+) | Class + confidence |
| **RNN** | Short sequences | Sequences (< 20 steps) | Prediction |
| **LSTM** | KPI forecasting, long sequences | Time-series | Multi-step forecast |
| **Transformer** | Policy analysis, NLP | Text | Class/embeddings |
| **Autoencoder** | Anomaly detection | Feature vectors | Anomaly flag + error |
| **GAN** | Synthetic data generation | - | Synthetic samples |

---

## 🔧 Configuration

### **Common Parameters**

```python
# Device selection
device = "cuda"  # or "cpu", "mps"

# Training
epochs = 50
batch_size = 32
learning_rate = 0.001

# Architecture
hidden_size = 64
num_layers = 2
dropout = 0.2
```

### **Model-Specific**

#### LSTM
```python
sequence_length = 30      # Input sequence length
forecast_horizon = 7      # Number of steps to predict
```

#### CNN
```python
num_classes = 10          # Number of output classes
input_channels = 1        # 1 (grayscale) or 3 (RGB)
image_size = 28           # Image height/width
```

#### Transformer
```python
model_name = "distilbert-base-uncased"  # Hugging Face model
max_length = 512                         # Max sequence length
```

#### Autoencoder
```python
latent_dim = 32                    # Compressed representation size
anomaly_threshold = 0.05           # Reconstruction error threshold
```

---

## 💾 Model Persistence

### **Save Model**

```python
# Save checkpoint
lstm.save_checkpoint(
    filepath="models/lstm_kpi_forecaster.pt",
    metadata={"version": "1.0", "trained_on": "2025-01-27"}
)
```

### **Load Model**

```python
# Load checkpoint
metadata = lstm.load_checkpoint("models/lstm_kpi_forecaster.pt")
print(f"Loaded model version: {metadata['version']}")
```

---

## 📊 Monitoring

### **Training Metrics**

All specialists track:
- Training loss per epoch
- Validation loss (if validation data provided)
- Training/validation accuracy (classification)
- Learning rate

```python
# Access training history
for metrics in lstm.training_history:
    print(f"Epoch {metrics.epoch}: "
          f"Train Loss={metrics.train_loss:.4f}, "
          f"Val Loss={metrics.val_loss:.4f}")
```

### **Model Summary**

```python
summary = lstm.get_model_summary()
print(summary)
# → Model: LSTMSpecialist
#   Total parameters: 45,321
#   Trainable parameters: 45,321
#   Device: cuda
#   Trained: True
#   Last training loss: 0.0234
```

---

## 🔬 Advanced Features

### **Custom Training Loop**

```python
# Train for one epoch
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

metrics = lstm.train_epoch(train_loader, val_loader)
```

### **Transfer Learning (Transformer)**

```python
# Load pretrained BERT
transformer = TransformerSpecialist(
    model_name="bert-base-uncased"
)

# Fine-tune on your data
# (standard fit() method)
```

### **Ensemble Predictions**

```python
# Train multiple models
models = [LSTMSpecialist() for _ in range(5)]
for model in models:
    model.fit(train_data, epochs=30)

# Average predictions
predictions = [model.forecast(history)[0] for model in models]
ensemble_forecast = np.mean(predictions, axis=0)
```

---

## 🐛 Troubleshooting

### **PyTorch not found**
```bash
pip install torch torchvision
```

### **CUDA out of memory**
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Or use CPU
device = "cpu"
```

### **Transformers not found**
```bash
pip install transformers
```

### **Model not training**
- Check data shape: `X_train.shape`
- Verify labels: `y_train.shape`
- Ensure data is numeric: `X_train.dtype`

---

## 📖 API Reference

### **BaseDeepLearningSpecialist**

- `build_model(input_shape, **kwargs)` - Build neural network
- `fit(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)` - Train model
- `predict_async(input_data, context)` - Make prediction
- `save_checkpoint(filepath, metadata)` - Save model
- `load_checkpoint(filepath)` - Load model
- `get_model_summary()` - Get model info

### **LSTMSpecialist**

- `forecast(history, steps)` - Multi-step forecast
- `create_sequences(data, seq_length, horizon)` - Prepare training data
- `normalize(data, fit)` - Normalize time-series

### **TransformerSpecialist**

- `predict_text(text)` - Classify text
- `get_embeddings(text)` - Get semantic embeddings
- `tokenize(texts)` - Tokenize text

### **AutoencoderSpecialist**

- `encode(X)` - Encode to latent space
- `reconstruct(X)` - Reconstruct input
- `detect_anomaly(X)` - Detect anomalies
- `calibrate_threshold(X_normal, percentile)` - Set threshold

### **GANSpecialist**

- `generate(n_samples)` - Generate synthetic data
- `train_gan_epoch(real_data, batch_size)` - Train GAN

---

## 🤝 Contributing

To add a new deep learning specialist:

1. Inherit from `BaseDeepLearningSpecialist`
2. Implement `build_model()`
3. Implement `_process_output()`
4. Add to `__init__.py`
5. Create examples
6. Update documentation

---

## 📄 License

Part of Grace AI Governance Platform.

---

## 🎓 Learn More

- **PyTorch**: https://pytorch.org/tutorials/
- **Hugging Face**: https://huggingface.co/docs/transformers/
- **Model Inventory**: See `MODEL_INVENTORY.md`
- **Integration Guide**: See `DEEP_LEARNING_COMPLETE.md`

---

**Status**: ✅ Production Ready  
**Coverage**: 100% of standard deep learning models  
**GPU Support**: CUDA, Apple MPS, CPU  
**Framework**: PyTorch 2.0+
