"""
Deep Learning Specialists - Integration Example

Demonstrates all neural network models working together for Grace governance:
- ANN for trust scoring
- LSTM for KPI forecasting
- CNN for document classification
- RNN for event sequences
- Transformer for policy analysis
- Autoencoder for anomaly detection
- GAN for synthetic data generation
"""

import asyncio
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_ann_trust_scoring():
    """Example: Use ANN for trust score prediction"""
    logger.info("\n" + "="*80)
    logger.info("Example 1: ANN for Trust Score Prediction")
    logger.info("="*80)
    
    try:
        from grace.mldl_specialists.deep_learning import ANNSpecialist
        
        # Initialize specialist
        ann = ANNSpecialist(
            specialist_id="trust_scorer_ann",
            hidden_sizes=[64, 32, 16],
            task_type='regression',
            num_classes=1
        )
        
        # Synthetic training data (features → trust score)
        np.random.seed(42)
        X_train = np.random.randn(1000, 10)  # 10 features
        y_train = np.random.rand(1000)  # Trust scores 0-1
        
        X_val = np.random.randn(200, 10)
        y_val = np.random.rand(200)
        
        # Train
        ann.fit(X_train, y_train, X_val, y_val, epochs=10, verbose=True)
        
        # Predict
        test_input = {'features': np.random.randn(10).tolist()}
        prediction = await ann.predict_async(test_input)
        
        logger.info(f"Trust Score Prediction: {prediction.prediction_value:.4f}")
        logger.info(f"Confidence: {prediction.confidence:.4f}")
        
    except Exception as e:
        logger.error(f"ANN example failed: {e}")


async def example_lstm_kpi_forecasting():
    """Example: Use LSTM for KPI forecasting"""
    logger.info("\n" + "="*80)
    logger.info("Example 2: LSTM for KPI Forecasting")
    logger.info("="*80)
    
    try:
        from grace.mldl_specialists.deep_learning import LSTMSpecialist
        
        # Initialize specialist
        lstm = LSTMSpecialist(
            specialist_id="kpi_forecaster_lstm",
            hidden_size=64,
            sequence_length=30,
            forecast_horizon=7
        )
        
        # Synthetic time-series data (e.g., daily KPI values)
        np.random.seed(42)
        t = np.linspace(0, 100, 500)
        time_series = np.sin(t) + 0.1 * np.random.randn(500)
        time_series = time_series.reshape(-1, 1)
        
        # Train
        lstm.fit(time_series, epochs=20, verbose=True)
        
        # Forecast next 7 days
        history = time_series[-30:]  # Last 30 days
        predictions, confidence = lstm.forecast(history, steps=7)
        
        logger.info(f"7-Day Forecast: {predictions}")
        logger.info(f"Confidence: {confidence}")
        
    except Exception as e:
        logger.error(f"LSTM example failed: {e}")


async def example_cnn_document_classification():
    """Example: Use CNN for document image classification"""
    logger.info("\n" + "="*80)
    logger.info("Example 3: CNN for Document Classification")
    logger.info("="*80)
    
    try:
        from grace.mldl_specialists.deep_learning import CNNSpecialist
        
        # Initialize specialist
        cnn = CNNSpecialist(
            specialist_id="document_classifier_cnn",
            num_classes=3,  # e.g., contract, policy, form
            image_size=28
        )
        
        # Synthetic document images (28x28 grayscale)
        np.random.seed(42)
        X_train = np.random.randint(0, 255, (1000, 28, 28), dtype=np.uint8)
        y_train = np.random.randint(0, 3, 1000)
        
        X_val = np.random.randint(0, 255, (200, 28, 28), dtype=np.uint8)
        y_val = np.random.randint(0, 3, 200)
        
        # Train
        cnn.fit(X_train, y_train, X_val, y_val, epochs=5, verbose=True)
        
        # Classify document
        test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        predicted_class, confidence, probs = cnn.predict_image(test_image)
        
        logger.info(f"Document Type: Class {predicted_class}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"Class Probabilities: {probs}")
        
    except Exception as e:
        logger.error(f"CNN example failed: {e}")


async def example_autoencoder_anomaly_detection():
    """Example: Use Autoencoder for anomaly detection"""
    logger.info("\n" + "="*80)
    logger.info("Example 4: Autoencoder for Anomaly Detection")
    logger.info("="*80)
    
    try:
        from grace.mldl_specialists.deep_learning import AutoencoderSpecialist
        
        # Initialize specialist
        autoencoder = AutoencoderSpecialist(
            specialist_id="anomaly_detector_ae",
            latent_dim=10,
            hidden_sizes=[64, 32]
        )
        
        # Synthetic normal data
        np.random.seed(42)
        X_normal = np.random.randn(1000, 20)  # Normal distribution
        
        # Train on normal data
        autoencoder.fit(X_normal, epochs=20, verbose=True)
        
        # Calibrate threshold
        autoencoder.calibrate_threshold(X_normal, percentile=95)
        
        # Test on normal and anomalous data
        normal_sample = np.random.randn(20)
        anomaly_sample = np.random.randn(20) * 5  # Much higher variance
        
        is_normal, error_normal = autoencoder.detect_anomaly(normal_sample)
        is_anomaly, error_anomaly = autoencoder.detect_anomaly(anomaly_sample)
        
        logger.info(f"Normal sample - Anomaly: {is_normal}, Error: {error_normal:.6f}")
        logger.info(f"Anomaly sample - Anomaly: {is_anomaly}, Error: {error_anomaly:.6f}")
        
    except Exception as e:
        logger.error(f"Autoencoder example failed: {e}")


async def example_gan_synthetic_data():
    """Example: Use GAN for synthetic data generation"""
    logger.info("\n" + "="*80)
    logger.info("Example 5: GAN for Synthetic Data Generation")
    logger.info("="*80)
    
    try:
        from grace.mldl_specialists.deep_learning import GANSpecialist
        
        # Initialize specialist
        gan = GANSpecialist(
            specialist_id="data_generator_gan",
            latent_dim=50,
            generator_hidden=[128, 256, 128],
            discriminator_hidden=[128, 64]
        )
        
        # Real data (e.g., governance events)
        np.random.seed(42)
        X_real = np.random.randn(1000, 10)
        
        # Train GAN
        gan.fit(X_real, epochs=50, verbose=True)
        
        # Generate synthetic data
        synthetic_data = gan.generate(n_samples=10)
        
        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        logger.info(f"Sample: {synthetic_data[0]}")
        
    except Exception as e:
        logger.error(f"GAN example failed: {e}")


async def example_transformer_policy_analysis():
    """Example: Use Transformer for policy text analysis"""
    logger.info("\n" + "="*80)
    logger.info("Example 6: Transformer for Policy Analysis")
    logger.info("="*80)
    
    try:
        from grace.mldl_specialists.deep_learning import TransformerSpecialist
        
        # Initialize specialist
        transformer = TransformerSpecialist(
            specialist_id="policy_analyzer_bert",
            model_name="distilbert-base-uncased",
            num_labels=2  # compliant/non-compliant
        )
        
        # Example policy text
        policy_text = "All user data must be encrypted at rest and in transit."
        
        # Get embeddings (for semantic similarity)
        embeddings = transformer.get_embeddings(policy_text)
        
        logger.info(f"Policy Embedding Dimension: {len(embeddings)}")
        logger.info(f"Embedding (first 5): {embeddings[:5]}")
        
        # Note: For classification, would need to fine-tune on labeled data
        
    except Exception as e:
        logger.error(f"Transformer example failed: {e}")


async def integrated_governance_pipeline():
    """
    Integrated pipeline using multiple DL specialists:
    1. LSTM forecasts KPI degradation
    2. If degradation predicted, ANN scores trust impact
    3. Autoencoder detects anomalous patterns
    4. Transformer analyzes policy compliance
    5. CNN validates document artifacts
    """
    logger.info("\n" + "="*80)
    logger.info("Integrated Deep Learning Governance Pipeline")
    logger.info("="*80)
    
    # Simulated workflow
    logger.info("\nStep 1: LSTM forecasts KPI degradation → DEGRADATION PREDICTED")
    logger.info("Step 2: ANN assesses trust impact → MEDIUM RISK (0.65)")
    logger.info("Step 3: Autoencoder detects anomalies → ANOMALY DETECTED")
    logger.info("Step 4: Transformer checks policy compliance → NON-COMPLIANT")
    logger.info("Step 5: CNN validates document → INVALID SIGNATURE")
    logger.info("\n🚨 GOVERNANCE ACTION: ESCALATE TO HUMAN REVIEW")


async def main():
    """Run all deep learning examples"""
    logger.info("="*80)
    logger.info("Grace Deep Learning Specialists - Integration Examples")
    logger.info("="*80)
    
    examples = [
        example_ann_trust_scoring,
        example_lstm_kpi_forecasting,
        example_cnn_document_classification,
        example_autoencoder_anomaly_detection,
        example_gan_synthetic_data,
        example_transformer_policy_analysis,
        integrated_governance_pipeline
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            logger.error(f"Example failed: {e}")
        
        await asyncio.sleep(0.5)  # Brief pause between examples
    
    logger.info("\n" + "="*80)
    logger.info("All Deep Learning Examples Completed!")
    logger.info("="*80)


if __name__ == '__main__':
    asyncio.run(main())
