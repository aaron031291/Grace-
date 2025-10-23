"""
End-to-End Integration Example for Grace ML/DL Cognitive Substrate

This demonstrates the complete integration of:
- Classical ML specialists (Random Forest, SVM, etc.)
- Deep Learning specialists (LSTM, Transformer, CNN, etc.)
- Operational Intelligence (monitoring, registry, active learning, uncertainty)
- Governance integration
- TriggerMesh workflows
- Complete pipeline: Ingress → Intelligence → ML/DL → Governance → Response
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from grace.mldl_specialists.cognitive_substrate import (
    CognitiveSubstrate,
    CognitiveFunction,
    IntegrationPoint,
    CognitiveEvent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndToEndIntegrationDemo:
    """
    Complete end-to-end demonstration of Grace's ML/DL cognitive substrate.
    
    Scenarios:
    1. Trust Score Prediction (Ingress → ML → Governance)
    2. KPI Forecasting (LSTM → Monitoring → Alert)
    3. Policy Analysis (Transformer → Governance → Compliance)
    4. Document Classification (CNN → Intelligence → Decision)
    5. Anomaly Detection (Autoencoder + Isolation Forest → Alert)
    6. Active Learning Loop (Uncertainty → Query → Retrain)
    """
    
    def __init__(self):
        self.substrate = CognitiveSubstrate(
            enable_gpu=True  # Use GPU if available
        )
    
    async def scenario_1_trust_score_prediction(self):
        """
        Scenario 1: Trust Score Prediction
        
        Flow:
        1. Ingress receives user action data
        2. Create ANN specialist for trust scoring
        3. Train on historical trust data
        4. Predict trust score for new action
        5. Governance validates prediction
        6. Return decision with trust score
        """
        logger.info("\n=== Scenario 1: Trust Score Prediction ===")
        
        # Step 1: Create ANN specialist
        ann_specialist = await self.substrate.create_specialist(
            specialist_type="ANN",
            specialist_id="trust_scorer_ann",
            cognitive_functions=[CognitiveFunction.TRUST_SCORING],
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1,
            task="regression"
        )
        
        # Step 2: Generate synthetic training data (in production: from database)
        X_train = np.random.randn(1000, 10)  # 1000 samples, 10 features
        y_train = (X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3 + 0.2).reshape(-1, 1)  # Trust score
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        # Step 3: Train model
        logger.info("Training trust score predictor...")
        train_result = await self.substrate.train_specialist(
            specialist_id="trust_scorer_ann",
            X_train=X_train,
            y_train=y_train,
            epochs=20,
            batch_size=32,
            learning_rate=0.001,
            verbose=False
        )
        logger.info(f"Training complete: {train_result}")
        
        # Step 4: Make prediction on new data
        new_action = np.random.randn(1, 10).astype(np.float32)
        prediction = await self.substrate.predict_with_specialist(
            specialist_id="trust_scorer_ann",
            X=new_action,
            detect_ood=True,
            calculate_uncertainty=True
        )
        
        logger.info(f"Trust Score: {prediction['prediction']}")
        logger.info(f"Confidence: {prediction.get('confidence', 'N/A')}")
        logger.info(f"Uncertainty: {prediction.get('uncertainty', 'N/A')}")
        logger.info(f"Out-of-Distribution: {prediction.get('is_ood', False)}")
        
        # Step 5: Governance validation (simulated)
        if prediction.get('confidence', 0) > 0.7 and not prediction.get('is_ood', False):
            logger.info("✓ Governance: Prediction approved")
            decision = "APPROVE"
        else:
            logger.info("✗ Governance: Prediction requires human review")
            decision = "REVIEW"
        
        return {
            'scenario': 'trust_score_prediction',
            'decision': decision,
            'trust_score': float(prediction['prediction'][0][0]) if hasattr(prediction['prediction'], '__iter__') else float(prediction['prediction']),
            'confidence': prediction.get('confidence'),
            'governance_approved': decision == "APPROVE"
        }
    
    async def scenario_2_kpi_forecasting(self):
        """
        Scenario 2: KPI Forecasting with LSTM
        
        Flow:
        1. LSTM specialist created for time-series forecasting
        2. Train on historical KPI data
        3. Forecast next 7 days
        4. Monitor for threshold violations
        5. Alert if predicted breach
        """
        logger.info("\n=== Scenario 2: KPI Forecasting (LSTM) ===")
        
        # Step 1: Create LSTM specialist
        lstm_specialist = await self.substrate.create_specialist(
            specialist_type="LSTM",
            specialist_id="kpi_forecaster_lstm",
            cognitive_functions=[CognitiveFunction.SIMULATION_FORECASTING],
            input_size=1,
            hidden_size=64,
            num_layers=2,
            sequence_length=30,
            forecast_horizon=7
        )
        
        # Step 2: Generate synthetic time-series (in production: from KPI tables)
        t = np.linspace(0, 100, 500)
        time_series = np.sin(t) + 0.1 * np.random.randn(500)
        time_series = time_series.reshape(-1, 1).astype(np.float32)
        
        # Step 3: Train LSTM
        logger.info("Training LSTM forecaster...")
        train_result = await self.substrate.train_specialist(
            specialist_id="kpi_forecaster_lstm",
            X_train=time_series,
            y_train=None,  # LSTM creates sequences internally
            epochs=30,
            batch_size=16,
            learning_rate=0.001,
            verbose=False
        )
        logger.info(f"Training complete: {train_result}")
        
        # Step 4: Forecast next 7 steps
        history = time_series[-30:]  # Last 30 timesteps
        forecast_result = await lstm_specialist.forecast(history, steps=7)
        predictions, confidence = forecast_result
        
        logger.info(f"7-Day Forecast: {predictions}")
        logger.info(f"Forecast Confidence: {confidence}")
        
        # Step 5: Monitor for threshold violations
        threshold = 1.5  # Example threshold
        violations = [p for p in predictions if p > threshold]
        
        if violations:
            logger.warning(f"⚠ Alert: {len(violations)} predicted threshold violations!")
            alert = True
        else:
            logger.info("✓ No threshold violations predicted")
            alert = False
        
        return {
            'scenario': 'kpi_forecasting',
            'forecast': predictions.tolist(),
            'confidence': confidence,
            'threshold': threshold,
            'violations': len(violations),
            'alert_triggered': alert
        }
    
    async def scenario_3_policy_analysis(self):
        """
        Scenario 3: Policy Analysis with Transformer
        
        Flow:
        1. Transformer specialist (BERT) for NLP
        2. Extract embeddings from policy documents
        3. Analyze semantic similarity
        4. Governance compliance check
        5. Flag non-compliant policies
        """
        logger.info("\n=== Scenario 3: Policy Analysis (Transformer) ===")
        
        # Step 1: Create Transformer specialist
        transformer_specialist = await self.substrate.create_specialist(
            specialist_type="Transformer",
            specialist_id="policy_analyzer_bert",
            cognitive_functions=[CognitiveFunction.PATTERN_INTERPRETATION],
            model_name="distilbert-base-uncased",
            num_labels=2  # compliant / non-compliant
        )
        
        # Step 2: Sample policy texts
        policies = [
            "All user data must be encrypted at rest and in transit",
            "Personal information shall be processed only with explicit consent",
            "We may share your data with third parties for marketing purposes"
        ]
        
        # Step 3: Extract embeddings
        logger.info("Extracting policy embeddings...")
        embeddings = []
        for policy in policies:
            embedding = await transformer_specialist.get_embeddings(policy)
            embeddings.append(embedding)
        
        logger.info(f"Extracted {len(embeddings)} policy embeddings (dim={len(embeddings[0])})")
        
        # Step 4: Compute semantic similarity (cosine similarity)
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        # Baseline compliant policy (first one)
        baseline = embeddings[0]
        similarities = [cosine_similarity(baseline, emb) for emb in embeddings]
        
        logger.info("Policy Similarity Scores:")
        for i, (policy, sim) in enumerate(zip(policies, similarities)):
            logger.info(f"  Policy {i+1}: {sim:.3f} - {policy[:50]}...")
        
        # Step 5: Governance compliance (policies with low similarity flagged)
        compliance_threshold = 0.85
        non_compliant = [
            {'policy': p, 'score': s}
            for p, s in zip(policies, similarities)
            if s < compliance_threshold
        ]
        
        if non_compliant:
            logger.warning(f"⚠ {len(non_compliant)} non-compliant policies detected")
        else:
            logger.info("✓ All policies compliant")
        
        return {
            'scenario': 'policy_analysis',
            'policies_analyzed': len(policies),
            'embeddings_extracted': len(embeddings),
            'similarities': similarities,
            'non_compliant_count': len(non_compliant),
            'non_compliant_policies': non_compliant
        }
    
    async def scenario_4_document_classification(self):
        """
        Scenario 4: Document Classification with CNN
        
        Flow:
        1. CNN specialist for image/document classification
        2. Train on document images (3 categories)
        3. Classify new document
        4. Route based on classification
        5. Update intelligence layer
        """
        logger.info("\n=== Scenario 4: Document Classification (CNN) ===")
        
        # Step 1: Create CNN specialist
        cnn_specialist = await self.substrate.create_specialist(
            specialist_type="CNN",
            specialist_id="doc_classifier_cnn",
            cognitive_functions=[CognitiveFunction.PATTERN_INTERPRETATION],
            num_classes=3,  # contract, policy, form
            input_channels=1,
            image_size=28
        )
        
        # Step 2: Generate synthetic document images
        X_train = np.random.randn(300, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 3, size=300)
        
        # Step 3: Train CNN
        logger.info("Training document classifier...")
        train_result = await self.substrate.train_specialist(
            specialist_id="doc_classifier_cnn",
            X_train=X_train,
            y_train=y_train,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            verbose=False
        )
        logger.info(f"Training complete: {train_result}")
        
        # Step 4: Classify new document
        new_document = np.random.randn(1, 1, 28, 28).astype(np.float32)
        prediction = await self.substrate.predict_with_specialist(
            specialist_id="doc_classifier_cnn",
            X=new_document,
            detect_ood=False
        )
        
        class_names = ["contract", "policy", "form"]
        predicted_class = int(prediction['prediction'])
        
        logger.info(f"Document Classification: {class_names[predicted_class]}")
        logger.info(f"Confidence: {prediction.get('confidence', 'N/A')}")
        
        # Step 5: Route based on classification
        routing = {
            0: "governance_kernel",  # contract
            1: "governance_kernel",  # policy
            2: "interface_kernel"    # form
        }
        
        destination = routing[predicted_class]
        logger.info(f"→ Routing to: {destination}")
        
        return {
            'scenario': 'document_classification',
            'document_type': class_names[predicted_class],
            'confidence': prediction.get('confidence'),
            'routed_to': destination
        }
    
    async def scenario_5_anomaly_detection_ensemble(self):
        """
        Scenario 5: Anomaly Detection with Ensemble (Autoencoder + Isolation Forest)
        
        Flow:
        1. Create Autoencoder (deep learning) and Isolation Forest (classical ML)
        2. Train both on normal data
        3. Detect anomalies using ensemble
        4. Alert on consensus anomalies
        5. Trigger governance review
        """
        logger.info("\n=== Scenario 5: Anomaly Detection Ensemble ===")
        
        # Step 1: Create Autoencoder specialist
        autoencoder = await self.substrate.create_specialist(
            specialist_type="Autoencoder",
            specialist_id="anomaly_autoencoder",
            cognitive_functions=[CognitiveFunction.ANOMALY_DETECTION],
            input_dim=20,
            latent_dim=5,
            hidden_sizes=[16, 8]
        )
        
        # Step 2: Create Isolation Forest specialist
        isolation_forest = await self.substrate.create_specialist(
            specialist_type="IsolationForest",
            specialist_id="anomaly_isolation_forest",
            cognitive_functions=[CognitiveFunction.ANOMALY_DETECTION],
            contamination=0.1,
            n_estimators=100
        )
        
        # Step 3: Generate normal data + anomalies
        X_normal = np.random.randn(1000, 20).astype(np.float32)
        X_test_normal = np.random.randn(100, 20).astype(np.float32)
        X_test_anomaly = np.random.randn(20, 20).astype(np.float32) * 5  # Anomalies
        
        # Step 4: Train both models
        logger.info("Training anomaly detectors...")
        
        await self.substrate.train_specialist(
            specialist_id="anomaly_autoencoder",
            X_train=X_normal,
            epochs=20,
            batch_size=32,
            verbose=False
        )
        
        await self.substrate.train_specialist(
            specialist_id="anomaly_isolation_forest",
            X_train=X_normal
        )
        
        # Calibrate autoencoder threshold
        await autoencoder.calibrate_threshold(X_normal, percentile=95)
        
        logger.info("Training complete")
        
        # Step 5: Ensemble prediction on test data
        X_test = np.vstack([X_test_normal, X_test_anomaly]).astype(np.float32)
        
        ensemble_result = await self.substrate.ensemble_predict(
            specialist_ids=["anomaly_autoencoder", "anomaly_isolation_forest"],
            X=X_test,
            weights=[0.6, 0.4]  # Give more weight to autoencoder
        )
        
        # Count anomalies
        predictions = ensemble_result['prediction']
        # Threshold ensemble scores
        threshold = 0.5
        anomalies_detected = sum(1 for p in predictions if p > threshold)
        
        logger.info(f"Tested {len(X_test)} samples")
        logger.info(f"Anomalies detected: {anomalies_detected}")
        logger.info(f"Ensemble confidence: {ensemble_result.get('confidence', 'N/A')}")
        
        # Step 6: Alert if anomalies found
        if anomalies_detected > 0:
            logger.warning(f"⚠ Alert: {anomalies_detected} anomalies detected!")
            logger.warning("→ Triggering governance review...")
            alert = True
        else:
            logger.info("✓ No anomalies detected")
            alert = False
        
        return {
            'scenario': 'anomaly_detection_ensemble',
            'samples_tested': len(X_test),
            'anomalies_detected': anomalies_detected,
            'ensemble_models': 2,
            'alert_triggered': alert
        }
    
    async def scenario_6_active_learning_loop(self):
        """
        Scenario 6: Active Learning Loop
        
        Flow:
        1. Train Random Forest classifier
        2. Query uncertain samples from unlabeled pool
        3. Simulate human labeling
        4. Retrain with new labels
        5. Measure improvement
        """
        logger.info("\n=== Scenario 6: Active Learning Loop ===")
        
        # Step 1: Create Random Forest specialist
        rf_specialist = await self.substrate.create_specialist(
            specialist_type="RandomForest",
            specialist_id="active_learner_rf",
            cognitive_functions=[CognitiveFunction.AUTONOMOUS_LEARNING],
            n_estimators=50,
            max_depth=10
        )
        
        # Step 2: Initial training data (small labeled set)
        X_labeled = np.random.randn(100, 10)
        y_labeled = (X_labeled[:, 0] > 0).astype(int)
        
        logger.info("Initial training with 100 labeled samples...")
        await self.substrate.train_specialist(
            specialist_id="active_learner_rf",
            X_train=X_labeled,
            y_train=y_labeled
        )
        
        # Step 3: Large unlabeled pool
        X_pool = np.random.randn(500, 10)
        y_pool_true = (X_pool[:, 0] > 0).astype(int)  # Ground truth (hidden)
        
        # Step 4: Active learning query
        logger.info("Querying most uncertain samples...")
        indices, samples = await self.substrate.active_learning_query(
            specialist_id="active_learner_rf",
            X_pool=X_pool,
            n_samples=20,
            strategy="uncertainty"
        )
        
        if indices is None:
            logger.warning("Active learning not available")
            return {'scenario': 'active_learning', 'status': 'not_available'}
        
        logger.info(f"Selected {len(indices)} samples for labeling")
        
        # Step 5: Simulate human labeling (use ground truth)
        new_labels = y_pool_true[indices]
        
        # Step 6: Retrain with augmented dataset
        X_augmented = np.vstack([X_labeled, samples])
        y_augmented = np.concatenate([y_labeled, new_labels])
        
        logger.info(f"Retraining with {len(X_augmented)} samples (100 → {len(X_augmented)})...")
        await self.substrate.train_specialist(
            specialist_id="active_learner_rf",
            X_train=X_augmented,
            y_train=y_augmented
        )
        
        # Step 7: Evaluate improvement
        X_test = np.random.randn(100, 10)
        y_test = (X_test[:, 0] > 0).astype(int)
        
        prediction = await self.substrate.predict_with_specialist(
            specialist_id="active_learner_rf",
            X=X_test
        )
        
        accuracy = (prediction['prediction'] == y_test).mean()
        logger.info(f"Test Accuracy after active learning: {accuracy:.2%}")
        
        return {
            'scenario': 'active_learning',
            'initial_samples': 100,
            'queried_samples': len(indices),
            'final_samples': len(X_augmented),
            'test_accuracy': float(accuracy)
        }
    
    async def run_all_scenarios(self):
        """Run all integration scenarios."""
        logger.info("=" * 60)
        logger.info("Grace ML/DL End-to-End Integration Demo")
        logger.info("=" * 60)
        
        results = {}
        
        try:
            results['scenario_1'] = await self.scenario_1_trust_score_prediction()
        except Exception as e:
            logger.error(f"Scenario 1 failed: {e}")
            results['scenario_1'] = {'error': str(e)}
        
        try:
            results['scenario_2'] = await self.scenario_2_kpi_forecasting()
        except Exception as e:
            logger.error(f"Scenario 2 failed: {e}")
            results['scenario_2'] = {'error': str(e)}
        
        try:
            results['scenario_3'] = await self.scenario_3_policy_analysis()
        except Exception as e:
            logger.error(f"Scenario 3 failed: {e}")
            results['scenario_3'] = {'error': str(e)}
        
        try:
            results['scenario_4'] = await self.scenario_4_document_classification()
        except Exception as e:
            logger.error(f"Scenario 4 failed: {e}")
            results['scenario_4'] = {'error': str(e)}
        
        try:
            results['scenario_5'] = await self.scenario_5_anomaly_detection_ensemble()
        except Exception as e:
            logger.error(f"Scenario 5 failed: {e}")
            results['scenario_5'] = {'error': str(e)}
        
        try:
            results['scenario_6'] = await self.scenario_6_active_learning_loop()
        except Exception as e:
            logger.error(f"Scenario 6 failed: {e}")
            results['scenario_6'] = {'error': str(e)}
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Integration Test Summary")
        logger.info("=" * 60)
        
        for scenario_name, result in results.items():
            status = "✓ PASS" if 'error' not in result else "✗ FAIL"
            logger.info(f"{scenario_name}: {status}")
            if 'error' in result:
                logger.error(f"  Error: {result['error']}")
        
        # List all specialists
        specialists = self.substrate.list_specialists()
        logger.info(f"\nTotal Specialists Created: {len(specialists)}")
        for spec in specialists:
            logger.info(f"  - {spec['specialist_id']} ({spec['specialist_type']})")
        
        # Substrate metrics
        metrics = self.substrate.get_metrics()
        logger.info(f"\nCognitive Substrate Metrics:")
        logger.info(f"  Events Processed: {metrics['total_events_processed']}")
        logger.info(f"  Deep Learning: {metrics['deep_learning_available']}")
        logger.info(f"  Device: {metrics['device']}")
        
        # Cleanup
        await self.substrate.cleanup()
        
        return results


async def main():
    """Main entry point."""
    demo = EndToEndIntegrationDemo()
    results = await demo.run_all_scenarios()
    return results


if __name__ == "__main__":
    asyncio.run(main())
