#!/usr/bin/env python3
"""
MLDL Kernel Demo - Demonstrates the model lifecycle management capabilities.
"""

import asyncio
import numpy as np
from datetime import datetime

from grace.mldl import (
    MLDLService,
    LogisticRegressionAdapter,
    XGBAdapter,
    KMeansAdapter,
    TrainingJobRunner,
    ModelRegistry,
    DeploymentManager,
    evaluate,
    calibration,
    fairness,
)


async def main():
    """Demonstrate MLDL Kernel capabilities."""
    print("🚀 MLDL Kernel Demo - Model Lifecycle Management")
    print("=" * 60)

    # 1. Model Adapter Demo
    print("\n📊 1. Model Adapter Demo")
    print("-" * 30)

    # Create sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # Test different model adapters
    adapters = [
        LogisticRegressionAdapter(model_key="demo_lr"),
        XGBAdapter(task="classification", model_key="demo_xgb"),
        KMeansAdapter(n_clusters=3, model_key="demo_kmeans"),
    ]

    for adapter in adapters:
        print(f"  Training {adapter.name}...")
        if adapter.task == "clustering":
            adapter.fit(X)
            predictions = adapter.predict(X[:10])
        else:
            adapter.fit(X, y)
            predictions = adapter.predict(X[:10])
            if hasattr(adapter, "predict_proba"):
                probabilities = adapter.predict_proba(X[:10])
                print(f"    Sample predictions: {predictions[:3]}")
                print(
                    f"    Sample probabilities: {probabilities[:3, 1] if probabilities is not None else 'N/A'}"
                )

        print(f"    ✅ {adapter.name} trained successfully")

    # 2. Training Job Demo
    print("\n🎯 2. Training Job Demo")
    print("-" * 30)

    runner = TrainingJobRunner()

    # Mock training job specification
    job_spec = {
        "job_id": "demo_job_001",
        "dataset_id": "demo_dataset",
        "version": "1.0.0",
        "spec": {
            "model_key": "demo.classification.lr",
            "family": "lr",
            "task": "classification",
            "adapter": "grace.mldl.adapters.classic.LogisticRegressionAdapter",
            "hyperparams": {"C": 1.0, "solver": "liblinear"},
            "feature_view": "demo_features",
        },
        "cv": {"folds": 3, "stratify": True},
        "hpo": {"strategy": "random", "max_trials": 5},
    }

    print("  Running training job...")
    trained_bundle = await runner.run(job_spec)
    print(
        f"    ✅ Training completed: {trained_bundle['model_key']}@{trained_bundle['version']}"
    )
    print(f"    📈 CV Score: {trained_bundle['metrics'].get('cv_score', 'N/A'):.4f}")

    # 3. Evaluation Demo
    print("\n📋 3. Model Evaluation Demo")
    print("-" * 30)

    # Generate sample predictions for evaluation
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize

    # Test evaluation functions
    metrics = evaluate("classification", y_true, y_pred, y_proba)
    print(f"  Classification Metrics: {metrics}")

    cal_result = calibration(y_proba, y_true)
    print(f"  Calibration ECE: {cal_result['ece']:.4f}")

    # Mock fairness evaluation
    groups = {
        "group_a": np.random.choice([True, False], 100, p=[0.4, 0.6]),
        "group_b": np.random.choice([True, False], 100, p=[0.6, 0.4]),
    }

    fairness_result = fairness(y_true, y_pred, groups, y_proba)
    print(
        f"  Fairness Delta: {fairness_result.get('metrics', {}).get('overall_delta', 'N/A'):.4f}"
    )

    # 4. Model Registry Demo
    print("\n🏛️ 4. Model Registry Demo")
    print("-" * 30)

    registry = ModelRegistry()

    # Register the trained model
    registration_result = await registry.register(trained_bundle)
    print(
        f"  ✅ Model registered: {registration_result['model_key']}@{registration_result['version']}"
    )

    # Query the registry
    registered_model = await registry.get(
        trained_bundle["model_key"], trained_bundle["version"]
    )
    if registered_model:
        print(f"  📚 Retrieved model: {registered_model['model_key']}")
        print(f"  📊 Model metrics: {list(registered_model.get('metrics', {}).keys())}")

    # 5. Deployment Demo
    print("\n🚀 5. Deployment Demo")
    print("-" * 30)

    deployment_manager = DeploymentManager()

    # Request deployment
    deployment_spec = {
        "target_env": "staging",
        "canary_pct": 10,
        "shadow": False,
        "guardrails": {
            "max_latency_p95_ms": 500,
            "min_calibration": 0.8,
            "rollback_on": ["metric_drop", "drift_spike"],
        },
    }

    deployment = await deployment_manager.request(
        trained_bundle["model_key"], trained_bundle["version"], deployment_spec
    )

    print(f"  ✅ Deployment requested: {deployment['deployment_id']}")
    print(f"  📈 Canary %: {deployment['canary_pct']}%")
    print(f"  🛡️ Environment: {deployment['environment']}")

    # 6. MLDL Service Demo
    print("\n🌐 6. MLDL Service Demo")
    print("-" * 30)

    mldl_service = MLDLService()

    print(f"  ✅ MLDL Service initialized")
    print(f"  🔗 FastAPI app available at: {mldl_service.app.title}")
    print(f"  📚 Available endpoints: /health, /train, /evaluate, /deploy, /registry/*")

    # 7. Integration Summary
    print("\n🎉 Integration Summary")
    print("-" * 30)
    print("  MLDL Kernel provides:")
    print("  ✅ Unified model adapters (LR, XGB, KMeans, etc.)")
    print("  ✅ Training job runner with HPO and CV")
    print("  ✅ Comprehensive evaluation (metrics, calibration, fairness)")
    print("  ✅ Model registry with lineage and version control")
    print("  ✅ Deployment manager with canary and shadow deployments")
    print("  ✅ Monitoring and alerting capabilities")
    print("  ✅ Snapshot and rollback functionality")
    print("  ✅ Bridge integrations to other Grace kernels")
    print("  ✅ FastAPI service with REST endpoints")

    print("\n🌟 MLDL Kernel Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
