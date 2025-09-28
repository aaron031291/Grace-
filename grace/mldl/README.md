# MLDL Kernel - Model Lifecycle Management

The MLDL (Machine Learning/Deep Learning) Kernel provides comprehensive model lifecycle management for the Grace AI system. It implements the complete model journey from training to deployment, monitoring, and rollback.

## 🎯 Purpose

- **Unified Model Lifecycle**: define → train → evaluate → register → deploy → monitor → rollback
- **Multi-Algorithm Support**: Classic ML (LR/SVM/KNN/Trees/GBM/SVR/Naive Bayes), Deep Learning (CNN/RNN/LSTM/Transformer/GAN), Reinforcement Learning (Q-learning/DQN/PG), Clustering (KMeans/Agglo/DBSCAN), and Dimensionality Reduction (PCA)
- **Enterprise Features**: HPO, calibration, fairness evaluation, drift monitoring, model registry, canary deployments, and automated rollback

## 🏗️ Architecture

```
mldl/
├── adapters/           # Model wrappers with consistent I/O
│   ├── base.py        # BaseModelAdapter interface
│   ├── classic.py     # Classic ML adapters (LR, SVM, KNN, XGB, etc.)
│   └── clustering.py  # Clustering & dimensionality reduction
├── training/          # Job runner with CV, HPO, early stopping
│   └── job.py
├── evaluation/        # Metrics, calibration, fairness evaluation
│   └── metrics.py
├── registry/          # Model registry with lineage tracking
│   └── registry.py
├── deployment/        # Canary/shadow deployment manager
│   └── manager.py
├── monitoring/        # Live metrics and SLO monitoring
│   └── collector.py
├── snapshots/         # State snapshots and rollback
│   └── manager.py
├── bridges/           # Integrations with other Grace kernels
│   ├── mesh_bridge.py
│   ├── gov_bridge.py
│   ├── mlt_bridge.py
│   ├── intel_bridge.py
│   ├── memory_bridge.py
│   └── ingress_bridge.py
├── contracts/         # Data contracts and schemas
│   ├── mldl.modelspec.schema.json
│   ├── mldl.events.yaml
│   └── mldl.api.openapi.yaml
└── mldl_service.py   # FastAPI service facade
```

## 🚀 Key Features

### Model Adapters
- **Consistent Interface**: All models use the same `BaseModelAdapter` interface
- **Save/Load Support**: Persistent model storage with metadata
- **Explanation Support**: Built-in model explainability features
- **Mock Support**: Graceful degradation when dependencies are missing

### Training & HPO
- **Cross-Validation**: Stratified, time-aware, and standard CV strategies
- **Hyperparameter Optimization**: Bayesian, grid, random, and Hyperband strategies
- **Early Stopping**: Configurable patience and improvement thresholds
- **Resource Management**: Cost tracking and constraint enforcement

### Evaluation & Quality
- **Task-Specific Metrics**: Classification, regression, clustering metrics
- **Calibration Assessment**: ECE calculation and isotonic/Platt calibration
- **Fairness Evaluation**: Group parity and bias detection
- **Robustness Testing**: Noise sensitivity and adversarial robustness

### Model Registry
- **Version Management**: Semantic versioning with lineage tracking
- **Metadata Storage**: Metrics, calibration, fairness, and robustness data
- **Governance Integration**: Approval workflows and compliance checks
- **Query Interface**: Search by task, constraints, and performance criteria

### Deployment Management
- **Canary Deployments**: Gradual rollout with automated promotion
- **Shadow Deployments**: Risk-free parallel evaluation
- **Blue/Green Deployments**: Zero-downtime deployment strategy
- **Guardrails**: Automatic rollback on SLO violations or drift

### Monitoring & Alerting
- **Live Metrics**: Performance, latency, accuracy, and cost tracking
- **SLO Management**: Configurable service level objectives
- **Drift Detection**: Statistical and embedding-based drift monitoring
- **Alert Management**: Severity-based alerting with resolution tracking

### Snapshots & Rollback
- **State Snapshots**: Complete system state capture
- **Rollback Capability**: Restore to any previous snapshot
- **Impact Assessment**: Risk analysis for rollback decisions
- **Audit Trail**: Complete rollback history and reasoning

## 📋 Data Contracts

### ModelSpec
```json
{
  "model_key": "tabular.classification.xgb",
  "family": "xgb",
  "task": "classification", 
  "adapter": "adapters.xgb.XGBAdapter",
  "hyperparams": {"max_depth": 6, "eta": 0.1},
  "feature_view": "customer_features_v1",
  "tags": ["baseline", "high_performance"],
  "constraints": {"latency_ms": 100, "cost_units": 0.01}
}
```

### TrainedBundle
```json
{
  "model_key": "tabular.classification.xgb",
  "version": "1.3.2",
  "artifact_uri": "/models/tabular.classification.xgb/20240315_143022",
  "metrics": {"f1": 0.87, "auroc": 0.94, "precision": 0.89},
  "calibration": {"ece": 0.03, "method": "isotonic"},
  "fairness": {"delta": 0.015, "groups": ["gender", "region"]},
  "robustness": {"noise_sensitivity": 0.12},
  "lineage": {
    "dataset_id": "customer_data_v5",
    "version": "2024.03.15",
    "feature_view": "customer_features_v1",
    "trainer_hash": "sha256:abc123..."
  },
  "validation_hash": "sha256:def456...",
  "created_at": "2024-03-15T14:30:22Z"
}
```

### DeploymentSpec
```json
{
  "target_env": "prod",
  "canary_pct": 5,
  "shadow": false,
  "guardrails": {
    "min_calibration": 0.8,
    "fairness_delta_max": 0.02,
    "max_latency_p95_ms": 500,
    "rollback_on": ["metric_drop", "drift_spike", "violation"]
  },
  "route": "single"
}
```

## 🔌 API Endpoints

### REST API (FastAPI)
Base URL: `/api/mldl/v1`

- `GET /health` - Health check
- `POST /train` - Start training job  
- `GET /jobs/{job_id}/status` - Get training status
- `POST /evaluate` - Evaluate model
- `POST /register` - Register trained model
- `GET /registry/{model_key}/{version}` - Get model details
- `POST /deploy` - Deploy model
- `GET /deployments/{deployment_id}` - Get deployment status
- `POST /canary/promote` - Promote canary deployment
- `POST /snapshot/export` - Create system snapshot
- `POST /rollback` - Rollback to snapshot
- `GET /metrics/live` - Get live metrics

## 🌐 Events & Integration

### Published Events
- `MLDL_TRAINING_STARTED` - Training job initiated
- `MLDL_CANDIDATE_READY` - Model training completed
- `MLDL_EVALUATED` - Model evaluation completed
- `MLDL_MODEL_REGISTERED` - Model registered in registry
- `MLDL_DEPLOYMENT_REQUESTED` - Deployment requested
- `MLDL_DEPLOYMENT_PROMOTED` - Canary promoted
- `MLDL_DRIFT_ALERT` - Model drift detected
- `MLDL_VIOLATION` - SLO or policy violation
- `ROLLBACK_REQUESTED` - System rollback requested

### Bridge Integrations
- **Governance Bridge**: Approval workflows and policy compliance
- **MLT Bridge**: Meta-learning experience sharing and adaptation
- **Intelligence Bridge**: Specialist consensus and recommendations  
- **Memory Bridge**: Feature view management and lineage
- **Ingress Bridge**: Data quality monitoring and alerts
- **Mesh Bridge**: Event routing and system coordination

## 🎮 Usage Example

```python
import asyncio
from grace.mldl import MLDLService, LogisticRegressionAdapter, TrainingJobRunner

async def main():
    # 1. Create and train a model
    adapter = LogisticRegressionAdapter(model_key="demo_lr")
    adapter.fit(X_train, y_train)
    predictions = adapter.predict(X_test)
    
    # 2. Run training job with HPO
    runner = TrainingJobRunner()
    job_spec = {
        "spec": {"model_key": "lr_model", "family": "lr", "task": "classification"},
        "hpo": {"strategy": "bayes", "max_trials": 50}
    }
    trained_bundle = await runner.run(job_spec)
    
    # 3. Start MLDL service
    service = MLDLService()
    await service.start()  # FastAPI server on port 8080
    
asyncio.run(main())
```

## 📊 KPIs & Metrics

### Quality Metrics
- Cross-validation scores and test performance
- Calibration error (ECE) and reliability
- Fairness delta across demographic groups
- Model robustness and adversarial resistance

### Operational Metrics  
- P95 latency and throughput (QPS)
- Canary pass rate and promotion success
- Deployment time and rollback frequency
- Resource utilization and cost per prediction

### Risk Metrics
- Drift alerts per week and severity distribution
- Policy violations and governance exceptions
- Unexplained variance and model degradation
- Security incidents and data breaches

## 🔧 Configuration

Default MLDL configuration:
```yaml
mldl:
  training:
    cv: {folds: 5, stratify: true}
    hpo: {strategy: "bayes", max_trials: 50, early_stop: true}
  evaluation:
    metrics:
      classification: [f1, auroc, logloss, calibration]
      regression: [rmse, mae, r2]
      clustering: [silhouette]
  calibration: {default: "isotonic"}
  fairness: {delta_max: 0.02, groups: ["gender", "region"]}
  deployment:
    canary_steps: [5, 25, 50, 100]
    promotion_window: 3600
    rollback_on: ["metric_drop", "drift_spike", "violation"]
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install numpy scikit-learn joblib fastapi uvicorn
   ```

2. **Run Demo**:
   ```bash
   python demo_mldl_kernel.py
   ```

3. **Start Service**:
   ```python
   from grace.mldl import MLDLService
   service = MLDLService()
   # Access FastAPI app at service.app
   ```

The MLDL Kernel provides production-ready model lifecycle management with enterprise-grade governance, monitoring, and operational capabilities.