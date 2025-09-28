# Grace Intelligence Kernel

The Intelligence Kernel is the brain of the Grace AI system that routes tasks → specialists → models, fuses outputs, reasons about uncertainty, and collaborates with Governance, MLT, Memory, and Ingress kernels.

## Purpose

- **Plan → Select → Execute → Explain** across 21-specialist quorum (classic ML/DL + deep + RL + XAI)
- **Guarantee policy safety** (governance gates), traceability (lineage/metrics), and adaptability (MLT loop)  
- **Serve online inference** (low-latency) and batch jobs with canary/shadow + rollback

## Architecture

```
intelligence/
├── router/                 # Task detection + policy-aware routing
├── planner/                # Candidate graph planning (search space, constraints)
├── adapters/               # Model wrappers (LR/SVM/Tree/GBM/NN/Transformer/RL/…)
├── specialists/            # 21 domain specialists (reports + risk + XAI)
├── ensembler/              # Stacking/blending/voting + uncertainty
├── explainer/              # SHAP/IG/attention maps/calibration
├── inference/              # Online service, canary/shadow, A/B routes
├── batch/                  # Offline pipelines (backfills, re-scores)
├── evaluation/             # Metrics, drift checks, fairness probes
├── governance_bridge/      # Policy validation + approvals
├── mlt_bridge/             # Insights/plans exchange
├── memory_bridge/          # Feature retrieval + precedent recall
├── contracts/              # JSON/YAML schemas
├── snapshots/              # State exports, lineage
├── db/                     # Database schemas
└── intelligence_service.py # FastAPI façade
```

## Core Components

### 1. Task Router (`router/task_router.py`)
- Detects task type from request or auto-detects via heuristics/metadata
- Policy screening (governance thresholds, blocklists, PII labels)
- Specialist selection based on task type and constraints
- Route optimization considering latency/cost/accuracy trade-offs

### 2. Plan Builder (`planner/plan_builder.py`)
- Creates execution plans from routing decisions
- Constraint validation and optimization
- Search space exploration
- Pre-flight checks for data/model availability

### 3. Meta Ensembler (`ensembler/meta_learner.py`)
- Stacking/blending/voting ensemble methods
- Uncertainty estimation: calibration, variance, entropy
- Meta-model training and prediction
- Dynamic ensemble weight adjustment

### 4. Inference Engine (`inference/engine.py`)
- Single model and ensemble inference execution
- Canary deployments with gradual traffic ramp-up
- Shadow deployments for comparison and validation
- A/B testing routes with traffic splitting
- Policy gating based on uncertainty thresholds

### 5. Governance Bridge (`governance_bridge.py`)
- Integration with Grace Governance Kernel
- Policy validation for plans and results
- Approval workflows for high-risk operations
- Compliance checking and audit trail

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /request` - Submit task request (returns 202 with req_id)
- `GET /result/{req_id}` - Get inference result
- `POST /plan/preview` - Preview execution plan

### Operations
- `POST /canary/promote` - Promote canary deployment  
- `POST /snapshot/export` - Export system snapshot
- `POST /rollback` - Rollback to snapshot
- `GET /metrics` - Get performance metrics

## Data Contracts

### TaskRequest
```json
{
  "req_id": "req_20240928_143022_1234",
  "task": "classification",
  "input": {
    "X": {"feature1": 0.5, "feature2": 1.2},
    "modality": "tabular"
  },
  "context": {
    "latency_budget_ms": 500,
    "explanation": true,
    "env": "prod"
  },
  "constraints": {
    "min_calibration": 0.95
  }
}
```

### InferenceResult
```json
{
  "req_id": "req_20240928_143022_1234",
  "outputs": {
    "y_hat": "approved",
    "confidence": 0.85,
    "proba": [0.15, 0.85]
  },
  "metrics": {"calibration": 0.96},
  "uncertainties": {"variance": 0.02},
  "lineage": {
    "plan_id": "plan_123",
    "models": ["xgb@1.3.2", "neural_net@2.1.0"],
    "ensemble": "vote"
  },
  "governance": {"approved": true},
  "timing": {"total_ms": 245}
}
```

## Usage

### Running the Service
```bash
# Start the Intelligence Service
cd intelligence
python intelligence_service.py

# Or with uvicorn
uvicorn intelligence_service:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run comprehensive tests
python test_intelligence_kernel.py

# Test health endpoint
curl http://localhost:8000/api/intel/v1/health

# Submit a classification task
curl -X POST http://localhost:8000/api/intel/v1/request \
  -H "Content-Type: application/json" \
  -d '{
    "task": "classification",
    "input": {"X": {"feature1": 0.5}, "modality": "tabular"},
    "context": {"latency_budget_ms": 500, "env": "dev"}
  }'

# Get result (use req_id from above)
curl http://localhost:8000/api/intel/v1/result/req_20240928_143022_1234
```

### Configuration
Edit `config/intelligence.yaml` to customize:
- Policy thresholds (confidence, calibration, fairness)
- Router hybrid weights (latency vs quality)
- Ensemble configuration
- Specialist weights and models
- Canary/shadow deployment settings

## Deployment Capabilities

### Canary Deployments
- Gradual traffic ramp-up: 5% → 25% → 50% → 100%
- Auto-promotion based on success metrics
- Automatic rollback on failure

### Shadow Deployments
- Mirror traffic to alternative models
- Compare agreement and performance
- Zero production impact

### Snapshots & Rollback
- Export complete system state
- Hash-verified integrity
- One-click rollback to previous state

## Monitoring

### Key Metrics
- **Quality**: success rate, regret_est, calibration, AUROC/F1/RMSE
- **Risk**: fairness delta, drift score, uncertainty
- **Ops**: P95 latency, cost units/query, canary pass rate, shadow agreement
- **Governance**: % plans auto-approved, violation rate, time-to-approval

### Database Schema
- `intel_requests` - Task requests
- `intel_plans` - Execution plans  
- `intel_results` - Inference results
- `intel_experiences` - Meta-learning data
- `intel_snapshots` - System snapshots
- `intel_policy_violations` - Governance events

## Integration

The Intelligence Kernel integrates with:
- **Governance Kernel**: Policy validation and approval
- **MLT Kernel**: Meta-learning insights and adaptation plans
- **Memory Kernel**: Feature retrieval and precedent recall
- **Event Mesh**: Async communication and event processing

## 21 Specialists

The system includes specialists for:
1. **Tabular**: Classification (XGB, RF, LGB), Regression (LR, Ridge, Lasso, RF, GBM), Gaussian Process, Bayesian
2. **Deep Learning**: Neural networks, DNNs, MLP
3. **Ensemble**: Meta-learners, stacking, voting
4. **Clustering**: K-means, hierarchical, DBSCAN, spectral, GMM  
5. **NLP**: BERT, RoBERTa, transformers
6. **Vision**: ResNet, EfficientNet, CNNs
7. **Time Series**: LSTM, GRU, Prophet
8. **RL**: DQN, Policy Gradient agents

## Security

- Policy enforcement at multiple levels
- PII detection and blocking
- Model allowlists/denylists
- Governance approval for high-risk operations
- Audit trail for all decisions
- Encrypted snapshots with hash verification