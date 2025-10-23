# ML/DL Specialists Module

## Overview

The ML/DL Specialists module provides a comprehensive suite of machine learning and deep learning models integrated into Grace's 4-layer architecture. Each specialist is a self-contained model with full implementation logic (not just API wrappers) that integrates seamlessly with Grace's governance, KPI monitoring, immutable audit logging, and memory systems.

## Architecture

### 4-Layer System

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Federated Meta-Learner                                │
│  • Continuous improvement via meta-learning                     │
│  • Trust score adjustment based on performance                  │
│  • Specialist ranking and recommendation                        │
│  • Concept drift detection                                      │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Governance Integration                                │
│  • Constitutional compliance validation                         │
│  • Policy enforcement                                           │
│  • Audit trail logging                                          │
│  • KPI tracking and reporting                                   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Consensus Engine                                       │
│  • Weighted voting across specialists                           │
│  • Confidence aggregation                                       │
│  • Capability-aware routing                                     │
│  • Conflict resolution                                          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Individual Specialists                                │
│  • Decision Tree    • SVM           • Random Forest             │
│  • Gradient Boost   • Naive Bayes   • K-Means                   │
│  • DBSCAN          • PCA            • Autoencoder               │
│  • (More to come: CNN, RNN, LSTM, GAN, Transformer, RL, etc.)  │
└─────────────────────────────────────────────────────────────────┘
```

## Implemented Specialists

### Supervised Learning (5 specialists)

#### DecisionTreeSpecialist
- **Algorithm**: Flowchart-like classification/regression
- **Strengths**: Interpretable, handles non-linear relationships
- **Use Cases**: Medical diagnosis, credit scoring, feature importance analysis
- **Metrics**: Accuracy, precision, recall, F1, tree depth, feature importances

#### SVMSpecialist
- **Algorithm**: Support Vector Machine with kernel tricks
- **Strengths**: Effective in high-dimensional spaces, memory efficient
- **Use Cases**: Text classification, image recognition, bioinformatics
- **Metrics**: Accuracy, support vector count, decision margin

#### RandomForestSpecialist
- **Algorithm**: Ensemble of decision trees
- **Strengths**: Reduces overfitting, provides feature importance
- **Use Cases**: Fraud detection, recommendation systems, risk assessment
- **Metrics**: Accuracy, tree consensus score, OOB error

#### GradientBoostingSpecialist
- **Algorithm**: Sequential ensemble learning
- **Strengths**: High predictive accuracy, handles various data types
- **Use Cases**: Ranking, time series prediction, anomaly detection
- **Metrics**: Accuracy, boosting stages, learning rate effectiveness

#### NaiveBayesSpecialist
- **Algorithm**: Probabilistic classification
- **Strengths**: Fast, works well with small datasets
- **Use Cases**: Spam filtering, sentiment analysis, real-time prediction
- **Metrics**: Accuracy, class probabilities, prior/posterior analysis

### Unsupervised Learning (4 specialists)

#### KMeansSpecialist
- **Algorithm**: K-means clustering
- **Strengths**: Simple, fast, guaranteed convergence
- **Use Cases**: Customer segmentation, image compression, pattern discovery
- **Metrics**: Silhouette score, Davies-Bouldin index, inertia

#### DBSCANSpecialist
- **Algorithm**: Density-based spatial clustering
- **Strengths**: Finds arbitrary shapes, robust to outliers
- **Use Cases**: Anomaly detection, geospatial analysis, network security
- **Metrics**: Clusters found, outlier count, density parameters

#### PCASpecialist
- **Algorithm**: Principal Component Analysis
- **Strengths**: Dimensionality reduction, removes correlation
- **Use Cases**: Visualization, noise reduction, feature extraction
- **Metrics**: Variance explained, compression ratio, reconstruction error

#### AutoencoderSpecialist
- **Algorithm**: Neural network encoder-decoder
- **Strengths**: Non-linear reduction, anomaly detection
- **Use Cases**: Feature learning, denoising, generative modeling
- **Metrics**: Reconstruction error, encoding dimension, anomaly threshold

## Grace Integration

### Governance Bridge
Every specialist integrates with Grace's governance system:

```python
async def validate_governance(self, X, prediction):
    """Validate prediction against constitutional rules"""
    if self.governance_bridge:
        result = await self.governance_bridge.validate({
            "specialist_id": self.specialist_id,
            "input_data": X,
            "prediction": prediction,
            "confidence": self.calculate_confidence(X),
            "timestamp": datetime.now()
        })
        return result.get("approved", True)
    return True
```

### Immutable Audit Logging
All operations logged to immutable trail:

```python
await self.log_to_immutable_trail(
    operation_type="prediction",
    operation_data={
        "samples": X.shape[0],
        "confidence": confidence,
        "compliance": compliance,
        "timestamp": datetime.now().isoformat()
    }
)
```

### KPI Monitoring
Performance metrics tracked continuously:

```python
await self.report_kpi_metrics(
    metrics=training_metrics,
    operation="training"
)
```

### Memory Bridge
Knowledge stored and retrieved:

```python
# Store model state
await self.store_in_memory(
    key="model_state",
    value={"weights": weights, "version": version}
)

# Retrieve for transfer learning
previous_state = await self.retrieve_from_memory(key="model_state")
```

## Usage Examples

### Basic Classification

```python
import numpy as np
from grace.mldl_specialists import RandomForestSpecialist, SpecialistCapability

# Initialize specialist
specialist = RandomForestSpecialist(
    task_type="classification",
    n_estimators=100
)

# Train
X_train, y_train = load_data()
metrics = await specialist.train(X_train, y_train)
print(f"Training Accuracy: {metrics.accuracy:.3f}")

# Predict
X_test = load_test_data()
prediction = await specialist.predict(X_test)

print(f"Prediction: {prediction.prediction}")
print(f"Confidence: {prediction.confidence:.3f}")
print(f"Trust Score: {prediction.trust_score:.3f}")
print(f"Constitutional Compliance: {prediction.constitutional_compliance}")
```

### Consensus-Based Prediction

```python
from grace.mldl_specialists import (
    DecisionTreeSpecialist,
    SVMSpecialist,
    RandomForestSpecialist,
    MLDLConsensusEngine,
    SpecialistCapability
)

# Initialize multiple specialists
specialists = [
    DecisionTreeSpecialist(task_type="classification"),
    SVMSpecialist(task_type="classification"),
    RandomForestSpecialist(task_type="classification")
]

# Train all
for specialist in specialists:
    await specialist.train(X_train, y_train)

# Set up consensus engine
consensus_engine = MLDLConsensusEngine(
    min_specialists_required=2,
    consensus_threshold=0.6
)

# Register specialists
for specialist in specialists:
    consensus_engine.register_specialist(specialist)

# Reach consensus
result = await consensus_engine.reach_consensus(
    X_test,
    required_capability=SpecialistCapability.CLASSIFICATION
)

print(f"Consensus Prediction: {result.final_prediction}")
print(f"Consensus Score: {result.consensus_score:.3f}")
print(f"Confidence: {result.confidence:.3f}")

# View contributions
for specialist_id, weight in result.weighted_contributions.items():
    print(f"{specialist_id}: {weight:.3f}")
```

### Federated Meta-Learning

```python
from grace.mldl_specialists import FederatedMetaLearner

# Initialize meta-learner
meta_learner = FederatedMetaLearner(
    learning_rate=0.1,
    trust_decay_rate=0.02,
    performance_window_days=7
)

# Record performance
for specialist_id, prediction in predictions.items():
    await meta_learner.record_specialist_performance(
        specialist_id=specialist_id,
        prediction=prediction,
        ground_truth=y_test,
        accuracy=calculate_accuracy(prediction, y_test)
    )

# Update trust scores
updates = await meta_learner.update_trust_scores(specialists)

for update in updates:
    print(f"{update.specialist_id}:")
    print(f"  Trust Adjustment: {update.trust_adjustment:+.3f}")
    print(f"  Trend: {update.performance_trend}")
    print(f"  Recommendations: {update.recommended_actions}")

# Federated learning round
await meta_learner.federated_learning_round(
    specialists=specialists,
    X_train=X_train,
    y_train=y_train,
    capability=SpecialistCapability.CLASSIFICATION
)

# Get rankings
rankings = meta_learner.get_specialist_ranking(
    capability=SpecialistCapability.CLASSIFICATION
)
print("Top Specialists:", rankings[:3])
```

### Clustering Pipeline

```python
from grace.mldl_specialists import (
    PCASpecialist,
    KMeansSpecialist,
    DBSCANSpecialist
)

# Dimensionality reduction
pca = PCASpecialist(n_components=0.95)  # Keep 95% variance
await pca.train(X)
pca_result = await pca.predict(X)
X_reduced = np.array(pca_result.prediction)

# Clustering
kmeans = KMeansSpecialist(n_clusters=5)
await kmeans.train(X_reduced)
cluster_result = await kmeans.predict(X_reduced)

print(f"Clusters: {cluster_result.prediction}")
print(f"Silhouette Score: {cluster_result.metadata['silhouette_score']:.3f}")
```

## Configuration

Each specialist accepts configuration parameters:

```python
specialist = RandomForestSpecialist(
    task_type="classification",
    n_estimators=200,           # Number of trees
    max_depth=15,               # Maximum tree depth
    min_samples_split=10,       # Min samples to split
    governance_bridge=gov,       # Governance integration
    kpi_monitor=kpi,            # KPI tracking
    immutable_logs=logs,        # Audit logging
    memory_bridge=memory,       # Memory storage
    min_confidence_threshold=0.8,  # Minimum confidence
    min_trust_threshold=0.7        # Minimum trust
)
```

## Metrics and Monitoring

### Training Metrics

```python
@dataclass
class TrainingMetrics:
    # Classification
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Regression
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Clustering
    silhouette_score: Optional[float] = None
    davies_bouldin_index: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
```

### Prediction Output

```python
@dataclass
class SpecialistPrediction:
    specialist_id: str
    specialist_type: str
    prediction: Any
    confidence: float  # 0-1
    reasoning: str
    capabilities_used: List[SpecialistCapability]
    execution_time_ms: float
    model_version: str
    constitutional_compliance: bool
    trust_score: float  # 0-1
    metadata: Dict[str, Any]
    timestamp: datetime
```

## Running the Demo

```bash
# Run the complete demo showcasing all 4 layers
python demo_mldl_specialists.py
```

The demo demonstrates:
1. **Layer 1**: Individual specialist training
2. **Layer 2**: Consensus aggregation
3. **Layer 3**: Governance validation
4. **Layer 4**: Meta-learning and trust updates

## Roadmap

### Planned Specialists

#### Deep Learning (6 specialists)
- ANNSpecialist (Artificial Neural Networks)
- CNNSpecialist (Convolutional Neural Networks)
- RNNSpecialist (Recurrent Neural Networks)
- LSTMSpecialist (Long Short-Term Memory)
- GANSpecialist (Generative Adversarial Networks)
- TransformerSpecialist (Attention-based models)

#### Reinforcement Learning (3 specialists)
- QLearningSpecialist (Q-Learning)
- DQNSpecialist (Deep Q-Networks)
- PPOSpecialist (Proximal Policy Optimization)

#### Ensemble Methods (3 specialists)
- StackingEnsembleSpecialist
- BaggingSpecialist
- AdaptiveBoostingSpecialist

#### Specialized Models (5 specialists)
- AnomalyDetectionSpecialist (Isolation Forest, One-Class SVM)
- TimeSeriesSpecialist (ARIMA, Prophet, LSTM)
- NLPSpecialist (BERT, GPT integration)
- GraphNeuralNetworkSpecialist (GCN, GraphSAGE)
- RecommenderSpecialist (Collaborative Filtering, Matrix Factorization)

## Development Guidelines

### Creating a New Specialist

1. **Inherit from BaseMLDLSpecialist**
2. **Implement required methods**: `train()`, `predict()`
3. **Integrate governance**: Call `validate_governance()`
4. **Log operations**: Use `log_to_immutable_trail()`
5. **Report metrics**: Call `report_kpi_metrics()`
6. **Store knowledge**: Use `store_in_memory()` / `retrieve_from_memory()`

Example template:

```python
class MySpecialist(BaseMLDLSpecialist):
    def __init__(self, **kwargs):
        super().__init__(
            specialist_id="my_specialist",
            specialist_type="MyModel",
            capabilities=[SpecialistCapability.CLASSIFICATION],
            **kwargs
        )
        self.model = initialize_model()
    
    async def train(self, X_train, y_train, **kwargs) -> TrainingMetrics:
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        metrics = TrainingMetrics(accuracy=score)
        self.training_history.append(metrics)
        
        # Log to audit trail
        await self.log_to_immutable_trail(
            operation_type="training",
            operation_data={"metrics": metrics.__dict__}
        )
        
        # Report to KPI
        await self.report_kpi_metrics(metrics, operation="training")
        
        return metrics
    
    async def predict(self, X, **kwargs) -> SpecialistPrediction:
        # Make prediction
        prediction = self.model.predict(X)
        confidence = calculate_confidence(X)
        
        # Validate governance
        compliance = await self.validate_governance(X, prediction)
        
        # Build result
        result = SpecialistPrediction(
            specialist_id=self.specialist_id,
            specialist_type=self.specialist_type,
            prediction=prediction,
            confidence=confidence,
            reasoning="...",
            capabilities_used=self.capabilities,
            execution_time_ms=elapsed_ms,
            model_version=self.model_version,
            constitutional_compliance=compliance,
            trust_score=self.current_trust_score,
            metadata={}
        )
        
        # Log prediction
        await self.log_to_immutable_trail(
            operation_type="prediction",
            operation_data={"confidence": confidence}
        )
        
        self.prediction_count += 1
        return result
```

## Testing

Run tests with pytest:

```bash
pytest tests/mldl_specialists/
```

## License

Part of the Grace project. See main LICENSE file.
