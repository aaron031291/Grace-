"""
Complete ML/DL Specialist System Demo

Demonstrates the full 4-layer architecture:
Layer 1: Individual specialists examine data independently
Layer 2: Consensus engine aggregates predictions
Layer 3: Governance integration validates results
Layer 4: Federated meta-learner improves system over time
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import specialists
from grace.mldl_specialists.supervised import (
    DecisionTreeSpecialist,
    SVMSpecialist,
    RandomForestSpecialist,
    GradientBoostingSpecialist,
    NaiveBayesSpecialist
)

from grace.mldl_specialists.unsupervised import (
    KMeansSpecialist,
    DBSCANSpecialist,
    PCASpecialist,
    AutoencoderSpecialist
)

from grace.mldl_specialists.base_specialist import SpecialistCapability
from grace.mldl_specialists.consensus_engine import MLDLConsensusEngine
from grace.mldl_specialists.federated_learning import FederatedMetaLearner


async def demo_classification_pipeline():
    """
    Demonstrate full classification pipeline through all 4 layers
    """
    print("=" * 80)
    print("GRACE ML/DL SPECIALIST SYSTEM - Classification Demo")
    print("=" * 80)
    
    # Generate synthetic classification dataset
    print("\n[1] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Layer 1: Initialize individual specialists
    print("\n[Layer 1] Initializing individual specialists...")
    specialists = {
        "decision_tree": DecisionTreeSpecialist(task_type="classification"),
        "svm": SVMSpecialist(task_type="classification"),
        "random_forest": RandomForestSpecialist(task_type="classification", n_estimators=50),
        "gradient_boosting": GradientBoostingSpecialist(task_type="classification", n_estimators=50),
        "naive_bayes": NaiveBayesSpecialist()
    }
    
    print(f"   Registered {len(specialists)} specialists")
    
    # Train all specialists
    print("\n[Layer 1] Training specialists independently...")
    for name, specialist in specialists.items():
        print(f"\n   Training {name}...")
        metrics = await specialist.train(X_train, y_train)
        
        if hasattr(metrics, 'accuracy'):
            print(f"      Accuracy: {metrics.accuracy:.3f}")
            print(f"      Precision: {metrics.precision:.3f}")
            print(f"      Recall: {metrics.recall:.3f}")
            print(f"      F1 Score: {metrics.f1_score:.3f}")
        
        print(f"      Trust Score: {specialist.current_trust_score:.3f}")
    
    # Layer 2: Set up consensus engine
    print("\n[Layer 2] Setting up consensus engine...")
    consensus_engine = MLDLConsensusEngine(
        min_specialists_required=3,
        consensus_threshold=0.6
    )
    
    # Register all specialists
    for specialist in specialists.values():
        consensus_engine.register_specialist(specialist)
    
    print(f"   Registered {len(specialists)} specialists with consensus engine")
    
    # Make predictions through consensus
    print("\n[Layer 2] Reaching consensus on test data...")
    consensus_result = await consensus_engine.reach_consensus(
        X_test,
        required_capability=SpecialistCapability.CLASSIFICATION
    )
    
    print(f"\n   Consensus Result:")
    print(f"      Final Prediction (first 10): {consensus_result.final_prediction[:10]}")
    print(f"      Consensus Score: {consensus_result.consensus_score:.3f}")
    print(f"      Overall Confidence: {consensus_result.confidence:.3f}")
    print(f"      Trust Score: {consensus_result.trust_score:.3f}")
    print(f"      Constitutional Compliance: {consensus_result.constitutional_compliance}")
    
    print(f"\n   Specialist Contributions:")
    for specialist_id, weight in sorted(
        consensus_result.weighted_contributions.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        pred = consensus_result.specialist_predictions[specialist_id]
        print(f"      {pred.specialist_type}: {weight:.3f} "
              f"(confidence={pred.confidence:.3f}, trust={pred.trust_score:.3f})")
    
    # Layer 3: Governance validation
    print("\n[Layer 3] Validating governance compliance...")
    governance_valid = await consensus_engine.validate_consensus_governance(consensus_result)
    print(f"   Governance Validation: {'PASSED' if governance_valid else 'FAILED'}")
    
    # Calculate actual accuracy
    accuracy = np.mean(np.array(consensus_result.final_prediction) == y_test)
    print(f"   Actual Accuracy: {accuracy:.3f}")
    
    # Layer 4: Meta-learning
    print("\n[Layer 4] Initializing federated meta-learner...")
    meta_learner = FederatedMetaLearner(
        learning_rate=0.1,
        trust_decay_rate=0.02
    )
    
    # Record performance for each specialist
    print("\n[Layer 4] Recording specialist performance...")
    for specialist_id, prediction in consensus_result.specialist_predictions.items():
        # Calculate accuracy for this specialist
        specialist_pred = np.array(prediction.prediction)
        specialist_accuracy = np.mean(specialist_pred == y_test)
        
        await meta_learner.record_specialist_performance(
            specialist_id=specialist_id,
            prediction=prediction,
            ground_truth=y_test,
            accuracy=specialist_accuracy
        )
        
        print(f"   {prediction.specialist_type}: accuracy={specialist_accuracy:.3f}")
    
    # Update trust scores based on performance
    print("\n[Layer 4] Updating trust scores via meta-learning...")
    updates = await meta_learner.update_trust_scores(specialists)
    
    print(f"\n   Meta-Learning Updates:")
    for update in updates:
        specialist = specialists[update.specialist_id]
        print(f"      {specialist.specialist_type}:")
        print(f"         Trust Adjustment: {update.trust_adjustment:+.3f}")
        print(f"         New Trust Score: {specialist.current_trust_score:.3f}")
        print(f"         Trend: {update.performance_trend}")
        if update.recommended_actions:
            print(f"         Recommendations: {', '.join(update.recommended_actions[:2])}")
    
    # Get specialist rankings
    print("\n[Layer 4] Specialist Rankings:")
    rankings = meta_learner.get_specialist_ranking(SpecialistCapability.CLASSIFICATION)
    for i, (specialist_id, score) in enumerate(rankings, 1):
        specialist = specialists[specialist_id]
        print(f"   {i}. {specialist.specialist_type}: {score:.3f}")
    
    # Federated learning round
    print("\n[Layer 4] Performing federated learning round...")
    await meta_learner.federated_learning_round(
        specialists,
        X_train,
        y_train,
        SpecialistCapability.CLASSIFICATION
    )
    
    # Get meta-learning stats
    meta_stats = meta_learner.get_meta_learning_stats()
    print(f"\n   Meta-Learning Statistics:")
    print(f"      Total Updates: {meta_stats['total_updates']}")
    print(f"      Avg Trust Adjustment: {meta_stats['avg_trust_adjustment']:+.4f}")
    print(f"      Improving Rate: {meta_stats['improving_rate']:.1%}")
    print(f"      Degrading Rate: {meta_stats['degrading_rate']:.1%}")
    print(f"      Avg Trust Score: {meta_stats['avg_trust_score']:.3f}")
    
    # Consensus stats
    consensus_stats = consensus_engine.get_consensus_stats()
    print(f"\n   Consensus Statistics:")
    print(f"      Total Consensus: {consensus_stats['total_consensus']}")
    print(f"      Avg Consensus Score: {consensus_stats['avg_consensus_score']:.3f}")
    print(f"      Avg Confidence: {consensus_stats['avg_confidence']:.3f}")
    print(f"      Compliance Rate: {consensus_stats['compliance_rate']:.1%}")
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION DEMO COMPLETE")
    print("=" * 80)


async def demo_clustering_pipeline():
    """
    Demonstrate clustering with unsupervised specialists
    """
    print("\n\n" + "=" * 80)
    print("GRACE ML/DL SPECIALIST SYSTEM - Clustering Demo")
    print("=" * 80)
    
    # Generate synthetic clustering dataset
    print("\n[1] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42
    )
    
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    
    # Layer 1: Initialize clustering specialists
    print("\n[Layer 1] Initializing clustering specialists...")
    clustering_specialists = {
        "kmeans": KMeansSpecialist(n_clusters=3),
        "dbscan": DBSCANSpecialist(eps=3.0, min_samples=10)
    }
    
    # Add dimensionality reduction
    pca_specialist = PCASpecialist(n_components=5)
    autoencoder_specialist = AutoencoderSpecialist(encoding_dim=5)
    
    print(f"   Registered {len(clustering_specialists)} clustering specialists")
    print(f"   Plus 2 dimensionality reduction specialists")
    
    # Train specialists
    print("\n[Layer 1] Training specialists...")
    
    # Train dimensionality reduction first
    print("\n   Training PCA...")
    pca_metrics = await pca_specialist.train(X)
    print(f"      Variance Explained: {pca_metrics.custom_metrics['variance_explained']:.1%}")
    
    print("\n   Training Autoencoder...")
    ae_metrics = await autoencoder_specialist.train(X, epochs=50)
    print(f"      Final Loss: {ae_metrics.custom_metrics['final_loss']:.4f}")
    
    # Get reduced representations
    pca_result = await pca_specialist.predict(X)
    X_reduced = np.array(pca_result.prediction)
    
    # Train clustering on reduced data
    for name, specialist in clustering_specialists.items():
        print(f"\n   Training {name}...")
        metrics = await specialist.train(X_reduced)
        
        if 'n_clusters_found' in metrics.custom_metrics:
            print(f"      Clusters Found: {metrics.custom_metrics['n_clusters_found']}")
        if 'silhouette_score' in metrics.custom_metrics:
            print(f"      Silhouette Score: {metrics.custom_metrics['silhouette_score']:.3f}")
    
    # Layer 2: Consensus on clustering
    print("\n[Layer 2] Reaching consensus on cluster assignments...")
    consensus_engine = MLDLConsensusEngine()
    
    for specialist in clustering_specialists.values():
        consensus_engine.register_specialist(specialist)
    
    consensus_result = await consensus_engine.reach_consensus(
        X_reduced,
        required_capability=SpecialistCapability.CLUSTERING
    )
    
    print(f"\n   Consensus Result:")
    print(f"      Consensus Score: {consensus_result.consensus_score:.3f}")
    print(f"      Confidence: {consensus_result.confidence:.3f}")
    
    print("\n" + "=" * 80)
    print("CLUSTERING DEMO COMPLETE")
    print("=" * 80)


async def main():
    """Run all demos"""
    # Classification demo
    await demo_classification_pipeline()
    
    # Clustering demo
    await demo_clustering_pipeline()
    
    print("\n\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("\nThe 4-layer architecture demonstrates:")
    print("  Layer 1: Individual specialists with full model logic")
    print("  Layer 2: Consensus engine for aggregated predictions")
    print("  Layer 3: Governance integration and validation")
    print("  Layer 4: Federated meta-learning for continuous improvement")
    print("\nAll operations logged to immutable audit trail")
    print("All metrics tracked via KPI monitors")
    print("All knowledge stored in memory bridge")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
