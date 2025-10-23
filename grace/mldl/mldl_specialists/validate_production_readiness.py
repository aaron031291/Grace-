"""
Production Readiness Validation for ML/DL Cognitive Substrate

Validates that all operational intelligence components are ready:
- Model interfaces implemented
- Uncertainty/OOD detection working
- Monitoring and alerting configured
- Active learning pipeline functional
- TriggerMesh workflows valid
- Model registry operational
- Governance integration complete
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ProductionReadinessValidator:
    """Validates production readiness of ML/DL system"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.passed = 0
        self.failed = 0
    
    async def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("=" * 80)
        logger.info("ML/DL Cognitive Substrate - Production Readiness Validation")
        logger.info("=" * 80)
        
        checks = [
            self.check_model_interface,
            self.check_uncertainty_ood,
            self.check_model_registry,
            self.check_monitoring,
            self.check_active_learning,
            self.check_triggermesh_workflows,
            self.check_specialists_interface_compliance,
            self.check_governance_integration,
            self.check_file_structure,
        ]
        
        for check in checks:
            try:
                await check()
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                self._record_result(check.__name__, False, str(e))
        
        self._print_summary()
        return self.failed == 0
    
    def _record_result(self, check_name: str, passed: bool, message: str = ""):
        """Record validation result"""
        self.results[check_name] = {
            'passed': passed,
            'message': message
        }
        if passed:
            self.passed += 1
            logger.info(f"✅ {check_name}: PASSED")
        else:
            self.failed += 1
            logger.error(f"❌ {check_name}: FAILED - {message}")
    
    async def check_model_interface(self):
        """Validate model interface implementation"""
        logger.info("\n1. Checking Model Interface...")
        
        try:
            from grace.mldl_specialists.model_interface import (
                ModelInterface,
                InferenceInput,
                InferenceOutput,
                ModelMetadata,
                ModelHealth,
                OODStatus
            )
            
            # Check required methods exist
            required_methods = ['predict', 'explain', 'metadata', 'health', 'validate_input']
            for method in required_methods:
                if not hasattr(ModelInterface, method):
                    self._record_result(
                        'check_model_interface',
                        False,
                        f"Missing required method: {method}"
                    )
                    return
            
            # Check dataclasses
            assert hasattr(InferenceInput, 'compute_hash')
            assert hasattr(InferenceOutput, 'ood_flag')
            assert hasattr(ModelHealth, 'is_healthy')
            
            self._record_result('check_model_interface', True)
            
        except ImportError as e:
            self._record_result('check_model_interface', False, f"Import failed: {e}")
        except AssertionError as e:
            self._record_result('check_model_interface', False, f"Missing attribute: {e}")
    
    async def check_uncertainty_ood(self):
        """Validate uncertainty and OOD detection"""
        logger.info("\n2. Checking Uncertainty & OOD Detection...")
        
        try:
            from grace.mldl_specialists.uncertainty_ood import (
                TemperatureScaling,
                MahalanobisOOD,
                EntropyOOD,
                UncertaintyAwareRouter,
                compute_calibration_error
            )
            
            # Test temperature scaling
            temp_scaler = TemperatureScaling()
            logits = np.random.randn(100, 5)
            probs = temp_scaler.apply(logits)
            assert probs.shape == logits.shape
            assert np.allclose(probs.sum(axis=1), 1.0)
            
            # Test Mahalanobis OOD
            mahal_detector = MahalanobisOOD(threshold=3.0)
            train_embeddings = np.random.randn(100, 10)
            mahal_detector.fit(train_embeddings)
            
            test_embedding = np.random.randn(10)
            result = mahal_detector.detect(test_embedding)
            assert hasattr(result, 'is_ood')
            assert hasattr(result, 'ood_score')
            
            # Test entropy OOD
            entropy_detector = EntropyOOD(num_classes=5)
            probs = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
            result = entropy_detector.detect(probs)
            assert hasattr(result, 'is_ood')
            
            # Test router
            router = UncertaintyAwareRouter()
            decision = router.route_decision(confidence=0.95)
            assert 'action' in decision
            assert 'requires_human_review' in decision
            
            self._record_result('check_uncertainty_ood', True)
            
        except Exception as e:
            self._record_result('check_uncertainty_ood', False, str(e))
    
    async def check_model_registry(self):
        """Validate model registry"""
        logger.info("\n3. Checking Model Registry...")
        
        try:
            from grace.mldl_specialists.model_registry import (
                ModelRegistry,
                ModelRegistryEntry,
                DeploymentStage,
                get_registry
            )
            
            # Create test registry in temp location
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                registry_path = f.name
            
            registry = ModelRegistry(registry_path=registry_path)
            
            # Test registration
            from datetime import datetime
            entry = ModelRegistryEntry(
                model_id='test_model',
                name='Test Model',
                version='1.0',
                artifact_path='/tmp/model.pkl',
                framework='sklearn',
                model_type='classification',
                owner='test_user',
                team='ml_team',
                training_data_hash='abc123',
                training_dataset_size=1000,
                training_timestamp=datetime.now(),
                deploy_status=DeploymentStage.DEVELOPMENT
            )
            
            assert registry.register_model(entry)
            retrieved = registry.get_model('test_model')
            assert retrieved is not None
            assert retrieved.model_id == 'test_model'
            
            # Test deployment status update
            assert registry.update_deployment_status(
                'test_model',
                DeploymentStage.PRODUCTION
            )
            
            # Cleanup
            Path(registry_path).unlink()
            
            self._record_result('check_model_registry', True)
            
        except Exception as e:
            self._record_result('check_model_registry', False, str(e))
    
    async def check_monitoring(self):
        """Validate monitoring system"""
        logger.info("\n4. Checking Monitoring & Observability...")
        
        try:
            from grace.mldl_specialists.monitoring import (
                ModelMonitor,
                TrustScoreLedger,
                LatencyMetrics,
                ThroughputMetrics,
                DriftMetrics
            )
            
            # Test model monitor
            monitor = ModelMonitor(model_id='test_model')
            
            # Record some inferences
            for i in range(100):
                monitor.record_inference(
                    latency_ms=50.0 + np.random.randn() * 10,
                    confidence=0.8 + np.random.rand() * 0.2,
                    prediction=i % 3,
                    ood_flag=np.random.rand() < 0.1,
                    input_features=np.random.randn(10),
                    success=True
                )
            
            # Check metrics
            latency = monitor.get_latency_metrics()
            assert isinstance(latency, LatencyMetrics)
            assert latency.sample_count == 100
            assert latency.p50_ms > 0
            assert latency.p95_ms > 0
            
            throughput = monitor.get_throughput_metrics()
            assert isinstance(throughput, ThroughputMetrics)
            assert throughput.total_requests == 100
            
            # Test trust ledger
            ledger = TrustScoreLedger()
            ledger.record_trust_score('test_model', 0.95)
            ledger.record_trust_score('test_model', 0.93)
            ledger.record_trust_score('test_model', 0.91)
            
            current_trust = ledger.get_current_trust_score('test_model')
            assert current_trust == 0.91
            
            trend = ledger.get_trust_trend('test_model')
            assert trend in ['improving', 'declining', 'stable']
            
            self._record_result('check_monitoring', True)
            
        except Exception as e:
            self._record_result('check_monitoring', False, str(e))
    
    async def check_active_learning(self):
        """Validate active learning pipeline"""
        logger.info("\n5. Checking Active Learning & HITL...")
        
        try:
            from grace.mldl_specialists.active_learning import (
                ReviewQueue,
                ActiveLearner,
                ReviewQueueItem,
                LabeledSample,
                ReviewStatus,
                SamplingStrategy
            )
            from datetime import datetime
            
            # Test review queue
            queue = ReviewQueue(max_queue_size=10)
            
            # Enqueue items
            for i in range(5):
                item = ReviewQueueItem(
                    item_id=f'item_{i}',
                    trace_id=f'trace_{i}',
                    input_data={'feature': i},
                    input_hash=f'hash_{i}',
                    model_id='test_model',
                    model_version='1.0',
                    prediction=i % 2,
                    confidence=0.6 + i * 0.05,
                    uncertainty_score=0.4 - i * 0.05
                )
                queue.enqueue(item)
            
            # Get next item
            next_item = queue.get_next_item(reviewer_id='test_reviewer')
            assert next_item is not None
            assert next_item.status == ReviewStatus.IN_REVIEW
            
            # Submit review
            success = queue.submit_review(
                item_id=next_item.item_id,
                status=ReviewStatus.APPROVED,
                label=1,
                feedback='Looks good',
                reviewer_id='test_reviewer'
            )
            assert success
            
            # Test active learner
            learner = ActiveLearner(
                review_queue=queue,
                sampling_strategy=SamplingStrategy.UNCERTAINTY
            )
            
            # Select samples
            candidates = [
                {'id': i, 'confidence': 0.5 + i * 0.1, 'features': np.random.randn(10)}
                for i in range(20)
            ]
            selected = learner.select_samples_for_labeling(candidates, n_samples=5)
            assert len(selected) == 5
            
            # Add labeled sample
            sample = LabeledSample(
                sample_id='sample_1',
                input_data={'feature': 1},
                ground_truth_label=1,
                labeled_by='human',
                labeled_at=datetime.now(),
                labeling_confidence=0.95
            )
            learner.add_labeled_sample(sample)
            
            stats = learner.get_statistics()
            assert stats['total_labeled_samples'] == 1
            
            self._record_result('check_active_learning', True)
            
        except Exception as e:
            self._record_result('check_active_learning', False, str(e))
    
    async def check_triggermesh_workflows(self):
        """Validate TriggerMesh workflow files"""
        logger.info("\n6. Checking TriggerMesh Workflows...")
        
        try:
            import yaml
            
            workflow_file = Path('orchestration/trigger_mesh_ml_workflows.yaml')
            if not workflow_file.exists():
                self._record_result(
                    'check_triggermesh_workflows',
                    False,
                    f"Workflow file not found: {workflow_file}"
                )
                return
            
            with open(workflow_file, 'r') as f:
                workflows = list(yaml.safe_load_all(f))
            
            # Check required workflows exist
            required_workflows = [
                'model_inference_workflow',
                'anomaly_detection_pipeline',
                'shadow_model_validation',
                'model_retraining_workflow',
                'model_rollback_workflow',
                'forecasting_kernel_workflow',
                'trust_scoring_workflow'
            ]
            
            workflow_names = [w.get('name') for w in workflows]
            
            for required in required_workflows:
                if required not in workflow_names:
                    self._record_result(
                        'check_triggermesh_workflows',
                        False,
                        f"Missing workflow: {required}"
                    )
                    return
            
            # Validate workflow structure
            for workflow in workflows:
                assert 'name' in workflow
                assert 'event' in workflow
                assert 'steps' in workflow
                assert isinstance(workflow['steps'], list)
            
            self._record_result('check_triggermesh_workflows', True)
            
        except Exception as e:
            self._record_result('check_triggermesh_workflows', False, str(e))
    
    async def check_specialists_interface_compliance(self):
        """Check that specialists implement the standard interface"""
        logger.info("\n7. Checking Specialists Interface Compliance...")
        
        try:
            from grace.mldl_specialists.supervised_specialists import (
                DecisionTreeSpecialist,
                RandomForestSpecialist
            )
            from grace.mldl_specialists.unsupervised_specialists import (
                KMeansClusteringSpecialist,
                IsolationForestAnomalySpecialist
            )
            
            # Check if specialists have predict_async method
            specialists = [
                DecisionTreeSpecialist(),
                RandomForestSpecialist(),
                KMeansClusteringSpecialist(),
                IsolationForestAnomalySpecialist()
            ]
            
            for specialist in specialists:
                assert hasattr(specialist, 'predict_async')
                assert hasattr(specialist, 'specialist_id')
                assert hasattr(specialist, 'capabilities')
            
            self._record_result('check_specialists_interface_compliance', True)
            
        except Exception as e:
            self._record_result('check_specialists_interface_compliance', False, str(e))
    
    async def check_governance_integration(self):
        """Check governance integration points"""
        logger.info("\n8. Checking Governance Integration...")
        
        try:
            from grace.mldl_specialists.cognitive_substrate import CognitiveSubstrate
            
            # Check that CognitiveSubstrate has governance integration
            substrate = CognitiveSubstrate(
                kpi_monitor=None,
                governance_engine=None,
                event_publisher=None,
                immutable_logs=None
            )
            
            assert hasattr(substrate, 'governance_engine')
            assert hasattr(substrate, 'kpi_monitor')
            assert hasattr(substrate, 'immutable_logs')
            assert hasattr(substrate, 'event_publisher')
            
            # Check process_cognitive_event has validation step
            import inspect
            source = inspect.getsource(substrate.process_cognitive_event)
            assert 'validate_governance' in source or '_validate_governance' in source
            
            self._record_result('check_governance_integration', True)
            
        except Exception as e:
            self._record_result('check_governance_integration', False, str(e))
    
    async def check_file_structure(self):
        """Validate file structure"""
        logger.info("\n9. Checking File Structure...")
        
        required_files = [
            'grace/mldl_specialists/model_interface.py',
            'grace/mldl_specialists/uncertainty_ood.py',
            'grace/mldl_specialists/model_registry.py',
            'grace/mldl_specialists/active_learning.py',
            'grace/mldl_specialists/monitoring.py',
            'grace/mldl_specialists/supervised_specialists.py',
            'grace/mldl_specialists/unsupervised_specialists.py',
            'grace/mldl_specialists/cognitive_substrate.py',
            'grace/mldl_specialists/cognitive_kernels.py',
            'grace/mldl_specialists/consensus_engine.py',
            'grace/mldl_specialists/integration_example.py',
            'orchestration/trigger_mesh_ml_workflows.yaml',
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self._record_result(
                'check_file_structure',
                False,
                f"Missing files: {', '.join(missing_files)}"
            )
        else:
            self._record_result('check_file_structure', True)
    
    def _print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        logger.info(f"Total Checks: {total}")
        logger.info(f"Passed: {self.passed} ✅")
        logger.info(f"Failed: {self.failed} ❌")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed > 0:
            logger.info("\nFailed Checks:")
            for check_name, result in self.results.items():
                if not result['passed']:
                    logger.info(f"  - {check_name}: {result['message']}")
        
        logger.info("\n" + "=" * 80)
        if self.failed == 0:
            logger.info("✅ ALL CHECKS PASSED - PRODUCTION READY")
        else:
            logger.info("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE PRODUCTION")
        logger.info("=" * 80)


async def main():
    """Run production readiness validation"""
    validator = ProductionReadinessValidator()
    success = await validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
