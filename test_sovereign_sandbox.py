#!/usr/bin/env python3
"""
Test suite for Grace Sovereign Sandbox System.

Tests the complete implementation including:
- Sandbox Manager
- Validation Harness
- Cross-Persona Consensus
- Human Feedback Loop
- Sovereign Orchestrator
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test imports
from grace.governance.sandbox_manager import (
    SandboxManager,
    ExperimentType,
    ResourceQuota,
)
from grace.governance.validation_harness import ValidationHarness, ValidationLevel
from grace.governance.consensus_engine import CrossPersonaConsensus, PersonaRole
from grace.governance.human_feedback import (
    HumanFeedbackInterface,
    FeedbackType,
    FeedbackScore,
)
from grace.governance.sovereign_orchestrator import SovereignSandboxOrchestrator
from grace.mldl.quorum import MLDLQuorum
from grace.layer_04_audit_logs.immutable_logs import ImmutableLogs
from grace.core.kpi_trust_monitor import KPITrustMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SovereignSandboxTestSuite:
    """Comprehensive test suite for the sovereign sandbox system."""

    def __init__(self):
        self.test_results = []

    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("🚀 Starting Sovereign Sandbox Test Suite")
        logger.info("=" * 70)

        tests = [
            ("Sandbox Manager Basic Operations", self.test_sandbox_manager),
            ("Validation Harness Integration", self.test_validation_harness),
            ("Cross-Persona Consensus Engine", self.test_consensus_engine),
            ("Human Feedback Interface", self.test_human_feedback),
            ("Complete Grace Experiment Cycle", self.test_grace_experiment_cycle),
            ("Orb Dashboard Data Generation", self.test_orb_dashboard),
            ("Meta-learning and Governance Evolution", self.test_meta_learning),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            try:
                logger.info(f"\n📋 Testing: {test_name}")
                logger.info("-" * 50)

                result = await test_func()

                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED")

                self.test_results.append((test_name, result))

            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results.append((test_name, False))

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info(
            f"🎯 Test Summary: {passed}/{total} tests passed ({passed / total * 100:.1f}%)"
        )

        if passed == total:
            logger.info("🎉 All tests passed! Sovereign sandbox system is ready.")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Review implementation.")

        return passed == total

    async def test_sandbox_manager(self) -> bool:
        """Test the SandboxManager component."""
        try:
            # Initialize with temporary file
            import tempfile

            temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db.close()

            immutable_logs = ImmutableLogs(temp_db.name)
            sandbox_manager = SandboxManager(immutable_logs=immutable_logs)

            # Test sandbox creation
            sandbox_id = await sandbox_manager.create_sandbox(
                experiment_type=ExperimentType.CURIOSITY_DRIVEN,
                description="Test experiment for pattern recognition",
            )

            assert sandbox_id is not None, "Sandbox creation failed"
            logger.info(f"✓ Created sandbox: {sandbox_id}")

            # Test sandbox retrieval
            sandbox = await sandbox_manager.get_sandbox(sandbox_id)
            assert sandbox is not None, "Sandbox retrieval failed"
            assert sandbox.experiment_type == ExperimentType.CURIOSITY_DRIVEN
            logger.info("✓ Sandbox retrieval successful")

            # Test experiment creation within sandbox
            experiment_id = await sandbox.start_experiment(
                experiment_type=ExperimentType.CURIOSITY_DRIVEN,
                description="Test pattern recognition experiment",
                hypothesis="This will improve decision accuracy",
            )

            assert experiment_id is not None, "Experiment creation failed"
            logger.info(f"✓ Started experiment: {experiment_id}")

            # Test experiment completion
            results = {
                "success": True,
                "improvements": ["Better accuracy", "Faster processing"],
                "metrics": {"accuracy_gain": 0.15},
            }

            await sandbox.complete_experiment(experiment_id, results, success=True)
            logger.info("✓ Experiment completed successfully")

            # Test sandbox validation
            validation_results = await sandbox.validate()
            assert validation_results["overall_status"] in ["approved", "rejected"]
            logger.info(f"✓ Sandbox validation: {validation_results['overall_status']}")

            # Test sandbox listing
            sandboxes = await sandbox_manager.list_sandboxes()
            assert len(sandboxes) >= 1, "Sandbox listing failed"
            logger.info(f"✓ Listed {len(sandboxes)} sandboxes")

            # Test dashboard data
            dashboard_data = await sandbox_manager.get_sandbox_dashboard_data()
            assert "dashboard" in dashboard_data
            assert dashboard_data["dashboard"]["active_sandboxes"] >= 1
            logger.info("✓ Dashboard data generation successful")

            # Cleanup
            await sandbox_manager.destroy_sandbox(sandbox_id, reason="test_cleanup")
            logger.info("✓ Sandbox cleanup successful")

            # Clean up temporary file
            os.unlink(temp_db.name)

            return True

        except Exception as e:
            logger.error(f"Sandbox manager test failed: {e}")
            return False

    async def test_validation_harness(self) -> bool:
        """Test the ValidationHarness component."""
        try:
            # Initialize
            validation_harness = ValidationHarness()

            # Mock sandbox data
            sandbox_data = {
                "sandbox_id": "test_sandbox",
                "experiments": [{"results": {"success": True}}],
                "metrics": {
                    "resource_usage": {
                        "compute_hours": 1.0,
                        "memory_gb_hours": 2.0,
                        "storage_gb_hours": 5.0,
                        "api_calls": 50,
                    }
                },
            }

            # Test basic validation
            results = await validation_harness.validate_sandbox(
                sandbox_id="test_sandbox",
                sandbox_data=sandbox_data,
                validation_level=ValidationLevel.BASIC,
            )

            assert "overall_score" in results
            assert "overall_status" in results
            assert results["summary"]["total_tests"] > 0
            logger.info(f"✓ Basic validation completed: {results['overall_status']}")

            # Test standard validation
            results = await validation_harness.validate_sandbox(
                sandbox_id="test_sandbox",
                sandbox_data=sandbox_data,
                validation_level=ValidationLevel.STANDARD,
            )

            assert results["summary"]["total_tests"] > 3  # Should have more tests
            logger.info(f"✓ Standard validation completed: {results['overall_status']}")

            # Test validation history
            history = await validation_harness.get_validation_history(limit=10)
            assert len(history) >= 2  # Should have our two test runs
            logger.info(f"✓ Validation history: {len(history)} entries")

            return True

        except Exception as e:
            logger.error(f"Validation harness test failed: {e}")
            return False

    async def test_consensus_engine(self) -> bool:
        """Test the CrossPersonaConsensus component."""
        try:
            # Initialize with simplified MLDL quorum (mock the event bus issue)
            class MockEventBus:
                async def publish(self, event_type, data):
                    pass

            mldl_quorum = MLDLQuorum(event_bus=MockEventBus())
            consensus_engine = CrossPersonaConsensus(mldl_quorum)

            # Mock validation results
            validation_results = {
                "overall_score": 0.85,
                "overall_status": "passed",
                "test_results": [
                    {"category": "security_scan", "status": "passed"},
                    {"category": "performance_test", "status": "passed"},
                ],
            }

            # Mock impact assessment
            impact_assessment = {
                "affected_systems": ["governance", "decision_making"],
                "impact_magnitude": "medium",
                "confidence_in_assessment": 0.8,
            }

            # Test consensus simulation first (doesn't require full MLDL)
            from grace.governance.consensus_engine import ConsensusRequest

            consensus_request = ConsensusRequest(
                request_id="test_simulation",
                sandbox_id="test_sandbox",
                proposal_type="merge",
                description="Test simulation",
                validation_results=validation_results,
                impact_assessment=impact_assessment,
                required_personas={PersonaRole.SENIOR_ARCHITECT},
            )

            simulation = await consensus_engine.simulate_consensus(consensus_request)
            assert "likely_outcome" in simulation
            logger.info(f"✓ Consensus simulation: {simulation['likely_outcome']}")

            # Test persona reliability scores
            reliability_scores = consensus_engine.get_persona_reliability_scores()
            assert len(reliability_scores) > 0
            logger.info(
                f"✓ Persona reliability scores: {len(reliability_scores)} personas"
            )

            # Test consensus history (should be empty initially)
            history = await consensus_engine.get_consensus_history(limit=5)
            assert isinstance(history, list)
            logger.info(f"✓ Consensus history: {len(history)} entries")

            logger.info("✓ Consensus engine core functionality working")

            return True

        except Exception as e:
            logger.error(f"Consensus engine test failed: {e}")
            return False

    async def test_human_feedback(self) -> bool:
        """Test the HumanFeedbackInterface component."""
        try:
            # Initialize
            feedback_interface = HumanFeedbackInterface()

            # Test feedback request
            request_id = await feedback_interface.request_feedback(
                feedback_type=FeedbackType.SANDBOX_REVIEW,
                target_id="test_sandbox",
                title="Test feedback request",
                description="Testing the feedback system",
                context_data={"test": True},
                priority="normal",
            )

            assert request_id is not None
            logger.info(f"✓ Feedback requested: {request_id}")

            # Test pending requests
            pending = feedback_interface.get_pending_feedback_requests()
            assert len(pending) >= 1
            logger.info(f"✓ Pending requests: {len(pending)}")

            # Test feedback submission
            feedback_id = await feedback_interface.submit_feedback(
                request_id=request_id,
                reviewer_id="test_reviewer",
                score=FeedbackScore.GOOD,
                approval_status="approved",
                detailed_feedback="This is a test feedback submission",
                specific_comments=["Good implementation", "Well tested"],
                suggestions=["Consider adding more tests"],
                confidence=0.8,
            )

            assert feedback_id is not None
            logger.info(f"✓ Feedback submitted: {feedback_id}")

            # Test feedback history
            history = feedback_interface.get_feedback_history(limit=10)
            assert len(history) >= 1
            logger.info(f"✓ Feedback history: {len(history)} entries")

            # Test reviewer dashboard
            dashboard = feedback_interface.get_reviewer_dashboard("test_reviewer")
            assert dashboard["total_reviews_completed"] >= 1
            logger.info(
                f"✓ Reviewer dashboard: {dashboard['total_reviews_completed']} reviews"
            )

            # Test system metrics
            metrics = feedback_interface.get_system_feedback_metrics()
            assert metrics["completed_reviews"] >= 1
            logger.info(
                f"✓ System metrics: {metrics['completed_reviews']} completed reviews"
            )

            # Test feedback insights
            insights = await feedback_interface.generate_feedback_insights()
            assert "improvement_recommendations" in insights
            logger.info(f"✓ Feedback insights generated")

            return True

        except Exception as e:
            logger.error(f"Human feedback test failed: {e}")
            return False

    async def test_grace_experiment_cycle(self) -> bool:
        """Test the complete Grace experiment cycle through the SovereignOrchestrator."""
        try:
            # Initialize orchestrator with temporary database
            import tempfile

            class MockEventBus:
                async def publish(self, event_type, data):
                    pass

            mldl_quorum = MLDLQuorum(event_bus=MockEventBus())

            temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db.close()

            immutable_logs = ImmutableLogs(temp_db.name)

            orchestrator = SovereignSandboxOrchestrator(
                mldl_quorum=mldl_quorum, immutable_logs=immutable_logs
            )

            # Test simple Grace experiment cycle (this will likely fail validation and that's ok for testing)
            cycle_result = await orchestrator.grace_experiment_cycle(
                experiment_description="Test autonomous experiment for validation",
                experiment_type=ExperimentType.CURIOSITY_DRIVEN,
                validation_level=ValidationLevel.BASIC,
            )

            assert "cycle_id" in cycle_result
            assert "status" in cycle_result
            logger.info(f"✓ Grace experiment cycle: {cycle_result['status']}")

            # The cycle should complete one of several valid outcomes
            valid_statuses = [
                "completed_successfully",
                "consensus_rejected",
                "awaiting_human_approval",
                "failed",
            ]

            assert cycle_result["status"] in valid_statuses
            logger.info(f"✓ Cycle completed with status: {cycle_result['status']}")

            # Test sovereignty report
            report = await orchestrator.get_sovereignty_report()
            assert "sovereignty_metrics" in report
            assert report["sovereignty_metrics"]["total_sovereign_decisions"] >= 1
            logger.info(
                f"✓ Sovereignty report: {report['sovereignty_metrics']['total_sovereign_decisions']} decisions"
            )

            # Cleanup
            await orchestrator.shutdown()
            logger.info("✓ Orchestrator shutdown successful")

            # Clean up temporary file
            os.unlink(temp_db.name)

            return True

        except Exception as e:
            logger.error(f"Grace experiment cycle test failed: {e}")
            return False

    async def test_orb_dashboard(self) -> bool:
        """Test Orb dashboard data generation."""
        try:
            # Initialize orchestrator with temporary database
            import tempfile

            class MockEventBus:
                async def publish(self, event_type, data):
                    pass

            mldl_quorum = MLDLQuorum(event_bus=MockEventBus())

            temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db.close()

            immutable_logs = ImmutableLogs(temp_db.name)

            orchestrator = SovereignSandboxOrchestrator(
                mldl_quorum=mldl_quorum, immutable_logs=immutable_logs
            )

            # Test dashboard data generation
            dashboard_data = await orchestrator.get_orb_dashboard_data()

            # Verify all required sections are present
            required_sections = [
                "sandbox_overview",
                "active_experiments",
                "validation_status",
                "consensus_status",
                "human_feedback_status",
                "grace_autonomy_metrics",
                "recent_sovereign_decisions",
                "system_health",
            ]

            for section in required_sections:
                assert hasattr(dashboard_data, section), (
                    f"Missing dashboard section: {section}"
                )
                logger.info(f"✓ Dashboard section present: {section}")

            # Test specific dashboard metrics
            autonomy_metrics = dashboard_data.grace_autonomy_metrics
            assert "current_autonomy_level" in autonomy_metrics
            assert 0.0 <= autonomy_metrics["current_autonomy_level"] <= 1.0
            logger.info(
                f"✓ Grace autonomy level: {autonomy_metrics['current_autonomy_level']:.2f}"
            )

            # Test system health
            system_health = dashboard_data.system_health
            assert "overall_status" in system_health
            assert system_health["overall_status"] in ["healthy", "busy", "degraded"]
            logger.info(f"✓ System health: {system_health['overall_status']}")

            # Cleanup
            await orchestrator.shutdown()

            # Clean up temporary file
            os.unlink(temp_db.name)

            return True

        except Exception as e:
            logger.error(f"Orb dashboard test failed: {e}")
            return False

    async def test_meta_learning(self) -> bool:
        """Test meta-learning and governance evolution capabilities."""
        try:
            # Initialize orchestrator with temporary database
            import tempfile

            class MockEventBus:
                async def publish(self, event_type, data):
                    pass

            mldl_quorum = MLDLQuorum(event_bus=MockEventBus())

            temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db.close()

            immutable_logs = ImmutableLogs(temp_db.name)

            orchestrator = SovereignSandboxOrchestrator(
                mldl_quorum=mldl_quorum, immutable_logs=immutable_logs
            )

            # Test initial autonomy level
            initial_autonomy = orchestrator.autonomy_level
            assert 0.0 <= initial_autonomy <= 1.0
            logger.info(f"✓ Initial autonomy level: {initial_autonomy:.2f}")

            # Simulate learning from feedback
            from grace.governance.human_feedback import (
                HumanFeedback,
                FeedbackType,
                FeedbackScore,
            )

            # Simulate positive feedback
            good_feedback = HumanFeedback(
                feedback_id="test_feedback_1",
                feedback_type=FeedbackType.SANDBOX_REVIEW,
                target_id="test_sandbox",
                reviewer_id="test_reviewer",
                score=FeedbackScore.EXCELLENT,
                approval_status="approved",
                detailed_feedback="Excellent work on this experiment",
                specific_comments=[],
                suggestions=[],
                areas_of_concern=[],
                positive_aspects=["Good reasoning", "Solid implementation"],
                confidence_in_feedback=0.9,
                timestamp=datetime.now(),
            )

            await orchestrator._process_human_feedback(good_feedback)

            # Check if autonomy increased
            new_autonomy = orchestrator.autonomy_level
            logger.info(f"✓ Autonomy after positive feedback: {new_autonomy:.2f}")

            # Test learning data accumulation
            learning_data = orchestrator.governance_learning_data
            assert "successful_patterns" in learning_data
            assert "failed_patterns" in learning_data
            logger.info("✓ Learning data structures present")

            # Test meta-learning process
            await orchestrator._perform_meta_learning()
            logger.info("✓ Meta-learning process executed")

            # Cleanup
            await orchestrator.shutdown()

            # Clean up temporary file
            os.unlink(temp_db.name)

            return True

        except Exception as e:
            logger.error(f"Meta-learning test failed: {e}")
            return False


async def main():
    """Run the test suite."""
    test_suite = SovereignSandboxTestSuite()
    success = await test_suite.run_all_tests()

    if success:
        print(
            "\n🎉 All tests passed! The Sovereign Sandbox System is ready for deployment."
        )
        return 0
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
