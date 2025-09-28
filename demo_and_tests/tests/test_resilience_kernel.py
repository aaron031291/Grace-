#!/usr/bin/env python3
"""
Test script for Grace Resilience Kernel implementation.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from grace.resilience.resilience_service import ResilienceService
from grace.resilience.controllers.circuit import CircuitBreaker
from grace.resilience.controllers.degradation import get_degradation_manager
from grace.resilience.detectors.slis import evaluate_sli, SLIMonitor
from grace.resilience.chaos.runner import get_chaos_runner
from grace.resilience.telemetry.budget import ErrorBudgetTracker
from grace.resilience.snapshots.manager import SnapshotManager


async def test_resilience_service():
    """Test basic resilience service functionality."""
    print("🛡️  Testing Grace Resilience Service...")
    
    try:
        # Initialize service
        service = ResilienceService()
        
        print("1. Testing service initialization...")
        print("   ✓ Service initialized successfully")
        
        # Test SLO policy
        print("2. Testing SLO policy management...")
        slo_policy = {
            "service_id": "test_service",
            "slos": [
                {"sli": "latency_p95_ms", "objective": 800, "window": "30d"},
                {"sli": "availability_pct", "objective": 99.9, "window": "30d"}
            ],
            "error_budget_days": 0.5
        }
        result = service.set_slo(slo_policy)
        print(f"   ✓ SLO policy set for service: {result['service_id']}")
        
        # Test resilience policy
        print("3. Testing resilience policy management...")
        resilience_policy = {
            "service_id": "test_service",
            "retries": {"max": 3, "backoff": "exp", "base_ms": 100},
            "circuit_breaker": {"failure_rate_threshold_pct": 30, "request_volume_threshold": 20},
            "degradation_modes": [
                {"mode_id": "lite_mode", "triggers": ["high_latency"], "actions": ["disable_explain"]}
            ]
        }
        result = service.set_resilience_policy(resilience_policy)
        print(f"   ✓ Resilience policy set for service: {result['service_id']}")
        
        # Test incident management
        print("4. Testing incident management...")
        incident = {
            "incident_id": "inc_001",
            "service_id": "test_service",
            "severity": "sev2",
            "signals": {"latency_p95": 1200, "error_rate": 3.5}
        }
        incident_id = service.open_incident(incident)
        print(f"   ✓ Incident opened: {incident_id}")
        
        # Test snapshot creation
        print("5. Testing snapshot management...")
        snapshot = service.export_snapshot()
        print(f"   ✓ Snapshot created: {snapshot['snapshot_id']}")
        
        print("✅ Resilience service test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Resilience service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n🔌 Testing Circuit Breaker...")
    
    try:
        # Initialize circuit breaker
        breaker = CircuitBreaker(
            failure_threshold_pct=50.0,
            volume_threshold=5,
            sleep_window_ms=1000,
            half_open_max_calls=3
        )
        
        print("1. Testing normal operation...")
        assert breaker.can_execute() == True
        breaker.on_success()
        assert breaker.state() == "closed"
        print("   ✓ Circuit breaker allows execution when closed")
        
        print("2. Testing failure handling...")
        # Simulate failures to open circuit
        for i in range(6):  # Exceed volume threshold
            breaker.on_failure()
        
        assert breaker.state() == "open"
        assert breaker.can_execute() == False
        print("   ✓ Circuit breaker opens after failures")
        
        print("3. Testing recovery...")
        # Force half-open for testing
        breaker.force_half_open()
        assert breaker.state() == "half_open"
        assert breaker.can_execute() == True
        
        # Successful calls should close the circuit
        for i in range(3):
            breaker.on_success()
        
        assert breaker.state() == "closed"
        print("   ✓ Circuit breaker closes after successful recovery")
        
        stats = breaker.get_stats()
        print(f"   ✓ Circuit breaker stats: {stats['state']}, {stats['failure_count']} failures")
        
        print("✅ Circuit breaker test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_degradation_manager():
    """Test degradation manager functionality."""
    print("\n📉 Testing Degradation Manager...")
    
    try:
        manager = get_degradation_manager()
        
        print("1. Testing degradation configuration...")
        degradation_modes = [
            {
                "mode_id": "lite_explanations",
                "triggers": ["high_latency"],
                "actions": ["disable_explain", "reduce_batch"]
            },
            {
                "mode_id": "cached_only",
                "triggers": ["dependency_down"],
                "actions": ["use_cache"]
            }
        ]
        manager.configure_modes("test_service", degradation_modes)
        print("   ✓ Degradation modes configured")
        
        print("2. Testing mode entry...")
        result = await manager.enter_mode("test_service", "lite_explanations", "test")
        assert result == True
        assert manager.is_in_mode("test_service", "lite_explanations")
        print("   ✓ Successfully entered degradation mode")
        
        print("3. Testing mode exit...")
        result = await manager.exit_mode("test_service", "lite_explanations", "test")
        assert result == True
        assert not manager.is_in_mode("test_service", "lite_explanations")
        print("   ✓ Successfully exited degradation mode")
        
        print("4. Testing trigger evaluation...")
        signals = {"latency_p95_ms": 1200, "error_rate_pct": 2.0}
        triggered_modes = await manager.evaluate_triggers("test_service", signals)
        print(f"   ✓ Triggered modes for high latency: {triggered_modes}")
        
        stats = manager.get_stats("test_service")
        print(f"   ✓ Degradation stats: {len(stats['configured_modes'])} modes configured")
        
        print("✅ Degradation manager test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Degradation manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sli_evaluation():
    """Test SLI evaluation functionality."""
    print("\n📊 Testing SLI Evaluation...")
    
    try:
        print("1. Testing latency P95 evaluation...")
        samples = {"latency_ms": [100, 150, 200, 300, 400, 500, 600, 700, 800, 900]}
        slo = {"sli": "latency_p95_ms", "objective": 800, "window": "30d"}
        result = evaluate_sli(samples, slo)
        
        assert result["sli"] == "latency_p95_ms"
        assert "current_value" in result
        print(f"   ✓ P95 latency: {result['current_value']:.1f}ms (status: {result['status']})")
        
        print("2. Testing availability evaluation...")
        samples = {"success_count": 995, "total_count": 1000}
        slo = {"sli": "availability_pct", "objective": 99.0, "window": "30d"}
        result = evaluate_sli(samples, slo)
        
        assert result["sli"] == "availability_pct"
        print(f"   ✓ Availability: {result['current_value']:.2f}% (status: {result['status']})")
        
        print("3. Testing SLI monitor...")
        monitor = SLIMonitor()
        monitor.register_slo("test_service", {
            "slos": [
                {"sli": "latency_p95_ms", "objective": 800, "window": "30d"}
            ],
            "error_budget_days": 0.5
        })
        
        # Add some samples
        monitor.add_samples("test_service", {"latency_ms": [100, 200, 300]})
        status = monitor.get_current_status("test_service")
        assert status["service_id"] == "test_service"
        print(f"   ✓ SLI monitor status: {status['status']}")
        
        print("✅ SLI evaluation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ SLI evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_budget():
    """Test error budget tracking."""
    print("\n💰 Testing Error Budget Tracking...")
    
    try:
        tracker = ErrorBudgetTracker()
        
        print("1. Testing budget configuration...")
        slo_policy = {
            "slos": [{"sli": "availability_pct", "objective": 99.9, "window": "30d"}],
            "error_budget_days": 0.5
        }
        tracker.set_policy("test_service", slo_policy)
        print("   ✓ Error budget policy configured")
        
        print("2. Testing breach recording...")
        tracker.record_breach("test_service", "availability_pct", 0.1, 300)  # 5 min breach
        budget_status = tracker.get_remaining_budget("test_service")
        
        assert budget_status["service_id"] == "test_service"
        print(f"   ✓ Remaining budget: {budget_status['remaining_pct']:.2f}%")
        
        print("3. Testing deployment decision...")
        deployment_decision = tracker.should_block_deployment("test_service", 10.0)
        print(f"   ✓ Deployment decision: {'BLOCK' if deployment_decision['should_block'] else 'ALLOW'}")
        
        print("4. Testing burn rate calculation...")
        burn_rate = tracker.get_budget_burn_rate("test_service")
        print(f"   ✓ Burn rate: {burn_rate['burn_rate_per_hour']:.4f} days/hour")
        
        print("✅ Error budget tracking test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error budget tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chaos_runner():
    """Test chaos engineering functionality."""
    print("\n🌪️  Testing Chaos Engineering...")
    
    try:
        runner = get_chaos_runner()
        
        print("1. Testing chaos experiment...")
        experiment_id = await runner.start_experiment(
            target="test_service",
            blast_radius_pct=2.0,
            duration_s=5,
            experiment_type="latency_injection",
            parameters={"latency_ms": 500}
        )
        
        print(f"   ✓ Chaos experiment started: {experiment_id}")
        
        # Wait for experiment to complete
        await asyncio.sleep(6)
        
        print("2. Testing experiment status...")
        status = runner.get_experiment_status(experiment_id)
        if status:
            print(f"   ✓ Experiment status: {status.get('status', 'unknown')}")
        
        print("3. Testing experiment history...")
        history = runner.get_experiment_history(limit=5)
        print(f"   ✓ Retrieved {len(history)} experiment records")
        
        print("✅ Chaos engineering test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Chaos engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_snapshot_manager():
    """Test snapshot and rollback functionality."""
    print("\n📸 Testing Snapshot Management...")
    
    try:
        manager = SnapshotManager()
        await manager.start()
        
        print("1. Testing snapshot creation...")
        snapshot = await manager.create_snapshot(
            description="Test snapshot",
            snapshot_type="manual"
        )
        
        snapshot_id = snapshot["snapshot_id"]
        print(f"   ✓ Snapshot created: {snapshot_id}")
        
        print("2. Testing snapshot retrieval...")
        retrieved_snapshot = manager.get_snapshot(snapshot_id)
        assert retrieved_snapshot is not None
        assert retrieved_snapshot["snapshot_id"] == snapshot_id
        print("   ✓ Snapshot retrieved successfully")
        
        print("3. Testing snapshot listing...")
        snapshots = manager.list_snapshots(limit=10)
        assert len(snapshots) > 0
        print(f"   ✓ Listed {len(snapshots)} snapshots")
        
        print("4. Testing rollback...")
        rollback_result = await manager.rollback(
            to_snapshot=snapshot_id,
            reason="test_rollback"
        )
        assert rollback_result["status"] == "completed"
        print(f"   ✓ Rollback completed: {rollback_result['rollback_id']}")
        
        print("5. Testing rollback history...")
        history = manager.get_rollback_history(limit=5)
        assert len(history) > 0
        print(f"   ✓ Retrieved {len(history)} rollback records")
        
        await manager.stop()
        print("✅ Snapshot management test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Snapshot management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    print("🚀 Starting Grace Resilience Kernel Tests...\n")
    
    tests = [
        test_resilience_service,
        test_circuit_breaker,
        test_degradation_manager,
        test_sli_evaluation,
        test_error_budget,
        test_chaos_runner,
        test_snapshot_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Grace Resilience Kernel is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)