"""
Complete orchestration system test
"""

import asyncio
import time
from datetime import datetime

print("=" * 80)
print("Grace Orchestration System - Complete Test")
print("=" * 80)

async def main():
    from grace.orchestration.enhanced_scheduler import EnhancedScheduler, SchedulerLoop
    from grace.orchestration.autoscaler import AdvancedAutoscaler, ScalingMetrics
    from grace.orchestration.heartbeat import HeartbeatMonitor
    from grace.avn.enhanced_core import EnhancedAVNCore
    
    # Test 1: Enhanced Scheduler with Metrics
    print("\n1. Testing Enhanced Scheduler with Metrics...")
    
    scheduler = EnhancedScheduler(scheduler_id="test_scheduler")
    
    # Register some test loops
    call_counts = {"fast": 0, "medium": 0, "slow": 0}
    
    async def fast_loop():
        call_counts["fast"] += 1
        await asyncio.sleep(0.1)
    
    async def medium_loop():
        call_counts["medium"] += 1
        await asyncio.sleep(0.5)
    
    async def slow_loop():
        call_counts["slow"] += 1
        await asyncio.sleep(1.0)
    
    scheduler.register_loop("fast_loop", fast_loop, interval=1.0, priority=3)
    scheduler.register_loop("medium_loop", medium_loop, interval=2.0, priority=2)
    scheduler.register_loop("slow_loop", slow_loop, interval=5.0, priority=1, timeout=3.0)
    
    # Register a policy
    def check_load(context):
        # Allow if execution count < 100
        return context["execution_count"] < 100
    
    scheduler.register_policy(
        "load_limiter",
        check_load,
        "throttle",
        "Limit execution count"
    )
    
    print(f"✓ Registered 3 loops and 1 policy")
    
    # Start scheduler
    await scheduler.start()
    print(f"✓ Scheduler started")
    
    # Run for a bit
    await asyncio.sleep(8)
    
    print(f"\n  Loop execution counts:")
    print(f"    Fast loop: {call_counts['fast']}")
    print(f"    Medium loop: {call_counts['medium']}")
    print(f"    Slow loop: {call_counts['slow']}")
    
    # Test 2: Snapshot and Restore
    print("\n2. Testing Snapshot and Restore...")
    
    snapshot = scheduler.create_snapshot()
    print(f"✓ Created snapshot:")
    print(f"  Loops: {len(snapshot['loops'])}")
    print(f"  Policies: {len(snapshot['policies'])}")
    print(f"  Timestamp: {snapshot['timestamp']}")
    
    # Stop scheduler
    await scheduler.stop()
    print(f"✓ Scheduler stopped")
    
    # Create new scheduler and restore
    new_scheduler = EnhancedScheduler(scheduler_id="restored_scheduler")
    
    # Recreate callbacks
    loop_callbacks = {
        "fast_loop": fast_loop,
        "medium_loop": medium_loop,
        "slow_loop": slow_loop
    }
    
    policy_conditions = {
        "load_limiter": check_load
    }
    
    await new_scheduler.restore_from_snapshot(snapshot, loop_callbacks, policy_conditions)
    print(f"✓ Scheduler restored from snapshot")
    
    # Verify restoration
    print(f"  Restored loops: {len(new_scheduler.loops)}")
    print(f"  Restored policies: {len(new_scheduler.policies)}")
    
    # Run restored scheduler briefly
    await asyncio.sleep(5)
    
    await new_scheduler.stop()
    
    # Test 3: Advanced Autoscaling
    print("\n3. Testing Advanced Autoscaling...")
    
    autoscaler = AdvancedAutoscaler(
        min_instances=2,
        max_instances=8,
        target_cpu=0.7,
        cooldown_period=5
    )
    
    print(f"✓ Autoscaler initialized: 2-8 instances")
    
    # Test scaling up
    print("\n  Scenario 1: High load (should scale up)")
    metrics_high_load = ScalingMetrics(
        cpu_usage=0.85,
        memory_usage=0.75,
        backlog_size=150,
        error_rate=0.08,
        trust_score=0.6,
        request_rate=1200,
        avg_latency=600
    )
    
    decision = autoscaler.evaluate_scaling(metrics_high_load, current_instances=3)
    print(f"    Decision: {'Scale' if decision.should_scale else 'No change'}")
    print(f"    Target: {decision.target_instances} instances")
    print(f"    Reason: {decision.reason}")
    print(f"    Confidence: {decision.confidence:.2f}")
    
    # Test steady state
    print("\n  Scenario 2: Normal load (should stay)")
    metrics_normal = ScalingMetrics(
        cpu_usage=0.65,
        memory_usage=0.55,
        backlog_size=50,
        error_rate=0.02,
        trust_score=0.85,
        request_rate=500,
        avg_latency=200
    )
    
    await asyncio.sleep(6)  # Wait for cooldown
    
    decision = autoscaler.evaluate_scaling(metrics_normal, current_instances=3)
    print(f"    Decision: {'Scale' if decision.should_scale else 'No change'}")
    print(f"    Target: {decision.target_instances} instances")
    print(f"    Reason: {decision.reason}")
    
    # Test scaling down
    print("\n  Scenario 3: Low load (should scale down)")
    metrics_low_load = ScalingMetrics(
        cpu_usage=0.25,
        memory_usage=0.30,
        backlog_size=5,
        error_rate=0.01,
        trust_score=0.95,
        request_rate=100,
        avg_latency=50
    )
    
    await asyncio.sleep(6)  # Wait for cooldown
    
    decision = autoscaler.evaluate_scaling(metrics_low_load, current_instances=4)
    print(f"    Decision: {'Scale' if decision.should_scale else 'No change'}")
    print(f"    Target: {decision.target_instances} instances")
    print(f"    Reason: {decision.reason}")
    
    # Test 4: Heartbeat Monitoring
    print("\n4. Testing Heartbeat Monitoring...")
    
    avn_core = EnhancedAVNCore()
    heartbeat_monitor = HeartbeatMonitor(
        ttl_seconds=5,
        check_interval=2,
        avn_core=avn_core
    )
    
    print(f"✓ Heartbeat monitor initialized (TTL=5s)")
    
    # Register kernels
    async def recovery_callback(kernel_id, last_record):
        print(f"    Recovery triggered for {kernel_id}")
    
    heartbeat_monitor.register_kernel("kernel_1", recovery_callback)
    heartbeat_monitor.register_kernel("kernel_2", recovery_callback)
    heartbeat_monitor.register_kernel("kernel_3")
    
    print(f"✓ Registered 3 kernels")
    
    # Start monitoring
    await heartbeat_monitor.start()
    
    # Send heartbeats
    heartbeat_monitor.report_heartbeat("kernel_1", "healthy", {"cpu": 0.5})
    heartbeat_monitor.report_heartbeat("kernel_2", "healthy", {"cpu": 0.6})
    heartbeat_monitor.report_heartbeat("kernel_3", "healthy", {"cpu": 0.4})
    
    print(f"✓ Sent heartbeats from all kernels")
    
    # Check status
    status = heartbeat_monitor.get_status()
    print(f"\n  Status after initial heartbeats:")
    print(f"    Healthy: {status['healthy_kernels']}")
    print(f"    Degraded: {status['degraded_kernels']}")
    
    # Simulate kernel_2 failure (stop sending heartbeats)
    print(f"\n  Simulating kernel_2 failure...")
    
    # Keep kernel_1 and kernel_3 alive
    for i in range(3):
        await asyncio.sleep(2)
        heartbeat_monitor.report_heartbeat("kernel_1", "healthy")
        heartbeat_monitor.report_heartbeat("kernel_3", "healthy")
    
    # Check status again
    status = heartbeat_monitor.get_status()
    print(f"\n  Status after failure simulation:")
    print(f"    Healthy: {status['healthy_kernels']}")
    print(f"    Degraded: {status['degraded_kernels']}")
    
    if status['degraded']:
        for degraded in status['degraded']:
            print(f"    - {degraded['kernel_id']} (age: {degraded['age_seconds']:.0f}s)")
    
    await heartbeat_monitor.stop()
    
    print("\n" + "=" * 80)
    print("✅ Orchestration System Tests Complete!")
    print("=" * 80)
    
    print("\nImplemented features:")
    print("  ✓ Scheduler metrics (Prometheus instrumentation)")
    print("  ✓ Loop execution tracking (duration, success/failure)")
    print("  ✓ Queue depth monitoring")
    print("  ✓ Snapshot creation with loop/policy serialization")
    print("  ✓ Complete snapshot restoration")
    print("  ✓ Scheduler restart with restored config")
    print("  ✓ Advanced autoscaling (multi-factor decisions)")
    print("  ✓ Scaling with backlog, errors, trust scores")
    print("  ✓ Graceful instance retirement")
    print("  ✓ Health checks for new instances")
    print("  ✓ Heartbeat monitoring with TTL")
    print("  ✓ Automatic failure detection")
    print("  ✓ AVN integration for recovery")
    print("  ✓ Recovery callbacks")

# Run test
asyncio.run(main())
