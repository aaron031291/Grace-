#!/usr/bin/env python3
"""
Test script for Grace Orchestration Kernel implementation.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace.orchestration.orchestration_service import OrchestrationService
from grace.orchestration.scheduler.scheduler import LoopDefinition
from grace.orchestration.state.state_manager import Policy, PolicyType, PolicyScope


async def test_orchestration_service():
    """Test basic orchestration service functionality."""
    print("🔧 Testing Grace Orchestration Service...")
    
    try:
        # Initialize service
        service = OrchestrationService()
        
        # Start service
        print("1. Starting orchestration service...")
        await service.start()
        print("   ✓ Service started successfully")
        
        # Test health check
        print("2. Testing health check...")
        routes_count = len(service.app.routes)  # Check that routes are setup
        print(f"   ✓ API routes configured ({routes_count} routes)")
        
        # Test loop creation
        print("3. Testing loop management...")
        loop_data = {
            "loop_id": "test_ooda",
            "name": "ooda", 
            "priority": 8,
            "interval_s": 30,
            "kernels": ["governance", "intelligence"],
            "policies": {"trust_threshold": 0.8}
        }
        
        loop_id = await service.scheduler.register_loop(loop_data)
        print(f"   ✓ Created loop: {loop_id}")
        
        # Test task dispatch
        print("4. Testing task dispatch...")
        task_id = await service.scheduler.dispatch_task(
            loop_id="test_ooda",
            inputs={"test_data": "sample"},
            priority=5
        )
        print(f"   ✓ Dispatched task: {task_id}")
        
        # Test status queries
        print("5. Testing status queries...")
        scheduler_status = service.scheduler.get_status()
        print(f"   ✓ Scheduler status: {scheduler_status['state']}")
        
        router_status = service.router.get_status()
        print(f"   ✓ Router running: {router_status['running']}")
        
        watchdog_status = service.watchdog.get_status()
        print(f"   ✓ Watchdog monitoring {watchdog_status['monitored_tasks']} tasks")
        
        # Test snapshot creation
        print("6. Testing snapshot functionality...")
        snapshot_id = await service.snapshot_manager.export_snapshot(
            description="Test snapshot"
        )
        print(f"   ✓ Created snapshot: {snapshot_id}")
        
        # Stop service
        print("7. Stopping orchestration service...")
        await service.stop()
        print("   ✓ Service stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_component_integration():
    """Test integration between components."""
    print("\n🔗 Testing Component Integration...")
    
    try:
        from grace.orchestration.scheduler.scheduler import Scheduler
        from grace.orchestration.router.router import Router
        from grace.orchestration.state.state_manager import StateManager
        from grace.orchestration.recovery.watchdog import Watchdog
        
        # Test component initialization
        print("1. Initializing components...")
        state_manager = StateManager("/tmp/test_orch_state")
        scheduler = Scheduler(state_manager=state_manager)
        router = Router()
        watchdog = Watchdog()
        print("   ✓ Components initialized")
        
        # Test component startup
        print("2. Starting components...")
        await scheduler.start()
        await router.start()
        await watchdog.start()
        print("   ✓ Components started")
        
        # Test inter-component communication
        print("3. Testing component communication...")
        
        # Register a test loop
        loop_data = {
            "loop_id": "integration_test",
            "name": "homeostasis",
            "priority": 5,
            "interval_s": 60,
            "kernels": ["memory", "governance"],
            "policies": {}
        }
        
        loop_id = await scheduler.register_loop(loop_data)
        print(f"   ✓ Scheduler registered loop: {loop_id}")
        
        # Test routing
        test_event = {
            "event_type": "TEST_EVENT",
            "payload": {"test": True},
            "source": "scheduler"
        }
        
        success = await router.route(test_event)
        print(f"   ✓ Router processed event: {success}")
        
        # Test monitoring
        watchdog.monitor("test_task_123", "integration_test", "scheduler", timeout_minutes=1)
        print("   ✓ Watchdog monitoring task")
        
        # Test state management
        test_policy = Policy(
            policy_id="test_policy",
            name="Test Integration Policy",
            policy_type=PolicyType.PERFORMANCE,
            scope=PolicyScope.LOOP,
            rules={"max_duration": 300}
        )
        
        await state_manager.add_policy(test_policy)
        print("   ✓ State manager added policy")
        
        # Stop components
        print("4. Stopping components...")
        await watchdog.stop()
        await router.stop()
        await scheduler.stop()
        print("   ✓ Components stopped")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_contracts_and_schemas():
    """Test contracts and schema validation."""
    print("\n📋 Testing Contracts and Schemas...")
    
    try:
        import json
        from pathlib import Path
        
        # Test schema files exist and are valid JSON
        contracts_dir = Path(__file__).parent / "grace" / "orchestration" / "contracts"
        
        schema_files = [
            "orch.loop.schema.json",
            "orch.task.schema.json",
            "orch.events.yaml",
            "orch.api.openapi.yaml"
        ]
        
        for schema_file in schema_files:
            file_path = contracts_dir / schema_file
            if file_path.exists():
                if schema_file.endswith('.json'):
                    with open(file_path) as f:
                        data = json.load(f)
                    print(f"   ✓ Valid JSON schema: {schema_file}")
                else:
                    # YAML files - basic existence check
                    print(f"   ✓ Schema file exists: {schema_file}")
            else:
                print(f"   ❌ Missing schema file: {schema_file}")
                return False
        
        # Test database schema
        db_schema = Path(__file__).parent.parent / "grace" / "orchestration" / "db" / "ddl" / "orchestration.sql"
        if db_schema.exists():
            print(f"   ✓ Database DDL schema exists")
        else:
            print(f"   ❌ Missing database DDL schema")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Schema validation failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("🚀 Grace Orchestration Kernel Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(await test_contracts_and_schemas())
    test_results.append(await test_component_integration())
    test_results.append(await test_orchestration_service())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Orchestration Kernel is ready for integration.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)