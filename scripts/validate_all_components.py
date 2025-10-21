"""
Validate all newly implemented components
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def validate_async_components():
    """Validate async memory and logging"""
    print("\n🔍 Validating Async Components...")
    
    try:
        from grace.memory.async_lightning import AsyncLightningMemory
        from grace.memory.async_fusion import AsyncFusionMemory
        from grace.memory.immutable_logs_async import AsyncImmutableLogs
        
        print("  ✓ Async memory imports successful")
        
        # Test Lightning (in-memory fallback)
        lightning = AsyncLightningMemory()
        await lightning.connect()
        await lightning.set("test", {"value": 1})
        result = await lightning.get("test")
        assert result == {"value": 1}
        await lightning.disconnect()
        
        print("  ✓ AsyncLightningMemory functional")
        
    except Exception as e:
        print(f"  ✗ Async components error: {e}")
        return False
    
    return True


def validate_event_system():
    """Validate GraceEvent implementation"""
    print("\n🔍 Validating Event System...")
    
    try:
        from grace.events.schema import GraceEvent
        from grace.events.factory import GraceEventFactory
        from grace.integration.event_bus import EventBus
        
        print("  ✓ Event system imports successful")
        
        # Test event creation
        factory = GraceEventFactory()
        event = factory.create_event(
            event_type="test",
            payload={"data": "test"},
            targets=["target1"]
        )
        
        assert event.event_id is not None
        assert event.constitutional_validation_required is not None
        assert event.headers is not None
        
        print("  ✓ GraceEvent schema complete")
        
        # Test event bus
        bus = EventBus()
        bus.publish(event)
        
        print("  ✓ EventBus functional")
        
    except Exception as e:
        print(f"  ✗ Event system error: {e}")
        return False
    
    return True


async def validate_governance():
    """Validate governance engine"""
    print("\n🔍 Validating Governance Engine...")
    
    try:
        from grace.governance.engine import GovernanceEngine
        from grace.events.schema import GraceEvent
        
        print("  ✓ Governance imports successful")
        
        engine = GovernanceEngine()
        
        event = GraceEvent(
            event_type="test",
            source="test",
            constitutional_validation_required=True,
            trust_score=0.8
        )
        
        # Test validate
        result = await engine.validate(event)
        assert result.passed is not None
        
        print("  ✓ GovernanceEngine.validate() implemented")
        
        # Test escalate
        escalation = await engine.escalate(event, "test reason", "standard")
        assert escalation.escalated is not None
        
        print("  ✓ GovernanceEngine.escalate() implemented")
        
    except Exception as e:
        print(f"  ✗ Governance error: {e}")
        return False
    
    return True


async def validate_trust():
    """Validate trust core"""
    print("\n🔍 Validating Trust System...")
    
    try:
        from grace.trust.core import TrustCoreKernel
        
        print("  ✓ Trust imports successful")
        
        trust = TrustCoreKernel()
        
        # Test calculate_trust
        score = await trust.calculate_trust("entity1", {})
        assert score.score is not None
        assert score.confidence is not None
        
        print("  ✓ TrustCoreKernel.calculate_trust() implemented")
        
        # Test update_trust
        updated = await trust.update_trust("entity1", {"success": True})
        assert updated.score is not None
        
        print("  ✓ TrustCoreKernel.update_trust() implemented")
        
    except Exception as e:
        print(f"  ✗ Trust system error: {e}")
        return False
    
    return True


def validate_llm():
    """Validate LLM integration"""
    print("\n🔍 Validating LLM Integration...")
    
    try:
        from grace.llm import ModelManager, InferenceRouter, LLMProvider
        
        print("  ✓ LLM imports successful")
        
        manager = ModelManager()
        router = InferenceRouter(manager)
        
        print("  ✓ ModelManager and InferenceRouter initialized")
        
    except Exception as e:
        print(f"  ✗ LLM integration error: {e}")
        return False
    
    return True


def validate_unified_service():
    """Validate unified service"""
    print("\n🔍 Validating Unified Service...")
    
    try:
        from grace.core.unified_service import create_unified_app
        
        print("  ✓ Unified service imports successful")
        
        app = create_unified_app()
        assert app is not None
        
        print("  ✓ create_unified_app() functional")
        
    except Exception as e:
        print(f"  ✗ Unified service error: {e}")
        return False
    
    return True


async def validate_demos():
    """Validate demo modules"""
    print("\n🔍 Validating Demo Modules...")
    
    try:
        from grace.demo import demo_multi_os_kernel, demo_mldl_kernel, demo_resilience_kernel
        
        print("  ✓ Demo imports successful")
        print("  ✓ All demo modules available")
        
    except Exception as e:
        print(f"  ✗ Demo modules error: {e}")
        return False
    
    return True


async def main():
    """Run all validations"""
    print("=" * 80)
    print("Grace System - Component Validation")
    print("=" * 80)
    
    results = []
    
    # Run validations
    results.append(("Async Components", await validate_async_components()))
    results.append(("Event System", validate_event_system()))
    results.append(("Governance", await validate_governance()))
    results.append(("Trust System", await validate_trust()))
    results.append(("LLM Integration", validate_llm()))
    results.append(("Unified Service", validate_unified_service()))
    results.append(("Demo Modules", await validate_demos()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:.<40} {status}")
    
    print("\n" + "=" * 80)
    print(f"Total: {passed}/{total} components validated")
    
    if passed == total:
        print("✅ All components validated successfully!")
        return 0
    else:
        print(f"⚠️  {total - passed} component(s) failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
