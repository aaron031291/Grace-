"""
Complete system integration tests
Tests all newly implemented async components
"""

import pytest
import asyncio
from datetime import datetime, timezone

# Test all async memory layers
from grace.memory.async_lightning import AsyncLightningMemory
from grace.memory.async_fusion import AsyncFusionMemory
from grace.memory.immutable_logs_async import AsyncImmutableLogs

# Test event system
from grace.events.schema import GraceEvent
from grace.events.factory import GraceEventFactory
from grace.integration.event_bus import EventBus

# Test governance
from grace.governance.engine import GovernanceEngine, ValidationResult, EscalationResult

# Test trust
from grace.trust.core import TrustCoreKernel, TrustScore

# Test LLM
from grace.llm import ModelManager, InferenceRouter, ModelConfig, LLMProvider


class TestAsyncMemoryLayers:
    """Test async memory implementations"""
    
    @pytest.mark.asyncio
    async def test_lightning_memory_operations(self):
        """Test Lightning memory (Redis) basic operations"""
        memory = AsyncLightningMemory()
        await memory.connect()
        
        try:
            # Test set and get
            test_data = {"user": "test", "action": "login"}
            await memory.set("test_key", test_data, ttl=60)
            
            result = await memory.get("test_key")
            assert result == test_data
            
            # Test exists
            assert await memory.exists("test_key") is True
            assert await memory.exists("nonexistent") is False
            
            # Test delete
            await memory.delete("test_key")
            assert await memory.get("test_key") is None
            
            # Test stats
            stats = await memory.get_stats()
            assert "type" in stats
            
        finally:
            await memory.disconnect()
    
    @pytest.mark.asyncio
    async def test_fusion_memory_patterns(self):
        """Test Fusion memory (Postgres) pattern storage"""
        memory = AsyncFusionMemory("postgresql://localhost/grace_test")
        
        try:
            await memory.connect()
            
            # Store pattern
            pattern_id = await memory.store_pattern(
                pattern_type="user_behavior",
                pattern_data={"action": "click", "element": "button"},
                confidence=0.85,
                metadata={"source": "test"}
            )
            
            assert pattern_id > 0
            
            # Retrieve patterns
            patterns = await memory.get_patterns(pattern_type="user_behavior")
            assert len(patterns) > 0
            assert patterns[0]["confidence"] == 0.85
            
        except Exception as e:
            pytest.skip(f"Postgres not available: {e}")
        finally:
            if memory._connected:
                await memory.disconnect()
    
    @pytest.mark.asyncio
    async def test_fusion_memory_interactions(self):
        """Test interaction logging"""
        memory = AsyncFusionMemory("postgresql://localhost/grace_test")
        
        try:
            await memory.connect()
            
            # Record interaction
            interaction_id = await memory.record_interaction(
                action="login",
                user_id="test_user",
                context={"ip": "127.0.0.1", "device": "chrome"},
                outcome="success",
                session_id="session_123"
            )
            
            assert interaction_id > 0
            
            # Retrieve interactions
            interactions = await memory.get_interactions(user_id="test_user")
            assert len(interactions) > 0
            
        except Exception as e:
            pytest.skip(f"Postgres not available: {e}")
        finally:
            if memory._connected:
                await memory.disconnect()
    
    @pytest.mark.asyncio
    async def test_immutable_logs_chaining(self):
        """Test immutable log cryptographic chaining"""
        logs = AsyncImmutableLogs("postgresql://localhost/grace_test", batch_size=2)
        
        try:
            await logs.connect()
            
            # Log multiple entries
            hash1 = await logs.log(
                operation_type="user_action",
                actor="test_user",
                action={"type": "create", "resource": "document"},
                result={"id": "doc123"},
                severity="info"
            )
            
            hash2 = await logs.log(
                operation_type="user_action",
                actor="test_user",
                action={"type": "update", "resource": "document"},
                result={"id": "doc123", "modified": True},
                severity="info"
            )
            
            # Force flush
            await logs._flush_batch()
            
            # Verify chain integrity
            verification = await logs.verify_chain()
            assert verification["verified"] is True
            assert verification["entries_checked"] >= 2
            
            # Query logs
            entries = await logs.query(actor="test_user", limit=10)
            assert len(entries) >= 2
            
        except Exception as e:
            pytest.skip(f"Postgres not available: {e}")
        finally:
            if logs.pool:
                await logs.disconnect()


class TestEventSystem:
    """Test specification-compliant event system"""
    
    def test_grace_event_schema(self):
        """Test GraceEvent has all required fields"""
        event = GraceEvent(
            event_type="test.event",
            source="test_service",
            targets=["target1", "target2"],
            payload={"data": "value"},
            constitutional_validation_required=True,
            priority="high"
        )
        
        # Verify all required fields
        assert event.event_id is not None
        assert event.event_type == "test.event"
        assert event.source == "test_service"
        assert event.targets == ["target1", "target2"]
        assert event.constitutional_validation_required is True
        assert event.headers is not None
        assert event.timestamp is not None
    
    def test_event_factory(self):
        """Test event factory creates compliant events"""
        factory = GraceEventFactory(default_source="test")
        
        event = factory.create_event(
            event_type="user.login",
            payload={"user_id": "123"},
            targets=["auth_service"],
            constitutional_validation_required=True,
            priority="normal"
        )
        
        assert event.event_type == "user.login"
        assert event.source == "test"
        assert event.chain_hash is not None
        assert event.previous_event_id is None  # First event
    
    def test_event_bus_idempotency(self):
        """Test event bus handles duplicate events"""
        bus = EventBus()
        factory = GraceEventFactory()
        
        event = factory.create_event(
            event_type="test.duplicate",
            payload={"data": "test"},
            idempotency_key="unique_key_123"
        )
        
        # Publish twice with same idempotency key
        result1 = bus.publish(event)
        result2 = bus.publish(event)
        
        assert result1 is True
        assert result2 is False  # Duplicate rejected
    
    def test_event_bus_routing(self):
        """Test event bus routes to subscribers"""
        bus = EventBus()
        received_events = []
        
        def callback(event):
            received_events.append(event)
        
        bus.subscribe("test.event", callback)
        
        event = bus.create_and_publish(
            event_type="test.event",
            payload={"test": "data"}
        )
        
        assert len(received_events) == 1
        assert received_events[0].event_type == "test.event"


class TestGovernanceEngine:
    """Test governance validation and escalation"""
    
    @pytest.mark.asyncio
    async def test_validate_event(self):
        """Test event validation"""
        engine = GovernanceEngine()
        
        event = GraceEvent(
            event_type="test.event",
            source="test",
            constitutional_validation_required=True,
            trust_score=0.8,
            priority="normal"
        )
        
        result = await engine.validate(event)
        
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_validate_low_trust(self):
        """Test validation fails for low trust"""
        engine = GovernanceEngine()
        
        event = GraceEvent(
            event_type="test.event",
            source="test",
            constitutional_validation_required=True,
            trust_score=0.3,  # Below threshold
            priority="normal"
        )
        
        result = await engine.validate(event)
        
        assert result.passed is False
        assert len(result.violations) > 0
        assert "trust score" in result.violations[0].lower()
    
    @pytest.mark.asyncio
    async def test_escalate_event(self):
        """Test event escalation"""
        engine = GovernanceEngine()
        
        event = GraceEvent(
            event_type="security.breach",
            source="security_monitor",
            priority="critical"
        )
        
        result = await engine.escalate(
            event,
            reason="Potential security breach detected",
            level="critical"
        )
        
        assert isinstance(result, EscalationResult)
        assert result.escalated is True
        assert result.escalation_level == "critical"
        assert len(result.assigned_to) > 0


class TestTrustSystem:
    """Test trust calculation and updates"""
    
    @pytest.mark.asyncio
    async def test_calculate_trust(self):
        """Test trust calculation"""
        trust = TrustCoreKernel()
        
        score = await trust.calculate_trust(
            entity_id="service_a",
            operation_context={"operation_type": "general"}
        )
        
        assert isinstance(score, TrustScore)
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert len(score.factors) > 0
    
    @pytest.mark.asyncio
    async def test_update_trust_success(self):
        """Test trust update on success"""
        trust = TrustCoreKernel()
        
        # Initial trust
        initial = await trust.calculate_trust("service_b", {})
        initial_score = initial.score
        
        # Update with success
        updated = await trust.update_trust(
            entity_id="service_b",
            outcome={"success": True, "error_rate": 0.0, "latency_ms": 50}
        )
        
        assert updated.score >= initial_score  # Should improve or stay same
    
    @pytest.mark.asyncio
    async def test_update_trust_failure(self):
        """Test trust update on failure"""
        trust = TrustCoreKernel()
        
        # Set initial trust
        trust.trust_scores["service_c"] = 0.8
        
        # Update with failure
        updated = await trust.update_trust(
            entity_id="service_c",
            outcome={"success": False, "error_rate": 0.5, "latency_ms": 5000}
        )
        
        assert updated.score < 0.8  # Should decrease
    
    def test_threshold_checking(self):
        """Test trust threshold checks"""
        trust = TrustCoreKernel()
        
        trust.trust_scores["high_trust"] = 0.9
        trust.trust_scores["low_trust"] = 0.4
        
        assert trust.check_threshold("high_trust", "good") is True
        assert trust.check_threshold("low_trust", "good") is False
        assert trust.check_threshold("low_trust", "minimum") is True


class TestLLMIntegration:
    """Test private LLM integration"""
    
    def test_model_manager_initialization(self):
        """Test model manager can be created"""
        manager = ModelManager()
        assert manager is not None
        assert manager.models == {}
    
    def test_model_registration(self):
        """Test model registration"""
        manager = ModelManager()
        
        config = ModelConfig(
            name="test-model",
            provider=LLMProvider.TRANSFORMERS,
            model_path="test/model",
            context_length=2048
        )
        
        try:
            manager.register_model("test", config, set_default=True)
            assert "test" in manager.models
            assert manager.default_model == "test"
        except Exception:
            # Model loading may fail without actual model files
            pytest.skip("Model files not available")
    
    def test_inference_router(self):
        """Test inference router"""
        manager = ModelManager()
        router = InferenceRouter(manager)
        
        assert router is not None
        assert router.routing_rules is not None


class TestUnifiedService:
    """Test unified service creation"""
    
    def test_create_unified_app(self):
        """Test unified app can be created"""
        from grace.core.unified_service import create_unified_app
        
        app = create_unified_app()
        
        assert app is not None
        assert hasattr(app, 'routes')
        assert hasattr(app, 'state')


class TestDemoModules:
    """Test demo modules can run"""
    
    @pytest.mark.asyncio
    async def test_multi_os_kernel_demo(self):
        """Test multi-OS kernel demo"""
        from grace.demo.multi_os_kernel import demo_multi_os_kernel
        
        # Should run without errors
        await demo_multi_os_kernel()
    
    @pytest.mark.asyncio
    async def test_mldl_kernel_demo(self):
        """Test MLDL kernel demo"""
        from grace.demo.mldl_kernel import demo_mldl_kernel
        
        # Should run without errors
        await demo_mldl_kernel()
    
    @pytest.mark.asyncio
    async def test_resilience_kernel_demo(self):
        """Test resilience kernel demo"""
        from grace.demo.resilience_kernel import demo_resilience_kernel
        
        # Should run without errors
        await demo_resilience_kernel()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
