#!/usr/bin/env python3
"""
Real Integration Tests - Validates Actual Functionality
======================================================

Tests that actually validate behavior, not just file existence.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEventBus:
    """Test event bus functionality"""
    
    @pytest.mark.asyncio
    async def test_event_publish_subscribe(self):
        """Test that events are published and delivered"""
        from grace.events import EventBus, Event, EventPriority
        
        bus = EventBus()
        await bus.start()
        
        # Track received events
        received = []
        
        def handler(event):
            received.append(event)
        
        # Subscribe to events
        bus.subscribe("test.event", handler)
        
        # Publish event
        await bus.emit("test.event", {"message": "Hello"})
        await asyncio.sleep(0.2)  # Allow processing
        
        # Verify event was received
        assert len(received) > 0, "Event not delivered to subscriber"
        assert received[0].type == "test.event"
        assert received[0].data["message"] == "Hello"
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        """Test wildcard event subscriptions"""
        from grace.events import EventBus
        
        bus = EventBus()
        await bus.start()
        
        received = []
        bus.subscribe("system.*", lambda e: received.append(e))
        
        await bus.emit("system.started", {})
        await bus.emit("system.stopped", {})
        await bus.emit("other.event", {})
        await asyncio.sleep(0.2)
        
        # Should receive system.* events but not other.event
        assert len(received) == 2
        assert all(e.type.startswith("system.") for e in received)
        
        await bus.stop()


class TestGovernanceKernel:
    """Test governance kernel functionality"""
    
    @pytest.mark.asyncio
    async def test_policy_enforcement(self):
        """Test that policies are enforced"""
        from grace.governance import GovernanceKernel
        
        gov = GovernanceKernel()
        await gov.start()
        
        # Test security policy - hardcoded secret should be blocked
        is_allowed, violations = await gov.validate_action(
            "deploy",
            {"code": "api_key = 'hardcoded_secret_123'"}
        )
        
        assert not is_allowed, "Should block hardcoded secrets"
        assert len(violations) > 0
        assert violations[0].policy_name == "No Hardcoded Secrets"
        
        await gov.stop()
    
    @pytest.mark.asyncio
    async def test_policy_stats(self):
        """Test governance statistics"""
        from grace.governance import GovernanceKernel
        
        gov = GovernanceKernel()
        stats = gov.get_stats()
        
        assert stats["total_policies"] > 0
        assert "enabled_policies" in stats
        assert stats["total_violations"] >= 0


class TestMTLEngine:
    """Test MTL engine functionality"""
    
    @pytest.mark.asyncio
    async def test_experience_learning(self):
        """Test that MTL learns from experiences"""
        from grace.mtl import MTLEngine
        
        mtl = MTLEngine()
        await mtl.start()
        
        # Log an experience
        experience = {
            "type": "task_completion",
            "domain": "nlp",
            "success": True,
            "metrics": {"accuracy": 0.92}
        }
        
        await mtl.learn_from_experience(experience)
        
        # Verify experience was stored
        stats = mtl.get_stats()
        assert stats["experiences"] == 1
        assert stats["active"] is True
        
        await mtl.stop()


class TestQuorumService:
    """Test quorum voting functionality"""
    
    @pytest.mark.asyncio
    async def test_voting_session(self):
        """Test that quorum voting works"""
        from grace.services.quorum_service import QuorumService, DecisionType, VoteChoice
        
        quorum = QuorumService(registry=None)
        
        # Start a session
        session_id = quorum.start_session(
            decision_type=DecisionType.MODEL_APPROVAL,
            context={"model": "test_v1"},
            required_quorum=2,
            required_consensus=0.75
        )
        
        assert session_id is not None
        
        # Cast votes
        assert quorum.cast_vote(session_id, "ml_specialist_1", VoteChoice.APPROVE, 0.9, "Looks good")
        assert quorum.cast_vote(session_id, "fairness_specialist", VoteChoice.APPROVE, 0.85, "Fair")
        
        # Get result
        result = quorum.get_result(session_id)
        
        assert result is not None
        assert result["approved"] is True
        assert result["total_votes"] == 2
        assert result["consensus"] > 0.75


class TestCodeGenerator:
    """Test code generation functionality"""
    
    def test_template_generation(self):
        """Test that code templates are generated"""
        from grace.mtl.collaborative_code_gen import CollaborativeCodeGenerator, CodeGenerationTask
        
        gen = CollaborativeCodeGenerator()
        
        task = CodeGenerationTask(
            task_id="test_001",
            requirements="Create a user authentication system",
            language="python"
        )
        
        # Generate code (sync method)
        code = gen._synthesize_code(task, {})
        
        # Verify code was generated
        assert code is not None
        assert len(code) > 100
        assert "class" in code.lower() or "def" in code
        assert task.language.lower() in code.lower() or "python" in code.lower()


class TestVoiceInterface:
    """Test voice interface functionality"""
    
    @pytest.mark.asyncio
    async def test_voice_initialization(self):
        """Test voice interface can be initialized"""
        from grace.interface import VoiceInterface
        
        voice = VoiceInterface()
        await voice.start()
        
        stats = voice.get_stats()
        assert stats["active"] is True
        assert stats["language"] == "en-US"
        
        await voice.stop()
        assert voice.active is False


class TestDisagreementConsensus:
    """Test multi-model consensus"""
    
    @pytest.mark.asyncio
    async def test_consensus_resolution(self):
        """Test consensus between multiple predictions"""
        from grace.mldl.disagreement_consensus import DisagreementConsensus
        
        consensus_engine = DisagreementConsensus()
        
        predictions = [
            {"result": "A", "confidence": 0.9, "model": "model1"},
            {"result": "A", "confidence": 0.85, "model": "model2"},
            {"result": "B", "confidence": 0.6, "model": "model3"}
        ]
        
        result = await consensus_engine.resolve(predictions)
        
        assert result["result"] == "A"  # A has higher weighted vote
        assert result["confidence"] > 0.5
        assert result["num_models"] == 3


class TestBreakthroughSystem:
    """Test breakthrough detection"""
    
    @pytest.mark.asyncio
    async def test_improvement_proposal(self):
        """Test improvement proposals"""
        from grace.core.breakthrough import BreakthroughMetaLoop
        
        breakthrough = BreakthroughMetaLoop()
        await breakthrough.start()
        
        proposal = await breakthrough.propose_improvement("accuracy", 0.85)
        
        assert proposal is not None
        assert proposal["area"] == "accuracy"
        assert proposal["current_performance"] == 0.85
        assert "expected_improvement" in proposal
        
        await breakthrough.stop()


class TestRuntimeIntegration:
    """Test runtime system integration"""
    
    @pytest.mark.asyncio
    async def test_runtime_bootstrap(self):
        """Test that runtime can bootstrap"""
        from grace.runtime import GraceRuntime, RuntimeConfig, RuntimeMode
        
        config = RuntimeConfig(
            mode=RuntimeMode.SINGLE_KERNEL,
            specific_kernel="orchestration",
            debug=True
        )
        
        runtime = GraceRuntime(config)
        success = await runtime.bootstrap()
        
        assert success is True
        assert runtime.phase == "complete" or "complete" in runtime.phase.value
        
        status = runtime.get_status()
        assert status["running"] is False  # Not started yet
        assert len(status["services"]) > 0


class TestDatabaseCompatibility:
    """Test database compatibility shim"""
    
    def test_database_imports(self):
        """Test that all database imports work"""
        from grace.database import Base, get_db, get_async_db, SessionLocal, init_db
        
        assert Base is not None
        assert callable(get_db)
        assert callable(get_async_db)
        assert callable(SessionLocal)
        assert callable(init_db)
    
    def test_config_compatibility(self):
        """Test that config imports work"""
        from grace.config import get_config, get_settings
        
        config1 = get_config()
        config2 = get_settings()
        
        assert config1 is not None
        assert config2 is not None
        assert config1 is config2  # Same instance


class TestAuthModels:
    """Test auth models are correct"""
    
    def test_user_model(self):
        """Test User model has all required fields"""
        from grace.auth.models import User
        
        # Check columns exist
        assert hasattr(User, 'id')
        assert hasattr(User, 'username')
        assert hasattr(User, 'email')
        assert hasattr(User, 'locked_until')  # Previously missing
        
        # Create instance (won't persist, just validate structure)
        user = User(
            username="test",
            email="test@example.com",
            hashed_password="hashed"
        )
        assert user.username == "test"
    
    def test_refresh_token_model(self):
        """Test RefreshToken model has all required fields"""
        from grace.auth.models import RefreshToken
        from datetime import datetime, timezone, timedelta
        
        # Check columns exist
        assert hasattr(RefreshToken, 'revoked')  # Previously missing
        
        # Create instance
        token = RefreshToken(
            token="test_token",
            user_id=1,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            revoked=False
        )
        
        assert token.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
