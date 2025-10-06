"""Tests for MTL Kernel components."""

import asyncio
import pytest
from datetime import datetime, timedelta

from grace.mtl_kernel.lightning_memory import LightningMemory
from grace.mtl_kernel.fusion_memory import FusionMemory
from grace.mtl_kernel.vector_memory import VectorMemory
from grace.mtl_kernel.trust_core import TrustCore
from grace.mtl_kernel.immutable_logger import ImmutableLogger
from grace.mtl_kernel.memory_orchestrator import MemoryOrchestrator
from grace.mtl_kernel.mtl_service import MTLService


class TestLightningMemory:
    """Test Lightning Memory (Redis-like cache)."""
    
    @pytest.mark.asyncio
    async def test_basic_operations(self):
        """Test basic get/set operations."""
        lightning = LightningMemory()
        
        # Set and get
        await lightning.set("test_key", "test_value")
        value = await lightning.get("test_key")
        assert value == "test_value"
        
        # Delete
        await lightning.delete("test_key")
        value = await lightning.get("test_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        lightning = LightningMemory(default_ttl=1)
        
        await lightning.set("expiring_key", "value")
        value = await lightning.get("expiring_key")
        assert value == "value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        value = await lightning.get("expiring_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch get/set."""
        lightning = LightningMemory()
        
        # Set many
        items = {"key1": "val1", "key2": "val2", "key3": "val3"}
        count = await lightning.set_many(items)
        assert count == 3
        
        # Get many
        result = await lightning.get_many(["key1", "key2", "key3"])
        assert result == items


class TestFusionMemory:
    """Test Fusion Memory (PostgreSQL-like storage)."""
    
    @pytest.mark.asyncio
    async def test_insert_query(self):
        """Test insert and query operations."""
        fusion = FusionMemory()
        
        # Insert
        entry_id = await fusion.insert(
            "learned_patterns",
            {
                "pattern_type": "test_pattern",
                "pattern_data": {"key": "value"},
                "confidence": 0.9,
                "usage_count": 5
            },
            trust_score=0.8
        )
        
        assert entry_id is not None
        
        # Query
        results = await fusion.query(
            "learned_patterns",
            filters={"pattern_type": "test_pattern"}
        )
        
        assert len(results) == 1
        assert results[0]["data"]["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_transaction(self):
        """Test transaction support."""
        fusion = FusionMemory()
        
        operations = [
            {
                "type": "insert",
                "table": "learned_patterns",
                "data": {"pattern_type": "tx_pattern_1"},
                "trust_score": 0.7
            },
            {
                "type": "insert",
                "table": "learned_patterns",
                "data": {"pattern_type": "tx_pattern_2"},
                "trust_score": 0.8
            }
        ]
        
        success = await fusion.transaction(operations)
        assert success is True
        
        # Verify both inserted
        results = await fusion.query("learned_patterns", limit=100)
        tx_patterns = [r for r in results if "tx_pattern" in r["data"].get("pattern_type", "")]
        assert len(tx_patterns) == 2


class TestVectorMemory:
    """Test Vector Memory (ChromaDB-like)."""
    
    @pytest.mark.asyncio
    async def test_add_search(self):
        """Test adding and searching vectors."""
        vector = VectorMemory()
        
        # Add vectors
        embedding1 = VectorMemory.generate_mock_embedding("machine learning")
        embedding2 = VectorMemory.generate_mock_embedding("deep learning")
        
        await vector.add("knowledge", embedding1, {"topic": "ML"})
        await vector.add("knowledge", embedding2, {"topic": "DL"})
        
        # Search
        query_embedding = VectorMemory.generate_mock_embedding("learning algorithms")
        results = await vector.search("knowledge", query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert "similarity" in results[0]
    
    @pytest.mark.asyncio
    async def test_batch_add(self):
        """Test batch adding vectors."""
        vector = VectorMemory()
        
        embeddings = [
            VectorMemory.generate_mock_embedding(f"text_{i}")
            for i in range(5)
        ]
        metadatas = [{"index": i} for i in range(5)]
        
        ids = await vector.add_many("patterns", embeddings, metadatas)
        assert len(ids) == 5


class TestTrustCore:
    """Test Trust Core."""
    
    @pytest.mark.asyncio
    async def test_register_entity(self):
        """Test entity registration."""
        trust = TrustCore()
        
        success = await trust.register_entity("test_entity", initial_trust=0.7)
        assert success is True
        
        score = await trust.get_trust_score("test_entity")
        assert score == 0.7
    
    @pytest.mark.asyncio
    async def test_trust_update(self):
        """Test trust score updates."""
        trust = TrustCore()
        
        await trust.register_entity("component_1", initial_trust=0.5)
        
        # Successful performance
        new_score = await trust.update_trust(
            "component_1",
            {
                "success": True,
                "error_count": 0,
                "response_time_ms": 50,
                "constitutional_compliant": True
            }
        )
        
        assert new_score > 0.5  # Trust should increase
    
    @pytest.mark.asyncio
    async def test_trust_decay(self):
        """Test trust decay over time."""
        trust = TrustCore()
        
        await trust.register_entity("decaying_entity", initial_trust=0.9)
        
        # Manually adjust last_updated to simulate time passage
        entity = trust._entities["decaying_entity"]
        entity.last_updated = datetime.utcnow() - timedelta(days=10)
        
        decayed_score = await trust.apply_decay("decaying_entity")
        assert decayed_score < 0.9  # Trust should decay


class TestImmutableLogger:
    """Test Immutable Logger."""
    
    @pytest.mark.asyncio
    async def test_logging(self):
        """Test basic logging."""
        logger = ImmutableLogger()
        
        audit_id = await logger.log(
            event_type="SYSTEM_EVENT",
            component_id="test_component",
            payload={"action": "test"},
            trust_score=0.8
        )
        
        assert audit_id is not None
        
        log_entry = await logger.get_log(audit_id)
        assert log_entry is not None
        assert log_entry["event_type"] == "SYSTEM_EVENT"
    
    @pytest.mark.asyncio
    async def test_chain_integrity(self):
        """Test blockchain chain integrity."""
        logger = ImmutableLogger()
        
        # Add multiple logs
        for i in range(5):
            await logger.log(
                event_type="SYSTEM_EVENT",
                component_id="test",
                payload={"index": i}
            )
        
        # Verify chain
        is_valid = await logger.verify_chain_integrity()
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_query_logs(self):
        """Test querying logs."""
        logger = ImmutableLogger()
        
        # Add logs
        await logger.log(
            event_type="MEMORY_STORED",
            component_id="mtl_kernel",
            payload={"key": "test1"}
        )
        await logger.log(
            event_type="TRUST_UPDATED",
            component_id="trust_core",
            payload={"entity": "test"}
        )
        
        # Query by event type
        results = await logger.query_logs(
            filters={"event_type": "MEMORY_STORED"}
        )
        
        assert len(results) >= 1
        assert all(r["event_type"] == "MEMORY_STORED" for r in results)


class TestMemoryOrchestrator:
    """Test Memory Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        """Test fallback from Lightning to Fusion."""
        orchestrator = MemoryOrchestrator()
        
        # Store in Fusion only
        await orchestrator.fusion.insert(
            "learned_patterns",
            {
                "pattern_type": "fallback_test",
                "pattern_data": {"data": "test"}
            }
        )
        
        # Get with fallback should find it and promote to Lightning
        value = await orchestrator.get("fallback_test", fallback_chain=True)
        assert value is not None
    
    @pytest.mark.asyncio
    async def test_tier_selection(self):
        """Test automatic tier selection."""
        orchestrator = MemoryOrchestrator()
        
        # Small data -> Lightning
        small_data = {"key": "value"}
        await orchestrator.set("small_key", small_data, storage_tier="auto")
        
        # Verify in Lightning
        value = await orchestrator.lightning.get("small_key")
        assert value == small_data


class TestMTLService:
    """Test MTL Service."""
    
    @pytest.mark.asyncio
    async def test_store_with_governance(self):
        """Test storing with governance checks."""
        mtl = MTLService()
        
        # Valid data
        entry_id = await mtl.store_with_governance(
            data={"key": "test_data", "value": "safe"},
            trust_score=0.8,
            constitutional_check=True,
            component_id="test"
        )
        
        assert entry_id is not None
    
    @pytest.mark.asyncio
    async def test_retrieve_with_trust(self):
        """Test retrieval with trust validation."""
        mtl = MTLService()
        
        # Register entity with high trust
        await mtl.trust.register_entity("trusted_key", initial_trust=0.9)
        
        # Store data
        await mtl.memory.lightning.set("trusted_key", "secret_data")
        
        # Retrieve with trust check
        value = await mtl.retrieve_with_trust(
            "trusted_key",
            min_trust=0.5,
            component_id="test"
        )
        
        assert value == "secret_data"
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health checks."""
        mtl = MTLService()
        
        health = await mtl.health_check()
        
        assert "lightning" in health
        assert "fusion" in health
        assert "vector" in health
        assert "trust" in health
        assert "logger" in health


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        print("Running MTL Kernel smoke tests...")
        
        # Test Lightning
        lightning = LightningMemory()
        await lightning.set("test", "value")
        assert await lightning.get("test") == "value"
        print("✓ Lightning Memory")
        
        # Test Fusion
        fusion = FusionMemory()
        await fusion.insert("learned_patterns", {"test": "data"})
        print("✓ Fusion Memory")
        
        # Test Vector
        vector = VectorMemory()
        emb = VectorMemory.generate_mock_embedding("test")
        await vector.add("knowledge", emb, {"test": "meta"})
        print("✓ Vector Memory")
        
        # Test Trust
        trust = TrustCore()
        await trust.register_entity("test", initial_trust=0.8)
        print("✓ Trust Core")
        
        # Test Logger
        logger = ImmutableLogger()
        await logger.log("SYSTEM_EVENT", "test", {"msg": "test"})
        assert await logger.verify_chain_integrity()
        print("✓ Immutable Logger")
        
        # Test Orchestrator
        orch = MemoryOrchestrator()
        await orch.set("key", "value")
        assert await orch.get("key") == "value"
        print("✓ Memory Orchestrator")
        
        # Test MTL Service
        mtl = MTLService()
        health = await mtl.health_check()
        assert all(v == "healthy" for v in health.values())
        print("✓ MTL Service")
        
        print("\n✅ All smoke tests passed!")
    
    asyncio.run(smoke_test())
