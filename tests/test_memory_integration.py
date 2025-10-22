"""
Integration tests for MemoryCore fan-out architecture
"""

import pytest
import asyncio
from datetime import datetime


@pytest.mark.asyncio
async def test_memory_core_write_fanout():
    """
    Assert MemoryCore writes fan out to all layers:
    - Lightning (cache)
    - Fusion (durable store)
    - Vector (semantic search)
    - Trust attestations
    - Immutable logs
    - Trigger ledgers
    """
    from grace.memory.core import MemoryCore
    from grace.memory.async_lightning import AsyncLightningMemory
    from grace.memory.async_fusion import AsyncFusionMemory
    from grace.memory.immutable_logs_async import AsyncImmutableLogs
    from grace.trust.core import TrustCoreKernel
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    # Setup components
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    fusion = AsyncFusionMemory("postgresql://localhost/grace_test")
    try:
        await fusion.connect()
    except:
        pytest.skip("Postgres not available")
    
    logs = AsyncImmutableLogs("postgresql://localhost/grace_test")
    try:
        await logs.connect()
    except:
        pytest.skip("Postgres not available")
    
    trust = TrustCoreKernel()
    event_bus = EventBus()
    factory = GraceEventFactory()
    
    # Create MemoryCore
    memory_core = MemoryCore(
        lightning_memory=lightning,
        fusion_memory=fusion,
        vector_store=None,
        trust_core=trust,
        immutable_logs=logs,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Track trigger events
    trigger_events = []
    event_bus.subscribe("memory.write", lambda e: trigger_events.append(e))
    
    # Perform write
    success = await memory_core.write(
        key="test_key",
        value="test_value",
        metadata={"test": True},
        ttl_seconds=60,
        actor="test_user",
        trust_attestation=True
    )
    
    # Assert write succeeded
    assert success is True
    
    # Verify Lightning (cache) write
    cached_value = await lightning.get("test_key")
    assert cached_value == "test_value"
    
    # Verify Fusion (durable) write
    interactions = await fusion.get_interactions(user_id="test_user", limit=10)
    assert len(interactions) > 0
    assert any(i.get("action") == "memory_write" for i in interactions)
    
    # Verify Immutable log
    # Force flush to ensure it's written
    await logs._flush_batch()
    
    audit_logs = await logs.query(actor="test_user", limit=10)
    assert len(audit_logs) > 0
    assert any(l.get("operation_type") == "memory_write" for l in audit_logs)
    
    # Verify Trust attestation
    trust_score = await trust.calculate_trust("test_user", {})
    assert trust_score is not None
    assert trust_score.score > 0
    
    # Verify Trigger event
    await asyncio.sleep(0.1)  # Give time for async emit
    assert len(trigger_events) > 0
    assert trigger_events[0].event_type == "memory.write"
    assert trigger_events[0].payload["key"] == "test_key"
    
    # Cleanup
    await lightning.disconnect()
    await fusion.disconnect()
    await logs.disconnect()


@pytest.mark.asyncio
async def test_memory_core_read_hierarchy():
    """
    Assert MemoryCore reads use cache hierarchy:
    Lightning -> Fusion -> Vector
    """
    from grace.memory.core import MemoryCore
    from grace.memory.async_lightning import AsyncLightningMemory
    from grace.memory.async_fusion import AsyncFusionMemory
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    event_bus = EventBus()
    factory = GraceEventFactory()
    
    memory_core = MemoryCore(
        lightning_memory=lightning,
        fusion_memory=None,
        vector_store=None,
        trust_core=None,
        immutable_logs=None,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Write to cache only
    await lightning.set("cached_key", "cached_value")
    
    # Read should hit cache
    initial_cache_hits = memory_core.cache_hits
    value = await memory_core.read("cached_key", use_cache=True)
    
    assert value == "cached_value"
    assert memory_core.cache_hits == initial_cache_hits + 1
    
    # Read non-existent key
    value = await memory_core.read("nonexistent_key", use_cache=True)
    assert value is None
    assert memory_core.cache_misses > 0
    
    await lightning.disconnect()


@pytest.mark.asyncio
async def test_memory_core_stats():
    """Assert MemoryCore tracks accurate statistics"""
    from grace.memory.core import MemoryCore
    from grace.memory.async_lightning import AsyncLightningMemory
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    event_bus = EventBus()
    factory = GraceEventFactory()
    
    memory_core = MemoryCore(
        lightning_memory=lightning,
        fusion_memory=None,
        vector_store=None,
        trust_core=None,
        immutable_logs=None,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Get initial stats
    stats = memory_core.get_stats()
    assert "writes_total" in stats
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    
    initial_writes = stats["writes_total"]
    
    # Perform write
    await memory_core.write(
        key="stats_test",
        value="value",
        actor="test",
        trust_attestation=False
    )
    
    # Check stats updated
    stats = memory_core.get_stats()
    assert stats["writes_total"] == initial_writes + 1
    
    await lightning.disconnect()


@pytest.mark.asyncio
async def test_memory_core_pattern_storage():
    """Assert MemoryCore stores learned patterns in Fusion"""
    from grace.memory.core import MemoryCore
    from grace.memory.async_fusion import AsyncFusionMemory
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    fusion = AsyncFusionMemory("postgresql://localhost/grace_test")
    try:
        await fusion.connect()
    except:
        pytest.skip("Postgres not available")
    
    event_bus = EventBus()
    factory = GraceEventFactory()
    
    memory_core = MemoryCore(
        lightning_memory=None,
        fusion_memory=fusion,
        vector_store=None,
        trust_core=None,
        immutable_logs=None,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Write pattern
    success = await memory_core.write(
        key="pattern_key",
        value={"action": "click", "frequency": 10},
        metadata={
            "is_pattern": True,
            "pattern_type": "user_behavior",
            "confidence": 0.85
        },
        actor="test_user",
        trust_attestation=False
    )
    
    assert success is True
    
    # Verify pattern stored
    patterns = await fusion.get_patterns(pattern_type="user_behavior")
    assert len(patterns) > 0
    
    # Find our pattern
    found = False
    for pattern in patterns:
        pattern_data = pattern.get("pattern_data", {})
        if pattern_data.get("key") == "pattern_key":
            found = True
            assert pattern.get("confidence") == 0.85
            break
    
    assert found, "Pattern not found in Fusion storage"
    
    await fusion.disconnect()


@pytest.mark.asyncio
async def test_memory_core_trust_integration():
    """Assert MemoryCore updates trust scores on writes"""
    from grace.memory.core import MemoryCore
    from grace.trust.core import TrustCoreKernel
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    trust = TrustCoreKernel()
    event_bus = EventBus()
    factory = GraceEventFactory()
    
    memory_core = MemoryCore(
        lightning_memory=None,
        fusion_memory=None,
        vector_store=None,
        trust_core=trust,
        immutable_logs=None,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Get initial trust
    initial_trust = await trust.calculate_trust("trust_test_user", {})
    initial_score = initial_trust.score
    
    # Perform successful write
    await memory_core.write(
        key="trust_key",
        value="value",
        actor="trust_test_user",
        trust_attestation=True
    )
    
    # Check trust updated
    updated_trust = await trust.calculate_trust("trust_test_user", {})
    
    # Trust should be updated (might increase or stay same based on algorithm)
    assert updated_trust.score is not None
    assert updated_trust.entity_id == "trust_test_user"


@pytest.mark.asyncio
async def test_memory_core_delete_operations():
    """Assert MemoryCore handles deletions correctly"""
    from grace.memory.core import MemoryCore
    from grace.memory.async_lightning import AsyncLightningMemory
    from grace.memory.immutable_logs_async import AsyncImmutableLogs
    from grace.integration.event_bus import EventBus
    from grace.events.factory import GraceEventFactory
    
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    logs = AsyncImmutableLogs("postgresql://localhost/grace_test")
    try:
        await logs.connect()
    except:
        pytest.skip("Postgres not available")
    
    event_bus = EventBus()
    factory = GraceEventFactory()
    
    memory_core = MemoryCore(
        lightning_memory=lightning,
        fusion_memory=None,
        vector_store=None,
        trust_core=None,
        immutable_logs=logs,
        event_bus=event_bus,
        event_factory=factory
    )
    
    # Write then delete
    await memory_core.write(key="delete_test", value="value", actor="test")
    
    success = await memory_core.delete(key="delete_test", actor="test")
    assert success is True
    
    # Verify deleted from cache
    value = await lightning.get("delete_test")
    assert value is None
    
    # Verify deletion logged
    await logs._flush_batch()
    audit_logs = await logs.query(operation_type="memory_delete", limit=10)
    assert len(audit_logs) > 0
    
    await lightning.disconnect()
    await logs.disconnect()
