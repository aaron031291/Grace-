#!/usr/bin/env python3
"""Standalone smoke test for MTL Kernel components."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct imports to avoid grace.__init__.py
from grace.mtl_kernel.lightning_memory import LightningMemory
from grace.mtl_kernel.fusion_memory import FusionMemory
from grace.mtl_kernel.vector_memory import VectorMemory
from grace.mtl_kernel.trust_core import TrustCore
from grace.mtl_kernel.immutable_logger import ImmutableLogger
from grace.mtl_kernel.memory_orchestrator import MemoryOrchestrator
from grace.mtl_kernel.mtl_service import MTLService


async def test_lightning_memory():
    """Test Lightning Memory component."""
    print("Testing Lightning Memory...")
    lightning = LightningMemory()
    
    # Basic operations
    await lightning.set('test_key', 'test_value')
    result = await lightning.get('test_key')
    assert result == 'test_value', f"Expected 'test_value', got {result}"
    
    # Batch operations
    items = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
    count = await lightning.set_many(items)
    assert count == 3
    
    results = await lightning.get_many(['key1', 'key2'])
    assert len(results) == 2
    
    stats = await lightning.get_stats()
    print(f"  ✓ Lightning: {stats['size']} entries, {stats['hit_rate']} hit rate")
    return True


async def test_fusion_memory():
    """Test Fusion Memory component."""
    print("Testing Fusion Memory...")
    fusion = FusionMemory()
    
    # Insert data
    entry_id = await fusion.insert(
        'learned_patterns',
        {
            'pattern_type': 'test_pattern',
            'pattern_data': {'key': 'value'},
            'confidence': 0.9
        },
        trust_score=0.8
    )
    assert entry_id is not None
    
    # Query data
    results = await fusion.query(
        'learned_patterns',
        filters={'pattern_type': 'test_pattern'}
    )
    assert len(results) > 0
    assert results[0]['data']['confidence'] == 0.9
    
    stats = await fusion.get_stats()
    print(f"  ✓ Fusion: {stats['total_entries']} entries across {len(stats['table_sizes'])} tables")
    return True


async def test_vector_memory():
    """Test Vector Memory component."""
    print("Testing Vector Memory...")
    vector = VectorMemory()
    
    # Generate embeddings
    emb1 = VectorMemory.generate_mock_embedding('machine learning')
    emb2 = VectorMemory.generate_mock_embedding('deep learning')
    
    # Add vectors
    await vector.add('knowledge', emb1, {'topic': 'ML'})
    await vector.add('knowledge', emb2, {'topic': 'DL'})
    
    # Search
    query_emb = VectorMemory.generate_mock_embedding('learning algorithms')
    results = await vector.search('knowledge', query_emb, top_k=2)
    assert len(results) <= 2
    assert 'similarity' in results[0]
    
    stats = await vector.get_stats()
    print(f"  ✓ Vector: {stats['total_vectors']} vectors across {len(stats['collections'])} collections")
    return True


async def test_trust_core():
    """Test Trust Core component."""
    print("Testing Trust Core...")
    trust = TrustCore()
    
    # Register entity
    await trust.register_entity('test_component', initial_trust=0.7)
    score = await trust.get_trust_score('test_component')
    assert score == 0.7
    
    # Update trust
    new_score = await trust.update_trust(
        'test_component',
        {
            'success': True,
            'error_count': 0,
            'response_time_ms': 50,
            'constitutional_compliant': True
        }
    )
    assert new_score > 0.7  # Should increase
    
    stats = await trust.get_stats()
    print(f"  ✓ Trust: {stats['total_entities']} entities, avg score {stats['avg_trust_score']}")
    return True


async def test_immutable_logger():
    """Test Immutable Logger component."""
    print("Testing Immutable Logger...")
    logger = ImmutableLogger()
    
    # Log events
    audit_id1 = await logger.log(
        'SYSTEM_EVENT',
        'test_component',
        {'action': 'test1'},
        trust_score=0.9
    )
    audit_id2 = await logger.log(
        'MEMORY_STORED',
        'mtl_kernel',
        {'key': 'test_key'},
        trust_score=0.8
    )
    
    # Verify chain integrity
    integrity = await logger.verify_chain_integrity()
    assert integrity is True
    
    # Query logs
    results = await logger.query_logs(
        filters={'event_type': 'SYSTEM_EVENT'}
    )
    assert len(results) >= 1
    
    stats = await logger.get_stats()
    print(f"  ✓ Logger: {stats['chain_length']} logs, integrity verified")
    return True


async def test_memory_orchestrator():
    """Test Memory Orchestrator component."""
    print("Testing Memory Orchestrator...")
    orch = MemoryOrchestrator()
    
    # Set and get
    await orch.set('orch_key', 'orch_value', storage_tier='auto')
    value = await orch.get('orch_key')
    assert value == 'orch_value'
    
    # Query
    results = await orch.query('test', search_type='hybrid')
    # Just verify it doesn't crash
    
    # Health check
    health = await orch.health_check()
    assert all(v == 'healthy' for v in health.values())
    
    stats = await orch.get_stats()
    print(f"  ✓ Orchestrator: {stats['orchestrator']['total_sets']} sets, {stats['orchestrator']['total_gets']} gets")
    return True


async def test_mtl_service():
    """Test MTL Service component."""
    print("Testing MTL Service...")
    mtl = MTLService()
    
    # Store with governance
    entry_id = await mtl.store_with_governance(
        data={'key': 'test_data', 'value': 'safe'},
        trust_score=0.8,
        constitutional_check=True,
        component_id='test'
    )
    assert entry_id is not None
    
    # Health check
    health = await mtl.health_check()
    all_healthy = all(v == 'healthy' for v in health.values())
    assert all_healthy, f"Health check failed: {health}"
    
    # Get stats
    stats = await mtl.get_stats()
    print(f"  ✓ MTL Service: All components healthy")
    print(f"    - Operations: {stats['mtl_service']['total_operations']}")
    print(f"    - Trust entities: {stats['trust']['total_entities']}")
    print(f"    - Audit logs: {stats['logger']['total_logs']}")
    
    return True


async def main():
    """Run all smoke tests."""
    print("\n🔬 MTL Kernel Component Smoke Tests\n")
    print("=" * 50)
    
    tests = [
        ("Lightning Memory", test_lightning_memory),
        ("Fusion Memory", test_fusion_memory),
        ("Vector Memory", test_vector_memory),
        ("Trust Core", test_trust_core),
        ("Immutable Logger", test_immutable_logger),
        ("Memory Orchestrator", test_memory_orchestrator),
        ("MTL Service", test_mtl_service),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed!\n")
        return 0
    else:
        print("❌ Some tests failed!\n")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
