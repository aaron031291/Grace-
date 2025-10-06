#!/usr/bin/env python3
"""
MTL Kernel Demonstration - Quick Start Guide

This demonstrates the core MTL (Memory, Trust, Learning) Kernel functionality.
"""

import asyncio
from grace.mtl_kernel import (
    LightningMemory,
    FusionMemory,
    VectorMemory,
    TrustCore,
    ImmutableLogger,
    MemoryOrchestrator,
)


async def demo_lightning_memory():
    """Demonstrate Lightning Memory - Fast in-memory cache."""
    print("\n" + "=" * 60)
    print("1. LIGHTNING MEMORY - High-speed cache (<1ms)")
    print("=" * 60)
    
    lightning = LightningMemory(max_size=1000, default_ttl=3600)
    
    # Store data
    await lightning.set('user:123', {'name': 'Alice', 'role': 'admin'})
    await lightning.set('session:abc', {'user_id': 123, 'expires': '2024-12-31'})
    
    # Retrieve data
    user = await lightning.get('user:123')
    print(f"✓ Retrieved user: {user}")
    
    # Batch operations
    items = {
        'config:theme': 'dark',
        'config:lang': 'en',
        'config:tz': 'UTC'
    }
    await lightning.set_many(items)
    configs = await lightning.get_many(['config:theme', 'config:lang'])
    print(f"✓ Batch retrieved: {configs}")
    
    # Stats
    stats = await lightning.get_stats()
    print(f"✓ Stats: {stats['size']} entries, hit rate {stats['hit_rate']}")


async def demo_fusion_memory():
    """Demonstrate Fusion Memory - Long-term structured storage."""
    print("\n" + "=" * 60)
    print("2. FUSION MEMORY - Structured storage (<50ms)")
    print("=" * 60)
    
    fusion = FusionMemory()
    
    # Store learned patterns
    await fusion.insert('learned_patterns', {
        'pattern_type': 'user_behavior',
        'pattern_data': {'action': 'login', 'frequency': 'daily'},
        'confidence': 0.92
    }, trust_score=0.85)
    
    # Store governance decision
    await fusion.insert('governance_decisions', {
        'decision_type': 'policy_approval',
        'context': 'User data retention policy',
        'outcome': 'approved',
        'reasoning': 'Complies with GDPR',
        'approved': True
    }, trust_score=0.95)
    
    # Query patterns
    patterns = await fusion.query('learned_patterns', 
                                   filters={'pattern_type': 'user_behavior'})
    print(f"✓ Found {len(patterns)} patterns")
    
    # Query decisions
    decisions = await fusion.query('governance_decisions',
                                    filters={'decision_type': 'policy_approval'})
    print(f"✓ Found {len(decisions)} decisions")
    
    stats = await fusion.get_stats()
    print(f"✓ Stats: {stats['total_entries']} total entries")


async def demo_vector_memory():
    """Demonstrate Vector Memory - Semantic search."""
    print("\n" + "=" * 60)
    print("3. VECTOR MEMORY - Semantic search")
    print("=" * 60)
    
    vector = VectorMemory()
    
    # Add knowledge with embeddings
    texts = [
        "Machine learning algorithms",
        "Data privacy regulations",
        "Neural network architectures",
        "GDPR compliance requirements"
    ]
    
    for text in texts:
        emb = VectorMemory.generate_mock_embedding(text, dimensions=64)
        await vector.add('knowledge', emb, {'text': text, 'category': 'docs'})
    
    print(f"✓ Added {len(texts)} knowledge entries")
    
    # Semantic search
    query = "AI and privacy"
    query_emb = VectorMemory.generate_mock_embedding(query, dimensions=64)
    results = await vector.search('knowledge', query_emb, top_k=2)
    
    print(f"✓ Search '{query}' found {len(results)} results:")
    for r in results:
        print(f"   - {r['metadata']['text']} (similarity: {r['similarity']:.3f})")
    
    stats = await vector.get_stats()
    print(f"✓ Stats: {stats['total_vectors']} vectors")


async def demo_trust_core():
    """Demonstrate Trust Core - Trust scoring system."""
    print("\n" + "=" * 60)
    print("4. TRUST CORE - Real-time trust management")
    print("=" * 60)
    
    trust = TrustCore()
    
    # Register entities
    await trust.register_entity('ml_model_v1', entity_type='component', initial_trust=0.7)
    await trust.register_entity('api_service', entity_type='component', initial_trust=0.8)
    
    print("✓ Registered 2 entities")
    
    # Update trust based on performance
    new_trust = await trust.update_trust('ml_model_v1', {
        'success': True,
        'error_count': 0,
        'response_time_ms': 45,
        'constitutional_compliant': True,
        'governance_approved': True
    })
    print(f"✓ ML model trust updated: 0.7 → {new_trust:.3f}")
    
    # Calculate context-aware trust
    context_trust = await trust.calculate_trust('api_service', {
        'sensitivity': 0.8,  # High sensitivity operation
        'risk_level': 0.3
    })
    print(f"✓ Context-aware trust for sensitive op: {context_trust:.3f}")
    
    # Validate threshold
    meets_threshold = await trust.validate_trust_threshold('api_service', 0.6)
    print(f"✓ Meets 0.6 threshold: {meets_threshold}")
    
    stats = await trust.get_stats()
    print(f"✓ Stats: {stats['total_entities']} entities, avg {stats['avg_trust_score']:.3f}")


async def demo_immutable_logger():
    """Demonstrate Immutable Logger - Audit trail."""
    print("\n" + "=" * 60)
    print("5. IMMUTABLE LOGGER - Blockchain-style audit")
    print("=" * 60)
    
    logger = ImmutableLogger()
    
    # Log various events
    events = [
        ('MEMORY_STORED', 'mtl_kernel', {'key': 'user:123', 'size_bytes': 256}),
        ('TRUST_UPDATED', 'trust_core', {'entity': 'ml_model', 'new_score': 0.85}),
        ('DECISION_MADE', 'governance', {'decision': 'approve', 'policy': 'data_retention'}),
        ('CONSTITUTIONAL_CHECK', 'governance', {'compliance': 1.0, 'article': 'privacy'}),
    ]
    
    for event_type, component, payload in events:
        await logger.log(event_type, component, payload, trust_score=0.9)
    
    print(f"✓ Logged {len(events)} events")
    
    # Verify chain integrity
    is_valid = await logger.verify_chain_integrity()
    print(f"✓ Chain integrity verified: {is_valid}")
    
    # Query logs
    decisions = await logger.query_logs(filters={'event_type': 'DECISION_MADE'})
    print(f"✓ Found {len(decisions)} decision logs")
    
    stats = await logger.get_stats()
    print(f"✓ Stats: {stats['chain_length']} logs, compliance {stats['avg_constitutional_compliance']:.3f}")


async def demo_memory_orchestrator():
    """Demonstrate Memory Orchestrator - Unified memory API."""
    print("\n" + "=" * 60)
    print("6. MEMORY ORCHESTRATOR - Unified memory API")
    print("=" * 60)
    
    orch = MemoryOrchestrator()
    
    # Auto-tier selection
    await orch.set('small_data', {'id': 1, 'name': 'test'}, storage_tier='auto')
    await orch.set('large_data', {'id': 2, 'content': 'x' * 20000}, storage_tier='auto')
    
    print("✓ Stored data with auto-tier selection")
    
    # Retrieve with fallback
    data = await orch.get('small_data', fallback_chain=True)
    print(f"✓ Retrieved: {data}")
    
    # Health check
    health = await orch.health_check()
    print(f"✓ Health: {', '.join(f'{k}={v}' for k, v in health.items())}")
    
    stats = await orch.get_stats()
    print(f"✓ Stats: {stats['orchestrator']['total_sets']} sets, " +
          f"{stats['orchestrator']['total_gets']} gets, " +
          f"hit rate {stats['orchestrator']['hit_rate']:.2f}")


async def main():
    """Run all demos."""
    print("\n" + "🚀" * 30)
    print("MTL KERNEL DEMONSTRATION")
    print("🚀" * 30)
    
    await demo_lightning_memory()
    await demo_fusion_memory()
    await demo_vector_memory()
    await demo_trust_core()
    await demo_immutable_logger()
    await demo_memory_orchestrator()
    
    print("\n" + "=" * 60)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Lightning Memory: <1ms hot cache for frequently accessed data")
    print("2. Fusion Memory: <50ms structured storage for patterns & decisions")
    print("3. Vector Memory: Semantic search for knowledge retrieval")
    print("4. Trust Core: Real-time trust scoring for all components")
    print("5. Immutable Logger: Tamper-proof audit trail for compliance")
    print("6. Memory Orchestrator: Unified API with smart routing\n")


if __name__ == "__main__":
    asyncio.run(main())
