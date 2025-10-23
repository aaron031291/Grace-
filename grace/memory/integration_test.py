"""
Integration Test for Production Memory System
Tests DB transactions, caching, health monitoring, and all integrations
"""

import pytest
import logging
from datetime import datetime
from grace.memory.enhanced_memory_core import EnhancedMemoryCore, MemoryHealth
from grace.clarity.memory_scoring import LoopMemoryBank
from grace.integration.avn_reporter import AVNReporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestProductionMemorySystem:
    """Integration tests for production memory system"""
    
    @pytest.fixture
    def setup_system(self):
        """Setup test system with all components"""
        # Note: In real tests, use test DB and Redis instances
        from grace.memory.production_demo import MockDBConnection, MockRedisClient
        
        db = MockDBConnection()
        redis = MockRedisClient()
        clarity = LoopMemoryBank()
        avn = AVNReporter()
        
        memory_core = EnhancedMemoryCore(
            db_connection=db,
            redis_client=redis,
            clarity_memory_bank=clarity,
            avn_reporter=avn
        )
        
        return {
            'memory_core': memory_core,
            'db': db,
            'redis': redis,
            'clarity': clarity,
            'avn': avn
        }
    
    def test_initialization_with_health_check(self, setup_system):
        """Test system initialization with health monitoring"""
        memory_core = setup_system['memory_core']
        
        # Initialize
        initialized = memory_core.initialize()
        
        assert initialized is True
        assert memory_core.health_status in [MemoryHealth.HEALTHY, MemoryHealth.DEGRADED]
        assert memory_core.metrics.db_connection_healthy is True
        
        logger.info("✓ Initialization with health check passed")
    
    def test_postgresql_transactions(self, setup_system):
        """Test PostgreSQL transaction operations"""
        memory_core = setup_system['memory_core']
        memory_core.initialize()
        
        # Store memory
        success = memory_core.store_structured_memory(
            memory_id='test_mem_001',
            content={'test': 'data', 'value': 42},
            memory_type='test',
            metadata={'source': 'integration_test'}
        )
        
        assert success is True
        
        # Update memory
        update_success = memory_core.update_structured_memory(
            'test_mem_001',
            {'value': 100, 'updated': True}
        )
        
        assert update_success is True
        
        # Verify in DB
        db = setup_system['db']
        assert 'test_mem_001' in db.data
        
        logger.info("✓ PostgreSQL transactions passed")
    
    def test_redis_caching(self, setup_system):
        """Test Redis caching operations"""
        memory_core = setup_system['memory_core']
        memory_core.initialize()
        
        # Store memory (should cache)
        memory_core.store_structured_memory(
            memory_id='cache_test_001',
            content={'cached': True},
            memory_type='test'
        )
        
        # First retrieval (cache miss)
        mem1 = memory_core.retrieve_memory('cache_test_001', use_cache=True)
        assert mem1 is not None
        assert memory_core.cache_misses == 1
        
        # Second retrieval (cache hit)
        mem2 = memory_core.retrieve_memory('cache_test_001', use_cache=True)
        assert mem2 is not None
        assert memory_core.cache_hits == 1
        
        # Invalidate cache
        memory_core._invalidate_memory_cache('cache_test_001')
        
        # Verify cache invalidated
        redis = setup_system['redis']
        assert f"memory:cache_test_001" not in redis.cache
        
        logger.info("✓ Redis caching passed")
    
    def test_embedding_fallback(self, setup_system):
        """Test embedding API fallback logic"""
        memory_core = setup_system['memory_core']
        memory_core.initialize()
        
        # Enable fallback
        memory_core.embedding_fallback_enabled = True
        
        # This should use fallback (mock OpenAI will work, but test fallback path)
        embedding = memory_core._fallback_embedding("test content")
        
        assert embedding is not None
        assert len(embedding) == 1536  # OpenAI embedding size
        assert embedding.max() <= 1.0  # Normalized
        
        logger.info("✓ Embedding fallback passed")
    
    def test_clarity_integration(self, setup_system):
        """Test Clarity Framework integration"""
        memory_core = setup_system['memory_core']
        clarity = setup_system['clarity']
        memory_core.initialize()
        
        # Store memory (should integrate with Clarity)
        memory_core.store_structured_memory(
            memory_id='clarity_test_001',
            content={'clarity_test': True, 'confidence': 0.95},
            memory_type='episodic',
            metadata={'clarity_integration': True}
        )
        
        # Check Clarity memory bank
        clarity_stats = clarity.get_memory_statistics()
        assert clarity_stats['total_memories'] > 0
        
        # Score memory in Clarity
        scores = clarity.score('clarity_test_001', {'test': True})
        assert 'clarity' in scores
        assert 'composite' in scores
        
        logger.info("✓ Clarity integration passed")
    
    def test_avn_reporting(self, setup_system):
        """Test AVN self-diagnostic reporting"""
        memory_core = setup_system['memory_core']
        avn = setup_system['avn']
        memory_core.initialize()
        
        # Check AVN received diagnostic report
        system_health = avn.get_system_health()
        assert 'enhanced_memory_core' in system_health.get('component_statuses', {})
        
        # Verify diagnostic history
        assert len(avn.diagnostic_history) > 0
        
        logger.info("✓ AVN reporting passed")
    
    def test_health_monitoring(self, setup_system):
        """Test comprehensive health monitoring"""
        memory_core = setup_system['memory_core']
        memory_core.initialize()
        
        # Get health status
        health = memory_core.get_health_status()
        
        assert 'status' in health
        assert 'metrics' in health
        assert 'db_healthy' in health['metrics']
        assert 'redis_healthy' in health['metrics']
        assert 'embedding_api_healthy' in health['metrics']
        
        # Run health check
        check_result = memory_core.run_health_check()
        assert check_result is True
        
        logger.info("✓ Health monitoring passed")
    
    def test_end_to_end_workflow(self, setup_system):
        """Test complete end-to-end workflow"""
        memory_core = setup_system['memory_core']
        memory_core.initialize()
        
        # 1. Store multiple memories
        memories = [
            ('e2e_001', {'type': 'decision', 'result': 'approved'}),
            ('e2e_002', {'type': 'knowledge', 'fact': 'AI safety'}),
            ('e2e_003', {'type': 'procedure', 'steps': [1, 2, 3]})
        ]
        
        for mem_id, content in memories:
            success = memory_core.store_structured_memory(
                memory_id=mem_id,
                content=content,
                memory_type='e2e_test'
            )
            assert success is True
        
        # 2. Update one memory
        update_success = memory_core.update_structured_memory(
            'e2e_001',
            {'result': 'approved_verified', 'verified': True}
        )
        assert update_success is True
        
        # 3. Retrieve with caching
        retrieved = memory_core.retrieve_memory('e2e_001', use_cache=True)
        assert retrieved is not None
        
        # 4. Check health
        health = memory_core.get_health_status()
        assert health['status'] in ['healthy', 'degraded']
        
        # 5. Verify all integrations
        clarity_stats = setup_system['clarity'].get_memory_statistics()
        avn_health = setup_system['avn'].get_system_health()
        
        assert clarity_stats['total_memories'] >= 3
        assert avn_health['total_reports'] > 0
        
        logger.info("✓ End-to-end workflow passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
