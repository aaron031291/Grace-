"""
Production Memory System Demo - Shows DB transactions, health monitoring, and integrations
"""

import logging
from datetime import datetime
from grace.memory.enhanced_memory_core import EnhancedMemoryCore
from grace.clarity.memory_scoring import LoopMemoryBank
from grace.integration.avn_reporter import AVNReporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock database connection for demo
class MockDBConnection:
    """Mock PostgreSQL connection"""
    def __init__(self):
        self.data = {}
        self.in_transaction = False
    
    def begin(self):
        self.in_transaction = True
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.in_transaction = False
    
    def commit(self):
        pass
    
    def execute(self, query, params=None):
        if "INSERT" in query or "UPDATE" in query:
            if params:
                self.data[params.get('memory_id', 'unknown')] = params
        elif "SELECT" in query:
            if params and 'memory_id' in params:
                result = self.data.get(params['memory_id'])
                return MockResult(result)
        return MockResult(None)


class MockResult:
    """Mock query result"""
    def __init__(self, data):
        self.data = data
    
    def fetchone(self):
        return self.data


# Mock Redis client for demo
class MockRedisClient:
    """Mock Redis connection"""
    def __init__(self):
        self.cache = {}
        self.sets = {}
    
    def ping(self):
        return True
    
    def pipeline(self):
        return MockPipeline(self)
    
    def get(self, key):
        return self.cache.get(key)
    
    def setex(self, key, ttl, value):
        self.cache[key] = value
    
    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
    
    def sadd(self, key, value):
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].add(value)
    
    def srem(self, key, value):
        if key in self.sets:
            self.sets[key].discard(value)


class MockPipeline:
    """Mock Redis pipeline"""
    def __init__(self, client):
        self.client = client
        self.commands = []
    
    def setex(self, key, ttl, value):
        self.commands.append(('setex', key, ttl, value))
        return self
    
    def sadd(self, key, value):
        self.commands.append(('sadd', key, value))
        return self
    
    def delete(self, key):
        self.commands.append(('delete', key))
        return self
    
    def srem(self, key, value):
        self.commands.append(('srem', key, value))
        return self
    
    def execute(self):
        for cmd in self.commands:
            if cmd[0] == 'setex':
                self.client.setex(cmd[1], cmd[2], cmd[3])
            elif cmd[0] == 'sadd':
                self.client.sadd(cmd[1], cmd[2])
            elif cmd[0] == 'delete':
                self.client.delete(cmd[1])
            elif cmd[0] == 'srem':
                self.client.srem(cmd[1], cmd[2])
        return [True] * len(self.commands)


def main():
    """Demonstrate production memory system"""
    
    logger.info("=== Production Memory System Demo ===\n")
    
    # Initialize components
    db = MockDBConnection()
    redis = MockRedisClient()
    clarity_memory = LoopMemoryBank()
    avn = AVNReporter()
    
    # Initialize Enhanced Memory Core
    memory_core = EnhancedMemoryCore(
        db_connection=db,
        redis_client=redis,
        clarity_memory_bank=clarity_memory,
        avn_reporter=avn
    )
    
    # Initialize with health monitoring
    logger.info("Step 1: Initialize with Health Monitoring")
    initialized = memory_core.initialize()
    print(f"Initialization: {'Success' if initialized else 'Failed'}")
    print(f"Health Status: {memory_core.health_status.value}\n")
    
    # Store structured memories with DB transactions
    logger.info("Step 2: Store Structured Memories (PostgreSQL Transactions)")
    
    memories = [
        {
            'memory_id': 'mem_001',
            'content': {
                'type': 'decision',
                'action': 'approved_deployment',
                'confidence': 0.95,
                'reasoning': 'All safety checks passed'
            },
            'memory_type': 'episodic',
            'metadata': {'domain': 'deployment', 'criticality': 'high'}
        },
        {
            'memory_id': 'mem_002',
            'content': {
                'type': 'knowledge',
                'fact': 'Ethical AI principles',
                'description': 'Always prioritize human wellbeing'
            },
            'memory_type': 'semantic',
            'metadata': {'domain': 'ethics', 'immutable': True}
        },
        {
            'memory_id': 'mem_003',
            'content': {
                'type': 'procedure',
                'steps': ['validate', 'execute', 'verify'],
                'context': 'Standard operating procedure'
            },
            'memory_type': 'procedural',
            'metadata': {'domain': 'operations'}
        }
    ]
    
    for mem in memories:
        success = memory_core.store_structured_memory(**mem)
        print(f"Stored {mem['memory_id']}: {'✓' if success else '✗'}")
    
    print()
    
    # Update memory with transaction
    logger.info("Step 3: Update Memory (PostgreSQL Transaction)")
    update_success = memory_core.update_structured_memory(
        'mem_001',
        {'confidence': 0.98, 'verified': True}
    )
    print(f"Updated mem_001: {'✓' if update_success else '✗'}\n")
    
    # Retrieve with caching
    logger.info("Step 4: Retrieve Memories (Redis Cache)")
    
    # First retrieval (cache miss)
    mem_1 = memory_core.retrieve_memory('mem_001', use_cache=True)
    print(f"Retrieved mem_001 (cache miss): {mem_1 is not None}")
    
    # Second retrieval (cache hit)
    mem_1_cached = memory_core.retrieve_memory('mem_001', use_cache=True)
    print(f"Retrieved mem_001 (cache hit): {mem_1_cached is not None}\n")
    
    # Test embedding fallback
    logger.info("Step 5: Test Embedding API Fallback")
    memory_core.embedding_fallback_enabled = True
    
    fallback_mem = memory_core.store_structured_memory(
        memory_id='mem_004',
        content={'text': 'Test fallback embedding'},
        memory_type='test',
        metadata={'fallback_test': True}
    )
    print(f"Stored with fallback embedding: {'✓' if fallback_mem else '✗'}\n")
    
    # Get health status
    logger.info("Step 6: Health Status Report")
    health = memory_core.get_health_status()
    
    print(f"Overall Status: {health['status']}")
    print(f"Cache Hit Rate: {health['metrics']['cache_hit_rate']}")
    print(f"Avg Retrieval Time: {health['metrics']['avg_retrieval_time']}")
    print(f"DB Healthy: {health['metrics']['db_healthy']}")
    print(f"Redis Healthy: {health['metrics']['redis_healthy']}")
    print(f"Embedding API Healthy: {health['metrics']['embedding_api_healthy']}")
    print(f"Total Retrievals: {health['total_retrievals']}")
    print(f"Cache Hits: {health['cache_hits']}")
    print(f"Cache Misses: {health['cache_misses']}\n")
    
    # Clarity integration
    logger.info("Step 7: Clarity Framework Integration")
    clarity_stats = clarity_memory.get_memory_statistics()
    print(f"Clarity Memories: {clarity_stats['total_memories']}")
    print(f"Avg Clarity: {clarity_stats.get('avg_clarity', 0):.2%}")
    print(f"Avg Ambiguity: {clarity_stats.get('avg_ambiguity', 0):.2%}\n")
    
    # AVN diagnostics
    logger.info("Step 8: AVN Self-Diagnostic Report")
    system_health = avn.get_system_health()
    print(f"System Status: {system_health['status']}")
    print(f"Monitored Components: {system_health['components']}")
    print(f"Total Diagnostic Reports: {system_health['total_reports']}\n")
    
    logger.info("=== Production Memory System Demo Complete ===")


if __name__ == "__main__":
    main()
