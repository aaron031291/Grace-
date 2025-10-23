"""
Redis Setup for Grace Memory System
Initializes Redis with proper configuration for memory caching
"""

import redis
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def initialize_redis(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    max_connections: int = 50
) -> redis.Redis:
    """
    Initialize Redis connection with production settings
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database number
        password: Redis password (optional)
        max_connections: Maximum connection pool size
        
    Returns:
        Redis client instance
    """
    
    connection_pool = redis.ConnectionPool(
        host=host,
        port=port,
        db=db,
        password=password,
        max_connections=max_connections,
        decode_responses=False  # We'll handle JSON encoding
    )
    
    client = redis.Redis(connection_pool=connection_pool)
    
    # Test connection
    try:
        client.ping()
        logger.info(f"Redis connected successfully: {host}:{port}/{db}")
        
        # Set recommended configs for memory system
        client.config_set('maxmemory-policy', 'allkeys-lru')
        client.config_set('maxmemory', '2gb')
        
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    
    return client


def setup_redis_keys():
    """Define Redis key patterns for Grace memory system"""
    
    KEY_PATTERNS = {
        'memory': 'memory:{memory_id}',
        'memory_index': 'memory:index',
        'memory_type_index': 'memory:type:{memory_type}',
        'memory_embedding': 'embedding:{memory_id}',
        'health_status': 'health:{component}',
        'cache_stats': 'stats:cache',
        'session': 'session:{session_id}'
    }
    
    return KEY_PATTERNS


def clear_memory_cache(client: redis.Redis):
    """Clear all memory-related cache entries"""
    
    patterns = ['memory:*', 'embedding:*']
    
    for pattern in patterns:
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            logger.info(f"Cleared {len(keys)} keys matching {pattern}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        redis_client = initialize_redis()
        patterns = setup_redis_keys()
        
        print("Redis initialized successfully")
        print(f"Key patterns: {patterns}")
        
    except Exception as e:
        print(f"Redis setup failed: {e}")
