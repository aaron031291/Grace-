"""
Event Mesh Configuration for Grace.

Provides configuration management for event transports and routing.
Supports environment-based configuration with sensible defaults.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class EventMeshConfig:
    """Configuration for Grace Event Mesh."""
    
    # Transport configuration
    transport_type: str = "in-memory"  # Options: in-memory, kafka, nats, redis
    transport_config: Dict[str, Any] = field(default_factory=dict)
    
    # Event bus configuration
    enable_deduplication: bool = True
    max_queue_size: int = 10000
    high_water_mark: int = 8000
    low_water_mark: int = 2000
    worker_count: int = 4
    
    # Retry configuration
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    enable_jitter: bool = True
    
    # Schema registry
    enable_schema_validation: bool = False
    schema_registry_url: Optional[str] = None
    
    # Monitoring and observability
    enable_metrics: bool = True
    metrics_port: int = 8080
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'EventMeshConfig':
        """Create configuration from environment variables."""
        
        # Transport configuration
        transport_type = os.getenv('GRACE_EVENT_TRANSPORT', 'in-memory')
        transport_config = cls._get_transport_config_from_env(transport_type)
        
        return cls(
            transport_type=transport_type,
            transport_config=transport_config,
            enable_deduplication=os.getenv('GRACE_EVENT_DEDUP', 'true').lower() == 'true',
            max_queue_size=int(os.getenv('GRACE_EVENT_MAX_QUEUE', '10000')),
            high_water_mark=int(os.getenv('GRACE_EVENT_HIGH_WATER', '8000')),
            low_water_mark=int(os.getenv('GRACE_EVENT_LOW_WATER', '2000')),
            worker_count=int(os.getenv('GRACE_EVENT_WORKERS', '4')),
            max_retries=int(os.getenv('GRACE_EVENT_MAX_RETRIES', '3')),
            base_delay=float(os.getenv('GRACE_EVENT_BASE_DELAY', '1.0')),
            max_delay=float(os.getenv('GRACE_EVENT_MAX_DELAY', '60.0')),
            exponential_base=float(os.getenv('GRACE_EVENT_EXP_BASE', '2.0')),
            enable_jitter=os.getenv('GRACE_EVENT_JITTER', 'true').lower() == 'true',
            enable_schema_validation=os.getenv('GRACE_SCHEMA_VALIDATION', 'false').lower() == 'true',
            schema_registry_url=os.getenv('GRACE_SCHEMA_REGISTRY_URL'),
            enable_metrics=os.getenv('GRACE_METRICS_ENABLED', 'true').lower() == 'true',
            metrics_port=int(os.getenv('GRACE_METRICS_PORT', '8080')),
            enable_tracing=os.getenv('GRACE_TRACING_ENABLED', 'false').lower() == 'true',
            jaeger_endpoint=os.getenv('GRACE_JAEGER_ENDPOINT')
        )
    
    @classmethod
    def _get_transport_config_from_env(cls, transport_type: str) -> Dict[str, Any]:
        """Get transport-specific configuration from environment."""
        
        if transport_type == "kafka":
            return {
                "bootstrap_servers": os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
                "security_protocol": os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT'),
                "sasl_mechanism": os.getenv('KAFKA_SASL_MECHANISM'),
                "sasl_username": os.getenv('KAFKA_SASL_USERNAME'),
                "sasl_password": os.getenv('KAFKA_SASL_PASSWORD'),
                "client_id": os.getenv('KAFKA_CLIENT_ID', 'grace-event-mesh'),
                "max_connections": int(os.getenv('KAFKA_MAX_CONNECTIONS', '10')),
                "connection_timeout": float(os.getenv('KAFKA_CONNECTION_TIMEOUT', '30.0')),
                "retry_attempts": int(os.getenv('KAFKA_RETRY_ATTEMPTS', '3')),
                "heartbeat_interval": float(os.getenv('KAFKA_HEARTBEAT_INTERVAL', '30.0'))
            }
        
        elif transport_type == "nats":
            return {
                "servers": os.getenv('NATS_SERVERS', 'nats://localhost:4222').split(','),
                "stream_name": os.getenv('NATS_STREAM_NAME', 'grace-events'),
                "durable_name": os.getenv('NATS_DURABLE_NAME', 'grace-consumer'),
                "max_deliver": int(os.getenv('NATS_MAX_DELIVER', '3')),
                "max_connections": int(os.getenv('NATS_MAX_CONNECTIONS', '10')),
                "connection_timeout": float(os.getenv('NATS_CONNECTION_TIMEOUT', '30.0')),
                "retry_attempts": int(os.getenv('NATS_RETRY_ATTEMPTS', '3')),
                "heartbeat_interval": float(os.getenv('NATS_HEARTBEAT_INTERVAL', '30.0'))
            }
        
        elif transport_type == "redis":
            return {
                "url": os.getenv('REDIS_URL', 'redis://localhost:6379'),
                "db": int(os.getenv('REDIS_DB', '0')),
                "stream_prefix": os.getenv('REDIS_STREAM_PREFIX', 'grace:events:'),
                "consumer_group": os.getenv('REDIS_CONSUMER_GROUP', 'grace-consumers'),
                "consumer_name": os.getenv('REDIS_CONSUMER_NAME'),
                "max_connections": int(os.getenv('REDIS_MAX_CONNECTIONS', '10')),
                "connection_timeout": float(os.getenv('REDIS_CONNECTION_TIMEOUT', '30.0')),
                "retry_attempts": int(os.getenv('REDIS_RETRY_ATTEMPTS', '3')),
                "heartbeat_interval": float(os.getenv('REDIS_HEARTBEAT_INTERVAL', '30.0'))
            }
        
        else:  # in-memory or unknown
            return {
                "max_connections": int(os.getenv('EVENT_MAX_CONNECTIONS', '10')),
                "connection_timeout": float(os.getenv('EVENT_CONNECTION_TIMEOUT', '30.0')),
                "retry_attempts": int(os.getenv('EVENT_RETRY_ATTEMPTS', '3')),
                "heartbeat_interval": float(os.getenv('EVENT_HEARTBEAT_INTERVAL', '30.0'))
            }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EventMeshConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
            
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'transport_type': self.transport_type,
            'transport_config': self.transport_config,
            'enable_deduplication': self.enable_deduplication,
            'max_queue_size': self.max_queue_size,
            'high_water_mark': self.high_water_mark,
            'low_water_mark': self.low_water_mark,
            'worker_count': self.worker_count,
            'max_retries': self.max_retries,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'exponential_base': self.exponential_base,
            'enable_jitter': self.enable_jitter,
            'enable_schema_validation': self.enable_schema_validation,
            'schema_registry_url': self.schema_registry_url,
            'enable_metrics': self.enable_metrics,
            'metrics_port': self.metrics_port,
            'enable_tracing': self.enable_tracing,
            'jaeger_endpoint': self.jaeger_endpoint
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information about configuration."""
        return {
            'transport_type': self.transport_type,
            'transport_available': self._check_transport_availability(),
            'configuration_source': 'environment' if self._is_from_env() else 'file',
            'schema_validation_enabled': self.enable_schema_validation,
            'metrics_enabled': self.enable_metrics,
            'tracing_enabled': self.enable_tracing
        }
    
    def _check_transport_availability(self) -> bool:
        """Check if the configured transport is available."""
        if self.transport_type == "kafka":
            try:
                import kafka
                return True
            except ImportError:
                return False
        elif self.transport_type == "nats":
            try:
                import nats
                return True
            except ImportError:
                return False
        elif self.transport_type == "redis":
            try:
                import redis
                return True
            except ImportError:
                return False
        else:  # in-memory
            return True
    
    def _is_from_env(self) -> bool:
        """Check if configuration was loaded from environment."""
        return 'GRACE_EVENT_TRANSPORT' in os.environ


# Default configurations for different environments
DEFAULT_CONFIGS = {
    "development": EventMeshConfig(
        transport_type="in-memory",
        enable_deduplication=False,
        max_queue_size=1000,
        worker_count=2,
        enable_metrics=True,
        enable_schema_validation=False
    ),
    
    "testing": EventMeshConfig(
        transport_type="in-memory",
        enable_deduplication=True,
        max_queue_size=500,
        worker_count=1,
        enable_metrics=False,
        enable_schema_validation=True
    ),
    
    "staging": EventMeshConfig(
        transport_type="kafka",
        transport_config={
            "bootstrap_servers": ["localhost:9092"],
            "security_protocol": "PLAINTEXT"
        },
        enable_deduplication=True,
        max_queue_size=5000,
        worker_count=3,
        enable_metrics=True,
        enable_schema_validation=True,
        enable_tracing=True
    ),
    
    "production": EventMeshConfig(
        transport_type="kafka",
        transport_config={
            "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
            "security_protocol": "SSL",
            "max_connections": 20,
            "retry_attempts": 5
        },
        enable_deduplication=True,
        max_queue_size=20000,
        high_water_mark=16000,
        low_water_mark=4000,
        worker_count=8,
        max_retries=5,
        enable_metrics=True,
        enable_schema_validation=True,
        enable_tracing=True
    )
}


def get_config(environment: Optional[str] = None) -> EventMeshConfig:
    """Get configuration for specified environment or from environment variables."""
    
    # If environment is specified, use default config for that environment
    if environment and environment in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[environment]
    
    # Try to load from environment variables
    try:
        return EventMeshConfig.from_env()
    except Exception:
        # Fall back to development defaults
        return DEFAULT_CONFIGS["development"]