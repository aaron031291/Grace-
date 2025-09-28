"""
Transport Abstraction Layer for Grace Event Mesh.

Provides pluggable transport implementations for production event routing:
- InMemoryTransport (default for dev/testing)
- KafkaTransport (production)  
- NATSTransport (production)
- RedisTransport (lightweight production)

This abstraction allows the same event bus logic to work with different
underlying transport mechanisms for different deployment scenarios.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
import uuid

from ..contracts.message_envelope import GraceMessageEnvelope


logger = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Base configuration for transports."""
    max_connections: int = 10
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    heartbeat_interval: float = 30.0


@dataclass
class KafkaConfig(TransportConfig):
    """Kafka transport configuration."""
    bootstrap_servers: List[str] = None
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    client_id: str = "grace-event-mesh"
    
    def __post_init__(self):
        if self.bootstrap_servers is None:
            self.bootstrap_servers = ["localhost:9092"]


@dataclass
class NATSConfig(TransportConfig):
    """NATS transport configuration."""
    servers: List[str] = None
    stream_name: str = "grace-events"
    durable_name: str = "grace-consumer"
    max_deliver: int = 3
    
    def __post_init__(self):
        if self.servers is None:
            self.servers = ["nats://localhost:4222"]


@dataclass
class RedisConfig(TransportConfig):
    """Redis transport configuration."""
    url: str = "redis://localhost:6379"
    db: int = 0
    stream_prefix: str = "grace:events:"
    consumer_group: str = "grace-consumers"
    consumer_name: Optional[str] = None
    
    def __post_init__(self):
        if self.consumer_name is None:
            self.consumer_name = f"grace-{uuid.uuid4().hex[:8]}"


class EventTransport(ABC):
    """Abstract base class for event transports."""
    
    def __init__(self, config: TransportConfig):
        self.config = config
        self.connected = False
        self.subscribers: Dict[str, Set[Callable]] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the transport."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the transport."""
        pass
    
    @abstractmethod
    async def publish(self, topic: str, message: GraceMessageEnvelope) -> bool:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic_pattern: str, handler: Callable) -> str:
        """Subscribe to messages matching topic pattern."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic_pattern: str, subscription_id: str):
        """Unsubscribe from topic."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Get transport health status."""
        pass


class InMemoryTransport(EventTransport):
    """In-memory transport for testing and development."""
    
    def __init__(self, config: Optional[TransportConfig] = None):
        super().__init__(config or TransportConfig())
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.subscription_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self) -> bool:
        """Connect (no-op for in-memory)."""
        self.connected = True
        logger.info("In-memory transport connected")
        return True
    
    async def disconnect(self):
        """Disconnect and cleanup."""
        # Cancel all subscription tasks
        for task in self.subscription_tasks.values():
            task.cancel()
        
        # Wait for tasks to finish
        if self.subscription_tasks:
            await asyncio.gather(*self.subscription_tasks.values(), return_exceptions=True)
        
        self.subscription_tasks.clear()
        self.message_queues.clear()
        self.connected = False
        logger.info("In-memory transport disconnected")
    
    async def publish(self, topic: str, message: GraceMessageEnvelope) -> bool:
        """Publish message to all matching topic queues."""
        if not self.connected:
            return False
        
        published = False
        for queue_topic, queue in self.message_queues.items():
            if self._matches_topic(topic, queue_topic):
                try:
                    await queue.put(message)
                    published = True
                except Exception as e:
                    logger.error(f"Failed to publish to queue {queue_topic}: {e}")
        
        return published
    
    async def subscribe(self, topic_pattern: str, handler: Callable) -> str:
        """Subscribe to topic pattern."""
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
        
        # Create queue for this subscription
        queue = asyncio.Queue()
        self.message_queues[topic_pattern] = queue
        
        # Start consumer task
        task = asyncio.create_task(
            self._consume_messages(queue, handler, subscription_id)
        )
        self.subscription_tasks[subscription_id] = task
        
        logger.info(f"Subscribed to {topic_pattern} (ID: {subscription_id})")
        return subscription_id
    
    async def unsubscribe(self, topic_pattern: str, subscription_id: str):
        """Unsubscribe from topic."""
        if subscription_id in self.subscription_tasks:
            task = self.subscription_tasks.pop(subscription_id)
            task.cancel()
        
        if topic_pattern in self.message_queues:
            del self.message_queues[topic_pattern]
        
        logger.info(f"Unsubscribed from {topic_pattern} (ID: {subscription_id})")
    
    async def _consume_messages(self, queue: asyncio.Queue, handler: Callable, subscription_id: str):
        """Consume messages from queue and execute handler."""
        logger.info(f"Message consumer started for subscription {subscription_id}")
        
        try:
            while True:
                message = await queue.get()
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Handler error in subscription {subscription_id}: {e}")
        except asyncio.CancelledError:
            logger.info(f"Message consumer stopped for subscription {subscription_id}")
    
    def _matches_topic(self, message_topic: str, subscription_pattern: str) -> bool:
        """Check if message topic matches subscription pattern."""
        if subscription_pattern == "*":
            return True
        if subscription_pattern.endswith("*"):
            return message_topic.startswith(subscription_pattern[:-1])
        return message_topic == subscription_pattern
    
    async def health_check(self) -> Dict[str, Any]:
        """Get transport health."""
        return {
            "transport": "in-memory",
            "connected": self.connected,
            "active_subscriptions": len(self.subscription_tasks),
            "message_queues": len(self.message_queues),
            "status": "healthy" if self.connected else "disconnected"
        }


class KafkaTransport(EventTransport):
    """Kafka transport for production event routing."""
    
    def __init__(self, config: KafkaConfig):
        super().__init__(config)
        self.config: KafkaConfig = config
        self.producer = None
        self.consumer = None
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> bool:
        """Connect to Kafka."""
        try:
            # Import kafka-python (optional dependency)
            from kafka import KafkaProducer, KafkaConsumer
            from kafka.errors import KafkaError
            
            # Create producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                security_protocol=self.config.security_protocol,
                sasl_mechanism=self.config.sasl_mechanism,
                sasl_plain_username=self.config.sasl_username,
                sasl_plain_password=self.config.sasl_password,
                client_id=self.config.client_id,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=self.config.retry_attempts,
                max_block_ms=int(self.config.connection_timeout * 1000)
            )
            
            self.connected = True
            logger.info(f"Kafka transport connected to {self.config.bootstrap_servers}")
            return True
            
        except ImportError:
            logger.error("kafka-python not installed. Install with: pip install kafka-python")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Kafka."""
        # Cancel all consumer tasks
        for task in self.consumer_tasks.values():
            task.cancel()
        
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks.values(), return_exceptions=True)
        
        # Close producer and consumer
        if self.producer:
            self.producer.close()
        
        self.connected = False
        logger.info("Kafka transport disconnected")
    
    async def publish(self, topic: str, message: GraceMessageEnvelope) -> bool:
        """Publish message to Kafka topic."""
        if not self.connected or not self.producer:
            return False
        
        try:
            # Convert GME to dict for JSON serialization
            message_dict = message.to_dict()
            
            # Send message
            future = self.producer.send(
                topic=topic,
                key=message.headers.source,
                value=message_dict,
                headers=[
                    ("event_type", message.headers.event_type.encode()),
                    ("priority", message.headers.priority.encode()),
                    ("msg_id", message.msg_id.encode())
                ]
            )
            
            # Wait for send to complete (with timeout)
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to Kafka topic {topic}, partition {record_metadata.partition}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish to Kafka topic {topic}: {e}")
            return False
    
    async def subscribe(self, topic_pattern: str, handler: Callable) -> str:
        """Subscribe to Kafka topic pattern."""
        subscription_id = f"kafka_sub_{uuid.uuid4().hex[:8]}"
        
        try:
            from kafka import KafkaConsumer
            
            # Create consumer for this subscription
            consumer = KafkaConsumer(
                bootstrap_servers=self.config.bootstrap_servers,
                security_protocol=self.config.security_protocol,
                sasl_mechanism=self.config.sasl_mechanism,
                sasl_plain_username=self.config.sasl_username,
                sasl_plain_password=self.config.sasl_password,
                group_id=f"{self.config.client_id}-{subscription_id}",
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Subscribe to topic pattern
            consumer.subscribe(pattern=topic_pattern)
            
            # Start consumer task
            task = asyncio.create_task(
                self._consume_kafka_messages(consumer, handler, subscription_id)
            )
            self.consumer_tasks[subscription_id] = task
            
            logger.info(f"Subscribed to Kafka pattern {topic_pattern} (ID: {subscription_id})")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to Kafka pattern {topic_pattern}: {e}")
            return ""
    
    async def unsubscribe(self, topic_pattern: str, subscription_id: str):
        """Unsubscribe from Kafka topic."""
        if subscription_id in self.consumer_tasks:
            task = self.consumer_tasks.pop(subscription_id)
            task.cancel()
        
        logger.info(f"Unsubscribed from Kafka pattern {topic_pattern} (ID: {subscription_id})")
    
    async def _consume_kafka_messages(self, consumer, handler: Callable, subscription_id: str):
        """Consume messages from Kafka and execute handler."""
        logger.info(f"Kafka consumer started for subscription {subscription_id}")
        
        try:
            loop = asyncio.get_event_loop()
            
            while True:
                # Poll for messages in thread pool to avoid blocking
                message_batch = await loop.run_in_executor(
                    None, lambda: consumer.poll(timeout_ms=1000, max_records=10)
                )
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Reconstruct GME from Kafka message
                            gme = GraceMessageEnvelope.from_dict(message.value)
                            
                            # Execute handler
                            if asyncio.iscoroutinefunction(handler):
                                await handler(gme)
                            else:
                                handler(gme)
                                
                        except Exception as e:
                            logger.error(f"Handler error in Kafka subscription {subscription_id}: {e}")
                
        except asyncio.CancelledError:
            consumer.close()
            logger.info(f"Kafka consumer stopped for subscription {subscription_id}")
        except Exception as e:
            logger.error(f"Kafka consumer error for subscription {subscription_id}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get Kafka transport health."""
        try:
            # Try to get cluster metadata to check connectivity
            if self.producer:
                cluster_metadata = self.producer.bootstrap.check_version()
                return {
                    "transport": "kafka",
                    "connected": self.connected,
                    "bootstrap_servers": self.config.bootstrap_servers,
                    "active_subscriptions": len(self.consumer_tasks),
                    "status": "healthy"
                }
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
        
        return {
            "transport": "kafka",
            "connected": False,
            "status": "unhealthy",
            "active_subscriptions": len(self.consumer_tasks)
        }


# Transport factory function
def create_transport(transport_type: str, config: Dict[str, Any]) -> EventTransport:
    """Create transport instance based on type and configuration."""
    
    if transport_type == "in-memory":
        return InMemoryTransport(TransportConfig(**config))
    
    elif transport_type == "kafka":
        return KafkaTransport(KafkaConfig(**config))
    
    elif transport_type == "nats":
        # Placeholder for NATS implementation
        logger.warning("NATS transport not yet implemented, using in-memory")
        return InMemoryTransport(TransportConfig(**config))
    
    elif transport_type == "redis":
        # Placeholder for Redis implementation
        logger.warning("Redis transport not yet implemented, using in-memory")
        return InMemoryTransport(TransportConfig(**config))
    
    else:
        logger.warning(f"Unknown transport type {transport_type}, using in-memory")
        return InMemoryTransport(TransportConfig(**config))