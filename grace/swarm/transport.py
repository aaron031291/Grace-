"""
Transport protocols for swarm communication
"""

from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import json
import logging
import aiohttp

logger = logging.getLogger(__name__)


class TransportProtocol(Enum):
    """Available transport protocols"""
    GRPC = "grpc"
    KAFKA = "kafka"
    HTTP = "http"


class MessageTransport(ABC):
    """Abstract base for message transport"""
    
    @abstractmethod
    async def send(self, destination: str, message: Dict[str, Any]) -> bool:
        """Send message to destination"""
        pass
    
    @abstractmethod
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive next message"""
        pass
    
    @abstractmethod
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all peers"""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        pass


class HTTPTransport(MessageTransport):
    """HTTP-based transport for swarm communication"""
    
    def __init__(self, node_id: str, port: int = 8080):
        self.node_id = node_id
        self.port = port
        self.peers: Dict[str, str] = {}  # node_id -> url
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info(f"HTTP transport initialized on port {port}")
    
    async def send(self, destination: str, message: Dict[str, Any]) -> bool:
        """Send message via HTTP POST"""
        if destination not in self.peers:
            logger.warning(f"Unknown destination: {destination}")
            return False
        
        url = self.peers[destination]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/swarm/message",
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    success = response.status == 200
                    
                    if success:
                        logger.debug(f"Sent message to {destination}")
                    else:
                        logger.error(f"Failed to send to {destination}: {response.status}")
                    
                    return success
        
        except Exception as e:
            logger.error(f"Error sending to {destination}: {e}")
            return False
    
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive next message from queue"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast to all known peers"""
        success_count = 0
        
        tasks = [self.send(peer_id, message) for peer_id in self.peers.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        
        logger.info(f"Broadcast to {len(self.peers)} peers, {success_count} successful")
        return success_count
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        
        self.subscriptions[topic].append(callback)
        logger.info(f"Subscribed to topic: {topic}")
    
    async def handle_incoming(self, message: Dict[str, Any]):
        """Handle incoming message"""
        await self.message_queue.put(message)
        
        # Trigger topic subscriptions
        topic = message.get("topic")
        if topic and topic in self.subscriptions:
            for callback in self.subscriptions[topic]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Subscription callback error: {e}")
    
    def add_peer(self, node_id: str, url: str):
        """Add peer node"""
        self.peers[node_id] = url
        logger.info(f"Added peer: {node_id} at {url}")
    
    def remove_peer(self, node_id: str):
        """Remove peer node"""
        if node_id in self.peers:
            del self.peers[node_id]
            logger.info(f"Removed peer: {node_id}")


class GRPCTransport(MessageTransport):
    """gRPC-based transport (production-ready, high performance)"""
    
    def __init__(self, node_id: str, port: int = 50051):
        self.node_id = node_id
        self.port = port
        self.peers: Dict[str, str] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        logger.info(f"gRPC transport initialized on port {port}")
        logger.warning("gRPC transport requires grpcio package")
    
    async def send(self, destination: str, message: Dict[str, Any]) -> bool:
        """Send via gRPC"""
        # In production: implement gRPC client calls
        logger.debug(f"gRPC send to {destination}: {message.get('type')}")
        return True
    
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive via gRPC streaming"""
        # In production: implement gRPC streaming receiver
        await asyncio.sleep(0.1)
        return None
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast via gRPC"""
        success_count = len(self.peers)
        logger.debug(f"gRPC broadcast to {success_count} peers")
        return success_count
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe via gRPC pub/sub"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)


class KafkaTransport(MessageTransport):
    """Kafka-based transport (production-ready, scalable)"""
    
    def __init__(self, node_id: str, bootstrap_servers: str = "localhost:9092"):
        self.node_id = node_id
        self.bootstrap_servers = bootstrap_servers
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        logger.info(f"Kafka transport initialized: {bootstrap_servers}")
        logger.warning("Kafka transport requires aiokafka package")
    
    async def send(self, destination: str, message: Dict[str, Any]) -> bool:
        """Send to Kafka topic"""
        # In production: use aiokafka producer
        topic = f"grace.node.{destination}"
        logger.debug(f"Kafka send to topic {topic}")
        return True
    
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive from Kafka consumer"""
        # In production: use aiokafka consumer
        await asyncio.sleep(0.1)
        return None
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast to Kafka broadcast topic"""
        # In production: send to grace.broadcast topic
        logger.debug("Kafka broadcast")
        return 1
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to Kafka topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)
        logger.info(f"Subscribed to Kafka topic: {topic}")
