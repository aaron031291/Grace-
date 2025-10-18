"""
Peer discovery and service registry for swarm
"""

from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a swarm node"""
    node_id: str
    host: str
    port: int
    protocol: str
    capabilities: List[str]
    trust_score: float
    last_seen: datetime
    metadata: Dict[str, Any]
    
    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """Check if node is healthy based on last_seen"""
        age = (datetime.now(timezone.utc) - self.last_seen).total_seconds()
        return age < timeout_seconds


class ServiceRegistry:
    """Registry of available services across swarm"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        logger.info("ServiceRegistry initialized")
    
    def register_service(
        self,
        service_name: str,
        node_id: str,
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a service"""
        if service_name not in self.services:
            self.services[service_name] = {}
        
        self.services[service_name][node_id] = {
            "endpoint": endpoint,
            "registered_at": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        
        logger.info(f"Registered service: {service_name} on {node_id}")
    
    def unregister_service(self, service_name: str, node_id: str):
        """Unregister a service"""
        if service_name in self.services and node_id in self.services[service_name]:
            del self.services[service_name][node_id]
            logger.info(f"Unregistered service: {service_name} on {node_id}")
    
    def get_service_nodes(self, service_name: str) -> List[str]:
        """Get nodes providing a service"""
        if service_name not in self.services:
            return []
        return list(self.services[service_name].keys())
    
    def get_service_endpoint(self, service_name: str, node_id: str) -> Optional[str]:
        """Get service endpoint for specific node"""
        if service_name in self.services and node_id in self.services[service_name]:
            return self.services[service_name][node_id]["endpoint"]
        return None


class PeerDiscovery:
    """
    Peer discovery mechanism for swarm nodes
    
    Supports:
    - Multicast discovery
    - Heartbeat monitoring
    - Automatic peer pruning
    """
    
    def __init__(self, node_id: str, heartbeat_interval: int = 30):
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        
        self.peers: Dict[str, NodeInfo] = {}
        self.discovery_callbacks: List[callable] = []
        self.running = False
        
        logger.info(f"PeerDiscovery initialized for node {node_id}")
    
    async def start(self):
        """Start peer discovery"""
        if self.running:
            return
        
        self.running = True
        
        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())
        
        # Start discovery loop
        asyncio.create_task(self._discovery_loop())
        
        logger.info("Peer discovery started")
    
    async def stop(self):
        """Stop peer discovery"""
        self.running = False
        logger.info("Peer discovery stopped")
    
    def register_peer(
        self,
        node_id: str,
        host: str,
        port: int,
        protocol: str = "http",
        capabilities: Optional[List[str]] = None,
        trust_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a discovered peer"""
        peer = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            protocol=protocol,
            capabilities=capabilities or [],
            trust_score=trust_score,
            last_seen=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.peers[node_id] = peer
        
        # Trigger discovery callbacks
        for callback in self.discovery_callbacks:
            try:
                await callback(peer)
            except Exception as e:
                logger.error(f"Discovery callback error: {e}")
        
        logger.info(f"Registered peer: {node_id} at {host}:{port}")
    
    def update_peer_heartbeat(self, node_id: str):
        """Update peer's last_seen timestamp"""
        if node_id in self.peers:
            self.peers[node_id].last_seen = datetime.now(timezone.utc)
    
    def get_healthy_peers(self) -> List[NodeInfo]:
        """Get list of healthy peers"""
        return [peer for peer in self.peers.values() if peer.is_healthy()]
    
    def get_peers_by_capability(self, capability: str) -> List[NodeInfo]:
        """Get peers with specific capability"""
        return [
            peer for peer in self.peers.values()
            if capability in peer.capabilities and peer.is_healthy()
        ]
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                # Broadcast heartbeat (implemented by coordinator)
                logger.debug(f"Heartbeat: {self.node_id}")
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _discovery_loop(self):
        """Periodic peer discovery and pruning"""
        while self.running:
            try:
                # Prune unhealthy peers
                to_remove = [
                    node_id for node_id, peer in self.peers.items()
                    if not peer.is_healthy(timeout_seconds=120)
                ]
                
                for node_id in to_remove:
                    del self.peers[node_id]
                    logger.info(f"Pruned unhealthy peer: {node_id}")
                
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
    
    def on_peer_discovered(self, callback: callable):
        """Register callback for peer discovery"""
        self.discovery_callbacks.append(callback)
