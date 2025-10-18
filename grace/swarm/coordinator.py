"""
Grace Node Coordinator - Orchestrates swarm nodes
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging

from grace.config import get_settings
from grace.swarm.transport import MessageTransport, HTTPTransport, TransportProtocol
from grace.swarm.discovery import PeerDiscovery, ServiceRegistry, NodeInfo
from grace.swarm.consensus import CollectiveConsensusEngine, ConsensusAlgorithm

logger = logging.getLogger(__name__)


class GraceNodeCoordinator:
    """
    Coordinates multiple Grace nodes in a swarm
    
    Features:
    - Peer discovery
    - Event exchange
    - Fault tolerance
    - Collective decision making
    """
    
    def __init__(
        self,
        node_id: str,
        transport_protocol: TransportProtocol = TransportProtocol.HTTP,
        port: int = 8080,
        consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.WEIGHTED_AVERAGE
    ):
        self.node_id = node_id
        self.port = port
        
        # Initialize transport
        if transport_protocol == TransportProtocol.HTTP:
            self.transport: MessageTransport = HTTPTransport(node_id, port)
        else:
            self.transport = HTTPTransport(node_id, port)  # Default
            logger.warning(f"Using HTTP transport (requested: {transport_protocol.value})")
        
        # Initialize discovery
        self.discovery = PeerDiscovery(node_id)
        
        # Initialize service registry
        self.service_registry = ServiceRegistry()
        
        # Initialize consensus engine
        self.consensus_engine = CollectiveConsensusEngine(consensus_algorithm)
        
        # State
        self.running = False
        self.local_trust_score = 0.8
        
        logger.info(f"GraceNodeCoordinator initialized: {node_id}")
    
    async def start(self):
        """Start coordinator"""
        if self.running:
            return
        
        self.running = True
        
        # Start discovery
        await self.discovery.start()
        
        # Register discovery callback
        self.discovery.on_peer_discovered(self._on_peer_discovered)
        
        # Start message handling
        asyncio.create_task(self._message_loop())
        
        logger.info(f"Coordinator started: {self.node_id}")
    
    async def stop(self):
        """Stop coordinator"""
        self.running = False
        await self.discovery.stop()
        logger.info(f"Coordinator stopped: {self.node_id}")
    
    async def _message_loop(self):
        """Handle incoming messages"""
        while self.running:
            try:
                message = await self.transport.receive()
                
                if message:
                    await self._handle_message(message)
                
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Message loop error: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming swarm message"""
        msg_type = message.get("type")
        
        if msg_type == "heartbeat":
            await self._handle_heartbeat(message)
        elif msg_type == "decision_request":
            await self._handle_decision_request(message)
        elif msg_type == "event":
            await self._handle_event(message)
        elif msg_type == "consensus_vote":
            await self._handle_consensus_vote(message)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat from peer"""
        peer_id = message.get("node_id")
        if peer_id:
            self.discovery.update_peer_heartbeat(peer_id)
            logger.debug(f"Heartbeat from {peer_id}")
    
    async def _handle_decision_request(self, message: Dict[str, Any]):
        """Handle decision request from peer"""
        request_id = message.get("request_id")
        decision_context = message.get("context")
        
        # Make local decision (integrate with local Grace instance)
        local_decision = await self._make_local_decision(decision_context)
        
        # Send response
        response = {
            "type": "decision_response",
            "request_id": request_id,
            "node_id": self.node_id,
            "decision": local_decision,
            "trust_score": self.local_trust_score
        }
        
        requester_id = message.get("node_id")
        await self.transport.send(requester_id, response)
    
    async def _handle_event(self, message: Dict[str, Any]):
        """Handle event from peer"""
        event_type = message.get("event_type")
        event_data = message.get("data")
        
        logger.info(f"Received event: {event_type} from {message.get('node_id')}")
        
        # Process event (integrate with local event bus)
    
    async def _handle_consensus_vote(self, message: Dict[str, Any]):
        """Handle consensus vote"""
        vote_data = message.get("vote")
        logger.debug(f"Consensus vote from {message.get('node_id')}")
    
    async def _make_local_decision(self, context: Dict[str, Any]) -> Any:
        """Make local decision (placeholder - integrate with Grace logic)"""
        # In production: integrate with local Unified Logic
        return {"decision": "local_response", "confidence": 0.8}
    
    async def _on_peer_discovered(self, peer: NodeInfo):
        """Callback when peer is discovered"""
        logger.info(f"Peer discovered: {peer.node_id}")
        
        # Add to transport
        if isinstance(self.transport, HTTPTransport):
            url = f"{peer.protocol}://{peer.host}:{peer.port}"
            self.transport.add_peer(peer.node_id, url)
    
    async def request_collective_decision(
        self,
        decision_context: Dict[str, Any],
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Request collective decision from swarm
        
        Returns consensus from all nodes
        """
        # Broadcast decision request
        request_id = f"req_{self.node_id}_{asyncio.get_event_loop().time()}"
        
        request_message = {
            "type": "decision_request",
            "request_id": request_id,
            "node_id": self.node_id,
            "context": decision_context
        }
        
        await self.transport.broadcast(request_message)
        
        # Collect responses (simplified - in production use proper async collection)
        await asyncio.sleep(timeout)
        
        # Get local decision
        local_decision = await self._make_local_decision(decision_context)
        
        # Compute consensus (in production, collect actual responses)
        node_decisions = {
            self.node_id: local_decision
        }
        
        node_trust_scores = {
            self.node_id: self.local_trust_score
        }
        
        consensus = self.consensus_engine.compute_consensus(
            node_decisions,
            node_trust_scores,
            context=decision_context
        )
        
        # Reconcile with local
        final_decision = self.consensus_engine.reconcile_with_local(
            consensus,
            local_decision,
            self.local_trust_score
        )
        
        return final_decision
    
    async def broadcast_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast event to all peers"""
        message = {
            "type": "event",
            "node_id": self.node_id,
            "event_type": event_type,
            "data": event_data
        }
        
        count = await self.transport.broadcast(message)
        logger.info(f"Broadcasted event {event_type} to {count} peers")
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        healthy_peers = self.discovery.get_healthy_peers()
        
        return {
            "node_id": self.node_id,
            "total_peers": len(self.discovery.peers),
            "healthy_peers": len(healthy_peers),
            "transport": self.transport.__class__.__name__,
            "consensus_algorithm": self.consensus_engine.algorithm.value,
            "peers": [
                {
                    "node_id": peer.node_id,
                    "host": peer.host,
                    "trust_score": peer.trust_score,
                    "capabilities": peer.capabilities
                }
                for peer in healthy_peers
            ]
        }
