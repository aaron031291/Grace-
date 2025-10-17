"""
Swarm Orchestrator - Coordinates all swarm intelligence components
"""

from typing import Dict, List, Any, Optional
import logging
from .node_coordinator import GraceNodeCoordinator, NodeRole, TaskPriority
from .consensus_engine import CollectiveConsensusEngine, ConsensusAlgorithm
from .knowledge_graph_manager import GlobalKnowledgeGraphManager

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """
    Main orchestrator for swarm intelligence layer
    Integrates node coordination, consensus, and knowledge federation
    """
    
    def __init__(self):
        self.coordinator = GraceNodeCoordinator()
        self.consensus = CollectiveConsensusEngine()
        self.knowledge = GlobalKnowledgeGraphManager()
        self.event_bus = None
        self.quorum_module = None
        logger.info("SwarmOrchestrator initialized")
    
    def connect_event_bus(self, event_bus):
        """Connect to EventBus for multi-agent signal routing"""
        self.event_bus = event_bus
        
        # Register swarm event handlers
        self._register_event_handlers()
        
        logger.info("Connected to EventBus for multi-agent signal routing")
    
    def connect_quorum(self, quorum_module):
        """Connect to MLDL Quorum module for consensus feedback"""
        self.quorum_module = quorum_module
        
        # Register consensus callbacks
        self._register_quorum_callbacks()
        
        logger.info("Connected to MLDL Quorum module")
    
    def _register_event_handlers(self):
        """Register handlers for EventBus events"""
        if not self.event_bus:
            return
        
        # Node lifecycle events
        self.coordinator.on_event("node_registered", self._handle_node_registered)
        self.coordinator.on_event("node_unregistered", self._handle_node_unregistered)
        self.coordinator.on_event("task_assigned", self._handle_task_assigned)
        self.coordinator.on_event("task_completed", self._handle_task_completed)
        self.coordinator.on_event("task_failed", self._handle_task_failed)
        
        logger.info("Registered EventBus handlers")
    
    def _register_quorum_callbacks(self):
        """Register callbacks for MLDL Quorum integration"""
        if not self.quorum_module:
            return
        
        # Register consensus feedback for different decision types
        self.consensus.register_quorum_callback(
            "model_selection",
            self._quorum_model_selection_callback
        )
        self.consensus.register_quorum_callback(
            "parameter_tuning",
            self._quorum_parameter_callback
        )
        self.consensus.register_quorum_callback(
            "strategy_decision",
            self._quorum_strategy_callback
        )
        
        logger.info("Registered Quorum callbacks")
    
    def _handle_node_registered(self, data: Dict[str, Any]):
        """Handle node registration event"""
        node = data['node']
        
        # Register node in consensus engine
        self.consensus.register_node(
            node.node_id,
            weight=node.capacity,
            reputation=1.0
        )
        
        # Publish to EventBus
        if self.event_bus:
            self.event_bus.publish("swarm.node.registered", {
                'node_id': node.node_id,
                'node_name': node.node_name,
                'role': node.role.value,
                'capabilities': list(node.capabilities)
            })
        
        logger.info(f"Handled node registration: {node.node_id}")
    
    def _handle_node_unregistered(self, data: Dict[str, Any]):
        """Handle node unregistration event"""
        if self.event_bus:
            self.event_bus.publish("swarm.node.unregistered", data)
    
    def _handle_task_assigned(self, data: Dict[str, Any]):
        """Handle task assignment event"""
        if self.event_bus:
            self.event_bus.publish("swarm.task.assigned", {
                'task_id': data['task'].task_id,
                'node_id': data['node'].node_id,
                'task_type': data['task'].task_type
            })
    
    def _handle_task_completed(self, data: Dict[str, Any]):
        """Handle task completion event"""
        task = data['task']
        
        # Update knowledge graph if applicable
        if task.task_type == 'knowledge_contribution':
            self._process_knowledge_contribution(task)
        
        if self.event_bus:
            self.event_bus.publish("swarm.task.completed", {
                'task_id': task.task_id,
                'task_type': task.task_type
            })
    
    def _handle_task_failed(self, data: Dict[str, Any]):
        """Handle task failure event"""
        if self.event_bus:
            self.event_bus.publish("swarm.task.failed", {
                'task_id': data['task'].task_id,
                'error': data['error']
            })
    
    def _process_knowledge_contribution(self, task):
        """Process knowledge contribution from completed task"""
        if 'knowledge_node' in task.result:
            node_data = task.result['knowledge_node']
            self.knowledge.add_knowledge_node(
                node_type=node_data.get('type', 'general'),
                content=node_data.get('content', {}),
                source_node=task.assigned_node or 'unknown',
                confidence=node_data.get('confidence', 1.0)
            )
    
    def _quorum_model_selection_callback(self, proposal):
        """Callback for model selection consensus"""
        if not self.quorum_module or proposal.status.value != 'approved':
            return
        
        # Feed consensus result to quorum
        try:
            self.quorum_module.update_model_preference(
                model_id=proposal.content.get('model_id'),
                confidence=len(proposal.votes) / proposal.required_votes,
                consensus_data=proposal.result
            )
            logger.info(f"Updated quorum with model selection consensus: {proposal.proposal_id}")
        except Exception as e:
            logger.error(f"Failed to update quorum: {e}")
    
    def _quorum_parameter_callback(self, proposal):
        """Callback for parameter tuning consensus"""
        if not self.quorum_module or proposal.status.value != 'approved':
            return
        
        try:
            self.quorum_module.apply_parameter_consensus(
                parameters=proposal.content.get('parameters', {}),
                consensus_strength=proposal.result.get('approval_ratio', 0)
            )
            logger.info(f"Applied parameter consensus to quorum: {proposal.proposal_id}")
        except Exception as e:
            logger.error(f"Failed to apply parameters: {e}")
    
    def _quorum_strategy_callback(self, proposal):
        """Callback for strategy decision consensus"""
        if not self.quorum_module or proposal.status.value != 'approved':
            return
        
        try:
            self.quorum_module.adopt_strategy(
                strategy=proposal.content.get('strategy'),
                support_level=proposal.result.get('approval_ratio', 0)
            )
            logger.info(f"Adopted strategy from consensus: {proposal.proposal_id}")
        except Exception as e:
            logger.error(f"Failed to adopt strategy: {e}")
    
    def distribute_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: Optional[set] = None
    ) -> str:
        """Distribute a task to the swarm"""
        task = self.coordinator.submit_task(
            task_type=task_type,
            payload=payload,
            priority=priority,
            required_capabilities=required_capabilities
        )
        
        return task.task_id
    
    def propose_to_swarm(
        self,
        proposer_id: str,
        proposal_type: str,
        content: Dict[str, Any],
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE
    ) -> str:
        """Create a proposal for swarm consensus"""
        proposal = self.consensus.create_proposal(
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            content=content,
            algorithm=algorithm
        )
        
        return proposal.proposal_id
    
    def contribute_knowledge(
        self,
        node_id: str,
        knowledge_type: str,
        content: Dict[str, Any],
        confidence: float = 1.0
    ) -> str:
        """Contribute knowledge to the shared graph"""
        node = self.knowledge.add_knowledge_node(
            node_type=knowledge_type,
            content=content,
            source_node=node_id,
            confidence=confidence
        )
        
        return node.node_id
    
    def get_swarm_intelligence(self) -> Dict[str, Any]:
        """Get collective swarm intelligence status"""
        return {
            'coordination': self.coordinator.get_swarm_status(),
            'consensus': self.consensus.get_consensus_statistics(),
            'knowledge': self.knowledge.get_graph_statistics(),
            'event_bus_connected': self.event_bus is not None,
            'quorum_connected': self.quorum_module is not None
        }
