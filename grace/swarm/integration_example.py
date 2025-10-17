"""
Swarm Integration Example - Demonstrates complete swarm functionality
"""

import logging
from .swarm_orchestrator import SwarmOrchestrator
from .node_coordinator import NodeRole, TaskPriority
from .consensus_engine import ConsensusAlgorithm, VoteType
from grace.integration.event_bus import EventBus
from grace.integration.quorum_integration import QuorumIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate swarm intelligence integration"""
    
    # Initialize components
    logger.info("=== Initializing Swarm Intelligence System ===")
    orchestrator = SwarmOrchestrator()
    event_bus = EventBus()
    quorum = QuorumIntegration()
    
    # Connect integrations
    orchestrator.connect_event_bus(event_bus)
    orchestrator.connect_quorum(quorum)
    
    # Register Grace nodes
    logger.info("\n=== Registering Grace Nodes ===")
    node1_id = "grace-node-1"
    node2_id = "grace-node-2"
    node3_id = "grace-node-3"
    
    orchestrator.coordinator.register_node(
        node_id=node1_id,
        node_name="Grace Alpha",
        capabilities={"reasoning", "analysis", "optimization"},
        specializations=["scientific_research"],
        role=NodeRole.WORKER
    )
    
    orchestrator.coordinator.register_node(
        node_id=node2_id,
        node_name="Grace Beta",
        capabilities={"reasoning", "ethics", "policy"},
        specializations=["ethical_analysis"],
        role=NodeRole.SPECIALIST
    )
    
    orchestrator.coordinator.register_node(
        node_id=node3_id,
        node_name="Grace Gamma",
        capabilities={"reasoning", "optimization", "coordination"},
        role=NodeRole.COORDINATOR
    )
    
    # Distribute tasks
    logger.info("\n=== Distributing Tasks ===")
    task1_id = orchestrator.distribute_task(
        task_type="optimization",
        payload={"problem": "route_optimization", "data": {"nodes": 100}},
        priority=TaskPriority.HIGH,
        required_capabilities={"optimization"}
    )
    
    task2_id = orchestrator.distribute_task(
        task_type="ethical_analysis",
        payload={"scenario": "AI_deployment", "stakeholders": 5},
        priority=TaskPriority.NORMAL,
        required_capabilities={"ethics"}
    )
    
    # Simulate heartbeats
    orchestrator.coordinator.heartbeat(node1_id, load=0.3)
    orchestrator.coordinator.heartbeat(node2_id, load=0.5)
    orchestrator.coordinator.heartbeat(node3_id, load=0.2)
    
    # Create consensus proposal
    logger.info("\n=== Creating Consensus Proposal ===")
    proposal_id = orchestrator.propose_to_swarm(
        proposer_id=node3_id,
        proposal_type="model_selection",
        content={
            "model_id": "gpt-4",
            "reason": "Superior reasoning capabilities",
            "use_case": "complex_analysis"
        },
        algorithm=ConsensusAlgorithm.WEIGHTED_VOTE
    )
    
    # Nodes vote on proposal
    logger.info("\n=== Voting on Proposal ===")
    orchestrator.consensus.submit_vote(
        proposal_id=proposal_id,
        voter_id=node1_id,
        vote_type=VoteType.APPROVE,
        rationale="Aligns with our performance requirements"
    )
    
    orchestrator.consensus.submit_vote(
        proposal_id=proposal_id,
        voter_id=node2_id,
        vote_type=VoteType.APPROVE,
        rationale="Good for ethical reasoning tasks"
    )
    
    orchestrator.consensus.submit_vote(
        proposal_id=proposal_id,
        voter_id=node3_id,
        vote_type=VoteType.APPROVE,
        rationale="Proven track record"
    )
    
    # Contribute knowledge
    logger.info("\n=== Contributing Knowledge ===")
    knowledge_id = orchestrator.contribute_knowledge(
        node_id=node1_id,
        knowledge_type="optimization_insight",
        content={
            "algorithm": "genetic_algorithm",
            "performance": 0.95,
            "use_case": "route_optimization"
        },
        confidence=0.9
    )
    
    # Create knowledge relationship
    orchestrator.knowledge.add_knowledge_edge(
        source_id=knowledge_id,
        target_id=knowledge_id,  # Self-reference for demo
        relationship_type="improves",
        weight=0.8,
        created_by=node1_id
    )
    
    # Get swarm intelligence status
    logger.info("\n=== Swarm Intelligence Status ===")
    status = orchestrator.get_swarm_intelligence()
    
    logger.info(f"Active Nodes: {status['coordination']['active_nodes']}")
    logger.info(f"Pending Tasks: {status['coordination']['pending_tasks']}")
    logger.info(f"Completed Tasks: {status['coordination']['completed_tasks']}")
    logger.info(f"Total Proposals: {status['consensus']['total_proposals']}")
    logger.info(f"Approved Proposals: {status['consensus']['approved']}")
    logger.info(f"Knowledge Nodes: {status['knowledge']['total_nodes']}")
    logger.info(f"Knowledge Edges: {status['knowledge']['total_edges']}")
    
    # Check quorum integration
    logger.info("\n=== Quorum Integration Status ===")
    quorum_status = quorum.get_quorum_status()
    logger.info(f"Preferred Model: {quorum_status['preferred_model']}")
    logger.info(f"Consensus Parameters: {quorum_status['consensus_parameters']}")
    
    # Get event bus statistics
    logger.info("\n=== Event Bus Statistics ===")
    event_stats = event_bus.get_statistics()
    logger.info(f"Total Events: {event_stats['total_events']}")
    logger.info(f"Event Types: {event_stats['event_types']}")
    
    logger.info("\n=== Swarm Intelligence Integration Complete ===")


if __name__ == "__main__":
    main()
