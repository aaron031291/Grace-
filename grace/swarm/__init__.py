"""
Swarm Intelligence Layer - Distributed coordination and collective intelligence
Enables multiple Grace instances to collaborate, reach consensus, and share knowledge.
"""

from .node_coordinator import GraceNodeCoordinator
from .consensus_engine import CollectiveConsensusEngine
from .knowledge_graph_manager import GlobalKnowledgeGraphManager
from .swarm_orchestrator import SwarmOrchestrator

__all__ = [
    'GraceNodeCoordinator',
    'CollectiveConsensusEngine',
    'GlobalKnowledgeGraphManager',
    'SwarmOrchestrator'
]

__version__ = '1.0.0'
