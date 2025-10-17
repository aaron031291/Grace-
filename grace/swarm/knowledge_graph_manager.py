"""
Global Knowledge Graph Manager - Shared knowledge federation across swarm
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    node_id: str
    node_type: str
    content: Dict[str, Any]
    source_node: str  # Which Grace node contributed this
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """Edge connecting knowledge nodes"""
    edge_id: str
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""


@dataclass
class KnowledgeUpdate:
    """Represents an update to the knowledge graph"""
    update_id: str
    update_type: str  # 'add', 'modify', 'delete'
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    contributor: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    synchronized: bool = False


class GlobalKnowledgeGraphManager:
    """
    Manages shared knowledge federation across distributed Grace instances
    Provides distributed knowledge graph with conflict resolution
    """
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.updates: List[KnowledgeUpdate] = []
        self.node_contributions: Dict[str, Set[str]] = defaultdict(set)
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # Type -> node IDs
        self.sync_status: Dict[str, datetime] = {}  # Node -> last sync time
        logger.info("GlobalKnowledgeGraphManager initialized")
    
    def add_knowledge_node(
        self,
        node_type: str,
        content: Dict[str, Any],
        source_node: str,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> KnowledgeNode:
        """Add a new knowledge node to the graph"""
        node = KnowledgeNode(
            node_id=str(uuid.uuid4()),
            node_type=node_type,
            content=content,
            source_node=source_node,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.nodes[node.node_id] = node
        self.node_contributions[source_node].add(node.node_id)
        self.node_index[node_type].add(node.node_id)
        
        # Record update
        update = KnowledgeUpdate(
            update_id=str(uuid.uuid4()),
            update_type='add',
            node_id=node.node_id,
            changes={'node': node.__dict__},
            contributor=source_node
        )
        self.updates.append(update)
        
        logger.info(f"Added knowledge node: {node.node_id} (type: {node_type}) from {source_node}")
        
        return node
    
    def add_knowledge_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        bidirectional: bool = False,
        created_by: str = "",
        properties: Optional[Dict] = None
    ) -> Optional[KnowledgeEdge]:
        """Add an edge between knowledge nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge: nodes not found ({source_id}, {target_id})")
            return None
        
        edge = KnowledgeEdge(
            edge_id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            bidirectional=bidirectional,
            properties=properties or {},
            created_by=created_by
        )
        
        self.edges[edge.edge_id] = edge
        
        # Record update
        update = KnowledgeUpdate(
            update_id=str(uuid.uuid4()),
            update_type='add',
            edge_id=edge.edge_id,
            changes={'edge': edge.__dict__},
            contributor=created_by
        )
        self.updates.append(update)
        
        logger.info(f"Added edge: {relationship_type} from {source_id} to {target_id}")
        
        return edge
    
    def update_knowledge_node(
        self,
        node_id: str,
        content: Optional[Dict] = None,
        confidence: Optional[float] = None,
        updater: str = ""
    ) -> bool:
        """Update an existing knowledge node"""
        if node_id not in self.nodes:
            logger.warning(f"Cannot update unknown node: {node_id}")
            return False
        
        node = self.nodes[node_id]
        changes = {}
        
        if content is not None:
            # Merge content
            node.content.update(content)
            changes['content'] = content
        
        if confidence is not None:
            # Update confidence (weighted average)
            node.confidence = (node.confidence + confidence) / 2
            changes['confidence'] = confidence
        
        node.updated_at = datetime.now()
        node.version += 1
        
        # Record update
        update = KnowledgeUpdate(
            update_id=str(uuid.uuid4()),
            update_type='modify',
            node_id=node_id,
            changes=changes,
            contributor=updater
        )
        self.updates.append(update)
        
        logger.info(f"Updated knowledge node: {node_id} (version: {node.version})")
        
        return True
    
    def merge_knowledge(
        self,
        node_id_1: str,
        node_id_2: str,
        merge_strategy: str = "weighted"
    ) -> Optional[KnowledgeNode]:
        """Merge two knowledge nodes with conflict resolution"""
        if node_id_1 not in self.nodes or node_id_2 not in self.nodes:
            return None
        
        node1 = self.nodes[node_id_1]
        node2 = self.nodes[node_id_2]
        
        # Only merge same types
        if node1.node_type != node2.node_type:
            logger.warning(f"Cannot merge different node types: {node1.node_type} vs {node2.node_type}")
            return None
        
        if merge_strategy == "weighted":
            # Weighted merge based on confidence
            total_confidence = node1.confidence + node2.confidence
            
            merged_content = {}
            for key in set(node1.content.keys()) | set(node2.content.keys()):
                if key in node1.content and key in node2.content:
                    # Weighted average for numeric values
                    if isinstance(node1.content[key], (int, float)) and isinstance(node2.content[key], (int, float)):
                        merged_content[key] = (
                            node1.content[key] * node1.confidence + 
                            node2.content[key] * node2.confidence
                        ) / total_confidence
                    else:
                        # Use higher confidence value
                        merged_content[key] = node1.content[key] if node1.confidence >= node2.confidence else node2.content[key]
                elif key in node1.content:
                    merged_content[key] = node1.content[key]
                else:
                    merged_content[key] = node2.content[key]
            
            merged_node = KnowledgeNode(
                node_id=str(uuid.uuid4()),
                node_type=node1.node_type,
                content=merged_content,
                source_node=f"merged:{node1.source_node},{node2.source_node}",
                confidence=total_confidence / 2,
                metadata={
                    'merged_from': [node_id_1, node_id_2],
                    'merge_strategy': merge_strategy
                }
            )
            
            self.nodes[merged_node.node_id] = merged_node
            
            # Transfer edges
            self._transfer_edges(node_id_1, merged_node.node_id)
            self._transfer_edges(node_id_2, merged_node.node_id)
            
            logger.info(f"Merged nodes {node_id_1} and {node_id_2} into {merged_node.node_id}")
            
            return merged_node
        
        return None
    
    def _transfer_edges(self, old_node_id: str, new_node_id: str):
        """Transfer edges from old node to new node"""
        for edge in self.edges.values():
            if edge.source_id == old_node_id:
                edge.source_id = new_node_id
            if edge.target_id == old_node_id:
                edge.target_id = new_node_id
    
    def query_knowledge(
        self,
        node_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.0
    ) -> List[KnowledgeNode]:
        """Query knowledge nodes"""
        results = []
        
        # Get candidate nodes
        if node_type:
            candidates = [self.nodes[nid] for nid in self.node_index.get(node_type, set())]
        else:
            candidates = list(self.nodes.values())
        
        for node in candidates:
            # Confidence filter
            if node.confidence < min_confidence:
                continue
            
            # Content filters
            if filters:
                matches = all(
                    node.content.get(key) == value
                    for key, value in filters.items()
                )
                if not matches:
                    continue
            
            results.append(node)
        
        # Sort by confidence
        results.sort(key=lambda n: n.confidence, reverse=True)
        
        return results
    
    def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 3,
        relationship_filter: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Traverse knowledge graph from a starting node"""
        if start_node_id not in self.nodes:
            return {}
        
        visited = set()
        result = {'nodes': {}, 'edges': []}
        
        def dfs(node_id: str, depth: int):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            result['nodes'][node_id] = self.nodes[node_id]
            
            # Find connected edges
            for edge in self.edges.values():
                if edge.source_id == node_id:
                    if relationship_filter and edge.relationship_type not in relationship_filter:
                        continue
                    
                    result['edges'].append(edge)
                    dfs(edge.target_id, depth + 1)
                
                elif edge.bidirectional and edge.target_id == node_id:
                    if relationship_filter and edge.relationship_type not in relationship_filter:
                        continue
                    
                    result['edges'].append(edge)
                    dfs(edge.source_id, depth + 1)
        
        dfs(start_node_id, 0)
        
        return result
    
    def synchronize_node(self, node_id: str, updates: List[Dict[str, Any]]) -> int:
        """Synchronize knowledge from a specific node"""
        sync_count = 0
        
        for update_data in updates:
            update = KnowledgeUpdate(**update_data)
            
            if update.update_type == 'add':
                if update.node_id and update.node_id not in self.nodes:
                    # Add node
                    node_data = update.changes.get('node', {})
                    self.nodes[update.node_id] = KnowledgeNode(**node_data)
                    sync_count += 1
                
                elif update.edge_id and update.edge_id not in self.edges:
                    # Add edge
                    edge_data = update.changes.get('edge', {})
                    self.edges[update.edge_id] = KnowledgeEdge(**edge_data)
                    sync_count += 1
            
            elif update.update_type == 'modify' and update.node_id:
                if update.node_id in self.nodes:
                    node = self.nodes[update.node_id]
                    if 'content' in update.changes:
                        node.content.update(update.changes['content'])
                    if 'confidence' in update.changes:
                        node.confidence = update.changes['confidence']
                    node.updated_at = update.timestamp
                    sync_count += 1
        
        self.sync_status[node_id] = datetime.now()
        
        logger.info(f"Synchronized {sync_count} updates from node {node_id}")
        
        return sync_count
    
    def get_unsynchronized_updates(self, node_id: str, since: Optional[datetime] = None) -> List[KnowledgeUpdate]:
        """Get updates that need to be synchronized to a node"""
        last_sync = since or self.sync_status.get(node_id, datetime.min)
        
        return [
            update for update in self.updates
            if update.timestamp > last_sync and update.contributor != node_id
        ]
    
    def get_node_contributions(self, node_id: str) -> Dict[str, Any]:
        """Get statistics on a node's contributions"""
        contributed_nodes = self.node_contributions.get(node_id, set())
        
        if not contributed_nodes:
            return {'node_id': node_id, 'total_contributions': 0}
        
        nodes = [self.nodes[nid] for nid in contributed_nodes if nid in self.nodes]
        
        return {
            'node_id': node_id,
            'total_contributions': len(nodes),
            'avg_confidence': sum(n.confidence for n in nodes) / len(nodes) if nodes else 0,
            'types': list(set(n.node_type for n in nodes)),
            'last_contribution': max(n.created_at for n in nodes) if nodes else None
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': len(self.node_index),
            'contributing_nodes': len(self.node_contributions),
            'total_updates': len(self.updates),
            'avg_node_confidence': sum(n.confidence for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'synchronized_nodes': len(self.sync_status)
        }
