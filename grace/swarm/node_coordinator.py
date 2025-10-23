"""
Grace Node Coordinator - Handles node orchestration and distributed intelligence
"""

from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a Grace node"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class NodeRole(Enum):
    """Roles that nodes can assume"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class GraceNode:
    """Represents a Grace instance in the swarm"""
    node_id: str
    node_name: str
    status: NodeStatus = NodeStatus.INITIALIZING
    role: NodeRole = NodeRole.WORKER
    capabilities: Set[str] = field(default_factory=set)
    capacity: float = 1.0  # 0.0 to 1.0
    current_load: float = 0.0
    specializations: List[str] = field(default_factory=list)
    location: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """Represents a task in the distributed system"""
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    required_capabilities: Set[str] = field(default_factory=set)


@dataclass
class NodePerformanceMetrics:
    """Performance metrics for a node"""
    node_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    uptime_percentage: float = 100.0
    specialization_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class GraceNodeCoordinator:
    """
    Coordinates multiple Grace instances in a distributed swarm
    Handles task distribution, load balancing, and node orchestration
    """
    
    def __init__(self, coordinator_id: Optional[str] = None):
        self.coordinator_id = coordinator_id or str(uuid.uuid4())
        self.nodes: Dict[str, GraceNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[DistributedTask] = []
        self.performance_metrics: Dict[str, NodePerformanceMetrics] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.heartbeat_timeout = timedelta(seconds=30)
        logger.info(f"GraceNodeCoordinator initialized: {self.coordinator_id}")
    
    def register_node(
        self,
        node_id: str,
        node_name: str,
        capabilities: Set[str],
        specializations: Optional[List[str]] = None,
        role: NodeRole = NodeRole.WORKER,
        metadata: Optional[Dict] = None
    ) -> GraceNode:
        """Register a new node in the swarm"""
        node = GraceNode(
            node_id=node_id,
            node_name=node_name,
            status=NodeStatus.ACTIVE,
            role=role,
            capabilities=capabilities,
            specializations=specializations or [],
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self.performance_metrics[node_id] = NodePerformanceMetrics(node_id=node_id)
        
        logger.info(f"Registered node: {node_name} ({node_id}) with role {role.value}")
        self._emit_event("node_registered", {"node": node})
        
        return node
    
    def unregister_node(self, node_id: str):
        """Unregister a node from the swarm"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Reassign tasks
            self._reassign_node_tasks(node_id)
            
            del self.nodes[node_id]
            logger.info(f"Unregistered node: {node_id}")
            self._emit_event("node_unregistered", {"node_id": node_id})
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update node status"""
        if node_id in self.nodes:
            old_status = self.nodes[node_id].status
            self.nodes[node_id].status = status
            self.nodes[node_id].last_heartbeat = datetime.now()
            
            logger.debug(f"Node {node_id} status: {old_status.value} -> {status.value}")
            
            if status == NodeStatus.OFFLINE:
                self._reassign_node_tasks(node_id)
    
    def heartbeat(self, node_id: str, load: float, metadata: Optional[Dict] = None):
        """Process heartbeat from a node"""
        if node_id not in self.nodes:
            logger.warning(f"Heartbeat from unknown node: {node_id}")
            return
        
        node = self.nodes[node_id]
        node.last_heartbeat = datetime.now()
        node.current_load = load
        
        if metadata:
            node.metadata.update(metadata)
        
        # Auto-adjust status based on load
        if load > 0.9:
            node.status = NodeStatus.BUSY
        elif load > 0.5:
            node.status = NodeStatus.ACTIVE
        else:
            node.status = NodeStatus.IDLE
    
    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: Optional[Set[str]] = None
    ) -> DistributedTask:
        """Submit a task to the swarm"""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            payload=payload,
            required_capabilities=required_capabilities or set()
        )
        
        self.tasks[task.task_id] = task
        self.task_queue.append(task)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        logger.info(f"Task submitted: {task.task_id} (type: {task_type}, priority: {priority.value})")
        
        # Try to assign immediately
        self._assign_tasks()
        
        return task
    
    def _assign_tasks(self):
        """Assign pending tasks to available nodes"""
        available_nodes = [
            n for n in self.nodes.values()
            if n.status in [NodeStatus.ACTIVE, NodeStatus.IDLE] and n.current_load < 0.8
        ]
        
        if not available_nodes:
            logger.debug("No available nodes for task assignment")
            return
        
        assigned_count = 0
        remaining_tasks = []
        
        for task in self.task_queue:
            if task.status != "pending":
                continue
            
            # Find best node for task
            best_node = self._select_node_for_task(task, available_nodes)
            
            if best_node:
                self._assign_task_to_node(task, best_node)
                assigned_count += 1
            else:
                remaining_tasks.append(task)
        
        self.task_queue = remaining_tasks
        
        if assigned_count > 0:
            logger.info(f"Assigned {assigned_count} tasks to nodes")
    
    def _select_node_for_task(
        self,
        task: DistributedTask,
        available_nodes: List[GraceNode]
    ) -> Optional[GraceNode]:
        """Select the best node for a task using intelligent routing"""
        # Filter nodes by required capabilities
        capable_nodes = [
            n for n in available_nodes
            if task.required_capabilities.issubset(n.capabilities)
        ]
        
        if not capable_nodes:
            return None
        
        # Score nodes
        scored_nodes = []
        for node in capable_nodes:
            score = self._calculate_node_score(node, task)
            scored_nodes.append((score, node))
        
        # Return highest scoring node
        scored_nodes.sort(reverse=True, key=lambda x: x[0])
        return scored_nodes[0][1] if scored_nodes else None
    
    def _calculate_node_score(self, node: GraceNode, task: DistributedTask) -> float:
        """Calculate node suitability score for a task"""
        score = 0.0
        
        # Load factor (prefer less loaded nodes)
        score += (1.0 - node.current_load) * 40
        
        # Capacity factor
        score += node.capacity * 20
        
        # Specialization bonus
        if task.task_type in node.specializations:
            score += 30
        
        # Performance history
        metrics = self.performance_metrics.get(node.node_id)
        if metrics and metrics.tasks_completed > 0:
            success_rate = metrics.tasks_completed / (metrics.tasks_completed + metrics.tasks_failed)
            score += success_rate * 10
        
        return score
    
    def _assign_task_to_node(self, task: DistributedTask, node: GraceNode):
        """Assign a task to a specific node"""
        task.assigned_node = node.node_id
        task.status = "assigned"
        task.started_at = datetime.now()
        
        # Update node load (estimate)
        node.current_load = min(1.0, node.current_load + 0.2)
        
        logger.info(f"Assigned task {task.task_id} to node {node.node_id}")
        self._emit_event("task_assigned", {"task": task, "node": node})
    
    def complete_task(
        self,
        task_id: str,
        result: Any,
        execution_time: Optional[float] = None
    ):
        """Mark a task as completed"""
        if task_id not in self.tasks:
            logger.warning(f"Unknown task completion: {task_id}")
            return
        
        task = self.tasks[task_id]
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now()
        
        # Update node metrics
        if task.assigned_node:
            self._update_node_metrics(task.assigned_node, success=True, execution_time=execution_time)
            
            # Free up node capacity
            if task.assigned_node in self.nodes:
                node = self.nodes[task.assigned_node]
                node.current_load = max(0.0, node.current_load - 0.2)
        
        logger.info(f"Task completed: {task_id}")
        self._emit_event("task_completed", {"task": task})
        
        # Try to assign more tasks
        self._assign_tasks()
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.error = error
        task.retries += 1
        
        if task.retries < task.max_retries:
            # Retry task
            task.status = "pending"
            task.assigned_node = None
            self.task_queue.append(task)
            logger.warning(f"Task {task_id} failed, retrying ({task.retries}/{task.max_retries})")
        else:
            task.status = "failed"
            logger.error(f"Task {task_id} failed permanently: {error}")
        
        # Update node metrics
        if task.assigned_node:
            self._update_node_metrics(task.assigned_node, success=False)
            
            if task.assigned_node in self.nodes:
                node = self.nodes[task.assigned_node]
                node.current_load = max(0.0, node.current_load - 0.2)
        
        self._emit_event("task_failed", {"task": task, "error": error})
        
        # Reassign tasks
        self._assign_tasks()
    
    def _update_node_metrics(
        self,
        node_id: str,
        success: bool,
        execution_time: Optional[float] = None
    ):
        """Update performance metrics for a node"""
        if node_id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[node_id]
        
        if success:
            metrics.tasks_completed += 1
        else:
            metrics.tasks_failed += 1
        
        if execution_time:
            # Update average response time
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            metrics.avg_response_time = (
                (metrics.avg_response_time * (total_tasks - 1) + execution_time) / total_tasks
            )
        
        metrics.last_updated = datetime.now()
    
    def _reassign_node_tasks(self, node_id: str):
        """Reassign tasks from a failed/offline node"""
        reassigned = []
        
        for task in self.tasks.values():
            if task.assigned_node == node_id and task.status in ["assigned", "running"]:
                task.assigned_node = None
                task.status = "pending"
                task.retries += 1
                
                if task.retries < task.max_retries:
                    self.task_queue.append(task)
                    reassigned.append(task.task_id)
                else:
                    task.status = "failed"
                    task.error = f"Node {node_id} failed"
        
        if reassigned:
            logger.info(f"Reassigned {len(reassigned)} tasks from node {node_id}")
            self._assign_tasks()
    
    def check_node_health(self):
        """Check health of all nodes and mark stale ones as offline"""
        now = datetime.now()
        
        for node_id, node in list(self.nodes.items()):
            if node.status != NodeStatus.OFFLINE:
                time_since_heartbeat = now - node.last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    logger.warning(f"Node {node_id} missed heartbeat, marking offline")
                    self.update_node_status(node_id, NodeStatus.OFFLINE)
    
    def rebalance_load(self):
        """Rebalance load across nodes"""
        # Find overloaded and underloaded nodes
        overloaded = [n for n in self.nodes.values() if n.current_load > 0.8]
        underloaded = [n for n in self.nodes.values() if n.current_load < 0.4]
        
        if not overloaded or not underloaded:
            return
        
        # Move pending tasks from overloaded nodes
        for node in overloaded:
            node_tasks = [
                t for t in self.tasks.values()
                if t.assigned_node == node.node_id and t.status == "assigned"
            ]
            
            # Reassign some tasks
            for task in node_tasks[:len(node_tasks)//2]:
                task.assigned_node = None
                task.status = "pending"
                self.task_queue.append(task)
                node.current_load = max(0.0, node.current_load - 0.2)
        
        self._assign_tasks()
        logger.info("Load rebalancing completed")
    
    def on_event(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get overall swarm status"""
        active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
        
        return {
            'coordinator_id': self.coordinator_id,
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == "pending"]),
            'running_tasks': len([t for t in self.tasks.values() if t.status in ["assigned", "running"]]),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == "completed"]),
            'failed_tasks': len([t for t in self.tasks.values() if t.status == "failed"]),
            'avg_node_load': sum(n.current_load for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'nodes': [
                {
                    'id': n.node_id,
                    'name': n.node_name,
                    'status': n.status.value,
                    'role': n.role.value,
                    'load': n.current_load,
                    'capabilities': list(n.capabilities)
                }
                for n in self.nodes.values()
            ]
        }
    
    def get_node_performance(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a node"""
        if node_id not in self.performance_metrics:
            return None
        
        metrics = self.performance_metrics[node_id]
        total_tasks = metrics.tasks_completed + metrics.tasks_failed
        
        return {
            'node_id': node_id,
            'tasks_completed': metrics.tasks_completed,
            'tasks_failed': metrics.tasks_failed,
            'success_rate': metrics.tasks_completed / total_tasks if total_tasks > 0 else 0,
            'avg_response_time': metrics.avg_response_time,
            'uptime_percentage': metrics.uptime_percentage,
            'last_updated': metrics.last_updated.isoformat()
        }
