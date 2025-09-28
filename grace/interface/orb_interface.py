"""
Grace Unified Orb Interface - Main User Interface
Comprehensive interface with chat, panels, memory management, and governance integration.
"""
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio

# Import components
from ..intelligence.grace_intelligence import GraceIntelligence, ReasoningContext, ReasoningResult
from .ide.grace_ide import GraceIDE

logger = logging.getLogger(__name__)


class PanelType(Enum):
    """Types of UI panels."""
    CHAT = "chat"
    TRADING = "trading"
    SALES = "sales"
    ANALYTICS = "analytics"
    MEMORY = "memory"
    GOVERNANCE = "governance"
    TASK_MANAGER = "task_manager"
    IDE = "ide"
    DASHBOARD = "dashboard"
    KNOWLEDGE_BASE = "knowledge_base"
    TASK_BOX = "task_box"
    COLLABORATION = "collaboration"
    LIBRARY_ACCESS = "library_access"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChatMessage:
    """Individual chat message."""
    message_id: str
    user_id: str
    content: str
    timestamp: str
    message_type: str = "user"  # user, assistant, system
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace_id: Optional[str] = None


@dataclass
class UIPanel:
    """UI Panel definition."""
    panel_id: str
    panel_type: PanelType
    title: str
    position: Dict[str, float]  # {"x": 0, "y": 0, "width": 400, "height": 300}
    data: Dict[str, Any] = field(default_factory=dict)
    is_closable: bool = True
    is_minimized: bool = False
    z_index: int = 1


@dataclass
class OrbNotification:
    """Proactive notification from Grace."""
    notification_id: str
    title: str
    message: str
    priority: NotificationPriority
    timestamp: str
    user_id: str
    action_required: bool = False
    actions: List[Dict[str, str]] = field(default_factory=list)
    auto_dismiss_seconds: Optional[int] = None


@dataclass
class MemoryFragment:
    """Fragment stored in memory."""
    fragment_id: str
    content: str
    fragment_type: str  # text, code, audio, video, document
    source: str
    trust_score: float
    timestamp: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceTask:
    """Governance approval or review task."""
    task_id: str
    title: str
    description: str
    task_type: str  # approval, review, audit
    priority: str
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TaskItem:
    """Individual task in the task box."""
    task_id: str
    title: str
    description: str
    status: str  # pending, in_progress, completed, failed
    priority: str  # low, medium, high, critical
    assigned_to: str  # grace, user, system
    created_at: str
    updated_at: str
    progress: float = 0.0  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    related_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEntry:
    """Knowledge base entry with library access."""
    entry_id: str
    title: str
    content: str
    source: str  # library, document, interaction
    domain: str  # coding, trading, analysis, etc.
    trust_score: float
    relevance_tags: List[str]
    created_at: str
    last_accessed: str
    access_count: int = 0
    related_libraries: List[str] = field(default_factory=list)


@dataclass
class MemoryExplorerItem:
    """File explorer-like memory item."""
    item_id: str
    name: str
    item_type: str  # folder, file, fragment
    content: Optional[str] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_editable: bool = True


@dataclass
class CollaborationSession:
    """Collaboration session for IDE-like development discussions."""
    session_id: str
    topic: str
    participants: List[str]  # user_ids
    status: str  # active, paused, completed
    discussion_points: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    shared_workspace: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OrbSession:
    """User session with the orb."""
    session_id: str
    user_id: str
    start_time: str
    last_activity: str
    chat_messages: List[ChatMessage] = field(default_factory=list)
    active_panels: List[UIPanel] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_memory: Dict[str, Any] = field(default_factory=dict)


class GraceUnifiedOrbInterface:
    """
    Grace Unified Orb Interface - Complete user interface system.
    
    Features:
    - Persistent chat with memory
    - Dynamic panels system
    - Memory & knowledge management
    - Governance integration
    - Task management
    - IDE integration
    - Proactive notifications
    - Multi-modal support
    """
    
    def __init__(self):
        self.version = "1.0.0"
        
        # Core components
        self.grace_intelligence = GraceIntelligence()
        self.grace_ide = GraceIDE()
        
        # Session management
        self.active_sessions: Dict[str, OrbSession] = {}
        
        # Global state
        self.memory_fragments: Dict[str, MemoryFragment] = {}
        self.governance_tasks: Dict[str, GovernanceTask] = {}
        self.notifications: Dict[str, OrbNotification] = {}
        
        # New enhanced features storage
        self.task_items: Dict[str, TaskItem] = {}
        self.knowledge_entries: Dict[str, KnowledgeEntry] = {}
        self.memory_explorer_items: Dict[str, MemoryExplorerItem] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        
        # System capabilities
        self.max_panels_per_session = 6
        self.supported_file_types = [
            "pdf", "doc", "docx", "txt", "csv", "json", "xml",
            "py", "js", "html", "css", "md",
            "jpg", "png", "gif", "mp4", "mp3", "wav"
        ]
        
        # Panel templates
        self.panel_templates = self._initialize_panel_templates()
        
        logger.info("Grace Unified Orb Interface initialized")

    def _initialize_panel_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize panel templates for different data types."""
        return {
            "trading_panel": {
                "title": "Trading Analysis",
                "default_size": {"width": 500, "height": 400},
                "components": ["price_chart", "indicators", "order_book"],
                "refresh_interval": 5
            },
            "sales_panel": {
                "title": "Sales Dashboard",
                "default_size": {"width": 450, "height": 350},
                "components": ["pipeline", "metrics", "recent_activities"],
                "refresh_interval": 30
            },
            "analytics_panel": {
                "title": "Analytics",
                "default_size": {"width": 600, "height": 450},
                "components": ["charts", "metrics", "filters"],
                "refresh_interval": 60
            },
            "memory_browser": {
                "title": "Memory Browser",
                "default_size": {"width": 400, "height": 500},
                "components": ["search", "fragment_list", "filters"],
                "refresh_interval": 0
            },
            "governance_center": {
                "title": "Governance Center",
                "default_size": {"width": 500, "height": 400},
                "components": ["pending_approvals", "audit_trail", "policies"],
                "refresh_interval": 15
            },
            "knowledge_base_panel": {
                "title": "Knowledge Base & Library Access",
                "default_size": {"width": 600, "height": 500},
                "components": ["search", "library_browser", "knowledge_graph", "access_logs"],
                "refresh_interval": 0
            },
            "task_box_panel": {
                "title": "Task Box",
                "default_size": {"width": 400, "height": 600},
                "components": ["active_tasks", "completed_tasks", "task_filters", "merge_tools"],
                "refresh_interval": 10
            },
            "collaboration_panel": {
                "title": "Collaboration Hub",
                "default_size": {"width": 800, "height": 600},
                "components": ["discussion_board", "action_items", "shared_workspace", "development_notes"],
                "refresh_interval": 5
            },
            "memory_explorer_panel": {
                "title": "Memory Explorer",
                "default_size": {"width": 500, "height": 700},
                "components": ["file_tree", "content_viewer", "editor", "metadata_panel"],
                "refresh_interval": 0
            }
        }

    # Session Management
    
    async def create_session(self, user_id: str, preferences: Optional[Dict[str, Any]] = None) -> str:
        """Create a new orb session."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        session = OrbSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow().isoformat(),
            last_activity=datetime.utcnow().isoformat(),
            preferences=preferences or {}
        )
        
        # Create default chat panel
        chat_panel = UIPanel(
            panel_id=f"chat_{session_id}",
            panel_type=PanelType.CHAT,
            title="Grace Chat",
            position={"x": 50, "y": 50, "width": 400, "height": 500},
            is_closable=False
        )
        session.active_panels.append(chat_panel)
        
        self.active_sessions[session_id] = session
        
        # Send welcome message
        await self.send_system_message(session_id, 
            "Hello! I'm Grace, your AI assistant. How can I help you today?")
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def end_session(self, session_id: str) -> bool:
        """End an orb session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Save session context to long-term memory
        await self._save_session_memory(session)
        
        # Clean up
        del self.active_sessions[session_id]
        
        logger.info(f"Ended session {session_id}")
        return True

    def get_session(self, session_id: str) -> Optional[OrbSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)

    async def update_session_activity(self, session_id: str):
        """Update last activity timestamp."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].last_activity = datetime.utcnow().isoformat()

    # Chat Interface
    
    async def send_chat_message(self, session_id: str, content: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        """Send chat message from user to Grace."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        # Create user message
        user_message = ChatMessage(
            message_id=message_id,
            user_id=session.user_id,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            message_type="user",
            attachments=attachments or []
        )
        
        session.chat_messages.append(user_message)
        await self.update_session_activity(session_id)
        
        # Process message with Grace Intelligence
        try:
            reasoning_context = ReasoningContext(
                user_id=session.user_id,
                session_id=session_id,
                metadata={"chat_history": len(session.chat_messages)}
            )
            
            reasoning_result = await self.grace_intelligence.process_request(content, reasoning_context)
            
            # Create Grace's response message
            response_message = ChatMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                user_id="grace",
                content=reasoning_result.response,
                timestamp=datetime.utcnow().isoformat(),
                message_type="assistant",
                reasoning_trace_id=f"trace_{message_id}"
            )
            
            session.chat_messages.append(response_message)
            
            # Handle UI instructions
            if reasoning_result.ui_instructions:
                await self._process_ui_instructions(session_id, reasoning_result.ui_instructions)
            
            # Store reasoning trace for later inspection
            session.context_memory[f"trace_{message_id}"] = reasoning_result.reasoning_trace
            
            return response_message.message_id
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_message = ChatMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                user_id="grace",
                content=f"I encountered an error while processing your request: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
                message_type="system"
            )
            session.chat_messages.append(error_message)
            return error_message.message_id

    async def send_system_message(self, session_id: str, content: str) -> str:
        """Send system message to user."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        system_message = ChatMessage(
            message_id=message_id,
            user_id="system",
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            message_type="system"
        )
        
        session.chat_messages.append(system_message)
        return message_id

    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get chat history for session."""
        if session_id not in self.active_sessions:
            return []
        
        messages = self.active_sessions[session_id].chat_messages
        
        if limit:
            return messages[-limit:]
        return messages

    # Panel Management
    
    async def create_panel(self, session_id: str, panel_type: PanelType, 
                          title: Optional[str] = None, data: Optional[Dict[str, Any]] = None,
                          position: Optional[Dict[str, float]] = None) -> str:
        """Create a new panel in the session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Check panel limit
        if len(session.active_panels) >= self.max_panels_per_session:
            raise ValueError(f"Maximum panels ({self.max_panels_per_session}) already open")
        
        panel_id = f"panel_{uuid.uuid4().hex[:8]}"
        
        # Get template for panel type
        template_key = f"{panel_type.value}_panel" if f"{panel_type.value}_panel" in self.panel_templates else "analytics_panel"
        template = self.panel_templates.get(template_key, {})
        
        # Set default position if not provided
        if not position:
            # Calculate position to avoid overlap
            base_x = 100 + (len(session.active_panels) % 3) * 200
            base_y = 100 + (len(session.active_panels) // 3) * 200
            default_size = template.get("default_size", {"width": 400, "height": 300})
            
            position = {
                "x": base_x,
                "y": base_y,
                "width": default_size["width"],
                "height": default_size["height"]
            }
        
        panel = UIPanel(
            panel_id=panel_id,
            panel_type=panel_type,
            title=title or template.get("title", f"{panel_type.value.title()} Panel"),
            position=position,
            data=data or {},
            z_index=len(session.active_panels) + 1
        )
        
        session.active_panels.append(panel)
        await self.update_session_activity(session_id)
        
        logger.info(f"Created {panel_type.value} panel {panel_id} in session {session_id}")
        return panel_id

    async def close_panel(self, session_id: str, panel_id: str) -> bool:
        """Close a panel."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Find and remove panel
        for i, panel in enumerate(session.active_panels):
            if panel.panel_id == panel_id and panel.is_closable:
                del session.active_panels[i]
                await self.update_session_activity(session_id)
                logger.info(f"Closed panel {panel_id} in session {session_id}")
                return True
        
        return False

    async def update_panel_data(self, session_id: str, panel_id: str, data: Dict[str, Any]) -> bool:
        """Update panel data."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        for panel in session.active_panels:
            if panel.panel_id == panel_id:
                panel.data.update(data)
                await self.update_session_activity(session_id)
                return True
        
        return False

    async def move_panel(self, session_id: str, panel_id: str, position: Dict[str, float]) -> bool:
        """Move/resize a panel."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        for panel in session.active_panels:
            if panel.panel_id == panel_id:
                panel.position.update(position)
                await self.update_session_activity(session_id)
                return True
        
        return False

    def get_panels(self, session_id: str) -> List[UIPanel]:
        """Get all panels for session."""
        if session_id not in self.active_sessions:
            return []
        
        return self.active_sessions[session_id].active_panels

    # Memory Management
    
    async def upload_document(self, user_id: str, file_path: str, file_type: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload and process a document into memory."""
        if file_type.lower() not in self.supported_file_types:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        fragment_id = f"fragment_{uuid.uuid4().hex[:8]}"
        
        # Process document based on type
        content = await self._process_document(file_path, file_type)
        
        # Calculate trust score
        trust_score = self._calculate_document_trust_score(file_path, file_type, metadata)
        
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            content=content,
            fragment_type=file_type,
            source=f"upload:{file_path}",
            trust_score=trust_score,
            timestamp=datetime.utcnow().isoformat(),
            tags=metadata.get("tags", []) if metadata else [],
            metadata=metadata or {}
        )
        
        # Use Grace Intelligence for document ingestion
        await self.grace_intelligence.ingest_batch_document(file_path, file_type)
        
        self.memory_fragments[fragment_id] = fragment
        logger.info(f"Uploaded document {file_path} as fragment {fragment_id}")
        
        return fragment_id

    async def search_memory(self, session_id: str, query: str, 
                           filters: Optional[Dict[str, Any]] = None) -> List[MemoryFragment]:
        """Search memory fragments."""
        # Basic search implementation
        results = []
        
        for fragment in self.memory_fragments.values():
            # Simple text matching
            if query.lower() in fragment.content.lower():
                # Apply filters
                if filters:
                    if "fragment_type" in filters and fragment.fragment_type != filters["fragment_type"]:
                        continue
                    if "min_trust_score" in filters and fragment.trust_score < filters["min_trust_score"]:
                        continue
                    if "tags" in filters and not any(tag in fragment.tags for tag in filters["tags"]):
                        continue
                
                results.append(fragment)
        
        # Sort by trust score and relevance
        results.sort(key=lambda f: f.trust_score, reverse=True)
        
        return results[:20]  # Limit results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "total_fragments": len(self.memory_fragments),
            "fragments_by_type": {},
            "average_trust_score": 0.0,
            "total_size": 0
        }
        
        if self.memory_fragments:
            trust_scores = []
            for fragment in self.memory_fragments.values():
                fragment_type = fragment.fragment_type
                stats["fragments_by_type"][fragment_type] = stats["fragments_by_type"].get(fragment_type, 0) + 1
                trust_scores.append(fragment.trust_score)
                stats["total_size"] += len(fragment.content)
            
            stats["average_trust_score"] = sum(trust_scores) / len(trust_scores)
        
        return stats

    # Governance Integration
    
    async def create_governance_task(self, title: str, description: str, task_type: str,
                                   requester_id: str, assignee_id: Optional[str] = None) -> str:
        """Create a governance task."""
        task_id = f"gov_task_{uuid.uuid4().hex[:8]}"
        
        task = GovernanceTask(
            task_id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority="medium",
            requester_id=requester_id,
            assignee_id=assignee_id
        )
        
        self.governance_tasks[task_id] = task
        
        # Create notification for assignee
        if assignee_id:
            await self.create_notification(
                user_id=assignee_id,
                title=f"New {task_type} Task",
                message=f"You have been assigned: {title}",
                priority=NotificationPriority.MEDIUM,
                action_required=True,
                actions=[
                    {"label": "Review", "action": f"governance_review:{task_id}"},
                    {"label": "Approve", "action": f"governance_approve:{task_id}"},
                    {"label": "Reject", "action": f"governance_reject:{task_id}"}
                ]
            )
        
        logger.info(f"Created governance task {task_id}: {title}")
        return task_id

    def get_governance_tasks(self, user_id: str, status_filter: Optional[str] = None) -> List[GovernanceTask]:
        """Get governance tasks for user."""
        tasks = []
        
        for task in self.governance_tasks.values():
            if task.assignee_id == user_id or task.requester_id == user_id:
                if not status_filter or task.status == status_filter:
                    tasks.append(task)
        
        return tasks

    async def update_governance_task_status(self, task_id: str, status: str, user_id: str) -> bool:
        """Update governance task status."""
        if task_id not in self.governance_tasks:
            return False
        
        task = self.governance_tasks[task_id]
        
        # Check permissions
        if task.assignee_id != user_id:
            return False
        
        task.status = status
        
        # Notify requester
        await self.create_notification(
            user_id=task.requester_id,
            title="Task Status Updated",
            message=f"Task '{task.title}' status changed to: {status}",
            priority=NotificationPriority.MEDIUM
        )
        
        return True

    # Notification System
    
    async def create_notification(self, user_id: str, title: str, message: str,
                                priority: NotificationPriority, action_required: bool = False,
                                actions: Optional[List[Dict[str, str]]] = None,
                                auto_dismiss_seconds: Optional[int] = None) -> str:
        """Create a proactive notification."""
        notification_id = f"notif_{uuid.uuid4().hex[:8]}"
        
        notification = OrbNotification(
            notification_id=notification_id,
            title=title,
            message=message,
            priority=priority,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            action_required=action_required,
            actions=actions or [],
            auto_dismiss_seconds=auto_dismiss_seconds
        )
        
        self.notifications[notification_id] = notification
        
        logger.info(f"Created notification {notification_id} for user {user_id}: {title}")
        return notification_id

    def get_notifications(self, user_id: str, unread_only: bool = True) -> List[OrbNotification]:
        """Get notifications for user."""
        user_notifications = []
        
        for notification in self.notifications.values():
            if notification.user_id == user_id:
                user_notifications.append(notification)
        
        # Sort by priority and timestamp
        user_notifications.sort(key=lambda n: (n.priority.value, n.timestamp), reverse=True)
        
        return user_notifications

    async def dismiss_notification(self, notification_id: str, user_id: str) -> bool:
        """Dismiss a notification."""
        if notification_id not in self.notifications:
            return False
        
        notification = self.notifications[notification_id]
        
        if notification.user_id != user_id:
            return False
        
        del self.notifications[notification_id]
        return True

    # IDE Integration
    
    async def open_ide_panel(self, session_id: str, flow_id: Optional[str] = None) -> str:
        """Open IDE panel in session."""
        panel_id = await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.IDE,
            title="Grace IDE",
            data={
                "flow_id": flow_id,
                "ide_stats": self.grace_ide.get_stats(),
                "block_registry": self.grace_ide.get_block_registry()
            }
        )
        
        return panel_id

    def get_ide_instance(self) -> GraceIDE:
        """Get the IDE instance for direct access."""
        return self.grace_ide

    # Utility Methods
    
    async def _process_ui_instructions(self, session_id: str, ui_instructions: Dict[str, Any]):
        """Process UI instructions from Grace Intelligence."""
        if "panels" in ui_instructions:
            for panel_config in ui_instructions["panels"]:
                panel_type_str = panel_config.get("type", "analytics")
                
                # Map string to enum
                panel_type = {
                    "trading_panel": PanelType.TRADING,
                    "sales_panel": PanelType.SALES,
                    "chart_panel": PanelType.ANALYTICS,
                    "analytics_panel": PanelType.ANALYTICS
                }.get(panel_type_str, PanelType.ANALYTICS)
                
                await self.create_panel(
                    session_id=session_id,
                    panel_type=panel_type,
                    title=panel_config.get("title"),
                    data=panel_config.get("data", {})
                )

    async def _process_document(self, file_path: str, file_type: str) -> str:
        """Process document content based on type."""
        # Simplified document processing
        if file_type.lower() in ["txt", "md"]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return f"Error reading file: {file_path}"
        else:
            return f"Binary file: {file_path} ({file_type})"

    def _calculate_document_trust_score(self, file_path: str, file_type: str, metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate trust score for uploaded document."""
        base_score = 0.8
        
        # Adjust based on file type
        if file_type.lower() in ["pdf", "doc", "docx"]:
            base_score += 0.1
        
        # Adjust based on metadata
        if metadata and "verified" in metadata and metadata["verified"]:
            base_score += 0.1
        
        return min(base_score, 1.0)

    async def _save_session_memory(self, session: OrbSession):
        """Save session context to long-term memory."""
        # Create session memory fragment
        session_data = {
            "user_id": session.user_id,
            "duration": "session_duration",  # Would calculate actual duration
            "message_count": len(session.chat_messages),
            "panels_used": [panel.panel_type.value for panel in session.active_panels],
            "key_topics": []  # Would extract from chat
        }
        
        fragment_id = f"session_{session.session_id}"
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            content=f"Session summary: {session_data}",
            fragment_type="session",
            source="session_memory",
            trust_score=0.9,
            timestamp=datetime.utcnow().isoformat(),
            tags=["session", session.user_id],
            metadata=session_data
        )
        
        self.memory_fragments[fragment_id] = fragment

    def get_orb_stats(self) -> Dict[str, Any]:
        """Get comprehensive orb interface statistics."""
        return {
            "sessions": {
                "active": len(self.active_sessions),
                "total_messages": sum(len(s.chat_messages) for s in self.active_sessions.values()),
                "total_panels": sum(len(s.active_panels) for s in self.active_sessions.values())
            },
            "memory": self.get_memory_stats(),
            "governance": {
                "total_tasks": len(self.governance_tasks),
                "pending_tasks": len([t for t in self.governance_tasks.values() if t.status == "pending"])
            },
            "notifications": {
                "total": len(self.notifications),
                "by_priority": {
                    priority.value: len([n for n in self.notifications.values() if n.priority == priority])
                    for priority in NotificationPriority
                }
            },
            "ide": self.grace_ide.get_stats(),
            "intelligence": {
                "version": self.grace_intelligence.version,
                "domain_pods": len(self.grace_intelligence.domain_pods),
                "models_available": len(self.grace_intelligence.model_registry)
            },
            "enhanced_features": {
                "task_items": len(self.task_items),
                "knowledge_entries": len(self.knowledge_entries),
                "memory_explorer_items": len(self.memory_explorer_items),
                "collaboration_sessions": len(self.collaboration_sessions)
            }
        }

    # Enhanced Features Methods
    
    # Knowledge Base & Library Access
    async def create_knowledge_entry(self, title: str, content: str, source: str, 
                                   domain: str, trust_score: float, 
                                   relevance_tags: List[str] = None,
                                   related_libraries: List[str] = None) -> str:
        """Create a new knowledge base entry with library access."""
        entry_id = f"knowledge_{uuid.uuid4().hex[:8]}"
        
        entry = KnowledgeEntry(
            entry_id=entry_id,
            title=title,
            content=content,
            source=source,
            domain=domain,
            trust_score=trust_score,
            relevance_tags=relevance_tags or [],
            created_at=datetime.utcnow().isoformat(),
            last_accessed=datetime.utcnow().isoformat(),
            related_libraries=related_libraries or []
        )
        
        self.knowledge_entries[entry_id] = entry
        logger.info(f"Created knowledge entry {entry_id}: {title}")
        return entry_id

    async def search_knowledge_base(self, query: str, domain: str = None, 
                                  min_trust_score: float = 0.0) -> List[KnowledgeEntry]:
        """Search knowledge base entries."""
        results = []
        query_lower = query.lower()
        
        for entry in self.knowledge_entries.values():
            if domain and entry.domain != domain:
                continue
            if entry.trust_score < min_trust_score:
                continue
                
            # Simple search in title, content, and tags
            if (query_lower in entry.title.lower() or 
                query_lower in entry.content.lower() or 
                any(query_lower in tag.lower() for tag in entry.relevance_tags)):
                
                # Update access count
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow().isoformat()
                results.append(entry)
        
        # Sort by trust score and access count
        results.sort(key=lambda x: (x.trust_score, x.access_count), reverse=True)
        return results

    async def access_library_data(self, library_name: str, topic: str) -> Dict[str, Any]:
        """Access library data for a specific topic (simulation)."""
        # This would integrate with actual library APIs in production
        return {
            "library": library_name,
            "topic": topic,
            "data_points": f"Retrieved {library_name} data for {topic}",
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": 0.85,
            "references": [f"{library_name} documentation", f"{library_name} examples"]
        }

    # Task Box Management  
    async def create_task_item(self, title: str, description: str, priority: str = "medium",
                             assigned_to: str = "grace") -> str:
        """Create a new task item."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = TaskItem(
            task_id=task_id,
            title=title,
            description=description,
            status="pending",
            priority=priority,
            assigned_to=assigned_to,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        self.task_items[task_id] = task
        logger.info(f"Created task {task_id}: {title}")
        return task_id

    async def update_task_status(self, task_id: str, status: str, progress: float = None) -> bool:
        """Update task status and progress."""
        if task_id not in self.task_items:
            return False
        
        task = self.task_items[task_id]
        task.status = status
        task.updated_at = datetime.utcnow().isoformat()
        
        if progress is not None:
            task.progress = min(1.0, max(0.0, progress))
        
        logger.info(f"Updated task {task_id} status to {status}")
        return True

    async def merge_task_data(self, task_id: str, data: Dict[str, Any]) -> bool:
        """Merge relevant information into task's related data."""
        if task_id not in self.task_items:
            return False
        
        task = self.task_items[task_id]
        task.related_data.update(data)
        task.updated_at = datetime.utcnow().isoformat()
        
        logger.info(f"Merged data into task {task_id}")
        return True

    def get_tasks_by_status(self, status: str = None) -> List[TaskItem]:
        """Get tasks filtered by status."""
        if status:
            return [task for task in self.task_items.values() if task.status == status]
        return list(self.task_items.values())

    # Memory Explorer (File System-like)
    async def create_memory_item(self, name: str, item_type: str, content: str = None,
                               parent_id: str = None, is_editable: bool = True) -> str:
        """Create a new memory explorer item."""
        item_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        item = MemoryExplorerItem(
            item_id=item_id,
            name=name,
            item_type=item_type,
            content=content,
            parent_id=parent_id,
            is_editable=is_editable
        )
        
        # Add to parent's children if parent exists
        if parent_id and parent_id in self.memory_explorer_items:
            parent = self.memory_explorer_items[parent_id]
            if item_id not in parent.children:
                parent.children.append(item_id)
        
        self.memory_explorer_items[item_id] = item
        logger.info(f"Created memory item {item_id}: {name}")
        return item_id

    async def update_memory_item_content(self, item_id: str, content: str) -> bool:
        """Update memory item content."""
        if item_id not in self.memory_explorer_items:
            return False
        
        item = self.memory_explorer_items[item_id]
        if not item.is_editable:
            return False
        
        item.content = content
        item.modified_at = datetime.utcnow().isoformat()
        
        logger.info(f"Updated memory item {item_id} content")
        return True

    def get_memory_tree(self, parent_id: str = None) -> List[MemoryExplorerItem]:
        """Get memory items in tree structure."""
        if parent_id:
            return [item for item in self.memory_explorer_items.values() 
                   if item.parent_id == parent_id]
        else:
            # Return root items (no parent)
            return [item for item in self.memory_explorer_items.values() 
                   if item.parent_id is None]

    # Collaboration System
    async def create_collaboration_session(self, topic: str, participants: List[str]) -> str:
        """Create a new collaboration session."""
        session_id = f"collab_{uuid.uuid4().hex[:8]}"
        
        session = CollaborationSession(
            session_id=session_id,
            topic=topic,
            participants=participants,
            status="active"
        )
        
        self.collaboration_sessions[session_id] = session
        logger.info(f"Created collaboration session {session_id}: {topic}")
        return session_id

    async def add_discussion_point(self, session_id: str, author: str, 
                                 point: str, point_type: str = "discussion") -> bool:
        """Add a discussion point to collaboration session."""
        if session_id not in self.collaboration_sessions:
            return False
        
        session = self.collaboration_sessions[session_id]
        discussion_point = {
            "id": f"point_{uuid.uuid4().hex[:6]}",
            "author": author,
            "point": point,
            "type": point_type,
            "timestamp": datetime.utcnow().isoformat(),
            "responses": []
        }
        
        session.discussion_points.append(discussion_point)
        session.updated_at = datetime.utcnow().isoformat()
        
        logger.info(f"Added discussion point to session {session_id}")
        return True

    async def add_action_item(self, session_id: str, title: str, description: str, 
                            assigned_to: str, priority: str = "medium") -> bool:
        """Add an action item to collaboration session."""
        if session_id not in self.collaboration_sessions:
            return False
        
        session = self.collaboration_sessions[session_id]
        action_item = {
            "id": f"action_{uuid.uuid4().hex[:6]}",
            "title": title,
            "description": description,
            "assigned_to": assigned_to,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        
        session.action_items.append(action_item)
        session.updated_at = datetime.utcnow().isoformat()
        
        logger.info(f"Added action item to session {session_id}")
        return True

    # Panel Management for New Features
    async def open_knowledge_base_panel(self, session_id: str) -> str:
        """Open knowledge base panel."""
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.KNOWLEDGE_BASE,
            title="Knowledge Base & Library Access",
            data={
                "total_entries": len(self.knowledge_entries),
                "domains": list(set(entry.domain for entry in self.knowledge_entries.values())),
                "recent_searches": []
            }
        )

    async def open_task_box_panel(self, session_id: str) -> str:
        """Open task box panel."""
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.TASK_BOX,
            title="Task Box",
            data={
                "pending_tasks": len(self.get_tasks_by_status("pending")),
                "in_progress_tasks": len(self.get_tasks_by_status("in_progress")),
                "completed_tasks": len(self.get_tasks_by_status("completed")),
                "task_summary": self.get_tasks_by_status()
            }
        )

    async def open_collaboration_panel(self, session_id: str, collab_session_id: str = None) -> str:
        """Open collaboration panel."""
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.COLLABORATION,
            title="Collaboration Hub",
            data={
                "collaboration_session_id": collab_session_id,
                "active_sessions": len([s for s in self.collaboration_sessions.values() 
                                     if s.status == "active"]),
                "discussion_points": 0 if not collab_session_id else 
                    len(self.collaboration_sessions.get(collab_session_id, CollaborationSession("", "", [], "")).discussion_points)
            }
        )

    async def open_memory_explorer_panel(self, session_id: str) -> str:
        """Open memory explorer panel."""
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.MEMORY,
            title="Memory Explorer",
            data={
                "total_items": len(self.memory_explorer_items),
                "folder_count": len([item for item in self.memory_explorer_items.values() 
                                   if item.item_type == "folder"]),
                "file_count": len([item for item in self.memory_explorer_items.values() 
                                 if item.item_type == "file"]),
                "root_items": self.get_memory_tree()
            }
        )