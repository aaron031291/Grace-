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
from .multimodal_interface import MultimodalInterface

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
    SCREEN_SHARE = "screen_share"
    RECORDING = "recording"
    VOICE_CONTROL = "voice_control"


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
    requester_id: str
    assignee_id: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "pending"
    details: Dict[str, Any] = field(default_factory=dict)


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
        self.multimodal_interface = MultimodalInterface(self)
        
        # Session management
        self.active_sessions: Dict[str, OrbSession] = {}
        
        # Global state
        self.memory_fragments: Dict[str, MemoryFragment] = {}
        self.governance_tasks: Dict[str, GovernanceTask] = {}
        self.notifications: Dict[str, OrbNotification] = {}
        
        # System capabilities
        self.max_panels_per_session = 6
        self.supported_file_types = [
            "pdf", "doc", "docx", "txt", "csv", "json", "xml",
            "py", "js", "html", "css", "md",
            "jpg", "png", "gif", "mp4", "mp3", "wav",
            "screen_recording", "audio_recording", "video_recording"
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
            "screen_share_panel": {
                "title": "Screen Share",
                "default_size": {"width": 800, "height": 600},
                "components": ["video_stream", "controls", "participants"],
                "refresh_interval": 0
            },
            "recording_panel": {
                "title": "Recording Studio",
                "default_size": {"width": 600, "height": 450},
                "components": ["recording_controls", "preview", "settings"],
                "refresh_interval": 0
            },
            "voice_control_panel": {
                "title": "Voice Control",
                "default_size": {"width": 400, "height": 300},
                "components": ["voice_settings", "speech_recognition", "tts_controls"],
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
            "multimodal": self.multimodal_interface.get_stats()
        }

    # Multimodal Interface Methods
    
    async def start_screen_share(self, user_id: str, quality_settings: Optional[Dict[str, Any]] = None) -> str:
        """Start screen sharing session."""
        return await self.multimodal_interface.start_screen_share(user_id, quality_settings)
    
    async def stop_screen_share(self, session_id: str) -> bool:
        """Stop screen sharing session."""
        return await self.multimodal_interface.stop_screen_share(session_id)
    
    async def start_recording(self, user_id: str, media_type: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start recording (audio, video, or screen)."""
        from .multimodal_interface import MediaType
        
        # Convert string to MediaType enum
        media_type_enum = MediaType(media_type.lower())
        return await self.multimodal_interface.start_recording(user_id, media_type_enum, metadata)
    
    async def stop_recording(self, session_id: str) -> Dict[str, Any]:
        """Stop recording and return session info."""
        return await self.multimodal_interface.stop_recording(session_id)
    
    async def set_voice_settings(self, user_id: str, settings: Dict[str, Any]):
        """Set voice input/output settings for user."""
        await self.multimodal_interface.set_voice_settings(user_id, settings)
    
    async def toggle_voice(self, user_id: str, enable: bool) -> bool:
        """Toggle voice input/output for user."""
        return await self.multimodal_interface.toggle_voice(user_id, enable)
    
    async def queue_background_task(self, task_type: str, metadata: Dict[str, Any]) -> str:
        """Queue a background processing task."""
        return await self.multimodal_interface.queue_background_task(task_type, metadata)
    
    def get_active_media_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active media sessions."""
        return self.multimodal_interface.get_active_sessions(user_id)
    
    def get_background_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of background task."""
        return self.multimodal_interface.get_background_task_status(task_id)
    
    def get_voice_settings(self, user_id: str) -> Dict[str, Any]:
        """Get voice settings for user."""
        return self.multimodal_interface.get_voice_settings(user_id)