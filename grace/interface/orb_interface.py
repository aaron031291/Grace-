"""
Grace Unified Orb Interface - Main User Interface
Comprehensive interface with chat, panels, memory management, multimodal, and governance.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# External components (assumed to exist in your repo)
from ..intelligence.grace_intelligence import (
    GraceIntelligence,
    ReasoningContext,
    ReasoningResult,
)
from .ide.grace_ide import GraceIDE
from .multimodal_interface import MultimodalInterface
from .enum_utils import create_enum_mapper
from .job_queue import job_queue, JobPriority
from ..gtrace import get_tracer, MemoryTracer

logger = logging.getLogger(__name__)


class PanelType(Enum):
    """Types of UI panels."""

    # Core
    CHAT = "chat"
    TRADING = "trading"
    SALES = "sales"
    ANALYTICS = "analytics"
    MEMORY = "memory"
    GOVERNANCE = "governance"
    TASK_MANAGER = "task_manager"
    IDE = "ide"
    DASHBOARD = "dashboard"
    # Enhanced features
    KNOWLEDGE_BASE = "knowledge_base"
    TASK_BOX = "task_box"
    COLLABORATION = "collaboration"
    LIBRARY_ACCESS = "library_access"
    # Multimodal
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
    read_at: Optional[str] = None  # Track when notification was read


@dataclass
class MemoryFragment:
    """Fragment stored in memory."""

    fragment_id: str
    content: str
    fragment_type: str  # text, code, audio, video, document, session
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
    requester_id: Optional[str] = None
    assignee_id: Optional[str] = None
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
        self.multimodal_interface = MultimodalInterface(self)

        # Initialize tracing
        self.tracer = get_tracer()
        self.memory_tracer = MemoryTracer(self.tracer)

        # Session management
        self.active_sessions: Dict[str, OrbSession] = {}

        # Global state
        self.memory_fragments: Dict[str, MemoryFragment] = {}
        self.governance_tasks: Dict[str, GovernanceTask] = {}
        self.notifications: Dict[str, OrbNotification] = {}

        # Enhanced features storage
        self.task_items: Dict[str, TaskItem] = {}
        self.knowledge_entries: Dict[str, KnowledgeEntry] = {}
        self.memory_explorer_items: Dict[str, MemoryExplorerItem] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}

        # System capabilities
        self.max_panels_per_session = 6
        self.supported_file_types = [
            "pdf",
            "doc",
            "docx",
            "txt",
            "csv",
            "json",
            "xml",
            "py",
            "js",
            "html",
            "css",
            "md",
            "jpg",
            "png",
            "gif",
            "mp4",
            "mp3",
            "wav",
            "screen_recording",
            "audio_recording",
            "video_recording",
        ]

        # Panel templates
        self.panel_templates = self._initialize_panel_templates()

        # Create safe enum mappers for robust parsing
        self.panel_type_mapper = create_enum_mapper(
            {
                "trading_panel": PanelType.TRADING,
                "sales_panel": PanelType.SALES,
                "chart_panel": PanelType.ANALYTICS,
                "analytics_panel": PanelType.ANALYTICS,
                "memory_explorer_panel": PanelType.MEMORY,
                "governance_panel": PanelType.GOVERNANCE,
                "task_panel": PanelType.TASK_MANAGER,
                "knowledge_panel": PanelType.KNOWLEDGE_BASE,
                "collab_panel": PanelType.COLLABORATION,
            },
            PanelType,
            PanelType.ANALYTICS,
        )

        self.notification_priority_mapper = create_enum_mapper(
            {}, NotificationPriority, NotificationPriority.MEDIUM
        )

        logger.info("Grace Unified Orb Interface initialized")

    # -----------------------------
    # Panel templates
    # -----------------------------
    def _initialize_panel_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize panel templates for different data types."""
        return {
            "trading_panel": {
                "title": "Trading Analysis",
                "default_size": {"width": 500, "height": 400},
                "components": ["price_chart", "indicators", "order_book"],
                "refresh_interval": 5,
            },
            "sales_panel": {
                "title": "Sales Dashboard",
                "default_size": {"width": 450, "height": 350},
                "components": ["pipeline", "metrics", "recent_activities"],
                "refresh_interval": 30,
            },
            "analytics_panel": {
                "title": "Analytics",
                "default_size": {"width": 600, "height": 450},
                "components": ["charts", "metrics", "filters"],
                "refresh_interval": 60,
            },
            "memory_browser": {
                "title": "Memory Browser",
                "default_size": {"width": 400, "height": 500},
                "components": ["search", "fragment_list", "filters"],
                "refresh_interval": 0,
            },
            "governance_center": {
                "title": "Governance Center",
                "default_size": {"width": 500, "height": 400},
                "components": ["pending_approvals", "audit_trail", "policies"],
                "refresh_interval": 15,
            },
            "knowledge_base_panel": {
                "title": "Knowledge Base & Library Access",
                "default_size": {"width": 600, "height": 500},
                "components": [
                    "search",
                    "library_browser",
                    "knowledge_graph",
                    "access_logs",
                ],
                "refresh_interval": 0,
            },
            "task_box_panel": {
                "title": "Task Box",
                "default_size": {"width": 400, "height": 600},
                "components": [
                    "active_tasks",
                    "completed_tasks",
                    "task_filters",
                    "merge_tools",
                ],
                "refresh_interval": 10,
            },
            "collaboration_panel": {
                "title": "Collaboration Hub",
                "default_size": {"width": 800, "height": 600},
                "components": [
                    "discussion_board",
                    "action_items",
                    "shared_workspace",
                    "development_notes",
                ],
                "refresh_interval": 5,
            },
            "memory_explorer_panel": {
                "title": "Memory Explorer",
                "default_size": {"width": 500, "height": 700},
                "components": [
                    "file_tree",
                    "content_viewer",
                    "editor",
                    "metadata_panel",
                ],
                "refresh_interval": 0,
            },
            "screen_share_panel": {
                "title": "Screen Share",
                "default_size": {"width": 800, "height": 600},
                "components": ["video_stream", "controls", "participants"],
                "refresh_interval": 0,
            },
            "recording_panel": {
                "title": "Recording Studio",
                "default_size": {"width": 600, "height": 450},
                "components": ["recording_controls", "preview", "settings"],
                "refresh_interval": 0,
            },
            "voice_control_panel": {
                "title": "Voice Control",
                "default_size": {"width": 400, "height": 300},
                "components": ["voice_settings", "speech_recognition", "tts_controls"],
                "refresh_interval": 0,
            },
        }

    # -----------------------------
    # Session Management
    # -----------------------------
    async def create_session(
        self, user_id: str, preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new orb session."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        session = OrbSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow().isoformat(),
            last_activity=datetime.utcnow().isoformat(),
            preferences=preferences or {},
        )

        # Default chat panel
        chat_panel = UIPanel(
            panel_id=f"chat_{session_id}",
            panel_type=PanelType.CHAT,
            title="Grace Chat",
            position={"x": 50, "y": 50, "width": 400, "height": 500},
            is_closable=False,
        )
        session.active_panels.append(chat_panel)

        self.active_sessions[session_id] = session

        # Welcome
        await self.send_system_message(
            session_id, "Hello! I'm Grace. How can I help you today?"
        )
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def end_session(self, session_id: str) -> bool:
        """End an orb session."""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        await self._save_session_memory(session)
        del self.active_sessions[session_id]
        logger.info(f"Ended session {session_id}")
        return True

    def get_session(self, session_id: str) -> Optional[OrbSession]:
        return self.active_sessions.get(session_id)

    async def update_session_activity(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions[
                session_id
            ].last_activity = datetime.utcnow().isoformat()

    # -----------------------------
    # Chat
    # -----------------------------
    async def send_chat_message(
        self,
        session_id: str,
        content: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        user_message = ChatMessage(
            message_id=message_id,
            user_id=session.user_id,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            message_type="user",
            attachments=attachments or [],
        )
        session.chat_messages.append(user_message)
        await self.update_session_activity(session_id)

        # Process with Intelligence
        try:
            ctx = ReasoningContext(
                user_id=session.user_id,
                session_id=session_id,
                metadata={"chat_history": len(session.chat_messages)},
            )
            result: ReasoningResult = await self.grace_intelligence.process_request(
                content, ctx
            )

            response_message = ChatMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                user_id="grace",
                content=result.response,
                timestamp=datetime.utcnow().isoformat(),
                message_type="assistant",
                reasoning_trace_id=f"trace_{message_id}",
            )
            session.chat_messages.append(response_message)

            # UI instructions
            if getattr(result, "ui_instructions", None):
                await self._process_ui_instructions(session_id, result.ui_instructions)

            # Store reasoning trace
            session.context_memory[f"trace_{message_id}"] = getattr(
                result, "reasoning_trace", {}
            )

            return response_message.message_id

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_message = ChatMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                user_id="grace",
                content=f"I encountered an error while processing your request: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
                message_type="system",
            )
            session.chat_messages.append(error_message)
            return error_message.message_id

    async def send_system_message(self, session_id: str, content: str) -> str:
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        system_message = ChatMessage(
            message_id=message_id,
            user_id="system",
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            message_type="system",
        )
        session.chat_messages.append(system_message)
        return message_id

    def get_chat_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        if session_id not in self.active_sessions:
            return []
        msgs = self.active_sessions[session_id].chat_messages
        return msgs[-limit:] if limit else msgs

    # -----------------------------
    # Panels
    # -----------------------------
    async def create_panel(
        self,
        session_id: str,
        panel_type: PanelType,
        title: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, float]] = None,
    ) -> str:
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        if len(session.active_panels) >= self.max_panels_per_session:
            raise ValueError(
                f"Maximum panels ({self.max_panels_per_session}) already open"
            )

        panel_id = f"panel_{uuid.uuid4().hex[:8]}"

        template_key = (
            f"{panel_type.value}_panel"
            if f"{panel_type.value}_panel" in self.panel_templates
            else "analytics_panel"
        )
        template = self.panel_templates.get(template_key, {})

        if not position:
            base_x = 100 + (len(session.active_panels) % 3) * 200
            base_y = 100 + (len(session.active_panels) // 3) * 200
            default_size = template.get("default_size", {"width": 400, "height": 300})
            position = {
                "x": base_x,
                "y": base_y,
                "width": default_size["width"],
                "height": default_size["height"],
            }

        panel = UIPanel(
            panel_id=panel_id,
            panel_type=panel_type,
            title=title or template.get("title", f"{panel_type.value.title()} Panel"),
            position=position,
            data=data or {},
            z_index=len(session.active_panels) + 1,
        )
        session.active_panels.append(panel)
        await self.update_session_activity(session_id)
        logger.info(
            f"Created {panel_type.value} panel {panel_id} in session {session_id}"
        )
        return panel_id

    async def close_panel(self, session_id: str, panel_id: str) -> bool:
        if session_id not in self.active_sessions:
            return False
        session = self.active_sessions[session_id]
        for i, panel in enumerate(session.active_panels):
            if panel.panel_id == panel_id and panel.is_closable:
                del session.active_panels[i]
                await self.update_session_activity(session_id)
                logger.info(f"Closed panel {panel_id} in session {session_id}")
                return True
        return False

    async def update_panel_data(
        self, session_id: str, panel_id: str, data: Dict[str, Any]
    ) -> bool:
        if session_id not in self.active_sessions:
            return False
        session = self.active_sessions[session_id]
        for panel in session.active_panels:
            if panel.panel_id == panel_id:
                panel.data.update(data)
                await self.update_session_activity(session_id)
                return True
        return False

    async def move_panel(
        self, session_id: str, panel_id: str, position: Dict[str, float]
    ) -> bool:
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
        if session_id not in self.active_sessions:
            return []
        return self.active_sessions[session_id].active_panels

    # -----------------------------
    # Memory
    # -----------------------------
    async def upload_document(
        self,
        user_id: str,
        file_path: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload and process a document into memory with background processing."""

        # Start copilot operation trace
        async with self.tracer.async_span(
            "copilot.upload_document",
            tags={
                "user.id": user_id,
                "file.path": file_path,
                "file.type": file_type,
                "operation.type": "document_upload",
            },
        ) as upload_span:
            if file_type.lower() not in self.supported_file_types:
                upload_span.set_tag("error.unsupported_type", file_type)
                raise ValueError(f"Unsupported file type: {file_type}")

            fragment_id = f"fragment_{uuid.uuid4().hex[:8]}"
            upload_span.set_tag("fragment.id", fragment_id)

            # Process document with tracing
            async with self.tracer.async_span(
                "copilot.process_document",
                parent_context=upload_span.context,
                tags={"file.type": file_type},
            ) as process_span:
                content = await self._process_document(file_path, file_type)
                process_span.set_tag("content.length", len(content))

            trust_score = self._calculate_document_trust_score(
                file_path, file_type, metadata
            )
            upload_span.set_tag("trust.score", trust_score)

            fragment = MemoryFragment(
                fragment_id=fragment_id,
                content=content,
                fragment_type=file_type,
                source=f"upload:{file_path}",
                trust_score=trust_score,
                timestamp=datetime.utcnow().isoformat(),
                tags=metadata.get("tags", []) if metadata else [],
                metadata={
                    **(metadata or {}),
                    "trace_id": upload_span.context.trace_id,
                    "copilot_operation": "document_upload",
                },
            )

            # Store fragment immediately
            self.memory_fragments[fragment_id] = fragment

            # Queue background ingest processing (idempotent)
            async with self.tracer.async_span(
                "copilot.queue_background_job", parent_context=upload_span.context
            ) as job_span:
                job_id = await job_queue.enqueue(
                    job_type="document_ingest",
                    payload={
                        "file_path": file_path,
                        "file_type": file_type,
                        "user_id": user_id,
                        "fragment_id": fragment_id,
                        "trace_id": upload_span.context.trace_id,
                    },
                    priority=JobPriority.MEDIUM,
                )

                job_span.set_tag("job.id", job_id)

            # Store job ID in fragment metadata for tracking
            fragment.metadata["background_job_id"] = job_id

            upload_span.log(
                "document_uploaded",
                {
                    "fragment_id": fragment_id,
                    "job_id": job_id,
                    "content_length": len(content),
                },
            )

            logger.info(
                f"Uploaded document {file_path} as fragment {fragment_id}, queued job {job_id}"
            )
            return fragment_id

    async def search_memory(
        self, session_id: str, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryFragment]:
        """Basic search over stored fragments."""

        # Start copilot memory search trace
        async with self.tracer.async_span(
            "copilot.search_memory",
            tags={
                "session.id": session_id,
                "search.query": query[:100],  # Truncate long queries
                "search.query_length": len(query),
                "operation.type": "memory_search",
            },
        ) as search_span:
            results: List[MemoryFragment] = []
            q = query.lower()
            search_span.set_tag("fragments.total", len(self.memory_fragments))

            # Apply filters and search
            filtered_count = 0
            for fragment in self.memory_fragments.values():
                if q in fragment.content.lower():
                    if filters:
                        if (
                            "fragment_type" in filters
                            and fragment.fragment_type != filters["fragment_type"]
                        ):
                            continue
                        if (
                            "min_trust_score" in filters
                            and fragment.trust_score < filters["min_trust_score"]
                        ):
                            continue
                        if "tags" in filters and not any(
                            tag in fragment.tags for tag in filters["tags"]
                        ):
                            continue
                    results.append(fragment)
                    filtered_count += 1

            # Sort by trust score
            results.sort(key=lambda f: f.trust_score, reverse=True)
            final_results = results[:20]

            # Add search result metrics to trace
            search_span.set_tag("search.matches_found", len(results))
            search_span.set_tag("search.results_returned", len(final_results))
            search_span.set_tag("search.filtered_count", filtered_count)

            if filters:
                search_span.set_tag("search.filters_applied", len(filters))
                for key, value in filters.items():
                    search_span.set_tag(f"filter.{key}", str(value))

            search_span.log(
                "memory_search_completed",
                {
                    "query": query[:100],
                    "matches": len(results),
                    "returned": len(final_results),
                    "avg_trust_score": sum(f.trust_score for f in final_results)
                    / len(final_results)
                    if final_results
                    else 0,
                },
            )

            return final_results

    def get_memory_stats(self) -> Dict[str, Any]:
        stats = {
            "total_fragments": len(self.memory_fragments),
            "fragments_by_type": {},
            "average_trust_score": 0.0,
            "total_size": 0,
        }
        if self.memory_fragments:
            scores = []
            for f in self.memory_fragments.values():
                stats["fragments_by_type"][f.fragment_type] = (
                    stats["fragments_by_type"].get(f.fragment_type, 0) + 1
                )
                scores.append(f.trust_score)
                stats["total_size"] += len(f.content)
            stats["average_trust_score"] = sum(scores) / len(scores)
        return stats

    # -----------------------------
    # Governance
    # -----------------------------
    async def create_governance_task(
        self,
        title: str,
        description: str,
        task_type: str,
        requester_id: str,
        assignee_id: Optional[str] = None,
    ) -> str:
        task_id = f"gov_task_{uuid.uuid4().hex[:8]}"
        task = GovernanceTask(
            task_id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority="medium",
            requester_id=requester_id,
            assignee_id=assignee_id,
        )
        self.governance_tasks[task_id] = task

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
                    {"label": "Reject", "action": f"governance_reject:{task_id}"},
                ],
            )
        logger.info(f"Created governance task {task_id}: {title}")
        return task_id

    def get_governance_tasks(
        self, user_id: str, status_filter: Optional[str] = None
    ) -> List[GovernanceTask]:
        tasks: List[GovernanceTask] = []
        for t in self.governance_tasks.values():
            if t.assignee_id == user_id or t.requester_id == user_id:
                if not status_filter or t.status == status_filter:
                    tasks.append(t)
        return tasks

    async def update_governance_task_status(
        self, task_id: str, status: str, user_id: str
    ) -> bool:
        if task_id not in self.governance_tasks:
            return False
        t = self.governance_tasks[task_id]
        if t.assignee_id != user_id:
            return False
        t.status = status
        t.updated_at = datetime.utcnow().isoformat()
        await self.create_notification(
            user_id=t.requester_id,
            title="Task Status Updated",
            message=f"Task '{t.title}' status changed to: {status}",
            priority=NotificationPriority.MEDIUM,
        )
        return True

    # -----------------------------
    # Notifications
    # -----------------------------
    async def create_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        priority: NotificationPriority,
        action_required: bool = False,
        actions: Optional[List[Dict[str, str]]] = None,
        auto_dismiss_seconds: Optional[int] = None,
    ) -> str:
        notif_id = f"notif_{uuid.uuid4().hex[:8]}"
        notif = OrbNotification(
            notification_id=notif_id,
            title=title,
            message=message,
            priority=priority,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            action_required=action_required,
            actions=actions or [],
            auto_dismiss_seconds=auto_dismiss_seconds,
        )
        self.notifications[notif_id] = notif
        logger.info(f"Created notification {notif_id} for user {user_id}: {title}")
        return notif_id

    def get_notifications(
        self, user_id: str, unread_only: bool = True
    ) -> List[OrbNotification]:
        # Filter notifications by user and read status
        items = [n for n in self.notifications.values() if n.user_id == user_id]

        if unread_only:
            items = [n for n in items if n.read_at is None]

        items.sort(key=lambda n: (n.priority.value, n.timestamp), reverse=True)
        return items

    async def dismiss_notification(self, notification_id: str, user_id: str) -> bool:
        if notification_id not in self.notifications:
            return False
        notif = self.notifications[notification_id]
        if notif.user_id != user_id:
            return False
        del self.notifications[notification_id]
        return True

    async def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark a notification as read without dismissing it."""
        if notification_id not in self.notifications:
            return False
        notif = self.notifications[notification_id]
        if notif.user_id != user_id:
            return False

        if notif.read_at is None:  # Only update if not already read
            notif.read_at = datetime.utcnow().isoformat()
            logger.info(
                f"Marked notification {notification_id} as read for user {user_id}"
            )

        return True

    # -----------------------------
    # IDE
    # -----------------------------
    async def open_ide_panel(
        self, session_id: str, flow_id: Optional[str] = None
    ) -> str:
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.IDE,
            title="Grace IDE",
            data={
                "flow_id": flow_id,
                "ide_stats": self.grace_ide.get_stats(),
                "block_registry": self.grace_ide.get_block_registry(),
            },
        )

    def get_ide_instance(self) -> GraceIDE:
        return self.grace_ide

    # -----------------------------
    # UI Helpers
    # -----------------------------
    async def _process_ui_instructions(
        self, session_id: str, ui_instructions: Dict[str, Any]
    ):
        if "panels" in ui_instructions:
            for panel_config in ui_instructions["panels"]:
                panel_type_str = panel_config.get("type", "analytics")
                # Use safe enum mapping with fallback
                panel_type = self.panel_type_mapper(panel_type_str)

                await self.create_panel(
                    session_id=session_id,
                    panel_type=panel_type,
                    title=panel_config.get("title"),
                    data=panel_config.get("data", {}),
                )

    async def _process_document(self, file_path: str, file_type: str) -> str:
        """Very simple document processing; replace with your real ingestion."""
        if file_type.lower() in ["txt", "md"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {file_path} ({e})"
        return f"Binary file: {file_path} ({file_type})"

    def _calculate_document_trust_score(
        self, file_path: str, file_type: str, metadata: Optional[Dict[str, Any]]
    ) -> float:
        base = 0.8
        if file_type.lower() in ["pdf", "doc", "docx"]:
            base += 0.1
        if metadata and metadata.get("verified"):
            base += 0.1
        return min(base, 1.0)

    async def _save_session_memory(self, session: OrbSession):
        session_data = {
            "user_id": session.user_id,
            "duration": "session_duration",  # TODO: compute actual duration
            "message_count": len(session.chat_messages),
            "panels_used": [p.panel_type.value for p in session.active_panels],
            "key_topics": [],  # TODO: extract key topics
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
            metadata=session_data,
        )
        self.memory_fragments[fragment_id] = fragment

    # -----------------------------
    # Stats
    # -----------------------------
    def get_orb_stats(self) -> Dict[str, Any]:
        return {
            "sessions": {
                "active": len(self.active_sessions),
                "total_messages": sum(
                    len(s.chat_messages) for s in self.active_sessions.values()
                ),
                "total_panels": sum(
                    len(s.active_panels) for s in self.active_sessions.values()
                ),
            },
            "memory": self.get_memory_stats(),
            "governance": {
                "total_tasks": len(self.governance_tasks),
                "pending_tasks": len(
                    [t for t in self.governance_tasks.values() if t.status == "pending"]
                ),
            },
            "notifications": {
                "total": len(self.notifications),
                "by_priority": {
                    p.value: len(
                        [n for n in self.notifications.values() if n.priority == p]
                    )
                    for p in NotificationPriority
                },
            },
            "ide": self.grace_ide.get_stats(),
            "intelligence": {
                "version": self.grace_intelligence.version,
                "domain_pods": len(self.grace_intelligence.domain_pods),
                "models_available": len(self.grace_intelligence.model_registry),
            },
            "enhanced_features": {
                "task_items": len(self.task_items),
                "knowledge_entries": len(self.knowledge_entries),
                "memory_explorer_items": len(self.memory_explorer_items),
                "collaboration_sessions": len(self.collaboration_sessions),
            },
            "multimodal": self.multimodal_interface.get_stats(),
        }

    # -----------------------------
    # Enhanced Features Methods
    # -----------------------------
    # Knowledge Base & Library Access
    async def create_knowledge_entry(
        self,
        title: str,
        content: str,
        source: str,
        domain: str,
        trust_score: float,
        relevance_tags: Optional[List[str]] = None,
        related_libraries: Optional[List[str]] = None,
    ) -> str:
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
            related_libraries=related_libraries or [],
        )
        self.knowledge_entries[entry_id] = entry
        logger.info(f"Created knowledge entry {entry_id}: {title}")
        return entry_id

    async def search_knowledge_base(
        self, query: str, domain: Optional[str] = None, min_trust_score: float = 0.0
    ) -> List[KnowledgeEntry]:
        results: List[KnowledgeEntry] = []
        q = query.lower()
        for e in self.knowledge_entries.values():
            if domain and e.domain != domain:
                continue
            if e.trust_score < min_trust_score:
                continue
            if (
                q in e.title.lower()
                or q in e.content.lower()
                or any(q in tag.lower() for tag in e.relevance_tags)
            ):
                e.access_count += 1
                e.last_accessed = datetime.utcnow().isoformat()
                results.append(e)
        results.sort(key=lambda x: (x.trust_score, x.access_count), reverse=True)
        return results

    async def access_library_data(
        self, library_name: str, topic: str
    ) -> Dict[str, Any]:
        # Placeholder; integrate real library clients here
        return {
            "library": library_name,
            "topic": topic,
            "data_points": f"Retrieved {library_name} data for {topic}",
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": 0.85,
            "references": [f"{library_name} documentation", f"{library_name} examples"],
        }

    # Task Box
    async def create_task_item(
        self,
        title: str,
        description: str,
        priority: str = "medium",
        assigned_to: str = "grace",
    ) -> str:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = TaskItem(
            task_id=task_id,
            title=title,
            description=description,
            status="pending",
            priority=priority,
            assigned_to=assigned_to,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        self.task_items[task_id] = task
        logger.info(f"Created task {task_id}: {title}")
        return task_id

    async def update_task_status(
        self, task_id: str, status: str, progress: Optional[float] = None
    ) -> bool:
        if task_id not in self.task_items:
            return False
        t = self.task_items[task_id]
        t.status = status
        t.updated_at = datetime.utcnow().isoformat()
        if progress is not None:
            t.progress = min(1.0, max(0.0, progress))
        logger.info(f"Updated task {task_id} status to {status}")
        return True

    async def merge_task_data(self, task_id: str, data: Dict[str, Any]) -> bool:
        if task_id not in self.task_items:
            return False
        t = self.task_items[task_id]
        t.related_data.update(data)
        t.updated_at = datetime.utcnow().isoformat()
        logger.info(f"Merged data into task {task_id}")
        return True

    def get_tasks_by_status(self, status: Optional[str] = None) -> List[TaskItem]:
        if status:
            return [t for t in self.task_items.values() if t.status == status]
        return list(self.task_items.values())

    # Memory Explorer
    async def create_memory_item(
        self,
        name: str,
        item_type: str,
        content: Optional[str] = None,
        parent_id: Optional[str] = None,
        is_editable: bool = True,
    ) -> str:
        item_id = f"mem_{uuid.uuid4().hex[:8]}"
        item = MemoryExplorerItem(
            item_id=item_id,
            name=name,
            item_type=item_type,
            content=content,
            parent_id=parent_id,
            is_editable=is_editable,
        )
        if parent_id and parent_id in self.memory_explorer_items:
            parent = self.memory_explorer_items[parent_id]
            if item_id not in parent.children:
                parent.children.append(item_id)
        self.memory_explorer_items[item_id] = item
        logger.info(f"Created memory item {item_id}: {name}")
        return item_id

    async def update_memory_item_content(self, item_id: str, content: str) -> bool:
        if item_id not in self.memory_explorer_items:
            return False
        item = self.memory_explorer_items[item_id]
        if not item.is_editable:
            return False
        item.content = content
        item.modified_at = datetime.utcnow().isoformat()
        logger.info(f"Updated memory item {item_id} content")
        return True

    def get_memory_tree(
        self, parent_id: Optional[str] = None
    ) -> List[MemoryExplorerItem]:
        if parent_id:
            return [
                i
                for i in self.memory_explorer_items.values()
                if i.parent_id == parent_id
            ]
        return [i for i in self.memory_explorer_items.values() if i.parent_id is None]

    # Collaboration
    async def create_collaboration_session(
        self, topic: str, participants: List[str]
    ) -> str:
        collab_id = f"collab_{uuid.uuid4().hex[:8]}"
        sess = CollaborationSession(
            session_id=collab_id,
            topic=topic,
            participants=participants,
            status="active",
        )
        self.collaboration_sessions[collab_id] = sess
        logger.info(f"Created collaboration session {collab_id}: {topic}")
        return collab_id

    async def add_discussion_point(
        self, session_id: str, author: str, point: str, point_type: str = "discussion"
    ) -> bool:
        if session_id not in self.collaboration_sessions:
            return False
        sess = self.collaboration_sessions[session_id]
        sess.discussion_points.append(
            {
                "id": f"point_{uuid.uuid4().hex[:6]}",
                "author": author,
                "point": point,
                "type": point_type,
                "timestamp": datetime.utcnow().isoformat(),
                "responses": [],
            }
        )
        sess.updated_at = datetime.utcnow().isoformat()
        logger.info(f"Added discussion point to session {session_id}")
        return True

    async def add_action_item(
        self,
        session_id: str,
        title: str,
        description: str,
        assigned_to: str,
        priority: str = "medium",
    ) -> bool:
        if session_id not in self.collaboration_sessions:
            return False
        sess = self.collaboration_sessions[session_id]
        sess.action_items.append(
            {
                "id": f"action_{uuid.uuid4().hex[:6]}",
                "title": title,
                "description": description,
                "assigned_to": assigned_to,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        sess.updated_at = datetime.utcnow().isoformat()
        logger.info(f"Added action item to session {session_id}")
        return True

    # Panels for enhanced features
    async def open_knowledge_base_panel(self, session_id: str) -> str:
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.KNOWLEDGE_BASE,
            title="Knowledge Base & Library Access",
            data={
                "total_entries": len(self.knowledge_entries),
                "domains": list({e.domain for e in self.knowledge_entries.values()}),
                "recent_searches": [],
            },
        )

    async def open_task_box_panel(self, session_id: str) -> str:
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.TASK_BOX,
            title="Task Box",
            data={
                "pending_tasks": len(self.get_tasks_by_status("pending")),
                "in_progress_tasks": len(self.get_tasks_by_status("in_progress")),
                "completed_tasks": len(self.get_tasks_by_status("completed")),
                "task_summary": self.get_tasks_by_status(),
            },
        )

    async def open_collaboration_panel(
        self, session_id: str, collab_session_id: Optional[str] = None
    ) -> str:
        discussion_count = 0
        if collab_session_id and collab_session_id in self.collaboration_sessions:
            discussion_count = len(
                self.collaboration_sessions[collab_session_id].discussion_points
            )
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.COLLABORATION,
            title="Collaboration Hub",
            data={
                "collaboration_session_id": collab_session_id,
                "active_sessions": len(
                    [
                        s
                        for s in self.collaboration_sessions.values()
                        if s.status == "active"
                    ]
                ),
                "discussion_points": discussion_count,
            },
        )

    async def open_memory_explorer_panel(self, session_id: str) -> str:
        return await self.create_panel(
            session_id=session_id,
            panel_type=PanelType.MEMORY,
            title="Memory Explorer",
            data={
                "total_items": len(self.memory_explorer_items),
                "folder_count": len(
                    [
                        i
                        for i in self.memory_explorer_items.values()
                        if i.item_type == "folder"
                    ]
                ),
                "file_count": len(
                    [
                        i
                        for i in self.memory_explorer_items.values()
                        if i.item_type == "file"
                    ]
                ),
                "root_items": self.get_memory_tree(),
            },
        )

    # -----------------------------
    # Multimodal Methods
    # -----------------------------
    async def start_screen_share(
        self, user_id: str, quality_settings: Optional[Dict[str, Any]] = None
    ) -> str:
        return await self.multimodal_interface.start_screen_share(
            user_id, quality_settings
        )

    async def stop_screen_share(self, session_id: str) -> bool:
        return await self.multimodal_interface.stop_screen_share(session_id)

    async def start_recording(
        self, user_id: str, media_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        from .multimodal_interface import MediaType

        media_type_enum = MediaType(media_type.lower())
        return await self.multimodal_interface.start_recording(
            user_id, media_type_enum, metadata
        )

    async def stop_recording(self, session_id: str) -> Dict[str, Any]:
        return await self.multimodal_interface.stop_recording(session_id)

    async def set_voice_settings(self, user_id: str, settings: Dict[str, Any]):
        await self.multimodal_interface.set_voice_settings(user_id, settings)

    async def toggle_voice(self, user_id: str, enable: bool) -> bool:
        return await self.multimodal_interface.toggle_voice(user_id, enable)

    async def queue_background_task(
        self, task_type: str, metadata: Dict[str, Any]
    ) -> str:
        return await self.multimodal_interface.queue_background_task(
            task_type, metadata
        )

    def get_active_media_sessions(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.multimodal_interface.get_active_sessions(user_id)

    def get_background_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.multimodal_interface.get_background_task_status(task_id)

    def get_voice_settings(self, user_id: str) -> Dict[str, Any]:
        return self.multimodal_interface.get_voice_settings(user_id)
