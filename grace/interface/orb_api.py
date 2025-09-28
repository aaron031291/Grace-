"""
Grace Unified Orb Interface - FastAPI Service
Complete REST API and WebSocket endpoints for the Grace Orb Interface.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .orb_interface import GraceUnifiedOrbInterface, PanelType, NotificationPriority
from .ide.grace_ide import BlockType  # referenced by IDE endpoints

logger = logging.getLogger(__name__)

# ---------------------------
# Pydantic models for API
# ---------------------------

class SessionCreateRequest(BaseModel):
    user_id: str
    preferences: Optional[Dict[str, Any]] = None


class ChatMessageRequest(BaseModel):
    session_id: str
    content: str
    attachments: Optional[List[Dict[str, Any]]] = None


class PanelCreateRequest(BaseModel):
    session_id: str
    panel_type: str
    title: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, float]] = None


class PanelUpdateRequest(BaseModel):
    session_id: str
    panel_id: str
    data: Dict[str, Any]


class MemorySearchRequest(BaseModel):
    session_id: str
    query: str
    filters: Optional[Dict[str, Any]] = None


class GovernanceTaskRequest(BaseModel):
    title: str
    description: str
    task_type: str
    requester_id: str
    assignee_id: Optional[str] = None


class NotificationRequest(BaseModel):
    user_id: str
    title: str
    message: str
    priority: str
    action_required: bool = False
    actions: Optional[List[Dict[str, str]]] = None
    auto_dismiss_seconds: Optional[int] = None


class IDEFlowRequest(BaseModel):
    name: str
    description: str
    creator_id: str
    template_id: Optional[str] = None


class IDEBlockRequest(BaseModel):
    flow_id: str
    block_type_id: str
    position: Dict[str, float]
    name: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None


# ---- Enhanced Features Request Models ----
class KnowledgeEntryRequest(BaseModel):
    title: str
    content: str
    source: str
    domain: str
    trust_score: float
    relevance_tags: Optional[List[str]] = None
    related_libraries: Optional[List[str]] = None


class KnowledgeSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    min_trust_score: float = 0.0


class TaskItemRequest(BaseModel):
    title: str
    description: str
    priority: str = "medium"
    assigned_to: str = "grace"


class TaskUpdateRequest(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None


class TaskDataMergeRequest(BaseModel):
    task_id: str
    data: Dict[str, Any]


class MemoryItemRequest(BaseModel):
    name: str
    item_type: str
    content: Optional[str] = None
    parent_id: Optional[str] = None
    is_editable: bool = True


class MemoryItemUpdateRequest(BaseModel):
    item_id: str
    content: str


class CollaborationSessionRequest(BaseModel):
    topic: str
    participants: List[str]


class DiscussionPointRequest(BaseModel):
    session_id: str
    author: str
    point: str
    point_type: str = "discussion"


class ActionItemRequest(BaseModel):
    session_id: str
    title: str
    description: str
    assigned_to: str
    priority: str = "medium"


# ---- Multimodal Models ----
class ScreenShareRequest(BaseModel):
    user_id: str
    quality_settings: Optional[Dict[str, Any]] = None


class RecordingRequest(BaseModel):
    user_id: str
    media_type: str  # "screen_recording", "audio_recording", "video_recording"
    metadata: Optional[Dict[str, Any]] = None


class VoiceSettingsRequest(BaseModel):
    user_id: str
    settings: Dict[str, Any]


class BackgroundTaskRequest(BaseModel):
    task_type: str
    metadata: Dict[str, Any]


# ---------------------------
# Global orb interface instance
# ---------------------------

orb_interface = GraceUnifiedOrbInterface()

# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(
    title="Grace Unified Orb Interface",
    description="Complete interface for Grace AI system with chat, panels, IDE, and governance",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
ws_connections: Dict[str, WebSocket] = {}

# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Grace Unified Orb Interface",
        "version": "1.0.0",
        "description": "Complete AI interface with chat, panels, IDE, memory, and governance",
        "endpoints": {
            "sessions": "/api/orb/v1/sessions/",
            "chat": "/api/orb/v1/chat/",
            "panels": "/api/orb/v1/panels/",
            "memory": "/api/orb/v1/memory/",
            "governance": "/api/orb/v1/governance/",
            "notifications": "/api/orb/v1/notifications/",
            "ide": "/api/orb/v1/ide/",
            "multimodal": {
                "screen_share": "/api/orb/v1/multimodal/screen-share/",
                "recording": "/api/orb/v1/multimodal/recording/",
                "voice": "/api/orb/v1/multimodal/voice/",
                "background_tasks": "/api/orb/v1/multimodal/tasks/"
            },
            "websocket": "/ws/{session_id}",
            "stats": "/api/orb/v1/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# ---------------------------
# Session Management
# ---------------------------

@app.post("/api/orb/v1/sessions/create")
async def create_session(request: SessionCreateRequest):
    """Create a new orb session."""
    try:
        session_id = await orb_interface.create_session(request.user_id, request.preferences)
        return {"session_id": session_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/orb/v1/sessions/{session_id}")
async def end_session(session_id: str):
    """End an orb session."""
    success = await orb_interface.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ended"}


@app.get("/api/orb/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    session = orb_interface.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "start_time": session.start_time,
        "last_activity": session.last_activity,
        "message_count": len(session.chat_messages),
        "active_panels": len(session.active_panels)
    }

# ---------------------------
# Chat Interface
# ---------------------------

@app.post("/api/orb/v1/chat/message")
async def send_chat_message(request: ChatMessageRequest):
    """Send a chat message."""
    try:
        message_id = await orb_interface.send_chat_message(
            request.session_id,
            request.content,
            request.attachments
        )
        return {"message_id": message_id, "status": "sent"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/chat/{session_id}/history")
async def get_chat_history(session_id: str, limit: Optional[int] = None):
    """Get chat history for session."""
    messages = orb_interface.get_chat_history(session_id, limit)
    return {
        "session_id": session_id,
        "messages": [
            {
                "message_id": msg.message_id,
                "user_id": msg.user_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_type": msg.message_type,
                "attachments": msg.attachments
            }
            for msg in messages
        ]
    }

# ---------------------------
# Panel Management
# ---------------------------

@app.post("/api/orb/v1/panels/create")
async def create_panel(request: PanelCreateRequest):
    """Create a new panel."""
    try:
        # Flexible enum parsing: allow value ("memory_explorer_panel") or name ("MEMORY_EXPLORER_PANEL")
        try:
            panel_type = PanelType(request.panel_type.lower())
        except ValueError:
            panel_type = PanelType[request.panel_type.upper()]

        panel_id = await orb_interface.create_panel(
            request.session_id,
            panel_type,
            request.title,
            request.data,
            request.position
        )
        return {"panel_id": panel_id, "status": "created"}
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown panel_type '{request.panel_type}'")
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/orb/v1/panels/{session_id}/{panel_id}")
async def close_panel(session_id: str, panel_id: str):
    """Close a panel."""
    success = await orb_interface.close_panel(session_id, panel_id)
    if not success:
        raise HTTPException(status_code=404, detail="Panel not found or not closable")
    return {"status": "closed"}


@app.put("/api/orb/v1/panels/update")
async def update_panel_data(request: PanelUpdateRequest):
    """Update panel data."""
    success = await orb_interface.update_panel_data(
        request.session_id,
        request.panel_id,
        request.data
    )
    if not success:
        raise HTTPException(status_code=404, detail="Panel not found")
    return {"status": "updated"}


@app.get("/api/orb/v1/panels/{session_id}")
async def get_panels(session_id: str):
    """Get all panels for session."""
    panels = orb_interface.get_panels(session_id)
    return {
        "session_id": session_id,
        "panels": [
            {
                "panel_id": panel.panel_id,
                "panel_type": panel.panel_type.value,
                "title": panel.title,
                "position": panel.position,
                "data": panel.data,
                "is_closable": panel.is_closable,
                "is_minimized": panel.is_minimized
            }
            for panel in panels
        ]
    }

# ---------------------------
# Memory Management
# ---------------------------

@app.post("/api/orb/v1/memory/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = "default"):
    """Upload a document to memory."""
    try:
        import tempfile
        import os

        # safer suffix handling if filename has no dot
        suffix = ""
        if "." in (file.filename or ""):
            suffix = f".{file.filename.rsplit('.', 1)[-1]}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            file_type = file.filename.rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "bin"
            fragment_id = await orb_interface.upload_document(
                user_id=user_id,
                file_path=temp_file_path,
                file_type=file_type,
                metadata={"original_filename": file.filename, "size": len(content)}
            )
            return {
                "fragment_id": fragment_id,
                "filename": file.filename,
                "file_type": file_type,
                "size": len(content),
                "status": "uploaded"
            }
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orb/v1/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search memory fragments."""
    try:
        results = await orb_interface.search_memory(
            request.session_id,
            request.query,
            request.filters
        )
        return {
            "query": request.query,
            "results": [
                {
                    "fragment_id": fragment.fragment_id,
                    "content": fragment.content[:200] + "..." if len(fragment.content) > 200 else fragment.content,
                    "fragment_type": fragment.fragment_type,
                    "source": fragment.source,
                    "trust_score": fragment.trust_score,
                    "timestamp": fragment.timestamp,
                    "tags": fragment.tags
                }
                for fragment in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/memory/stats")
async def get_memory_stats():
    """Get memory usage statistics."""
    return orb_interface.get_memory_stats()

# ---------------------------
# Governance
# ---------------------------

@app.post("/api/orb/v1/governance/tasks")
async def create_governance_task(request: GovernanceTaskRequest):
    """Create a governance task."""
    try:
        task_id = await orb_interface.create_governance_task(
            request.title,
            request.description,
            request.task_type,
            request.requester_id,
            request.assignee_id
        )
        return {"task_id": task_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/governance/tasks/{user_id}")
async def get_governance_tasks(user_id: str, status_filter: Optional[str] = None):
    """Get governance tasks for user."""
    tasks = orb_interface.get_governance_tasks(user_id, status_filter)
    return {
        "user_id": user_id,
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "task_type": task.task_type,
                "priority": task.priority,
                "status": task.status,
                "requester_id": task.requester_id,
                "assignee_id": task.assignee_id
            }
            for task in tasks
        ]
    }


@app.put("/api/orb/v1/governance/tasks/{task_id}/status")
async def update_governance_task_status(task_id: str, status: str, user_id: str):
    """Update governance task status."""
    success = await orb_interface.update_governance_task_status(task_id, status, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or permission denied")
    return {"status": "updated"}

# ---------------------------
# Notifications
# ---------------------------

@app.post("/api/orb/v1/notifications")
async def create_notification(request: NotificationRequest):
    """Create a notification."""
    try:
        priority = NotificationPriority(request.priority.lower())
        notification_id = await orb_interface.create_notification(
            request.user_id,
            request.title,
            request.message,
            priority,
            request.action_required,
            request.actions,
            request.auto_dismiss_seconds
        )
        return {"notification_id": notification_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/notifications/{user_id}")
async def get_notifications(user_id: str, unread_only: bool = True):
    """Get notifications for user."""
    notifications = orb_interface.get_notifications(user_id, unread_only)
    return {
        "user_id": user_id,
        "notifications": [
            {
                "notification_id": notif.notification_id,
                "title": notif.title,
                "message": notif.message,
                "priority": notif.priority.value,
                "timestamp": notif.timestamp,
                "action_required": notif.action_required,
                "actions": notif.actions
            }
            for notif in notifications
        ]
    }


@app.delete("/api/orb/v1/notifications/{notification_id}")
async def dismiss_notification(notification_id: str, user_id: str):
    """Dismiss a notification."""
    success = await orb_interface.dismiss_notification(notification_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "dismissed"}

# ---------------------------
# IDE Integration
# ---------------------------

@app.post("/api/orb/v1/ide/panels/{session_id}")
async def open_ide_panel(session_id: str, flow_id: Optional[str] = None):
    """Open IDE panel in session."""
    try:
        panel_id = await orb_interface.open_ide_panel(session_id, flow_id)
        return {"panel_id": panel_id, "status": "opened"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orb/v1/ide/flows")
async def create_ide_flow(request: IDEFlowRequest):
    """Create a new IDE flow."""
    try:
        ide = orb_interface.get_ide_instance()
        flow_id = ide.create_flow(
            request.name,
            request.description,
            request.creator_id,
            request.template_id
        )
        return {"flow_id": flow_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/ide/flows/{flow_id}")
async def get_ide_flow(flow_id: str):
    """Get IDE flow details."""
    ide = orb_interface.get_ide_instance()
    flow = ide.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    return {
        "flow_id": flow.flow_id,
        "name": flow.name,
        "description": flow.description,
        "creator_id": flow.creator_id,
        "tags": flow.tags,
        "blocks": [
            {
                "block_id": block.block_id,
                "name": block.name,
                "description": block.description,
                "block_type": block.block_type.value if hasattr(block.block_type, "value") else str(block.block_type),
                "position": block.position,
                "configuration": block.configuration
            }
            for block in flow.blocks
        ],
        "connections": flow.connections
    }


@app.post("/api/orb/v1/ide/flows/blocks")
async def add_block_to_flow(request: IDEBlockRequest):
    """Add a block to IDE flow."""
    try:
        ide = orb_interface.get_ide_instance()
        block_id = ide.add_block_to_flow(
            request.flow_id,
            request.block_type_id,
            request.position,
            request.name,
            request.configuration
        )
        return {"block_id": block_id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/orb/v1/ide/blocks")
async def get_block_registry():
    """Get available IDE blocks."""
    ide = orb_interface.get_ide_instance()
    registry = ide.get_block_registry()
    return {
        "blocks": {
            block_id: {
                "name": block_info["name"],
                "description": block_info["description"],
                "block_type": block_info["block_type"].value if hasattr(block_info["block_type"], "value") else str(block_info["block_type"]),
                "inputs": block_info["inputs"],
                "outputs": block_info["outputs"]
            }
            for block_id, block_info in registry.items()
        }
    }

# ---------------------------
# Multimodal Interface Endpoints
# ---------------------------

@app.post("/api/orb/v1/multimodal/screen-share/start")
async def start_screen_share(request: ScreenShareRequest):
    """Start screen sharing session."""
    try:
        session_id = await orb_interface.start_screen_share(
            request.user_id,
            request.quality_settings
        )
        return {
            "session_id": session_id,
            "status": "started",
            "quality_settings": request.quality_settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orb/v1/multimodal/screen-share/stop/{session_id}")
async def stop_screen_share(session_id: str):
    """Stop screen sharing session."""
    success = await orb_interface.stop_screen_share(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Screen sharing session not found")
    return {"status": "stopped"}


@app.post("/api/orb/v1/multimodal/recording/start")
async def start_recording(request: RecordingRequest):
    """Start recording (audio, video, or screen)."""
    try:
        session_id = await orb_interface.start_recording(
            request.user_id,
            request.media_type,
            request.metadata
        )
        return {
            "session_id": session_id,
            "media_type": request.media_type,
            "status": "started"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orb/v1/multimodal/recording/stop/{session_id}")
async def stop_recording(session_id: str):
    """Stop recording and return session info."""
    try:
        result = await orb_interface.stop_recording(session_id)
        return {"status": "stopped", **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/multimodal/sessions")
async def get_active_media_sessions(user_id: Optional[str] = None):
    """Get active media sessions."""
    sessions = orb_interface.get_active_media_sessions(user_id)
    return {"sessions": sessions, "total": len(sessions)}


@app.post("/api/orb/v1/multimodal/voice/settings")
async def set_voice_settings(request: VoiceSettingsRequest):
    """Set voice input/output settings for user."""
    try:
        await orb_interface.set_voice_settings(request.user_id, request.settings)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/multimodal/voice/settings/{user_id}")
async def get_voice_settings(user_id: str):
    """Get voice settings for user."""
    settings = orb_interface.get_voice_settings(user_id)
    return {"user_id": user_id, "settings": settings}


@app.post("/api/orb/v1/multimodal/voice/toggle/{user_id}")
async def toggle_voice(user_id: str, enable: bool):
    """Toggle voice input/output for user."""
    try:
        result = await orb_interface.toggle_voice(user_id, enable)
        return {"user_id": user_id, "voice_enabled": result, "status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orb/v1/multimodal/tasks")
async def create_background_task(request: BackgroundTaskRequest):
    """Queue a background processing task."""
    try:
        task_id = await orb_interface.queue_background_task(
            request.task_type,
            request.metadata
        )
        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orb/v1/multimodal/tasks/{task_id}")
async def get_background_task_status(task_id: str):
    """Get status of background task."""
    task_status = orb_interface.get_background_task_status(task_id)
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_status

# ---------------------------
# Statistics
# ---------------------------

@app.get("/api/orb/v1/stats")
async def get_orb_stats():
    """Get comprehensive orb statistics."""
    return orb_interface.get_orb_stats()


@app.get("/api/orb/v1/stats/ide")
async def get_ide_stats():
    """Get IDE statistics."""
    ide = orb_interface.get_ide_instance()
    return ide.get_stats()

# ---------------------------
# WebSocket Support
# ---------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    ws_connections[session_id] = websocket

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "")

            if message_type == "chat_message":
                content = data.get("content", "")
                attachments = data.get("attachments", [])
                try:
                    message_id = await orb_interface.send_chat_message(session_id, content, attachments)
                    messages = orb_interface.get_chat_history(session_id, 2)
                    await websocket.send_json({
                        "type": "chat_response",
                        "message_id": message_id,
                        "messages": [
                            {
                                "message_id": msg.message_id,
                                "user_id": msg.user_id,
                                "content": msg.content,
                                "timestamp": msg.timestamp,
                                "message_type": msg.message_type
                            }
                            for msg in messages[-1:]
                        ]
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "error": str(e)})

            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "error": f"Unknown message type: {message_type}"})

    except WebSocketDisconnect:
        ws_connections.pop(session_id, None)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        ws_connections.pop(session_id, None)

# ---------------------------
# Lifecycle Hooks
# ---------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Grace Unified Orb Interface API starting up...")
    logger.info(f"Orb Interface version: {orb_interface.version}")
    logger.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Grace Unified Orb Interface API shutting down...")
    for sid in list(orb_interface.active_sessions.keys()):
        await orb_interface.end_session(sid)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)