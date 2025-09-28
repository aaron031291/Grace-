"""Main Interface Service implementation matching problem statement specification."""
import asyncio
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid

from .models import *
from .sessions import SessionManager
from .rbac import RBACEvaluator
from .taskcards import TaskCardManager
from .consent import ConsentService
from .notifications import NotificationService
from .snapshots import SnapshotManager
from .bridges import GovernanceBridge, MemoryBridge, MLTBridge

logger = logging.getLogger(__name__)


class InterfaceService:
    """Main interface service bootstrap (HTTP + WS) as specified in problem statement."""
    
    def __init__(self):
        # Core services
        self.session_manager = SessionManager()
        self.rbac_evaluator = RBACEvaluator()
        self.taskcard_manager = TaskCardManager()
        self.consent_service = ConsentService()
        self.notification_service = NotificationService()
        self.snapshot_manager = SnapshotManager()
        
        # Bridge services
        self.governance_bridge = GovernanceBridge()
        self.memory_bridge = MemoryBridge()
        self.mlt_bridge = MLTBridge()
        
        # WebSocket connections
        self.ws_connections: Dict[str, WebSocket] = {}
        
        # FastAPI app
        self.app = self._create_fastapi_app()
        
        # Setup default notification dispatchers
        self.notification_service.setup_default_dispatchers()
        
        logger.info("Interface Service initialized")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Grace Interface Kernel",
            description="Real-time, trusted UX for tasks, telemetry, governance prompts, memory search, and results",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register all API routes as specified in problem statement."""
        
        # Health check
        @app.get("/api/ui/v1/health")
        async def health_check():
            return {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": iso_format()
            }
        
        # Session management
        @app.post("/api/ui/v1/session/start")
        async def start_session(session_data: dict):
            try:
                session = self.session_manager.create(
                    user=session_data["user"],
                    client=session_data["client"]
                )
                
                # Emit session started event
                await self._emit_event("UI_SESSION_STARTED", {
                    "session": session
                })
                
                return session
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # UI Actions
        @app.post("/api/ui/v1/action")
        async def dispatch_action(action_data: dict):
            action_id = str(uuid.uuid4())
            
            try:
                # Create UI action
                ui_action = UIAction(
                    action_id=action_id,
                    session_id=action_data["session_id"],
                    type=action_data["type"],
                    payload=action_data["payload"]
                )
                
                # Check RBAC
                session = self.session_manager.get_session(action_data["session_id"])
                if not session:
                    raise HTTPException(status_code=401, detail="Invalid session")
                
                if not self._check_action_permission(session, ui_action):
                    raise HTTPException(status_code=403, detail="Action not permitted")
                
                # Dispatch action
                result = await self.dispatch_action(ui_action.dict())
                
                # Emit action event
                await self._emit_event("UI_ACTION", {
                    "action": ui_action.dict()
                })
                
                return {"action_id": action_id, "result": result}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Action dispatch failed: {e}")
                raise HTTPException(status_code=500, detail="Action failed")
        
        # TaskCard management
        @app.get("/api/ui/v1/taskcards/{card_id}")
        async def get_taskcard(card_id: str):
            card = self.taskcard_manager.get_card(card_id)
            if not card:
                raise HTTPException(status_code=404, detail="TaskCard not found")
            return card.dict()
        
        @app.post("/api/ui/v1/taskcards")
        async def create_taskcard(card_data: dict):
            try:
                card = self.taskcard_manager.new_card(
                    title=card_data["title"],
                    kind=card_data["kind"],
                    owner=card_data["owner"],
                    ctx=card_data.get("context", {})
                )
                
                # Emit card update event
                await self._emit_event("UI_TASKCARD_UPDATED", {
                    "card": card
                })
                
                return {"card_id": card["card_id"]}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Consent management
        @app.post("/api/ui/v1/consent")
        async def manage_consent(consent_data: dict):
            try:
                if consent_data["action"] == "grant":
                    consent_id = self.consent_service.grant_consent(
                        user_id=consent_data["user_id"],
                        scope=consent_data["scope"],
                        expires_days=consent_data.get("expires_days"),
                        evidence_uri=consent_data.get("evidence_uri")
                    )
                elif consent_data["action"] == "revoke":
                    success = self.consent_service.revoke_consent(consent_data["consent_id"])
                    consent_id = consent_data["consent_id"] if success else None
                else:
                    raise ValueError("Invalid consent action")
                
                return {"consent_id": consent_id}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Notifications
        @app.get("/api/ui/v1/notifications")
        async def get_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
            notifications = self.notification_service.get_user_notifications(
                user_id=user_id,
                unread_only=unread_only,
                limit=limit
            )
            return [n.dict() for n in notifications]
        
        # Snapshot management
        @app.post("/api/ui/v1/snapshot/export")
        async def export_snapshot():
            try:
                result = self.snapshot_manager.export_snapshot()
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/ui/v1/rollback")
        async def rollback(rollback_data: dict):
            try:
                # Emit rollback requested event
                await self._emit_event("ROLLBACK_REQUESTED", {
                    "target": "interface",
                    "to_snapshot": rollback_data["to_snapshot"]
                })
                
                result = await self.rollback(rollback_data["to_snapshot"])
                
                # Emit rollback completed event
                await self._emit_event("ROLLBACK_COMPLETED", result)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoints
        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            await self._handle_websocket_stream(websocket)
        
        @app.websocket("/ws/telemetry")
        async def websocket_telemetry(websocket: WebSocket):
            await self._handle_websocket_telemetry(websocket)
        
        @app.websocket("/ws/approvals")
        async def websocket_approvals(websocket: WebSocket):
            await self._handle_websocket_approvals(websocket)
    
    def _check_action_permission(self, session: UISession, action: UIAction) -> bool:
        """Check if session user can perform action."""
        context = {
            "user_id": session.user.user_id,
            "labels": session.user.labels or [],
            "current_time": utc_now()
        }
        
        return self.rbac_evaluator.evaluate(
            user_roles=session.user.roles,
            action=action.type,
            resource=f"{action.type.split('.')[0]}:*",
            ctx=context
        )
    
    async def _handle_websocket_stream(self, websocket: WebSocket):
        """Handle multiplexed stream WebSocket."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.ws_connections[connection_id] = websocket
        
        try:
            while True:
                # Keep connection alive
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": iso_format()
                })
                await asyncio.sleep(20)  # Heartbeat every 20 seconds
                
        except WebSocketDisconnect:
            if connection_id in self.ws_connections:
                del self.ws_connections[connection_id]
    
    async def _handle_websocket_telemetry(self, websocket: WebSocket):
        """Handle telemetry WebSocket (opt-in, privacy-safe)."""
        await websocket.accept()
        
        try:
            while True:
                # Receive telemetry data
                data = await websocket.receive_json()
                
                # Process telemetry (privacy-safe)
                await self._process_telemetry(data)
                
        except WebSocketDisconnect:
            pass
    
    async def _handle_websocket_approvals(self, websocket: WebSocket):
        """Handle governance approvals WebSocket."""
        await websocket.accept()
        
        try:
            while True:
                # Send pending approvals
                pending = self.governance_bridge.list_pending_approvals()
                
                if pending:
                    await websocket.send_json({
                        "type": "pending_approvals",
                        "payload": pending
                    })
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except WebSocketDisconnect:
            pass
    
    async def _process_telemetry(self, telemetry_data: dict):
        """Process incoming telemetry data."""
        try:
            # Create UI experience
            experience = UIExperience(
                stage=telemetry_data["stage"],
                metrics=UIExperienceMetrics(**telemetry_data["metrics"]),
                segment=telemetry_data.get("segment")
            )
            
            # Emit telemetry event
            await self._emit_event("UI_EXPERIENCE", {
                "schema_version": "1.0.0",
                "experience": experience.dict()
            })
            
        except Exception as e:
            logger.error(f"Telemetry processing failed: {e}")
    
    async def _emit_event(self, event_type: str, payload: dict):
        """Emit event to WebSocket connections."""
        event_data = {
            "type": event_type,
            "payload": payload,
            "timestamp": iso_format()
        }
        
        # Send to all connected WebSocket clients
        disconnected = []
        for connection_id, websocket in self.ws_connections.items():
            try:
                await websocket.send_json(event_data)
            except:
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            if connection_id in self.ws_connections:
                del self.ws_connections[connection_id]
    
    # Main interface methods from problem statement
    def start_session(self, user: dict, client: dict) -> dict:
        """Start a new UI session."""
        return self.session_manager.create(user, client)
    
    async def dispatch_action(self, action: dict) -> str:
        """Dispatch a UI action."""
        action_type = action.get("type", "")
        
        if action_type.startswith("task."):
            return await self._handle_task_action(action)
        elif action_type.startswith("memory."):
            return await self._handle_memory_action(action)
        elif action_type.startswith("governance."):
            return await self._handle_governance_action(action)
        elif action_type.startswith("consent."):
            return await self._handle_consent_action(action)
        elif action_type.startswith("snapshot."):
            return await self._handle_snapshot_action(action)
        else:
            return f"Handled action: {action_type}"
    
    async def _handle_task_action(self, action: dict) -> str:
        """Handle task-related actions."""
        action_type = action["type"]
        payload = action["payload"]
        
        if action_type == "task.create":
            result = self.taskcard_manager.new_card(
                title=payload["title"],
                kind=payload["kind"],
                owner=payload["owner"],
                ctx=payload.get("context", {})
            )
            return f"Created task {result['card_id']}"
        
        elif action_type in ["task.run", "task.pause", "task.cancel"]:
            card_id = payload["card_id"]
            state_map = {"task.run": "running", "task.pause": "paused", "task.cancel": "done"}
            
            self.taskcard_manager.set_state(card_id, state_map[action_type])
            return f"Task {card_id} state updated"
        
        return "Task action handled"
    
    async def _handle_memory_action(self, action: dict) -> str:
        """Handle memory-related actions."""
        if action["type"] == "memory.search":
            result = await self.memory_bridge.search_memory(
                query=action["payload"]["query"],
                user_id=action["payload"]["user_id"],
                filters=action["payload"].get("filters")
            )
            result_count = result.get('result_count', 0)
            return f"Memory search completed: {result_count} results"
        
        return "Memory action handled"
    
    async def _handle_governance_action(self, action: dict) -> str:
        """Handle governance-related actions."""
        if action["type"] == "governance.request_approval":
            approval_id = await self.governance_bridge.request_approval(action["payload"])
            return f"Approval requested: {approval_id}"
        
        return "Governance action handled"
    
    async def _handle_consent_action(self, action: dict) -> str:
        """Handle consent-related actions."""
        payload = action["payload"]
        
        if action["type"] == "consent.grant":
            consent_id = self.consent_service.grant_consent(
                user_id=payload["user_id"],
                scope=payload["scope"]
            )
            return f"Consent granted: {consent_id}"
        
        elif action["type"] == "consent.revoke":
            self.consent_service.revoke_consent(payload["consent_id"])
            return "Consent revoked"
        
        return "Consent action handled"
    
    async def _handle_snapshot_action(self, action: dict) -> str:
        """Handle snapshot-related actions."""
        if action["type"] == "snapshot.export":
            result = self.snapshot_manager.export_snapshot()
            return f"Snapshot exported: {result['snapshot_id']}"
        
        elif action["type"] == "snapshot.rollback":
            await self.rollback(action["payload"]["to_snapshot"])
            return "Rollback completed"
        
        return "Snapshot action handled"
    
    def export_snapshot(self) -> dict:
        """Export current interface state snapshot."""
        return self.snapshot_manager.export_snapshot()
    
    async def rollback(self, to_snapshot: str) -> dict:
        """Rollback interface to specific snapshot."""
        return await self.snapshot_manager.rollback(to_snapshot)
    
    def set_kernel_references(self, mtl_kernel=None, governance_kernel=None, intelligence_kernel=None):
        """Set kernel references for bridges."""
        if mtl_kernel:
            self.memory_bridge.set_mtl_kernel(mtl_kernel)
            self.mlt_bridge.set_mlt_kernel(mtl_kernel)
        
        if governance_kernel:
            self.governance_bridge.set_governance_kernel(governance_kernel)
        
        if intelligence_kernel:
            self.memory_bridge.set_intelligence_kernel(intelligence_kernel)
    
    def get_stats(self) -> dict:
        """Get comprehensive interface service statistics."""
        return {
            "sessions": self.session_manager.get_stats(),
            "taskcards": self.taskcard_manager.get_stats(),
            "consent": self.consent_service.get_stats(),
            "notifications": self.notification_service.get_stats(),
            "snapshots": self.snapshot_manager.get_stats(),
            "bridges": {
                "governance": self.governance_bridge.get_stats(),
                "memory": self.memory_bridge.get_stats(),
                "mlt": self.mlt_bridge.get_stats()
            },
            "websockets": len(self.ws_connections)
        }