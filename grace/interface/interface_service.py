"""Main Interface Service implementation with elite-level NLP capabilities."""

import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
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

# Elite NLP imports
try:
    from ..mldl.specialists.elite_nlp_specialist import EliteNLPSpecialist
    from ..mtl_kernel.enhanced_w5h_indexer import EnhancedW5HIndexer

    ELITE_NLP_AVAILABLE = True
except ImportError:
    ELITE_NLP_AVAILABLE = False

logger = logging.getLogger(__name__)


class InterfaceService:
    """Main interface service bootstrap (HTTP + WS) with elite-level NLP capabilities."""

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

        # Elite NLP services
        if ELITE_NLP_AVAILABLE:
            self.nlp_specialist = EliteNLPSpecialist()
            self.enhanced_indexer = EnhancedW5HIndexer()
            logger.info("Elite NLP capabilities initialized")
        else:
            self.nlp_specialist = None
            self.enhanced_indexer = None
            logger.warning(
                "Elite NLP capabilities not available - using basic processing"
            )

        # WebSocket connections
        self.ws_connections: Dict[str, WebSocket] = {}

        # Conversation tracking for advanced NLP
        self.active_conversations = {}

        # FastAPI app
        self.app = self._create_fastapi_app()

        # Setup default notification dispatchers
        self.notification_service.setup_default_dispatchers()

        logger.info("Interface Service initialized with elite NLP capabilities")

    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Grace Interface Kernel",
            description="Real-time, trusted UX for tasks, telemetry, governance prompts, memory search, and results",
            version="1.0.0",
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
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Session management
        @app.post("/api/ui/v1/session/start")
        async def start_session(session_data: dict):
            try:
                session = self.session_manager.create(
                    user=session_data["user"], client=session_data["client"]
                )

                # Emit session started event
                await self._emit_event("UI_SESSION_STARTED", {"session": session})

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
                    payload=action_data["payload"],
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
                await self._emit_event("UI_ACTION", {"action": ui_action.dict()})

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
                    ctx=card_data.get("context", {}),
                )

                # Emit card update event
                await self._emit_event("UI_TASKCARD_UPDATED", {"card": card})

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
                        evidence_uri=consent_data.get("evidence_uri"),
                    )
                elif consent_data["action"] == "revoke":
                    success = self.consent_service.revoke_consent(
                        consent_data["consent_id"]
                    )
                    consent_id = consent_data["consent_id"] if success else None
                else:
                    raise ValueError("Invalid consent action")

                return {"consent_id": consent_id}

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Notifications
        @app.get("/api/ui/v1/notifications")
        async def get_notifications(
            user_id: str, unread_only: bool = False, limit: int = 50
        ):
            notifications = self.notification_service.get_user_notifications(
                user_id=user_id, unread_only=unread_only, limit=limit
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
                await self._emit_event(
                    "ROLLBACK_REQUESTED",
                    {
                        "target": "interface",
                        "to_snapshot": rollback_data["to_snapshot"],
                    },
                )

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

        # Elite NLP endpoints
        if ELITE_NLP_AVAILABLE:

            @app.post("/api/ui/v1/nlp/analyze")
            async def analyze_text(analysis_request: dict):
                """Perform elite-level NLP analysis on text."""
                try:
                    text = analysis_request.get("text", "")
                    context = analysis_request.get("context", {})

                    if not text:
                        raise HTTPException(status_code=400, detail="Text is required")

                    # Perform comprehensive NLP analysis
                    analysis = await self.nlp_specialist.analyze_text(text, context)

                    return {
                        "analysis_id": str(uuid.uuid4()),
                        "analysis": analysis.to_dict(),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"NLP analysis failed: {str(e)}"
                    )

            @app.post("/api/ui/v1/nlp/conversation")
            async def manage_conversation(conversation_request: dict):
                """Manage conversation context with advanced NLP understanding."""
                try:
                    conversation_id = conversation_request.get(
                        "conversation_id", str(uuid.uuid4())
                    )
                    user_input = conversation_request.get("user_input", "")
                    user_profile = conversation_request.get("user_profile", {})
                    domain = conversation_request.get("domain")

                    if not user_input:
                        raise HTTPException(
                            status_code=400, detail="User input is required"
                        )

                    # Manage conversation context
                    context = await self.nlp_specialist.manage_conversation_context(
                        conversation_id, user_input, user_profile, domain
                    )

                    # Perform intent-aware response
                    response = await self._generate_intelligent_response(
                        user_input, context
                    )

                    return {
                        "conversation_id": conversation_id,
                        "context": context.to_dict(),
                        "response": response,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Conversation management failed: {str(e)}",
                    )

            @app.post("/api/ui/v1/nlp/extract")
            async def extract_w5h(extraction_request: dict):
                """Extract Who/What/When/Where/Why/How using enhanced NLP."""
                try:
                    text = extraction_request.get("text", "")
                    context = extraction_request.get("context", {})

                    if not text:
                        raise HTTPException(status_code=400, detail="Text is required")

                    # Extract W5H using enhanced indexer
                    w5h_index = self.enhanced_indexer.extract(text, context)

                    # Perform intent analysis
                    intent_analysis = self.enhanced_indexer.analyze_intent(text)

                    return {
                        "extraction_id": str(uuid.uuid4()),
                        "w5h_index": w5h_index.__dict__,
                        "intent_analysis": intent_analysis,
                        "text_length": len(text),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"W5H extraction failed: {str(e)}"
                    )

            @app.post("/api/ui/v1/nlp/toxicity")
            async def detect_toxicity(toxicity_request: dict):
                """Detect toxicity and harmful content."""
                try:
                    text = toxicity_request.get("text", "")

                    if not text:
                        raise HTTPException(status_code=400, detail="Text is required")

                    # Detect toxicity
                    toxicity_result = await self.nlp_specialist.detect_toxicity(text)

                    return {
                        "analysis_id": str(uuid.uuid4()),
                        "toxicity": toxicity_result,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Toxicity detection failed: {str(e)}"
                    )

            @app.get("/api/ui/v1/nlp/stats")
            async def get_nlp_stats():
                """Get NLP performance statistics."""
                try:
                    stats = {
                        "specialist_stats": self.nlp_specialist.get_performance_stats(),
                        "indexer_stats": self.enhanced_indexer.get_stats(),
                        "active_conversations": len(self.active_conversations),
                    }

                    return stats

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Failed to get NLP stats: {str(e)}"
                    )

    def _check_action_permission(self, session: UISession, action: UIAction) -> bool:
        """Check if session user can perform action."""
        context = {
            "user_id": session.user.user_id,
            "labels": session.user.labels or [],
            "current_time": datetime.utcnow(),
        }

        return self.rbac_evaluator.evaluate(
            user_roles=session.user.roles,
            action=action.type,
            resource=f"{action.type.split('.')[0]}:*",
            ctx=context,
        )

    async def _handle_websocket_stream(self, websocket: WebSocket):
        """Handle multiplexed stream WebSocket."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.ws_connections[connection_id] = websocket

        try:
            while True:
                # Keep connection alive
                await websocket.send_json(
                    {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}
                )
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
                    await websocket.send_json(
                        {"type": "pending_approvals", "payload": pending}
                    )

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
                segment=telemetry_data.get("segment"),
            )

            # Emit telemetry event
            await self._emit_event(
                "UI_EXPERIENCE",
                {"schema_version": "1.0.0", "experience": experience.dict()},
            )

        except Exception as e:
            logger.error(f"Telemetry processing failed: {e}")

    async def _emit_event(self, event_type: str, payload: dict):
        """Emit event to WebSocket connections."""
        event_data = {
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
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
        """Dispatch a UI action with enhanced NLP understanding."""
        action_type = action.get("type", "")

        # Use NLP to understand action intent if available
        if ELITE_NLP_AVAILABLE and self.nlp_specialist:
            action_text = f"{action_type} {action.get('payload', {})}"
            intent_analysis = await self.nlp_specialist.analyze_text(action_text)

            # Add intent information to action metadata
            action["intent_analysis"] = intent_analysis.to_dict()

            # Use intent to improve routing
            if intent_analysis.intent.get("urgency") == "high":
                logger.info(f"High urgency action detected: {action_type}")

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
                ctx=payload.get("context", {}),
            )
            return f"Created task {result['card_id']}"

        elif action_type in ["task.run", "task.pause", "task.cancel"]:
            card_id = payload["card_id"]
            state_map = {
                "task.run": "running",
                "task.pause": "paused",
                "task.cancel": "done",
            }

            self.taskcard_manager.set_state(card_id, state_map[action_type])
            return f"Task {card_id} state updated"

        return "Task action handled"

    async def _handle_memory_action(self, action: dict) -> str:
        """Handle memory-related actions with enhanced NLP processing."""
        if action["type"] == "memory.search":
            query = action["payload"]["query"]
            user_id = action["payload"]["user_id"]
            filters = action["payload"].get("filters")

            # Enhance query with NLP understanding
            if ELITE_NLP_AVAILABLE and self.enhanced_indexer:
                # Extract W5H elements to improve search
                w5h_index = self.enhanced_indexer.extract(query)

                # Add semantic filters based on extracted elements
                if not filters:
                    filters = {}

                # Add entity-based filters
                if w5h_index.who:
                    filters["entities"] = w5h_index.who[:3]  # Top 3 entities

                if w5h_index.what:
                    filters["topics"] = w5h_index.what[:3]  # Top 3 topics

                if w5h_index.when:
                    filters["temporal"] = w5h_index.when[:2]  # Top 2 temporal refs

            result = await self.memory_bridge.search_memory(
                query=query, user_id=user_id, filters=filters
            )
            result_count = result.get("result_count", 0)
            return f"Enhanced memory search completed: {result_count} results"

        return "Memory action handled"

    async def _handle_governance_action(self, action: dict) -> str:
        """Handle governance-related actions."""
        if action["type"] == "governance.request_approval":
            approval_id = await self.governance_bridge.request_approval(
                action["payload"]
            )
            return f"Approval requested: {approval_id}"

        return "Governance action handled"

    async def _handle_consent_action(self, action: dict) -> str:
        """Handle consent-related actions."""
        payload = action["payload"]

        if action["type"] == "consent.grant":
            consent_id = self.consent_service.grant_consent(
                user_id=payload["user_id"], scope=payload["scope"]
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

    def set_kernel_references(
        self, mtl_kernel=None, governance_kernel=None, intelligence_kernel=None
    ):
        """Set kernel references for bridges."""
        if mtl_kernel:
            self.memory_bridge.set_mtl_kernel(mtl_kernel)
            self.mlt_bridge.set_mlt_kernel(mtl_kernel)

        if governance_kernel:
            self.governance_bridge.set_governance_kernel(governance_kernel)

        if intelligence_kernel:
            self.memory_bridge.set_intelligence_kernel(intelligence_kernel)

    def get_stats(self) -> dict:
        """Get comprehensive interface service statistics with NLP metrics."""
        base_stats = {
            "sessions": self.session_manager.get_stats(),
            "taskcards": self.taskcard_manager.get_stats(),
            "consent": self.consent_service.get_stats(),
            "notifications": self.notification_service.get_stats(),
            "snapshots": self.snapshot_manager.get_stats(),
            "bridges": {
                "governance": self.governance_bridge.get_stats(),
                "memory": self.memory_bridge.get_stats(),
                "mlt": self.mlt_bridge.get_stats(),
            },
            "websockets": len(self.ws_connections),
        }

        # Add NLP statistics if available
        if ELITE_NLP_AVAILABLE and self.nlp_specialist and self.enhanced_indexer:
            base_stats["nlp"] = {
                "specialist_stats": self.nlp_specialist.get_performance_stats(),
                "indexer_stats": self.enhanced_indexer.get_stats(),
                "active_conversations": len(self.active_conversations),
            }

        return base_stats

    async def _generate_intelligent_response(
        self, user_input: str, context
    ) -> Dict[str, Any]:
        """Generate intelligent response based on NLP analysis and context."""
        response = {
            "text": "I understand your request.",
            "confidence": 0.7,
            "suggested_actions": [],
            "follow_up_questions": [],
        }

        if not ELITE_NLP_AVAILABLE or not self.nlp_specialist:
            return response

        try:
            # Analyze the current input
            analysis = await self.nlp_specialist.analyze_text(
                user_input, {"domain": context.domain}
            )

            # Generate response based on intent
            intent_type = analysis.intent.get("intent_type", "informational")
            intent = analysis.intent.get("intent", "unknown")

            if intent_type == "informational":
                if intent == "question":
                    response["text"] = (
                        "I'll help you find that information. Let me search our knowledge base."
                    )
                    response["suggested_actions"] = [
                        {"action": "memory.search", "label": "Search Knowledge Base"},
                        {
                            "action": "governance.request_info",
                            "label": "Request Official Information",
                        },
                    ]
                else:
                    response["text"] = (
                        "Thank you for the information. I've processed and understood your input."
                    )

            elif intent_type == "transactional":
                if intent == "action_request":
                    response["text"] = (
                        "I understand you'd like me to perform an action. Let me help you with that."
                    )
                    response["suggested_actions"] = [
                        {"action": "task.create", "label": "Create Task"},
                        {
                            "action": "governance.request_approval",
                            "label": "Request Approval",
                        },
                    ]

                    # Check urgency
                    if analysis.intent.get("urgency") == "high":
                        response["text"] += (
                            " I notice this seems urgent, so I'll prioritize this request."
                        )
                        response["confidence"] = 0.9

            # Add sentiment-aware responses
            sentiment = analysis.sentiment.get("label", "neutral")
            if sentiment == "negative":
                response["text"] = (
                    "I sense some frustration. "
                    + response["text"]
                    + " How can I make this easier for you?"
                )
                response["follow_up_questions"] = [
                    "What specific challenges are you facing?"
                ]
            elif sentiment == "positive":
                response["text"] = "Great! " + response["text"]

            # Add follow-up questions based on extracted entities
            if analysis.entities:
                entity_types = [entity["label"] for entity in analysis.entities]
                if "PERSON" in entity_types:
                    response["follow_up_questions"].append(
                        "Would you like me to notify the mentioned people?"
                    )
                if "DATE" in entity_types:
                    response["follow_up_questions"].append(
                        "Should I set a reminder for the mentioned date?"
                    )

            # Set confidence based on analysis quality
            response["confidence"] = analysis.confidence

        except Exception as e:
            logger.error(f"Error generating intelligent response: {e}")
            response["text"] = (
                "I'm processing your request. How can I assist you further?"
            )

        return response

    async def cleanup_conversations(self):
        """Clean up old conversation contexts."""
        if ELITE_NLP_AVAILABLE and self.nlp_specialist:
            self.nlp_specialist.cleanup_old_conversations()
