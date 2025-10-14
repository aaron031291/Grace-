"""Interface kernel - REST API and WebSocket interface using FastAPI."""

import asyncio
import time
import logging
from typing import Dict, Optional
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..contracts.governed_request import GovernedRequest
from ..contracts.governed_decision import GovernedDecision
from ..contracts.rag_query import RAGQuery, RAGResult
from ..contracts.dto_common import MemoryEntry

# Import the comprehensive Interface Service
from ..interface.interface_service import InterfaceService

# Import voice service
from .voice_service import create_voice_service

# Import health dashboard
from .health_dashboard import HealthDashboard

# Import book ingestion service
from ..memory.book_ingestion import BookIngestionService


logger = logging.getLogger(__name__)


# Request/Response models for API
class MemoryRequest(BaseModel):
    content: str
    content_type: str = "text/plain"
    metadata: Optional[Dict] = None


class MemoryResponse(BaseModel):
    id: str
    message: str = "Memory stored successfully"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "0.1.0"
    kernels: Dict[str, bool]


class AuditProofResponse(BaseModel):
    audit_id: str
    proof: Optional[Dict] = None
    verified: bool


class VoiceToggleResponse(BaseModel):
    success: bool
    message: str
    state: str


class VoiceStatusResponse(BaseModel):
    voice_enabled: bool
    state: str
    voice_available: bool
    conversation_length: int


class BookIngestionRequest(BaseModel):
    content: str
    title: str
    author: Optional[str] = None
    metadata: Optional[Dict] = None


class BookIngestionResponse(BaseModel):
    job_id: str
    status: str
    title: str
    message: Optional[str] = None


class InterfaceKernel:
    """Main interface kernel providing REST API and WebSocket endpoints."""

    def __init__(
        self,
        mtl_kernel=None,
        governance_kernel=None,
        orchestration_kernel=None,
        resilience_kernel=None,
    ):
        self.mtl_kernel = mtl_kernel
        self.governance_kernel = governance_kernel
        self.orchestration_kernel = orchestration_kernel
        self.resilience_kernel = resilience_kernel

        # Initialize comprehensive Interface Service
        self.interface_service = InterfaceService()
        self.interface_service.set_kernel_references(
            mtl_kernel=mtl_kernel,
            governance_kernel=governance_kernel,
            intelligence_kernel=None,  # Would be passed if available
        )

        # Use Interface Service app
        self.app = self.interface_service.app

        # Legacy WebSocket connections for backward compatibility
        self.websocket_connections = []

        # Initialize voice service
        self.voice_service = create_voice_service(self._handle_voice_communication)

        # Initialize health dashboard
        self.health_dashboard = HealthDashboard()

        # Initialize book ingestion service
        self.book_ingestion = BookIngestionService()

        # Register additional legacy routes for compatibility
        self._register_legacy_routes()

    def _register_legacy_routes(self):
        """Register all API routes."""

        # Health check endpoint
        @self.app.get("/api/runtime/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(
                status="healthy",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                kernels={
                    "mtl": bool(self.mtl_kernel),
                    "governance": bool(self.governance_kernel),
                    "orchestration": bool(self.orchestration_kernel),
                    "resilience": bool(self.resilience_kernel),
                },
            )

        # MTL Memory endpoints
        @self.app.post("/api/mtl/memory", response_model=MemoryResponse)
        async def store_memory(request: MemoryRequest):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")

            try:
                # Create memory entry
                entry = MemoryEntry(
                    content=request.content,
                    content_type=request.content_type,
                    metadata=request.metadata,
                )

                # Store using MTL kernel
                memory_id = self.mtl_kernel.write(entry)

                return MemoryResponse(
                    id=memory_id, message="Memory stored successfully"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to store memory: {str(e)}"
                )

        @self.app.get("/api/mtl/memory")
        async def recall_memory(query: str, limit: int = 10):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")

            try:
                results = self.mtl_kernel.recall(query, filters=None)
                return {
                    "query": query,
                    "results": [r.dict() for r in results[:limit]],
                    "total": len(results),
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to recall memory: {str(e)}"
                )

        @self.app.get(
            "/api/mtl/audit/{audit_id}/proof", response_model=AuditProofResponse
        )
        async def get_audit_proof(audit_id: str):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")

            try:
                proof = self.mtl_kernel.get_audit_proof(audit_id)
                return AuditProofResponse(
                    audit_id=audit_id,
                    proof=proof.dict() if proof else None,
                    verified=bool(proof),
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get audit proof: {str(e)}"
                )

        # RAG endpoint
        @self.app.post("/api/mtl/rag", response_model=RAGResult)
        async def rag_query(query: RAGQuery):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")

            try:
                start_time = time.time()

                # Use librarian for search and ranking
                results = self.mtl_kernel.librarian.search_and_rank(
                    query=query.query, limit=query.limit
                )

                # Apply minimum relevance filter (mock implementation)
                filtered_results = (
                    results  # In real implementation, filter by relevance
                )

                # Distill if requested
                distilled_summary = None
                if len(filtered_results) > 1:
                    distilled_summary = self.mtl_kernel.librarian.distill_content(
                        filtered_results, context=query.query
                    )

                processing_time = (time.time() - start_time) * 1000

                return RAGResult(
                    query=query.query,
                    items=filtered_results,
                    total_found=len(results),
                    processing_time_ms=processing_time,
                    distilled_summary=distilled_summary,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"RAG query failed: {str(e)}"
                )

        # Governance endpoint
        @self.app.post("/api/governance/evaluate", response_model=GovernedDecision)
        async def evaluate_governance(request: GovernedRequest):
            if not self.governance_kernel:
                raise HTTPException(
                    status_code=503, detail="Governance kernel not available"
                )

            try:
                decision = self.governance_kernel.evaluate(request)
                return decision
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Governance evaluation failed: {str(e)}"
                )

        # Voice interface endpoints
        @self.app.post("/api/voice/toggle", response_model=VoiceToggleResponse)
        async def toggle_voice_mode():
            """Toggle voice communication mode on/off."""
            try:
                result = await self.voice_service.toggle_voice_mode()
                return VoiceToggleResponse(**result)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Voice toggle failed: {str(e)}"
                )

        @self.app.get("/api/voice/status", response_model=VoiceStatusResponse)
        async def get_voice_status():
            """Get current voice interface status."""
            try:
                status = self.voice_service.get_voice_status()
                return VoiceStatusResponse(**status)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Voice status failed: {str(e)}"
                )

        @self.app.get("/api/voice/conversation")
        async def get_voice_conversation():
            """Get voice conversation history."""
            try:
                history = self.voice_service.get_conversation_history()
                return {"conversation": history, "total_messages": len(history)}
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Voice conversation failed: {str(e)}"
                )

        @self.app.post("/api/voice/clear")
        async def clear_voice_conversation():
            """Clear voice conversation history."""
            try:
                self.voice_service.conversation_history.clear()
                return {"message": "Voice conversation history cleared"}
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Clear conversation failed: {str(e)}"
                )

        # Enhanced Health Dashboard endpoints
        @self.app.get("/api/health/comprehensive")
        async def get_comprehensive_health():
            """Get comprehensive system health status."""
            try:
                return self.health_dashboard.get_comprehensive_health()
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Health check failed: {str(e)}"
                )

        @self.app.post("/api/health/check")
        async def perform_health_check():
            """Trigger immediate health check of all systems."""
            try:
                result = await self.health_dashboard.perform_health_check()
                return result
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Health check failed: {str(e)}"
                )

        @self.app.get("/api/health/alerts")
        async def get_health_alerts(resolved: Optional[bool] = None, hours: int = 24):
            """Get system health alerts."""
            try:
                alerts = self.health_dashboard.get_alerts(
                    resolved=resolved, hours=hours
                )
                return {"alerts": alerts, "total": len(alerts)}
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Get alerts failed: {str(e)}"
                )

        @self.app.get("/api/health/component/{component_name}")
        async def get_component_health(component_name: str, hours: int = 24):
            """Get detailed health information for a specific component."""
            try:
                history = self.health_dashboard.get_component_history(
                    component_name, hours
                )
                diagnostic = await self.health_dashboard.run_diagnostic(component_name)

                return {
                    "component": component_name,
                    "history": history,
                    "diagnostic": diagnostic,
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Component health check failed: {str(e)}"
                )

        @self.app.post("/api/health/diagnostic")
        async def run_system_diagnostic(component_name: Optional[str] = None):
            """Run comprehensive system diagnostic."""
            try:
                result = await self.health_dashboard.run_diagnostic(component_name)
                return result
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"System diagnostic failed: {str(e)}"
                )

        @self.app.post("/api/health/monitoring/start")
        async def start_health_monitoring():
            """Start continuous health monitoring."""
            try:
                await self.health_dashboard.start_monitoring()
                return {
                    "message": "Health monitoring started",
                    "monitoring_active": True,
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Start monitoring failed: {str(e)}"
                )

        @self.app.post("/api/health/monitoring/stop")
        async def stop_health_monitoring():
            """Stop continuous health monitoring."""
            try:
                await self.health_dashboard.stop_monitoring()
                return {
                    "message": "Health monitoring stopped",
                    "monitoring_active": False,
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Stop monitoring failed: {str(e)}"
                )

        # Book Ingestion endpoints
        @self.app.post("/api/books/ingest", response_model=BookIngestionResponse)
        async def ingest_book(request: BookIngestionRequest):
            """Ingest a large book or document (up to 500+ pages)."""
            try:
                result = await self.book_ingestion.ingest_book(
                    content=request.content,
                    title=request.title,
                    author=request.author,
                    metadata=request.metadata,
                )

                return BookIngestionResponse(
                    job_id=result["job_id"],
                    status=result["status"],
                    title=result["title"],
                    message=result.get("processing_summary", {}).get(
                        "chapters_processed", 0
                    ),
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Book ingestion failed: {str(e)}"
                )

        @self.app.get("/api/books/{job_id}/progress")
        async def get_ingestion_progress(job_id: str):
            """Get progress for a book ingestion job."""
            try:
                progress = self.book_ingestion.get_ingestion_progress(job_id)
                if not progress:
                    raise HTTPException(status_code=404, detail="Job not found")
                return progress
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Progress check failed: {str(e)}"
                )

        @self.app.get("/api/books/{job_id}/insights")
        async def get_book_insights(job_id: str, limit: int = 20):
            """Get insights extracted from a processed book."""
            try:
                insights = await self.book_ingestion.get_book_insights(job_id)
                return {
                    "job_id": job_id,
                    "insights": insights[:limit],
                    "total": len(insights),
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Get insights failed: {str(e)}"
                )

        @self.app.get("/api/books/{job_id}/chapters")
        async def get_book_chapters(job_id: str):
            """Get chapter summaries for a processed book."""
            try:
                chapters = await self.book_ingestion.get_book_chapters(job_id)
                return {"job_id": job_id, "chapters": chapters, "total": len(chapters)}
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Get chapters failed: {str(e)}"
                )

        @self.app.get("/api/books/{job_id}/search")
        async def search_book_content(job_id: str, query: str, limit: int = 10):
            """Search within a specific book's content."""
            try:
                results = await self.book_ingestion.search_book_content(
                    job_id, query, limit
                )
                return {
                    "job_id": job_id,
                    "query": query,
                    "results": results,
                    "total": len(results),
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Book search failed: {str(e)}"
                )

        # WebSocket endpoint for events
        @self.app.websocket("/ws/events")
        async def websocket_events(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                # Keep connection alive and stream trigger events
                while True:
                    # Get recent trigger events from MTL
                    if self.mtl_kernel:
                        recent_events = (
                            self.mtl_kernel.trigger_ledger.get_recent_events(5)
                        )
                        for event in recent_events:
                            await websocket.send_json(
                                {"type": "trigger_event", "data": event.dict()}
                            )

                    # Wait before next batch
                    await asyncio.sleep(5)

            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)

    async def broadcast_event(self, event_data: Dict):
        """Broadcast event to all connected WebSocket clients."""
        if not self.websocket_connections:
            return

        # Remove disconnected connections
        active_connections = []
        for connection in self.websocket_connections:
            try:
                await connection.send_json(event_data)
                active_connections.append(connection)
            except Exception:
                # Connection is dead, skip it
                pass

        self.websocket_connections = active_connections

    async def _handle_voice_communication(self, text: str) -> Dict:
        """Handle voice communication through Grace's processing pipeline."""
        try:
            # Create a basic structured query similar to grace_communication_demo
            response = await self._process_structured_query(text)
            return response
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error processing your voice input: {str(e)}",
                "confidence": 0.0,
                "source": "error_handler",
            }

    async def _process_structured_query(self, query: str) -> Dict:
        """Process a structured query and return response (simplified version from demo)."""
        import random

        # Simple query processing based on keywords
        query_lower = query.lower()

        if any(word in query_lower for word in ["health", "status", "system"]):
            answer = "All Grace systems are operational. Governance protocols active, memory systems functional, and trust mechanisms working normally."
            confidence = 0.92

        elif any(
            word in query_lower for word in ["help", "what can you do", "capabilities"]
        ):
            answer = "I can assist with governance decisions, system monitoring, memory management, and now voice communication. Ask me about system health, policies, or any questions you have."
            confidence = 0.95

        elif any(word in query_lower for word in ["governance", "policy", "decision"]):
            answer = "Grace governance uses multi-layer democratic processes with constitutional compliance checking and trust-weighted voting mechanisms."
            confidence = 0.88

        elif any(word in query_lower for word in ["memory", "remember", "recall"]):
            answer = "Grace memory systems can store, index, and recall information using vector embeddings and constitutional filtering for safe knowledge management."
            confidence = 0.85

        else:
            # General response
            answer = f"I understand you said: '{query}'. I'm Grace, your AI governance assistant. How can I help you with system management or decision-making?"
            confidence = 0.70

        return {
            "answer": answer,
            "confidence": confidence,
            "query": query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_sources": random.randint(2, 6),
            "response_time": 0.01,
        }

    def get_stats(self) -> Dict:
        """Get interface kernel statistics."""
        # Get comprehensive stats from Interface Service
        interface_stats = self.interface_service.get_stats()

        # Add legacy compatibility data
        interface_stats.update(
            {
                "active_websocket_connections": len(self.websocket_connections),
                "kernels_connected": {
                    "mtl": bool(self.mtl_kernel),
                    "governance": bool(self.governance_kernel),
                    "orchestration": bool(self.orchestration_kernel),
                    "resilience": bool(self.resilience_kernel),
                },
                "api_version": "1.0.0",  # Updated version
            }
        )

        return interface_stats

    def set_mtl_kernel(self, mtl_kernel):
        """Set the MTL kernel."""
        self.mtl_kernel = mtl_kernel

    def set_governance_kernel(self, governance_kernel):
        """Set the governance kernel."""
        self.governance_kernel = governance_kernel

    async def cleanup(self):
        """Cleanup resources including voice service and health monitoring."""
        try:
            # Cleanup voice service
            await self.voice_service.cleanup()
        except Exception as e:
            logger.error(f"Voice service cleanup error: {e}")

        try:
            # Stop health monitoring
            await self.health_dashboard.stop_monitoring()
        except Exception as e:
            logger.error(f"Health dashboard cleanup error: {e}")


# Create FastAPI app instance for uvicorn
def create_app():
    """Create and configure the FastAPI application."""
    from ..mtl_kernel.kernel import MTLKernel
    from ..governance.grace_governance_kernel import (
        GraceGovernanceKernel as GovernanceKernel,
    )
    from ..intelligence.kernel.kernel import IntelligenceKernel

    # Initialize kernels
    mtl_kernel = MTLKernel()
    intelligence_kernel = IntelligenceKernel(mtl_kernel)
    governance_kernel = GovernanceKernel(mtl_kernel, intelligence_kernel)

    # Link kernels
    governance_kernel.set_mtl_kernel(mtl_kernel)
    governance_kernel.set_intelligence_kernel(intelligence_kernel)
    intelligence_kernel.set_mtl_kernel(mtl_kernel)

    # Create interface
    interface = InterfaceKernel(
        mtl_kernel=mtl_kernel, governance_kernel=governance_kernel
    )

    return interface.app


# FastAPI app for uvicorn
app = create_app()
