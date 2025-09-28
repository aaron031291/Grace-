"""Interface kernel - REST API and WebSocket interface using FastAPI."""
import asyncio
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..contracts.governed_request import GovernedRequest
from ..contracts.governed_decision import GovernedDecision
from ..contracts.rag_query import RAGQuery, RAGResult
from ..contracts.dto_common import MemoryEntry


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


class InterfaceKernel:
    """Main interface kernel providing REST API and WebSocket endpoints."""
    
    def __init__(self, 
                 mtl_kernel=None, 
                 governance_kernel=None, 
                 orchestration_kernel=None,
                 resilience_kernel=None):
        self.mtl_kernel = mtl_kernel
        self.governance_kernel = governance_kernel
        self.orchestration_kernel = orchestration_kernel
        self.resilience_kernel = resilience_kernel
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Grace Kernel API",
            description="Multi-kernel AI governance system",
            version="0.1.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # WebSocket connection manager
        self.websocket_connections = []
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
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
                    "resilience": bool(self.resilience_kernel)
                }
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
                    metadata=request.metadata
                )
                
                # Store using MTL kernel
                memory_id = self.mtl_kernel.write(entry)
                
                return MemoryResponse(
                    id=memory_id,
                    message="Memory stored successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")
        
        @self.app.get("/api/mtl/memory")
        async def recall_memory(query: str, limit: int = 10):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")
            
            try:
                results = self.mtl_kernel.recall(query, filters=None)
                return {
                    "query": query,
                    "results": [r.dict() for r in results[:limit]],
                    "total": len(results)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to recall memory: {str(e)}")
        
        @self.app.get("/api/mtl/audit/{audit_id}/proof", response_model=AuditProofResponse)
        async def get_audit_proof(audit_id: str):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")
            
            try:
                proof = self.mtl_kernel.get_audit_proof(audit_id)
                return AuditProofResponse(
                    audit_id=audit_id,
                    proof=proof.dict() if proof else None,
                    verified=bool(proof)
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get audit proof: {str(e)}")
        
        # RAG endpoint
        @self.app.post("/api/mtl/rag", response_model=RAGResult)
        async def rag_query(query: RAGQuery):
            if not self.mtl_kernel:
                raise HTTPException(status_code=503, detail="MTL kernel not available")
            
            try:
                start_time = time.time()
                
                # Use librarian for search and ranking
                results = self.mtl_kernel.librarian.search_and_rank(
                    query=query.query, 
                    limit=query.limit
                )
                
                # Apply minimum relevance filter (mock implementation)
                filtered_results = results  # In real implementation, filter by relevance
                
                # Distill if requested
                distilled_summary = None
                if len(filtered_results) > 1:
                    distilled_summary = self.mtl_kernel.librarian.distill_content(
                        filtered_results, 
                        context=query.query
                    )
                
                processing_time = (time.time() - start_time) * 1000
                
                return RAGResult(
                    query=query.query,
                    items=filtered_results,
                    total_found=len(results),
                    processing_time_ms=processing_time,
                    distilled_summary=distilled_summary
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
        
        # Governance endpoint
        @self.app.post("/api/governance/evaluate", response_model=GovernedDecision)
        async def evaluate_governance(request: GovernedRequest):
            if not self.governance_kernel:
                raise HTTPException(status_code=503, detail="Governance kernel not available")
            
            try:
                decision = self.governance_kernel.evaluate(request)
                return decision
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Governance evaluation failed: {str(e)}")
        
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
                        recent_events = self.mtl_kernel.trigger_ledger.get_recent_events(5)
                        for event in recent_events:
                            await websocket.send_json({
                                "type": "trigger_event",
                                "data": event.dict()
                            })
                    
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
            except:
                # Connection is dead, skip it
                pass
        
        self.websocket_connections = active_connections
    
    def get_stats(self) -> Dict:
        """Get interface kernel statistics."""
        return {
            "active_websocket_connections": len(self.websocket_connections),
            "kernels_connected": {
                "mtl": bool(self.mtl_kernel),
                "governance": bool(self.governance_kernel),
                "orchestration": bool(self.orchestration_kernel),
                "resilience": bool(self.resilience_kernel)
            },
            "api_version": "0.1.0"
        }
    
    def set_mtl_kernel(self, mtl_kernel):
        """Set the MTL kernel."""
        self.mtl_kernel = mtl_kernel
    
    def set_governance_kernel(self, governance_kernel):
        """Set the governance kernel."""
        self.governance_kernel = governance_kernel


# Create FastAPI app instance for uvicorn
def create_app():
    """Create and configure the FastAPI application."""
    from ..mtl_kernel.kernel import MTLKernel
    from ..governance_kernel.kernel import GovernanceKernel
    from ..intelligence_kernel.kernel import IntelligenceKernel
    
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
        mtl_kernel=mtl_kernel,
        governance_kernel=governance_kernel
    )
    
    return interface.app


# FastAPI app for uvicorn
app = create_app()