#!/usr/bin/env python3
"""
Grace Enhanced Interface Server
===============================

Simple web server to host the Grace Enhanced Interface and provide API endpoints
for testing the voice toggle, book ingestion, and health monitoring features.

Usage:
    python grace_interface_server.py
    
Then open: http://localhost:8080
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from grace.interface_kernel.kernel import InterfaceKernel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraceInterfaceServer:
    """Web server for Grace Enhanced Interface."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Grace Enhanced Interface",
            description="Voice toggle, book ingestion, and health monitoring interface",
            version="1.0.0"
        )
        
        # Initialize Grace kernel
        self.grace_kernel = InterfaceKernel()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Grace Interface Server initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_interface():
            """Serve the main interface HTML."""
            html_file = Path(__file__).parent / "grace_enhanced_interface.html"
            if html_file.exists():
                return HTMLResponse(html_file.read_text())
            else:
                return HTMLResponse("""
                <html>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>Grace Enhanced Interface</h1>
                    <p>Interface file not found. Please ensure grace_enhanced_interface.html is in the same directory.</p>
                    <p>Available API endpoints:</p>
                    <ul style="display: inline-block; text-align: left;">
                        <li><a href="/api/voice/status">/api/voice/status</a></li>
                        <li><a href="/api/health/comprehensive">/api/health/comprehensive</a></li>
                        <li><a href="/docs">/docs</a> (API Documentation)</li>
                    </ul>
                </body>
                </html>
                """)
        
        @self.app.get("/api/status")
        async def system_status():
            """Get overall system status."""
            return {
                "status": "operational",
                "timestamp": "2025-09-28T18:25:00Z",
                "services": {
                    "voice": "available",
                    "health_monitoring": "active", 
                    "book_ingestion": "ready"
                },
                "version": "1.0.0"
            }
        
        # Voice API endpoints (delegate to kernel)
        @self.app.post("/api/voice/toggle")
        async def voice_toggle():
            """Toggle voice mode."""
            try:
                result = await self.grace_kernel.voice_service.toggle_voice_mode()
                return result
            except Exception as e:
                logger.error(f"Voice toggle error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/voice/status")
        async def voice_status():
            """Get voice system status."""
            try:
                status = self.grace_kernel.voice_service.get_voice_status()
                return status
            except Exception as e:
                logger.error(f"Voice status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/voice/conversation")
        async def voice_conversation():
            """Get voice conversation history."""
            try:
                history = self.grace_kernel.voice_service.get_conversation_history()
                return {
                    "conversation": history,
                    "total_messages": len(history)
                }
            except Exception as e:
                logger.error(f"Voice conversation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Health API endpoints
        @self.app.get("/api/health/comprehensive")
        async def health_comprehensive():
            """Get comprehensive health status."""
            try:
                health = self.grace_kernel.health_dashboard.get_comprehensive_health()
                return health
            except Exception as e:
                logger.error(f"Health check error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/health/check")
        async def health_check():
            """Perform immediate health check."""
            try:
                result = await self.grace_kernel.health_dashboard.perform_health_check()
                return result
            except Exception as e:
                logger.error(f"Health check error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/health/diagnostic")
        async def health_diagnostic():
            """Run system diagnostic."""
            try:
                result = await self.grace_kernel.health_dashboard.run_diagnostic()
                return result
            except Exception as e:
                logger.error(f"Diagnostic error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Book ingestion API endpoints
        @self.app.post("/api/books/ingest")
        async def book_ingest(request: Request):
            """Ingest a book from JSON payload."""
            try:
                data = await request.json()
                
                result = await self.grace_kernel.book_ingestion.ingest_book(
                    content=data.get('content', ''),
                    title=data.get('title', 'Untitled Book'),
                    author=data.get('author'),
                    metadata=data.get('metadata', {})
                )
                
                return result
            except Exception as e:
                logger.error(f"Book ingestion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/books/{job_id}/progress")
        async def book_progress(job_id: str):
            """Get book ingestion progress."""
            try:
                progress = self.grace_kernel.book_ingestion.get_ingestion_progress(job_id)
                if not progress:
                    raise HTTPException(status_code=404, detail="Job not found")
                return progress
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Book progress error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/books/{job_id}/insights")
        async def book_insights(job_id: str):
            """Get book insights."""
            try:
                insights = await self.grace_kernel.book_ingestion.get_book_insights(job_id)
                return {
                    "job_id": job_id,
                    "insights": insights,
                    "total": len(insights)
                }
            except Exception as e:
                logger.error(f"Book insights error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/books/{job_id}/search")
        async def book_search(job_id: str, query: str, limit: int = 10):
            """Search within book content."""
            try:
                results = await self.grace_kernel.book_ingestion.search_book_content(
                    job_id, query, limit
                )
                return {
                    "job_id": job_id,
                    "query": query,
                    "results": results,
                    "total": len(results)
                }
            except Exception as e:
                logger.error(f"Book search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Demo endpoints for testing
        @self.app.post("/api/demo/simulate-voice-message")
        async def simulate_voice_message(request: Request):
            """Simulate receiving a voice message for testing."""
            try:
                data = await request.json()
                message = data.get('message', 'Hello Grace')
                
                # Process through Grace's communication handler
                response = await self.grace_kernel._handle_voice_communication(message)
                
                return {
                    "user_message": message,
                    "grace_response": response,
                    "timestamp": "2025-09-28T18:25:00Z"
                }
            except Exception as e:
                logger.error(f"Simulate voice error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.grace_kernel.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_app():
    """Create the FastAPI application."""
    server = GraceInterfaceServer()
    return server.app, server


async def main():
    """Main server function."""
    app, server = create_app()
    
    try:
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
        
        server_instance = uvicorn.Server(config)
        
        logger.info("🚀 Starting Grace Enhanced Interface Server")
        logger.info("📱 Open your browser to: http://localhost:8080")
        logger.info("📚 API Documentation: http://localhost:8080/docs")
        
        await server_instance.serve()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())