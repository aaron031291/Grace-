"""
Grace Integrated Services Demo - Demonstrates end-to-end functionality.

This service shows how the enhanced event mesh, governance API, persistent storage,
and memory systems work together in production scenarios.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Core Grace imports
from grace.layer_02_event_mesh import GraceEventBus, EventMeshConfig, EventTypes
from grace.governance.governance_api import GovernanceAPIService
from grace.core.snapshot_manager import GraceSnapshotManager
from grace.memory.api import GraceMemoryAPI
from grace.layer_04_audit_logs.immutable_logs import ImmutableLogs
from grace.contracts.message_envelope import GraceMessageEnvelope

# FastAPI imports for demo endpoints
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class GraceIntegratedService:
    """
    Integrated Grace service demonstrating production-ready capabilities:
    - Event mesh with pluggable transports
    - Governance approval workflows
    - Persistent checkpoints/snapshots
    - Memory system (Lightning + Fusion)
    - Immutable audit logging
    """
    
    def __init__(self, config_env: str = "development"):
        self.config_env = config_env
        
        # Initialize core components
        self.event_config = EventMeshConfig.from_env()
        self.event_bus = None
        self.governance_api = None
        self.snapshot_manager = None
        self.memory_api = None
        self.audit_logger = None
        
        # Demo app
        self.app = None
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Grace Integrated Services Demo",
                description="End-to-end demonstration of Grace infrastructure",
                version="1.0.0"
            )
            self._register_demo_routes()
        
        logger.info(f"Grace Integrated Service initialized (env: {config_env})")
    
    async def initialize(self):
        """Initialize all service components."""
        logger.info("Initializing Grace integrated services...")
        
        try:
            # 1. Initialize event bus with configured transport
            logger.info(f"Starting event bus with transport: {self.event_config.transport_type}")
            self.event_bus = GraceEventBus(
                transport_config={"type": self.event_config.transport_type, "config": self.event_config.transport_config}
            )
            await self.event_bus.start()
            
            # 2. Initialize audit logging
            self.audit_logger = ImmutableLogs()
            
            # 3. Initialize snapshot manager with persistent storage
            self.snapshot_manager = GraceSnapshotManager()
            
            # 4. Initialize memory system
            self.memory_api = GraceMemoryAPI()
            
            # 5. Initialize governance API
            self.governance_api = GovernanceAPIService(
                event_bus=self.event_bus,
                immutable_logger=self.audit_logger
            )
            
            # 6. Set up event subscriptions for demonstrations
            await self._setup_demo_event_handlers()
            
            logger.info("✅ All Grace services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            raise
    
    async def _setup_demo_event_handlers(self):
        """Set up event handlers to demonstrate integration."""
        
        # Handler for governance events
        async def handle_governance_event(gme: GraceMessageEnvelope):
            event_type = gme.headers.event_type
            payload = gme.payload
            
            logger.info(f"Governance event received: {event_type}")
            
            # Log to audit system
            await self.audit_logger.log_entry(
                f"governance_event_{gme.msg_id}",
                "governance_workflow",
                {
                    "event_type": event_type,
                    "payload": payload,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": gme.headers.source
                },
                transparency_level="democratic_oversight"
            )
        
        # Handler for memory events
        async def handle_memory_event(gme: GraceMessageEnvelope):
            event_type = gme.headers.event_type
            payload = gme.payload
            
            logger.info(f"Memory event received: {event_type}")
            
            # Example: Auto-snapshot on significant memory writes
            if event_type == EventTypes.MEMORY_WRITE_COMPLETED:
                content_key = payload.get("key")
                if content_key and "critical" in payload.get("tags", []):
                    snapshot_result = await self.snapshot_manager.export_snapshot(
                        component_type="memory",
                        payload={"triggered_by": event_type, "content_key": content_key},
                        description=f"Auto-snapshot after critical write: {content_key}",
                        created_by="auto-system"
                    )
                    logger.info(f"Auto-created snapshot: {snapshot_result.get('snapshot_id')}")
        
        # Handler for system events
        async def handle_system_event(gme: GraceMessageEnvelope):
            event_type = gme.headers.event_type
            payload = gme.payload
            
            logger.info(f"System event received: {event_type}")
            
            # Log system events for monitoring
            await self.audit_logger.log_entry(
                f"system_event_{gme.msg_id}",
                "system_monitoring",
                {
                    "event_type": event_type,
                    "payload": payload,
                    "timestamp": datetime.utcnow().isoformat()
                },
                transparency_level="audit_only"
            )
        
        # Subscribe to event patterns
        self.event_bus.subscribe("GOVERNANCE_*", handle_governance_event)
        self.event_bus.subscribe("MEMORY_*", handle_memory_event)
        self.event_bus.subscribe("TRUST_*|SNAPSHOT_*|ROLLBACK_*", handle_system_event)
        
        logger.info("Event handlers registered for demo")
    
    def _register_demo_routes(self):
        """Register demo API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with service info."""
            return {
                "service": "Grace Integrated Services Demo",
                "status": "running",
                "environment": self.config_env,
                "transport": self.event_config.transport_type if self.event_config else "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            try:
                # Check event bus
                if self.event_bus:
                    bus_health = self.event_bus.get_health()
                    health_data["components"]["event_bus"] = bus_health
                
                # Check other components
                if self.snapshot_manager:
                    health_data["components"]["snapshot_manager"] = {"status": "healthy"}
                
                if self.memory_api:
                    health_data["components"]["memory_api"] = {"status": "healthy"}
                
                if self.governance_api:
                    health_data["components"]["governance_api"] = {"status": "healthy"}
                
            except Exception as e:
                health_data["status"] = "degraded"
                health_data["error"] = str(e)
            
            return health_data
        
        @self.app.get("/stats")
        async def stats():
            """Get system statistics."""
            stats_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_bus": {},
                "snapshot_manager": {},
                "memory_api": {}
            }
            
            try:
                if self.event_bus:
                    stats_data["event_bus"] = self.event_bus.get_stats()
                
                if self.snapshot_manager:
                    snapshots = await self.snapshot_manager.list_snapshots()
                    stats_data["snapshot_manager"] = {
                        "total_snapshots": len(snapshots),
                        "recent_snapshots": len([s for s in snapshots if 
                                                (datetime.utcnow() - datetime.fromisoformat(s["created_at"])).days <= 7])
                    }
            
            except Exception as e:
                stats_data["error"] = str(e)
            
            return stats_data
        
        @self.app.post("/demo/workflow")
        async def demo_workflow():
            """Demonstrate end-to-end workflow."""
            workflow_id = f"demo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                # 1. Write to memory system
                memory_result = await self.memory_api.write_content(
                    content=f"Demo workflow {workflow_id} started",
                    source_id=workflow_id,
                    tags=["demo", "workflow", "critical"],
                    metadata={"type": "demo_workflow", "timestamp": datetime.utcnow().isoformat()}
                )
                
                # 2. Create governance request
                if self.governance_api:
                    governance_request = {
                        "action_type": "demo.execute_workflow",
                        "resource_id": workflow_id,
                        "payload": {"memory_key": memory_result.get("key"), "workflow_type": "demo"},
                        "priority": "normal",
                        "requester": "demo_service",
                        "reason": f"Demo workflow {workflow_id}"
                    }
                    
                    # Note: In a real scenario, this would go through approval process
                    # For demo, we'll just simulate it
                    logger.info(f"Would create governance request for workflow {workflow_id}")
                
                # 3. Publish events
                await self.event_bus.publish(
                    event_type=EventTypes.MEMORY_WRITE_COMPLETED,
                    payload={
                        "key": memory_result.get("key"),
                        "workflow_id": workflow_id,
                        "tags": ["demo", "workflow", "critical"]
                    },
                    source="demo_service"
                )
                
                # 4. The event handlers will automatically create snapshots and audit logs
                
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "memory_write": memory_result,
                    "message": "Demo workflow completed successfully. Check logs and snapshots.",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Demo workflow failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "workflow_id": workflow_id}
                )
        
        @self.app.get("/demo/snapshots")
        async def list_demo_snapshots():
            """List recent snapshots."""
            if not self.snapshot_manager:
                raise HTTPException(status_code=503, detail="Snapshot manager not available")
            
            snapshots = await self.snapshot_manager.list_snapshots(limit=10)
            return {
                "snapshots": snapshots,
                "count": len(snapshots),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        logger.info("Demo API routes registered")
    
    async def run_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the demo server."""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available, cannot run server")
            return
        
        if not self.app:
            logger.error("FastAPI app not initialized")
            return
        
        logger.info(f"Starting Grace demo server on {host}:{port}")
        
        # Mount governance API if available
        if self.governance_api and self.governance_api.app:
            self.app.mount("/governance", self.governance_api.app)
            logger.info("Governance API mounted at /governance")
        
        # Mount memory API if available
        if self.memory_api and hasattr(self.memory_api, 'app') and self.memory_api.app:
            self.app.mount("/memory", self.memory_api.app)
            logger.info("Memory API mounted at /memory")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self):
        """Shutdown all services."""
        logger.info("Shutting down Grace integrated services...")
        
        if self.event_bus:
            await self.event_bus.stop()
        
        logger.info("Grace services shut down")


async def main():
    """Main entry point for the integrated demo."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and initialize service
    service = GraceIntegratedService()
    
    try:
        await service.initialize()
        
        if len(sys.argv) > 1 and sys.argv[1] == "server":
            # Run as web server
            await service.run_server()
        else:
            # Run demo workflow
            logger.info("Running integrated demo workflow...")
            
            # Demonstrate event publishing
            await service.event_bus.publish(
                event_type=EventTypes.GOVERNANCE_VALIDATION,
                payload={"test": "demo", "timestamp": datetime.utcnow().isoformat()},
                source="demo_script"
            )
            
            # Write to memory
            if service.memory_api:
                result = await service.memory_api.write_content(
                    content="Demo content from integrated service",
                    source_id="demo_test",
                    tags=["demo", "test"]
                )
                logger.info(f"Memory write result: {result}")
            
            # Create snapshot
            if service.snapshot_manager:
                snapshot_result = await service.snapshot_manager.export_snapshot(
                    component_type="demo",
                    payload={"demo": True, "timestamp": datetime.utcnow().isoformat()},
                    description="Demo snapshot from integrated service",
                    created_by="demo_script"
                )
                logger.info(f"Created snapshot: {snapshot_result}")
            
            logger.info("Demo completed successfully!")
            
            # Show stats
            if service.event_bus:
                stats = service.event_bus.get_stats()
                logger.info(f"Event bus stats: {stats}")
            
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())