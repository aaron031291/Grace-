import asyncio
import logging
from datetime import datetime
from typing import Optional

from grace.mcp import MCPClient, MCPMessageType, MCPPriority

logger = logging.getLogger(__name__)


class MultiOSKernel:
    """
    Multi-OS Kernel with MCP integration
    
    Dependencies injected by Unified Service
    """
    
    def __init__(self, event_bus, event_factory, trigger_mesh=None):
        self.event_bus = event_bus
        self.event_factory = event_factory
        self.trigger_mesh = trigger_mesh  # Use TriggerMesh if available
        
        # State
        self._running = False
        self._start_time: Optional[datetime] = None
        self._events_processed = 0
        self._error_count = 0
        self._warning_count = 0
        self._heartbeat_task: Optional[asyncio.Task] = None
    
        # MCP Client
        self.mcp_client = MCPClient(
            kernel_name="multi_os_kernel",
            event_bus=event_bus,
            trigger_mesh=trigger_mesh,
            minimum_trust=0.5
        )
    
    async def _heartbeat_loop(self):
        """Heartbeat loop with MCP"""
        while self._running:
            try:
                # Send MCP-validated heartbeat
                await self.mcp_client.send_message(
                    destination="health_monitor",
                    payload={
                        "kernel": "multi_os",
                        "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
                        "events_processed": self._events_processed
                    },
                    message_type=MCPMessageType.HEARTBEAT,
                    schema_name="heartbeat",
                    trust_score=0.95
                )
                
                self._events_processed += 1
                logger.debug(f"Heartbeat emitted: {event.event_id}")
            
            except Exception as e:
                self._error_count += 1
                logger.error(f"Heartbeat error: {e}", exc_info=True)
            
            await asyncio.sleep(5)
    
    async def start(self):
        """Start multi-OS kernel"""
        if self._running:
            logger.warning("Multi-OS kernel already running")
            return
        
        self._running = True
        self._start_time = datetime.utcnow()
        
        logger.info("Multi-OS kernel starting", extra={
            "kernel": "multi_os",
            "start_time": self._start_time.isoformat(),
            "trigger_mesh_enabled": self.trigger_mesh is not None
        })
        
        # Register event subscriptions via TriggerMesh or EventBus
        if self.trigger_mesh:
            self.trigger_mesh.subscribe("kernel.heartbeat", self._on_heartbeat, "multi_os_kernel")
            self.trigger_mesh.subscribe("kernel.command", self._on_command, "multi_os_kernel")
        else:
            self.event_bus.subscribe("kernel.heartbeat", self._on_heartbeat)
            self.event_bus.subscribe("kernel.command", self._on_command)
        
        # Start heartbeat loop
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("Multi-OS kernel started successfully")
    
    async def _on_heartbeat(self, event):
        """Handle heartbeat events from other kernels"""
        logger.debug(f"Heartbeat received: {event.event_id} from {event.source}")
    
    async def _on_command(self, event):
        """Handle commands sent to this kernel"""
        command = event.payload.get("command")
        logger.info(f"Received command: {command}")
        
        # Process command and emit response
        response = self.event_factory.create_event(
            event_type="kernel.command.response",
            payload={"command": command, "status": "processed"},
            correlation_id=event.correlation_id,
            source="multi_os_kernel"
        )
        
        if self.trigger_mesh:
            await self.trigger_mesh.emit(response)
        else:
            await self.event_bus.emit(response)
    
    async def stop(self):
        """Graceful shutdown with cleanup"""
        if not self._running:
            logger.warning("Multi-OS kernel not running")
            return
        
        logger.info("Multi-OS kernel stopping", extra={
            "events_processed": self._events_processed,
            "errors": self._error_count
        })
        
        self._running = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe("kernel.heartbeat", self._on_heartbeat)
        self.event_bus.unsubscribe("kernel.command", self._on_command)
        
        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await asyncio.wait_for(self._heartbeat_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        logger.info("Multi-OS kernel stopped")
    
    def get_health(self) -> dict:
        """Health check with MCP stats"""
        uptime = 0.0
        if self._running and self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        health = {
            "status": "healthy" if self._running else "stopped",
            "running": self._running,
            "uptime_seconds": uptime,
            "heartbeat_interval_seconds": 5,
            "events_processed": self._events_processed,
            "errors": self._error_count,
            "warnings": self._warning_count,
            "heartbeat_task_alive": self._heartbeat_task is not None and not self._heartbeat_task.done() if self._heartbeat_task else False,
            "trigger_mesh_enabled": self.trigger_mesh is not None,
            "mcp_stats": self.mcp_client.get_stats()
        }
        
        return health


# Global instance for backwards compatibility
_instance: Optional[MultiOSKernel] = None


async def start():
    """Legacy start function for backwards compatibility"""
    global _instance
    if _instance is None:
        from grace.integration.event_bus import get_event_bus
        from grace.events.factory import GraceEventFactory
        _instance = MultiOSKernel(get_event_bus(), GraceEventFactory())
    await _instance.start()


async def stop():
    """Legacy stop function"""
    if _instance:
        await _instance.stop()


def get_health():
    """Legacy health function"""
    if _instance:
        return _instance.get_health()
    return {"status": "not_initialized", "running": False}
