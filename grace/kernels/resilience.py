import asyncio
import logging
from datetime import datetime
from typing import Optional

from grace.mcp import MCPClient

logger = logging.getLogger(__name__)


class ResilienceKernel:
    """
    Resilience Kernel with MCP integration
    """
    
    def __init__(self, event_bus, event_factory, governance_engine, trigger_mesh=None):
        self.event_bus = event_bus
        self.event_factory = event_factory
        self.governance = governance_engine
        self.trigger_mesh = trigger_mesh
        
        # Inject trigger_mesh into governance for consensus
        if self.governance and trigger_mesh:
            self.governance.trigger_mesh = trigger_mesh
        
        # State
        self._running = False
        self._start_time: Optional[datetime] = None
        self._events_processed = 0
        self._error_count = 0
        self._escalations = 0
        self._validations_passed = 0
        self._validations_failed = 0
        
        # MCP Client
        self.mcp_client = MCPClient(
            kernel_name="resilience_kernel",
            event_bus=event_bus,
            trigger_mesh=trigger_mesh,
            minimum_trust=0.7
        )
    
    async def start(self):
        """Start resilience kernel"""
        if self._running:
            logger.warning("Resilience kernel already running")
            return
        
        if not self.governance:
            raise RuntimeError("GovernanceEngine required for resilience kernel")
        
        self._running = True
        self._start_time = datetime.utcnow()
        
        logger.info("Resilience kernel starting", extra={"start_time": self._start_time.isoformat()})
        
        # Subscribe to error events
        self.event_bus.subscribe("system.error", self._on_error)
        self.event_bus.subscribe("kernel.failure", self._on_failure)
        
        logger.info("Resilience kernel started")
    
    async def _on_error(self, event):
        """Handle system error events"""
        try:
            self._events_processed += 1
            
            logger.info(f"Processing error event: {event.event_id}", extra={
                "event_type": event.event_type,
                "trust_score": event.trust_score
            })
            
            # Validate event (will request MLDL consensus if needed)
            result = await self.governance.validate(
                event,
                context={"error_severity": "high"},
                request_mldl_consensus=True  # Enable consensus for errors
            )
            
            if result.passed:
                self._validations_passed += 1
                logger.info(f"Validation passed for {event.event_id}")
            else:
                self._validations_failed += 1
                logger.warning(f"Validation failed for {event.event_id}", extra={
                    "violations": result.violations,
                    "mldl_consensus": result.decision.get("mldl_consensus") if result.decision else None
                })
                
                # Escalate if failed
                escalation = await self.governance.escalate(
                    event,
                    reason="Auto-resilience: validation failed",
                    level="high"
                )
                self._escalations += 1
                
                logger.warning(f"Escalated to {escalation.assigned_to}", extra={
                    "escalation_level": escalation.escalation_level,
                    "actions": escalation.actions_taken
                })
                
                # Emit escalation event
                escalation_event = self.event_factory.create_event(
                    event_type="resilience.escalation",
                    payload={
                        "original_event_id": event.event_id,
                        "escalation_level": escalation.escalation_level,
                        "assigned_to": escalation.assigned_to,
                        "reason": escalation.reason
                    },
                    source="resilience_kernel",
                    priority="high"
                )
                await self.event_bus.emit(escalation_event)
        
        except Exception as e:
            self._error_count += 1
            logger.error(f"Resilience processing error: {e}", exc_info=True)
    
    async def _on_failure(self, event):
        """Handle kernel failure events"""
        failed_kernel = event.payload.get("kernel")
        reason = event.payload.get("reason", "unknown")
        
        logger.error(f"Kernel failure detected: {failed_kernel}, reason: {reason}")
        
        # Emit recovery event
        recovery_event = self.event_factory.create_event(
            event_type="resilience.recovery",
            payload={
                "failed_kernel": failed_kernel,
                "reason": reason,
                "action": "restart_initiated"
            },
            source="resilience_kernel",
            priority="high"
        )
        await self.event_bus.emit(recovery_event)
    
    async def stop(self):
        """Graceful shutdown"""
        if not self._running:
            logger.warning("Resilience kernel not running")
            return
        
        logger.info("Resilience kernel stopping", extra={
            "escalations": self._escalations,
            "validations": self._validations_passed + self._validations_failed
        })
        
        self._running = False
        
        # Unsubscribe
        self.event_bus.unsubscribe("system.error", self._on_error)
        self.event_bus.unsubscribe("kernel.failure", self._on_failure)
        
        logger.info("Resilience kernel stopped")
    
    def get_health(self) -> dict:
        """Health check with actual metrics"""
        uptime = 0.0
        if self._running and self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        total_validations = self._validations_passed + self._validations_failed
        pass_rate = 0.0
        if total_validations > 0:
            pass_rate = self._validations_passed / total_validations
        
        return {
            "status": "healthy" if self._running else "stopped",
            "running": self._running,
            "uptime_seconds": uptime,
            "escalations": self._escalations,
            "validations_passed": self._validations_passed,
            "validations_failed": self._validations_failed,
            "validation_pass_rate": pass_rate,
            "errors": self._error_count,
            "events_processed": self._events_processed,
            "governance_engine_active": self.governance is not None
        }


# Global instance for backwards compatibility
_instance: Optional[ResilienceKernel] = None


async def start():
    """Legacy start function"""
    global _instance
    if _instance is None:
        from grace.integration.event_bus import get_event_bus
        from grace.events.factory import GraceEventFactory
        from grace.governance.engine import GovernanceEngine
        _instance = ResilienceKernel(get_event_bus(), GraceEventFactory(), GovernanceEngine())
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
