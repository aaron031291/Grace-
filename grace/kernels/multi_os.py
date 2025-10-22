import asyncio
import logging
from datetime import datetime
from typing import Optional

from grace.integration.event_bus import get_event_bus
from grace.events.factory import GraceEventFactory

logger = logging.getLogger(__name__)
_factory = GraceEventFactory()
_bus = get_event_bus()

_running = False
_start_time: Optional[datetime] = None
_events_processed = 0
_error_count = 0
_warning_count = 0

async def _heartbeat_loop():
    global _events_processed, _error_count
    
    while _running:
        try:
            event = _factory.create_event(
                event_type="kernel.heartbeat",
                payload={
                    "timestamp": datetime.utcnow().isoformat(),
                    "kernel": "multi_os",
                    "events_processed": _events_processed
                },
                targets=[]
            )
            _bus.publish(event)
            _events_processed += 1
        except Exception as e:
            _error_count += 1
            logger.error(f"Heartbeat error: {e}")
        
        await asyncio.sleep(5)

async def start():
    global _running, _start_time
    if _running:
        return
    _running = True
    _start_time = datetime.utcnow()
    logger.info("multi_os kernel starting")
    
    # Register sample subscriber
    def _on_heartbeat(e):
        logger.debug(f"heartbeat received {e.event_id}")
    
    _bus.subscribe("kernel.heartbeat", _on_heartbeat)
    
    # Kick off loop
    asyncio.create_task(_heartbeat_loop())

async def stop():
    global _running
    _running = False
    logger.info("multi_os kernel stopped")

def get_health():
    """Return kernel-specific health metrics"""
    return {
        "heartbeat_interval": 5,
        "events_processed": _events_processed,
        "errors": _error_count
    }
