import asyncio
import logging
from datetime import datetime
from typing import Optional

from grace.integration.event_bus import get_event_bus
from grace.events.factory import GraceEventFactory
from grace.governance.engine import GovernanceEngine

logger = logging.getLogger(__name__)
_bus = get_event_bus()
_factory = GraceEventFactory()
_engine = GovernanceEngine()

_running = False
_start_time: Optional[datetime] = None
_events_processed = 0
_error_count = 0
_escalations = 0
_validations_passed = 0
_validations_failed = 0

async def _on_error(event):
    global _events_processed, _error_count, _escalations, _validations_passed, _validations_failed
    
    try:
        _events_processed += 1
        
        # validate and escalate if needed
        result = await _engine.validate(event)
        
        if result.passed:
            _validations_passed += 1
        else:
            _validations_failed += 1
            await _engine.escalate(event, reason="auto-resilience escalation", level="high")
            _escalations += 1
    
    except Exception as e:
        _error_count += 1
        logger.error(f"Resilience error: {e}")

async def start():
    global _running, _start_time
    if _running:
        return
    _running = True
    _start_time = datetime.utcnow()
    logger.info("resilience kernel starting")
    
    _bus.subscribe("system.error", lambda e: asyncio.create_task(_on_error(e)))

async def stop():
    global _running
    _running = False
    logger.info("resilience kernel stopping")

def get_health():
    """Return kernel-specific health metrics"""
    return {
        "escalations": _escalations,
        "validations_passed": _validations_passed,
        "validations_failed": _validations_failed,
        "errors": _error_count
    }
