"""
Compatibility shim for MemoryOrchestrator used by MCP pushback.

This is a minimal implementation that provides the interface expected by the
existing MCP code: a singleton accessor and a `request_healing` method.

For real behavior, replace this with the project's memory orchestration logic.
"""
import asyncio
from typing import Any, Dict


class MemoryOrchestrator:
    _instance = None

    def __init__(self):
        # placeholder state
        self._queue = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def request_healing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pretend to schedule a healing job and return a ticket dict."""
        # simulate async work
        await asyncio.sleep(0.01)
        ticket = {
            "ticket_id": f"heal_{int(asyncio.get_event_loop().time()*1000)}",
            "status": "scheduled",
            "context": context,
        }
        self._queue.append(ticket)
        return ticket

    def get_pending(self):
        return list(self._queue)


# convenience module-level accessor
def get_instance():
    return MemoryOrchestrator.get_instance()
