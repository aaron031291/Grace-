"""
Grace Canonical Core Runner
Boots: EventBus → Governance → Trigger Mesh → MLDL quorum → API
Keeps the loop alive with graceful shutdown.
"""

import asyncio
import logging
from grace.core.event_bus import EventBus
from grace.governance.grace_governance_kernel import GraceGovernanceKernel
from grace.layer_02_event_mesh.trigger_mesh import TriggerMesh
from grace.mldl.quorum import MLDLQuorum
from grace.api.api_service import GraceAPIService

logger = logging.getLogger("grace_core_runner")


async def main():
    logger.info("Booting Grace Core Runner...")
    event_bus = EventBus()
    governance = GraceGovernanceKernel(event_bus=event_bus)
    trigger_mesh = TriggerMesh(event_bus=event_bus)
    mldl_quorum = MLDLQuorum(event_bus=event_bus)
    api_service = GraceAPIService(governance=governance, event_bus=event_bus)

    # Boot sequence
    await event_bus.start()
    await governance.start()
    await trigger_mesh.start()
    await mldl_quorum.start()
    await api_service.start()

    logger.info("Grace Core Runner started. Keeping loop alive...")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Grace Core Runner shutting down...")
        await api_service.shutdown()
        await mldl_quorum.shutdown()
        await trigger_mesh.shutdown()
        await governance.shutdown()
        await event_bus.shutdown()
        logger.info("Grace Core Runner shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
