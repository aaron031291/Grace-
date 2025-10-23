"""
Grace AI - Main Entry Point
This script initializes and starts the entire Grace AI system.
"""

import asyncio
import logging
import sys
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Grace-AI")

from grace.core import EventBus, ImmutableLogger, KPITrustMonitor, ComponentRegistry
from grace.services.task_manager import TaskManager
from grace.services.notification_service import NotificationService
from grace.services.communication_channel import CommunicationChannel
from grace.services.llm_service import LLMService
from grace.agents.remote_agent import RemoteAgent
from grace.consciousness import Consciousness
from grace.kernels.cognitive_cortex import CognitiveCortex
from grace.kernels.sentinel_kernel import SentinelKernel

async def initialize_system():
    """Initialize all Grace system components."""
    logger.info("Initializing Grace AI System...")
    
    # Core infrastructure
    event_bus = EventBus()
    immutable_logger = ImmutableLogger()
    kpi_monitor = KPITrustMonitor()
    component_registry = ComponentRegistry()
    
    # Services
    task_manager = TaskManager()
    notification_service = NotificationService()
    communication_channel = CommunicationChannel()
    llm_service = LLMService()
    remote_agent = RemoteAgent()
    
    # Kernels
    cognitive_cortex = CognitiveCortex(
        event_bus=event_bus,
        task_manager=task_manager,
        communication_channel=communication_channel,
        sandbox_manager=None,  # Will be set later
        llm_service=llm_service
    )
    
    sentinel_kernel = SentinelKernel(
        event_bus=event_bus,
        llm_service=llm_service
    )
    
    # Consciousness loop
    consciousness = Consciousness(
        cognitive_cortex=cognitive_cortex,
        task_manager=task_manager,
        kpi_monitor=kpi_monitor,
        component_registry=component_registry,
        event_bus=event_bus,
        immutable_logger=immutable_logger
    )
    
    # Register components
    component_registry.register("event_bus", event_bus)
    component_registry.register("immutable_logger", immutable_logger)
    component_registry.register("kpi_monitor", kpi_monitor)
    component_registry.register("task_manager", task_manager)
    component_registry.register("communication_channel", communication_channel)
    component_registry.register("llm_service", llm_service)
    component_registry.register("cognitive_cortex", cognitive_cortex)
    component_registry.register("sentinel_kernel", sentinel_kernel)
    component_registry.register("consciousness", consciousness)
    
    logger.info("All components initialized successfully")
    
    return {
        "event_bus": event_bus,
        "immutable_logger": immutable_logger,
        "kpi_monitor": kpi_monitor,
        "task_manager": task_manager,
        "notification_service": notification_service,
        "communication_channel": communication_channel,
        "llm_service": llm_service,
        "remote_agent": remote_agent,
        "cognitive_cortex": cognitive_cortex,
        "sentinel_kernel": sentinel_kernel,
        "consciousness": consciousness,
        "component_registry": component_registry
    }

async def run_consciousness_loop(consciousness):
    """Run the main consciousness loop."""
    await consciousness.run(tick_interval=2.0)

def run_sentinel_loop(sentinel_kernel):
    """Run the sentinel monitoring loop in a separate thread."""
    asyncio.run(sentinel_kernel.monitor(check_interval=10.0))

async def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Starting Grace AI System")
    logger.info("="*60)
    
    try:
        # Initialize system
        components = await initialize_system()
        consciousness = components["consciousness"]
        sentinel_kernel = components["sentinel_kernel"]
        
        # Start Sentinel in a separate thread
        sentinel_thread = Thread(target=run_sentinel_loop, args=(sentinel_kernel,), daemon=True)
        sentinel_thread.start()
        logger.info("Sentinel Kernel started in background thread")
        
        # Start Consciousness loop
        logger.info("Starting Consciousness loop...")
        await run_consciousness_loop(consciousness)
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())