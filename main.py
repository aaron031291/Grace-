"""
Grace AI - Main Entry Point
This script initializes and starts the entire Grace AI system.
"""
import asyncio
import logging
import sys
import os
from threading import Thread
from flask import Flask

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
from grace.kernels.resilience_kernel import ResilienceKernel
from grace.kernels.swarm_kernel import SwarmKernel
from grace.mcp.manager import MCPManager
from grace.immune_system import ImmuneSystem, ThreatDetector
from grace.services.observability import ObservabilityService
from grace.services.policy_engine import PolicyEngine
from grace.services.trust_ledger import TrustLedger
from grace.services.websocket_service import WebSocketService
from grace.api import create_app

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
    mcp_manager = MCPManager(event_bus=event_bus, llm_service=llm_service)
    observability_service = ObservabilityService()
    policy_engine = PolicyEngine()
    trust_ledger = TrustLedger()
    websocket_service = WebSocketService()
    
    # Immune System
    immune_system = ImmuneSystem(event_bus=event_bus)
    threat_detector = ThreatDetector(kpi_monitor=kpi_monitor, immutable_logger=immutable_logger)
    
    # Kernels
    cognitive_cortex = CognitiveCortex(
        event_bus=event_bus,
        task_manager=task_manager,
        communication_channel=communication_channel,
        sandbox_manager=None,
        llm_service=llm_service
    )
    
    sentinel_kernel = SentinelKernel(
        event_bus=event_bus,
        llm_service=llm_service
    )
    
    resilience_kernel = ResilienceKernel(event_bus=event_bus)
    swarm_kernel = SwarmKernel(event_bus=event_bus)
    
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
    component_registry.register("mcp_manager", mcp_manager)
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
        "mcp_manager": mcp_manager,
        "observability_service": observability_service,
        "policy_engine": policy_engine,
        "trust_ledger": trust_ledger,
        "websocket_service": websocket_service,
        "immune_system": immune_system,
        "threat_detector": threat_detector,
        "cognitive_cortex": cognitive_cortex,
        "sentinel_kernel": sentinel_kernel,
        "resilience_kernel": resilience_kernel,
        "swarm_kernel": swarm_kernel,
        "consciousness": consciousness,
        "component_registry": component_registry
    }

async def run_consciousness_loop(consciousness):
    """Run the main consciousness loop."""
    await consciousness.run(tick_interval=2.0)

def run_sentinel_loop(sentinel_kernel):
    """Run the sentinel monitoring loop in a separate thread."""
    asyncio.run(sentinel_kernel.monitor(check_interval=10.0))

def run_flask_api(app):
    """Run the Flask API server."""
    app.run(host='0.0.0.0', port=5000, debug=False)

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
        
        # Create Flask app with components
        flask_app = create_app(components)
        
        # Serve static files
        @flask_app.route('/')
        def index():
            return flask_app.send_static_file('index.html')
        
        @flask_app.route('/static/<path:path>')
        def send_static(path):
            return flask_app.send_from_directory('frontend', path)
        
        # Start Flask API in a separate thread
        logger.info("Starting Flask API on http://0.0.0.0:5000")
        flask_thread = Thread(target=run_flask_api, args=(flask_app,), daemon=True)
        flask_thread.start()
        
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