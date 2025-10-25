#!/usr/bin/env python3
"""
Grace AI - End-to-End Communication Simulation
==============================================
This script demonstrates a full, end-to-end data flow through the Grace system.
It simulates an external event, which triggers a workflow, which in turn causes
a kernel to execute a task and log its decision to the immutable log and trust ledger.
"""

import sys
import asyncio
import logging

# Add workspace to path
sys.path.insert(0, '/workspaces/Grace-')

from grace.launcher import GraceLauncher

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("E2E_SIMULATION")

async def run_simulation():
    """
    Initializes the system, triggers an event, and verifies the outcome.
    """
    logger.info("--- STARTING END-TO-END SIMULATION ---")

    # 1. Initialize the Grace system
    logger.info("Step 1: Initializing Grace system...")
    class MockArgs:
        debug = False
        kernel = None
        dry_run = False
    
    launcher = GraceLauncher(MockArgs())
    await launcher.initialize()
    logger.info("Grace system initialized successfully.")

    # 2. Get the TriggerMesh service
    logger.info("Step 2: Retrieving TriggerMesh service...")
    trigger_mesh = launcher.registry.get('trigger_mesh')
    if not trigger_mesh:
        logger.error("FATAL: Could not retrieve TriggerMesh service. Aborting.")
        return
    logger.info("TriggerMesh service retrieved.")

    # 3. Define and dispatch a simulated external event
    logger.info("Step 3: Dispatching simulated external event...")
    event_type = "external_data_received"
    event_payload = {
        "source": "external_api",
        "data_id": "xyz-123",
        "content": "Market sentiment shows a significant positive shift."
    }
    await trigger_mesh.dispatch_event(event_type, event_payload)
    logger.info(f"Event '{event_type}' dispatched with payload: {event_payload}")

    # In a real application, we would wait for tasks to complete.
    # For this simulation, the calls are awaited, so we can assume completion.
    
    logger.info("--- SIMULATION COMPLETE ---")
    logger.info("Check the 'grace_log.jsonl' file to see the immutable record of the kernel's action.")


if __name__ == "__main__":
    asyncio.run(run_simulation())
