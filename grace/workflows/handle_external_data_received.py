"""
Grace AI - Demo Workflow Handler for external_data_received events
"""
import logging

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "handle_external_data_received"
EVENTS = ["external_data_received"]


class _DemoWorkflow:
    """Minimal demo workflow that executes when external_data_received event fires."""
    name = WORKFLOW_NAME
    EVENTS = EVENTS

    async def execute(self, event):
        """Execute the workflow handler."""
        event_id = event.get("id", "unknown")
        logger.info(f"HANDLER_START {WORKFLOW_NAME} event_id={event_id}")
        
        # Do something visible (processing the external data)
        payload = event.get("payload", {})
        logger.info(f"Processing external data: {payload}")
        
        logger.info(f"HANDLER_DONE {WORKFLOW_NAME} event_id={event_id}")
        return {"status": "success", "workflow": WORKFLOW_NAME}


# Export the workflow instance
workflow = _DemoWorkflow()
