"""
Grace AI Sentinel Kernel - Environmental monitoring and threat detection
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from grace.kernels.base_kernel import BaseKernel

logger = logging.getLogger(__name__)

class SentinelKernel(BaseKernel):
    """Monitors the external environment for threats and opportunities."""
    
    def __init__(self, service_registry=None):
        super().__init__("sentinel_kernel", service_registry)
        self.comm_channel = self.get_service('communication_channel')
        self.notifier = self.get_service('notification_service')
        self.immune_system = self.get_service('immune_system') # Assuming this is registered
        self.events_monitored = 0
        self.logger.info("Sentinel Kernel initialized and services wired.")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a monitoring or alerting task.

        Args:
            task: A dictionary defining the task.
                  Example: {'type': 'monitor_log', 'source': '/var/log/syslog'}
                           {'type': 'alert', 'message': 'High CPU usage detected.'}
        """
        task_type = task.get('type', 'unknown')
        self.logger.info(f"Executing sentinel task of type: {task_type}")

        if not all([self.comm_channel, self.notifier, self.immune_system]):
            error_msg = "One or more required services are not available."
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        try:
            if task_type == 'monitor_event':
                result = await self._monitor_event(task)
            elif task_type == 'send_alert':
                result = await self._send_alert(task)
            else:
                result = {'success': False, 'error': f"Unknown task type: {task_type}"}

            self.events_monitored += 1
            return result
        except Exception as e:
            self.logger.error(f"Error during sentinel execution: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _monitor_event(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor a specific event or data source."""
        source = task.get('source', 'unknown_source')
        self.logger.info(f"Monitoring event source: {source}")

        # Simulate monitoring
        await asyncio.sleep(0.1)
        
        # If a threat is detected, report it to the immune system
        threat_detected = True # Placeholder for real detection logic
        if threat_detected:
            threat_info = {'source': source, 'level': 'medium', 'description': 'Suspicious activity detected.'}
            await self.immune_system.report_threat(threat_info)
            self.logger.warning(f"Threat reported to Immune System from source: {source}")
            return {'success': True, 'monitored': True, 'threat_reported': True, 'source': source}

        return {'success': True, 'monitored': True, 'threat_reported': False, 'source': source}

    async def _send_alert(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send an alert through the notification and communication services."""
        message = task.get('message', 'No message provided.')
        channel = task.get('channel', 'default')
        self.logger.info(f"Sending alert to channel '{channel}': {message}")

        # Use NotificationService to format and log the alert
        await self.notifier.send(message, level='alert')

        # Use CommunicationChannel to broadcast the alert
        await self.comm_channel.broadcast(event='alert', data={'message': message})

        return {'success': True, 'alert_sent': True, 'channel': channel}

    async def health_check(self) -> Dict[str, Any]:
        """Return the health status of the kernel."""
        return {
            'name': self.name,
            'running': self.is_running,
            'services': {
                'comm_channel': 'wired' if self.comm_channel else 'missing',
                'notifier': 'wired' if self.notifier else 'missing',
                'immune_system': 'wired' if self.immune_system else 'missing',
            },
            'events_monitored': self.events_monitored,
        }
