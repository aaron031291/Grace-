#!/usr/bin/env python3
"""
Grace Live Status Dashboard
Real-time monitoring and alerting for Grace system health.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import our system check capabilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_check import GraceSystemHealthChecker, SystemHealthReport, ComponentStatus

logger = logging.getLogger(__name__)

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str  # 'component_unhealthy', 'overall_degraded', 'ooda_failed', 'governance_failed'
    severity: str  # 'critical', 'warning', 'info'
    threshold: Optional[float] = None
    enabled: bool = True

@dataclass
class Alert:
    """System alert."""
    id: str
    rule: str
    severity: str
    message: str
    component: Optional[str] = None
    timestamp: str = None
    acknowledged: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class GraceLiveDashboard:
    """Live status dashboard with real-time monitoring and alerting."""
    
    def __init__(self, update_interval: int = 30):
        self.app = FastAPI(title="Grace Live Status Dashboard", version="1.0.0")
        self.update_interval = update_interval
        self.running = False
        self.websocket_connections: Set[WebSocket] = set()
        
        # Alert management
        self.alerts: List[Alert] = []
        self.alert_rules = self._create_default_alert_rules()
        
        # Monitoring data
        self.current_report: Optional[SystemHealthReport] = None
        self.history: List[SystemHealthReport] = []
        self.max_history = 288  # 24 hours at 5-minute intervals
        
        # Setup routes
        self._setup_routes()
    
    def _create_default_alert_rules(self) -> List[AlertRule]:
        """Create default alerting rules."""
        return [
            AlertRule(
                name="Component Unhealthy",
                condition="component_unhealthy",
                severity="critical"
            ),
            AlertRule(
                name="Overall System Degraded",
                condition="overall_degraded",
                severity="warning"
            ),
            AlertRule(
                name="OODA Loop Failed",
                condition="ooda_failed",
                severity="critical"
            ),
            AlertRule(
                name="Governance System Failed",
                condition="governance_failed",
                severity="critical"
            ),
            AlertRule(
                name="Telemetry Offline",
                condition="telemetry_failed",
                severity="warning"
            )
        ]
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def dashboard():
            """Main dashboard page."""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/status")
        async def get_current_status():
            """Get current system status."""
            if not self.current_report:
                raise HTTPException(status_code=503, detail="No status report available")
            return asdict(self.current_report)
        
        @self.app.get("/api/history")
        async def get_status_history(hours: int = 24):
            """Get status history."""
            cutoff = datetime.now() - timedelta(hours=hours)
            cutoff_str = cutoff.isoformat()
            
            filtered_history = [
                report for report in self.history 
                if report.timestamp >= cutoff_str
            ]
            
            return [asdict(report) for report in filtered_history]
        
        @self.app.get("/api/alerts")
        async def get_alerts(active_only: bool = True):
            """Get system alerts."""
            if active_only:
                return [asdict(alert) for alert in self.alerts if not alert.acknowledged]
            return [asdict(alert) for alert in self.alerts]
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert."""
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    await self._broadcast_alert_update()
                    return {"status": "acknowledged"}
            raise HTTPException(status_code=404, detail="Alert not found")
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get system metrics summary."""
            if not self.current_report:
                raise HTTPException(status_code=503, detail="No status report available")
            
            # Calculate metrics
            total_components = len(self.current_report.components)
            healthy_components = sum(1 for c in self.current_report.components if c.status == "healthy")
            
            # Response time statistics
            response_times = [c.response_time for c in self.current_report.components if c.response_time is not None]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                "total_components": total_components,
                "healthy_components": healthy_components,
                "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
                "avg_response_time": avg_response_time,
                "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
                "ooda_functional": self.current_report.ooda_loop_functional,
                "governance_functional": self.current_report.governance_functional,
                "telemetry_active": self.current_report.telemetry_active
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                # Send current status immediately
                if self.current_report:
                    await websocket.send_json({
                        "type": "status_update",
                        "data": asdict(self.current_report)
                    })
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                self.websocket_connections.discard(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get the dashboard HTML page."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Grace System Health Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .header h1 { margin: 0; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-healthy { color: #27ae60; }
        .status-degraded { color: #f39c12; }
        .status-unhealthy { color: #e74c3c; }
        .status-missing { color: #95a5a6; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .alert { 
            padding: 15px; margin: 10px 0; border-radius: 4px; 
            background: #fff3cd; border: 1px solid #ffeaa7; color: #856404;
        }
        .alert.critical { background: #f8d7da; border-color: #f5c6cb; color: #721c24; }
        .component { 
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 0; border-bottom: 1px solid #eee;
        }
        .component:last-child { border-bottom: none; }
        .badge { 
            padding: 4px 8px; border-radius: 12px; font-size: 12px; 
            background: #ddd; color: #666; font-weight: bold;
        }
        .badge.healthy { background: #d4edda; color: #155724; }
        .badge.degraded { background: #fff3cd; color: #856404; }
        .badge.unhealthy { background: #f8d7da; color: #721c24; }
        .badge.missing { background: #e2e3e5; color: #6c757d; }
        .refresh-btn { 
            background: #3498db; color: white; border: none; 
            padding: 10px 20px; border-radius: 4px; cursor: pointer; 
        }
        #connection-status { 
            position: fixed; top: 10px; right: 10px; 
            padding: 5px 10px; border-radius: 4px; font-size: 12px;
        }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Grace System Health Dashboard</h1>
        <p>Real-time monitoring of Grace governance system</p>
    </div>
    
    <div id="connection-status" class="disconnected">Connecting...</div>
    
    <div class="status-grid">
        <div class="card">
            <h3>System Overview</h3>
            <div class="metric">
                <span>Overall Status:</span>
                <span id="overall-status" class="status-healthy">Loading...</span>
            </div>
            <div class="metric">
                <span>Components Healthy:</span>
                <span id="component-health">Loading...</span>
            </div>
            <div class="metric">
                <span>OODA Loop:</span>
                <span id="ooda-status">Loading...</span>
            </div>
            <div class="metric">
                <span>Governance:</span>
                <span id="governance-status">Loading...</span>
            </div>
            <div class="metric">
                <span>Telemetry:</span>
                <span id="telemetry-status">Loading...</span>
            </div>
            <button class="refresh-btn" onclick="refreshData()">Refresh Now</button>
        </div>
        
        <div class="card">
            <h3>Active Alerts</h3>
            <div id="alerts-list">No alerts</div>
        </div>
        
        <div class="card">
            <h3>Component Status</h3>
            <div id="components-list">Loading...</div>
        </div>
        
        <div class="card">
            <h3>Performance Metrics</h3>
            <div class="metric">
                <span>Avg Response Time:</span>
                <span id="avg-response-time">Loading...</span>
            </div>
            <div class="metric">
                <span>Last Update:</span>
                <span id="last-update">Loading...</span>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('connection-status').className = 'connected';
                document.getElementById('connection-status').textContent = 'Connected';
                reconnectAttempts = 0;
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'status_update') {
                    updateDashboard(message.data);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').className = 'disconnected';
                document.getElementById('connection-status').textContent = 'Disconnected';
                
                // Attempt to reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    setTimeout(connectWebSocket, 5000);
                    reconnectAttempts++;
                }
            };
        }
        
        function updateDashboard(data) {
            // Update overall status
            document.getElementById('overall-status').textContent = data.overall_status.toUpperCase();
            document.getElementById('overall-status').className = `status-${data.overall_status}`;
            
            // Update component health
            const total = data.components.length;
            const healthy = data.components.filter(c => c.status === 'healthy').length;
            document.getElementById('component-health').textContent = `${healthy}/${total} (${Math.round(healthy/total*100)}%)`;
            
            // Update system flags
            document.getElementById('ooda-status').textContent = data.ooda_loop_functional ? 'Functional' : 'Failed';
            document.getElementById('governance-status').textContent = data.governance_functional ? 'Functional' : 'Failed';
            document.getElementById('telemetry-status').textContent = data.telemetry_active ? 'Active' : 'Inactive';
            
            // Update components list
            const componentsList = document.getElementById('components-list');
            componentsList.innerHTML = data.components.map(component => `
                <div class="component">
                    <span>${component.name}</span>
                    <span class="badge ${component.status}">${component.status.toUpperCase()}</span>
                </div>
            `).join('');
            
            // Update performance metrics
            const responseTimes = data.components.filter(c => c.response_time !== null).map(c => c.response_time);
            const avgResponseTime = responseTimes.length > 0 ? 
                (responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length).toFixed(3) : 'N/A';
            document.getElementById('avg-response-time').textContent = avgResponseTime + 's';
            
            document.getElementById('last-update').textContent = new Date().toLocaleString();
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }
        
        // Initialize
        connectWebSocket();
        refreshData();
    </script>
</body>
</html>
        """
    
    async def start_monitoring(self):
        """Start the monitoring loop."""
        self.running = True
        logger.info("Starting Grace live monitoring...")
        
        while self.running:
            try:
                # Run system health check
                checker = GraceSystemHealthChecker()
                report = await checker.run_comprehensive_check()
                
                # Update current report
                old_report = self.current_report
                self.current_report = report
                
                # Add to history
                self.history.append(report)
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                
                # Check for alerts
                await self._check_alerts(report, old_report)
                
                # Broadcast update to WebSocket clients
                await self._broadcast_status_update(report)
                
                logger.info(f"System status: {report.overall_status} - {len([c for c in report.components if c.status == 'healthy'])}/{len(report.components)} components healthy")
                
            except Exception as e:
                logger.error(f"Error during monitoring cycle: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.update_interval)
    
    async def _check_alerts(self, current_report: SystemHealthReport, previous_report: Optional[SystemHealthReport]):
        """Check for new alerts based on system status."""
        new_alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check conditions
            alert_triggered = False
            alert_message = ""
            component_name = None
            
            if rule.condition == "component_unhealthy":
                unhealthy_components = [c for c in current_report.components if c.status == "unhealthy"]
                if unhealthy_components:
                    alert_triggered = True
                    component_name = unhealthy_components[0].name
                    alert_message = f"Component '{component_name}' is unhealthy: {unhealthy_components[0].error}"
            
            elif rule.condition == "overall_degraded" and current_report.overall_status == "degraded":
                alert_triggered = True
                alert_message = "Overall system status is degraded"
            
            elif rule.condition == "ooda_failed" and not current_report.ooda_loop_functional:
                alert_triggered = True
                alert_message = "OODA loop is not functional"
            
            elif rule.condition == "governance_failed" and not current_report.governance_functional:
                alert_triggered = True
                alert_message = "Governance system is not functional"
            
            elif rule.condition == "telemetry_failed" and not current_report.telemetry_active:
                alert_triggered = True
                alert_message = "Telemetry system is not active"
            
            # Create alert if triggered and not already existing
            if alert_triggered:
                alert_id = f"{rule.condition}_{component_name or 'system'}_{current_report.timestamp}"
                
                # Check if we already have this alert
                existing_alert = any(
                    a.rule == rule.name and not a.acknowledged 
                    for a in self.alerts
                )
                
                if not existing_alert:
                    alert = Alert(
                        id=alert_id,
                        rule=rule.name,
                        severity=rule.severity,
                        message=alert_message,
                        component=component_name
                    )
                    self.alerts.append(alert)
                    new_alerts.append(alert)
        
        # Broadcast new alerts
        if new_alerts:
            await self._broadcast_alert_update()
    
    async def _broadcast_status_update(self, report: SystemHealthReport):
        """Broadcast status update to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "status_update",
            "data": asdict(report)
        }
        
        # Remove disconnected clients
        disconnected = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send_json(message)
            except:
                disconnected.add(ws)
        
        self.websocket_connections -= disconnected
    
    async def _broadcast_alert_update(self):
        """Broadcast alert update to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "alerts_update",
            "data": [asdict(alert) for alert in self.alerts if not alert.acknowledged]
        }
        
        # Remove disconnected clients
        disconnected = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send_json(message)
            except:
                disconnected.add(ws)
        
        self.websocket_connections -= disconnected
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False

async def main():
    """Main entry point for the live dashboard."""
    dashboard = GraceLiveDashboard(update_interval=30)  # Update every 30 seconds
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(dashboard.start_monitoring())
    
    # Start the web server
    config = uvicorn.Config(
        dashboard.app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("🚀 Grace Live Dashboard starting on http://localhost:8080")
    
    try:
        await server.serve()
    finally:
        dashboard.stop_monitoring()
        monitor_task.cancel()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())