"""
Grace AI Backend API - Flask REST API for the web dashboard
"""
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_app(components: Dict[str, Any] = None):
    """Create and configure the Flask app."""
    app = Flask(__name__)
    CORS(app)
    
    # Store components in app context
    app.components = components or {}
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "online",
            "message": "Grace API is running"
        }), 200
    
    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Get current system status."""
        try:
            kpi_monitor = app.components.get('kpi_monitor')
            task_manager = app.components.get('task_manager')
            component_registry = app.components.get('component_registry')
            
            kpis = kpi_monitor.get_all_kpis() if kpi_monitor else {}
            trust = kpi_monitor.get_overall_trust() if kpi_monitor else 0
            tasks = len(task_manager.get_open_tasks()) if task_manager else 0
            
            return jsonify({
                "status": "running",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "kpis": kpis,
                "trust_score": trust,
                "active_tasks": tasks,
                "components": component_registry.list_components() if component_registry else {}
            }), 200
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/tasks', methods=['GET'])
    def get_tasks():
        """Get all tasks."""
        try:
            task_manager = app.components.get('task_manager')
            if not task_manager:
                return jsonify({"tasks": []}), 200
            
            tasks = [t.to_dict() for t in task_manager.tasks.values()]
            return jsonify({"tasks": tasks}), 200
        except Exception as e:
            logger.error(f"Error getting tasks: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/tasks', methods=['POST'])
    def create_task():
        """Create a new task."""
        try:
            data = request.get_json()
            task_manager = app.components.get('task_manager')
            
            if not task_manager:
                return jsonify({"error": "Task manager not available"}), 500
            
            title = data.get('title', 'Untitled Task')
            description = data.get('description', '')
            
            task_id = task_manager.create_task(title, description, created_by='user')
            return jsonify({"task_id": task_id, "success": True}), 201
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/tasks/<task_id>', methods=['GET'])
    def get_task(task_id):
        """Get a specific task."""
        try:
            task_manager = app.components.get('task_manager')
            if not task_manager:
                return jsonify({"error": "Task manager not available"}), 500
            
            task = task_manager.get_task(task_id)
            if not task:
                return jsonify({"error": "Task not found"}), 404
            
            return jsonify(task.to_dict()), 200
        except Exception as e:
            logger.error(f"Error getting task: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/mcp/tools', methods=['GET'])
    def get_mcp_tools():
        """Get available MCP tools."""
        try:
            mcp_manager = app.components.get('mcp_manager')
            if not mcp_manager:
                return jsonify({"error": "MCP manager not available"}), 500
            
            tools = mcp_manager.get_available_tools()
            return jsonify(tools), 200
        except Exception as e:
            logger.error(f"Error getting MCP tools: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/mcp/execute', methods=['POST'])
    def execute_mcp_tool():
        """Execute an MCP tool."""
        try:
            data = request.get_json()
            mcp_manager = app.components.get('mcp_manager')
            
            if not mcp_manager:
                return jsonify({"error": "MCP manager not available"}), 500
            
            tool_id = data.get('tool_id')
            parameters = data.get('parameters', {})
            correlation_id = data.get('correlation_id')
            
            # Execute tool asynchronously
            import asyncio
            loop = asyncio.new_event_loop()
            response = loop.run_until_complete(
                mcp_manager.execute_tool(tool_id, parameters, correlation_id)
            )
            loop.close()
            
            return jsonify(response.to_dict()), 200
        except Exception as e:
            logger.error(f"Error executing MCP tool: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app
