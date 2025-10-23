"""
Grace AI MCP (Model Context Protocol) - Integration layer for external tools and services
"""
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Types of tools available through MCP."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    SEARCH = "search"
    VECTOR_STORE = "vector_store"
    EXTERNAL_API = "external_api"

@dataclass
class Tool:
    """Represents a tool available through MCP."""
    id: str
    name: str
    description: str
    tool_type: ToolType
    endpoint: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_auth: bool = False
    auth_token: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "endpoint": self.endpoint,
            "parameters": self.parameters,
            "requires_auth": self.requires_auth
        }

@dataclass
class ToolRequest:
    """A request to use a tool."""
    request_id: str
    tool_id: str
    parameters: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_id": self.tool_id,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }

@dataclass
class ToolResponse:
    """Response from a tool execution."""
    request_id: str
    tool_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_id": self.tool_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp
        }

class MCPRegistry:
    """Registry of all tools available through the Model Context Protocol."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.request_history: List[ToolRequest] = []
        self.response_history: List[ToolResponse] = []
    
    def register_tool(self, tool: Tool, handler: Callable) -> bool:
        """Register a new tool and its handler."""
        if tool.id in self.tools:
            logger.warning(f"Tool {tool.id} already registered, overwriting")
        
        self.tools[tool.id] = tool
        self.tool_handlers[tool.id] = handler
        logger.info(f"Registered MCP tool: {tool.name} (id: {tool.id})")
        return True
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)
    
    def list_tools(self) -> List[Tool]:
        """List all available tools."""
        return list(self.tools.values())
    
    def list_tools_by_type(self, tool_type: ToolType) -> List[Tool]:
        """List all tools of a specific type."""
        return [t for t in self.tools.values() if t.tool_type == tool_type]
    
    async def execute_tool(self, request: ToolRequest) -> ToolResponse:
        """Execute a tool request."""
        import time
        start_time = time.time()
        
        tool = self.tools.get(request.tool_id)
        if not tool:
            response = ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                success=False,
                result=None,
                error=f"Tool {request.tool_id} not found"
            )
            self.response_history.append(response)
            return response
        
        handler = self.tool_handlers.get(request.tool_id)
        if not handler:
            response = ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                success=False,
                result=None,
                error=f"No handler for tool {request.tool_id}"
            )
            self.response_history.append(response)
            return response
        
        try:
            result = await handler(request.parameters) if asyncio.iscoroutinefunction(handler) else handler(request.parameters)
            execution_time = (time.time() - start_time) * 1000
            
            response = ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
            logger.info(f"Tool {request.tool_id} executed successfully in {execution_time:.1f}ms")
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            response = ToolResponse(
                request_id=request.request_id,
                tool_id=request.tool_id,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time
            )
            logger.error(f"Tool {request.tool_id} failed: {str(e)}")
        
        self.request_history.append(request)
        self.response_history.append(response)
        return response
    
    def get_tool_context(self) -> Dict[str, Any]:
        """Get the current MCP context (all available tools and their capabilities)."""
        return {
            "available_tools": len(self.tools),
            "tools": [t.to_dict() for t in self.tools.values()],
            "tools_by_type": {
                tool_type.value: [t.to_dict() for t in self.list_tools_by_type(tool_type)]
                for tool_type in ToolType
            }
        }
