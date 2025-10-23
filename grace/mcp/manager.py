"""
Grace AI MCP Manager - Central orchestration of Model Context Protocol
"""
import logging
from typing import Dict, Any, Optional
import asyncio

from grace.mcp.protocol import MCPRegistry, Tool, ToolType, ToolRequest
from grace.mcp.vector_store import vector_store_handler
from grace.mcp.search_tool import search_handler
from grace.mcp.code_generation import code_generation_handler

logger = logging.getLogger(__name__)

class MCPManager:
    """Central manager for the Model Context Protocol."""
    
    def __init__(self, event_bus=None, llm_service=None):
        self.registry = MCPRegistry()
        self.event_bus = event_bus
        self.llm_service = llm_service
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default built-in tools."""
        
        # Vector Store Tool
        vector_tool = Tool(
            id="vector_store",
            name="Vector Store",
            description="Semantic search and vector operations",
            tool_type=ToolType.VECTOR_STORE,
            endpoint="/mcp/tools/vector_store"
        )
        self.registry.register_tool(vector_tool, vector_store_handler)
        
        # Search Tool
        search_tool = Tool(
            id="search",
            name="Search",
            description="Web and semantic search",
            tool_type=ToolType.SEARCH,
            endpoint="/mcp/tools/search"
        )
        self.registry.register_tool(search_tool, search_handler)
        
        # Code Generation Tool
        code_tool = Tool(
            id="code_generation",
            name="Code Generation",
            description="Generate and refactor code using LLM",
            tool_type=ToolType.CODE_GENERATION,
            endpoint="/mcp/tools/code_generation"
        )
        self.registry.register_tool(code_tool, code_generation_handler)
        
        logger.info("Registered 3 default MCP tools")
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any], correlation_id: Optional[str] = None):
        """Execute a tool through MCP."""
        import uuid
        request_id = str(uuid.uuid4())
        
        request = ToolRequest(
            request_id=request_id,
            tool_id=tool_id,
            parameters=parameters,
            timestamp=__import__('datetime').datetime.now().isoformat(),
            correlation_id=correlation_id
        )
        
        logger.info(f"MCP: Executing tool {tool_id} (request_id: {request_id})")
        response = await self.registry.execute_tool(request)
        
        # Publish event if event_bus is available
        if self.event_bus:
            await self.event_bus.publish("mcp.tool_executed", {
                "tool_id": tool_id,
                "request_id": request_id,
                "success": response.success,
                "correlation_id": correlation_id
            })
        
        return response
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about all available tools."""
        return self.registry.get_tool_context()
    
    def get_tool_by_id(self, tool_id: str) -> Optional[Tool]:
        """Get a specific tool by ID."""
        return self.registry.get_tool(tool_id)
    
    def get_tools_by_type(self, tool_type: ToolType):
        """Get all tools of a specific type."""
        return self.registry.list_tools_by_type(tool_type)
    
    def get_execution_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get history of tool executions."""
        return {
            "total_requests": len(self.registry.request_history),
            "total_responses": len(self.registry.response_history),
            "recent_requests": [r.to_dict() for r in self.registry.request_history[-limit:]],
            "recent_responses": [r.to_dict() for r in self.registry.response_history[-limit:]]
        }
