"""
Grace Meta-Control Protocol (MCP) System

The MCP is Grace's canonical API façade layer providing:
- Unified API contracts for all domain tables
- Automatic governance validation
- Meta-Loop integration (OODA cycle)
- Vectorization and semantic search
- Event emission and audit logging
- Trust scoring and provenance tracking
- Resilient error handling with learning

Quick Start:
    from grace.mcp import BaseMCP, mcp_endpoint, MCPContext
    from grace.mcp.handlers import PatternsMCP
    
    # Use existing handler
    patterns_mcp = PatternsMCP()
    
    # Or create your own
    class MyMCP(BaseMCP):
        domain = "my_domain"
        
        @mcp_endpoint(manifest="my_domain.yaml", endpoint="create")
        async def create(self, request, context):
            record = await self.db.insert('my_table', request.dict())
            return {"id": record.id}

See README.md for full documentation.
"""

__version__ = "1.0.0"
__author__ = "Grace AI Team"
__license__ = "Proprietary"

# Core classes
from grace.mcp.base_mcp import (
    BaseMCP,
    mcp_endpoint,
    MCPContext,
    MCPResponse,
    MCPException,
    Caller,
    Severity,
    OwnerType,
    GovernanceRejection,
    ServiceUnavailable
)

# Pushback system
from grace.mcp.pushback import (
    PushbackHandler,
    PushbackPayload,
    PushbackSeverity,
    PushbackCategory,
    handle_governance_rejection,
    handle_service_unavailable,
    handle_index_failure,
    retry_with_backoff
)

# Handlers (add more as implemented)
try:
    from grace.mcp.handlers.patterns_mcp import PatternsMCP
    _patterns_available = True
except ImportError:
    _patterns_available = False

__all__ = [
    # Core
    "BaseMCP",
    "mcp_endpoint",
    "MCPContext",
    "MCPResponse",
    "MCPException",
    "Caller",
    "Severity",
    "OwnerType",
    "GovernanceRejection",
    "ServiceUnavailable",
    
    # Pushback
    "PushbackHandler",
    "PushbackPayload",
    "PushbackSeverity",
    "PushbackCategory",
    "handle_governance_rejection",
    "handle_service_unavailable",
    "handle_index_failure",
    "retry_with_backoff",
]

# Add handlers if available
if _patterns_available:
    __all__.append("PatternsMCP")


def get_version() -> str:
    """Get MCP system version"""
    return __version__


def get_available_handlers() -> list:
    """Get list of available MCP handlers"""
    handlers = []
    if _patterns_available:
        handlers.append("PatternsMCP")
    return handlers


# System health check
def health_check() -> dict:
    """
    Check MCP system health.
    
    Returns:
        dict: Health status with component availability
    """
    return {
        "status": "healthy",
        "version": __version__,
        "handlers_available": get_available_handlers(),
        "core_components": {
            "base_mcp": True,
            "pushback": True,
            "manifests": True
        }
    }


# Convenience function for quick MCP creation
def create_mcp(domain: str, manifest_path: str = None) -> BaseMCP:
    """
    Create a basic MCP instance for a domain.
    
    Args:
        domain: Domain name (e.g., "patterns", "experiences")
        manifest_path: Optional custom manifest path
        
    Returns:
        BaseMCP instance
        
    Example:
        >>> mcp = create_mcp("patterns")
        >>> # Now add endpoints with @mcp_endpoint decorator
    """
    class CustomMCP(BaseMCP):
        pass
    
    CustomMCP.domain = domain
    if manifest_path:
        CustomMCP.manifest_path = manifest_path
    
    return CustomMCP()


"""
Message Control Protocol (MCP) - Schema validation and routing
"""

from .client import MCPClient
from .schema import MCPMessage, MCPSchema
from .validator import MCPValidator

__all__ = ['MCPClient', 'MCPMessage', 'MCPSchema', 'MCPValidator']


"""
Grace AI MCP (Model Context Protocol) Module
Provides integration with external tools, services, and specialized models
"""
from grace.mcp.protocol import MCPRegistry, Tool, ToolType, ToolRequest, ToolResponse
from grace.mcp.vector_store import VectorStore, vector_store_handler
from grace.mcp.search_tool import SearchTool, search_handler
from grace.mcp.code_generation import CodeGenerationTool, code_generation_handler
from grace.mcp.manager import MCPManager

__all__ = [
    "MCPRegistry",
    "Tool",
    "ToolType",
    "ToolRequest",
    "ToolResponse",
    "VectorStore",
    "vector_store_handler",
    "SearchTool",
    "search_handler",
    "CodeGenerationTool",
    "code_generation_handler",
    "MCPManager",
]
