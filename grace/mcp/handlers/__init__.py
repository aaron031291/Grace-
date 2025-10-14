"""
Grace MCP Handlers

Domain-specific MCP implementations for Grace's core data types.

Available Handlers:
- PatternsMCP: Pattern observation and semantic search
- (More to be added: ExperiencesMCP, DecisionsMCP, ObservationsMCP, KnowledgeMCP)
"""

from grace.mcp.handlers.patterns_mcp import (
    PatternsMCP,
    PatternCreateRequest,
    PatternResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    PatternSearchResult,
    PatternMetadataRequest,
    PatternMetadataResponse
)

__all__ = [
    # Handlers
    "PatternsMCP",
    
    # Schemas
    "PatternCreateRequest",
    "PatternResponse",
    "SemanticSearchRequest",
    "SemanticSearchResponse",
    "PatternSearchResult",
    "PatternMetadataRequest",
    "PatternMetadataResponse",
]
