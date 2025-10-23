"""
MCP Message Validator
"""

from typing import Dict, Any, List, Optional
import logging

from .schema import MCPMessage, MCPSchema

logger = logging.getLogger(__name__)


class MCPValidator:
    """
    Validates MCP messages against schemas and policies
    """
    
    def __init__(self):
        self.schemas: Dict[str, MCPSchema] = {}
        self._load_default_schemas()
    
    def _load_default_schemas(self):
        """Load default schemas"""
        # Heartbeat schema
        self.schemas["heartbeat"] = MCPSchema(
            name="heartbeat",
            version="1.0",
            fields={
                "kernel": {"type": "string"},
                "uptime_seconds": {"type": "number"},
                "events_processed": {"type": "number"}
            },
            required_fields=["kernel"]
        )
        
        # Consensus request schema
        self.schemas["consensus_request"] = MCPSchema(
            name="consensus_request",
            version="1.0",
            fields={
                "decision_context": {"type": "object"},
                "options": {"type": "array"}
            },
            required_fields=["decision_context", "options"]
        )
        
        # Consensus response schema
        self.schemas["consensus_response"] = MCPSchema(
            name="consensus_response",
            version="1.0",
            fields={
                "consensus": {"type": "object"},
                "processing_time_ms": {"type": "number"}
            },
            required_fields=["consensus"]
        )
    
    def register_schema(self, name: str, schema: MCPSchema):
        """Register custom schema"""
        self.schemas[name] = schema
        logger.info(f"Registered MCP schema: {name}")
    
    def validate_message(self, message: MCPMessage) -> tuple[bool, List[str]]:
        """
        Validate message structure and content
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # Validate basic structure
        if not message.message_id:
            errors.append("Missing message_id")
        
        if not message.source:
            errors.append("Missing source")
        
        if not message.destination:
            errors.append("Missing destination")
        
        # Validate trust score
        if message.trust_score < 0 or message.trust_score > 1:
            errors.append(f"Invalid trust_score: {message.trust_score} (must be 0-1)")
        
        # Validate payload against schema if available
        schema_name = message.headers.get("schema")
        if schema_name and schema_name in self.schemas:
            schema = self.schemas[schema_name]
            is_valid, schema_errors = schema.validate(message.payload)
            errors.extend(schema_errors)
        
        return len(errors) == 0, errors
    
    def validate_trust_score(self, message: MCPMessage, minimum_trust: float = 0.5) -> bool:
        """Validate trust score meets minimum threshold"""
        return message.trust_score >= minimum_trust
