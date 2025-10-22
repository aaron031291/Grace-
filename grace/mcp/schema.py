"""
MCP Message Schema - Enforces consistency across all communications
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MCPMessageType(Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    HEARTBEAT = "heartbeat"


class MCPPriority(Enum):
    """MCP message priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MCPSchema:
    """
    Schema definition for MCP messages
    
    Defines expected structure and validation rules
    """
    name: str
    version: str
    fields: Dict[str, Dict[str, Any]]
    required_fields: List[str]
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate data against schema
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field_name, field_def in self.fields.items():
            if field_name in data:
                expected_type = field_def.get("type")
                actual_value = data[field_name]
                
                if expected_type == "string" and not isinstance(actual_value, str):
                    errors.append(f"Field '{field_name}' must be string")
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    errors.append(f"Field '{field_name}' must be number")
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    errors.append(f"Field '{field_name}' must be boolean")
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    errors.append(f"Field '{field_name}' must be object")
                elif expected_type == "array" and not isinstance(actual_value, list):
                    errors.append(f"Field '{field_name}' must be array")
        
        return len(errors) == 0, errors


@dataclass
class MCPMessage:
    """
    Canonical MCP message structure
    
    All kernel communications must use this format
    """
    message_id: str
    message_type: MCPMessageType
    source: str
    destination: str
    payload: Dict[str, Any]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: MCPPriority = MCPPriority.NORMAL
    
    # Governance
    schema_version: str = "1.0"
    trust_score: float = 1.0
    requires_validation: bool = False
    
    # Headers
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "schema_version": self.schema_version,
            "trust_score": self.trust_score,
            "requires_validation": self.requires_validation,
            "headers": self.headers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create from dictionary"""
        return cls(
            message_id=data["message_id"],
            message_type=MCPMessageType(data["message_type"]),
            source=data["source"],
            destination=data["destination"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            correlation_id=data.get("correlation_id"),
            priority=MCPPriority(data.get("priority", "normal")),
            schema_version=data.get("schema_version", "1.0"),
            trust_score=data.get("trust_score", 1.0),
            requires_validation=data.get("requires_validation", False),
            headers=data.get("headers", {})
        )
