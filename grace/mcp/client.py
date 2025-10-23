"""
MCP Client - Kernel communication interface
"""

from typing import Dict, Any, Optional, Callable
import uuid
import logging
from datetime import datetime

from .schema import MCPMessage, MCPMessageType, MCPPriority
from .validator import MCPValidator

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP Client for kernel communications
    
    Provides:
    - Schema validation
    - Message routing
    - Trust score checking
    """
    
    def __init__(
        self,
        kernel_name: str,
        event_bus,
        trigger_mesh=None,
        minimum_trust: float = 0.5
    ):
        self.kernel_name = kernel_name
        self.event_bus = event_bus
        self.trigger_mesh = trigger_mesh
        self.minimum_trust = minimum_trust
        
        self.validator = MCPValidator()
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.validation_failures = 0
    
    async def send_message(
        self,
        destination: str,
        payload: Dict[str, Any],
        message_type: MCPMessageType = MCPMessageType.REQUEST,
        priority: MCPPriority = MCPPriority.NORMAL,
        correlation_id: Optional[str] = None,
        trust_score: float = 1.0,
        schema_name: Optional[str] = None
    ) -> MCPMessage:
        """
        Send MCP message to destination kernel
        
        Args:
            destination: Target kernel name
            payload: Message payload
            message_type: Type of message
            priority: Message priority
            correlation_id: Optional correlation ID
            trust_score: Trust score (0-1)
            schema_name: Schema to validate against
        
        Returns:
            Sent MCPMessage
        
        Raises:
            ValueError: If validation fails
        """
        # Create message
        message = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            source=self.kernel_name,
            destination=destination,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority,
            trust_score=trust_score,
            requires_validation=schema_name is not None
        )
        
        # Set schema header
        if schema_name:
            message.headers["schema"] = schema_name
        
        # Validate message
        is_valid, errors = self.validator.validate_message(message)
        if not is_valid:
            self.validation_failures += 1
            error_msg = f"MCP validation failed: {', '.join(errors)}"
            logger.error(error_msg, extra={
                "message_id": message.message_id,
                "source": message.source,
                "destination": message.destination
            })
            raise ValueError(error_msg)
        
        # Check trust score
        if not self.validator.validate_trust_score(message, self.minimum_trust):
            self.validation_failures += 1
            error_msg = f"Trust score {message.trust_score} below minimum {self.minimum_trust}"
            logger.warning(error_msg, extra={"message_id": message.message_id})
            raise ValueError(error_msg)
        
        # Convert to event and send
        from grace.schemas.events import GraceEvent
        
        event = GraceEvent(
            event_type=f"mcp.{message_type.value}",
            source=message.source,
            targets=[message.destination],
            payload=message.to_dict(),
            correlation_id=message.correlation_id,
            priority=message.priority.value,
            trust_score=message.trust_score
        )
        
        # Send via TriggerMesh if available, else EventBus
        if self.trigger_mesh:
            await self.trigger_mesh.emit(event)
        else:
            await self.event_bus.emit(event)
        
        self.messages_sent += 1
        
        logger.info(f"MCP message sent", extra={
            "message_id": message.message_id,
            "destination": destination,
            "type": message_type.value
        })
        
        return message
    
    async def receive_message(self, event) -> Optional[MCPMessage]:
        """
        Receive and validate MCP message from event
        
        Args:
            event: GraceEvent containing MCP message
        
        Returns:
            Validated MCPMessage or None if invalid
        """
        try:
            # Extract MCP message from event payload
            mcp_data = event.payload
            message = MCPMessage.from_dict(mcp_data)
            
            # Validate
            is_valid, errors = self.validator.validate_message(message)
            if not is_valid:
                self.validation_failures += 1
                logger.error(f"Received invalid MCP message: {', '.join(errors)}")
                return None
            
            self.messages_received += 1
            
            logger.debug(f"MCP message received", extra={
                "message_id": message.message_id,
                "source": message.source
            })
            
            return message
        
        except Exception as e:
            logger.error(f"Failed to parse MCP message: {e}")
            return None
    
    def register_schema(self, name: str, schema):
        """Register custom schema"""
        self.validator.register_schema(name, schema)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MCP client statistics"""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "validation_failures": self.validation_failures,
            "minimum_trust": self.minimum_trust
        }
