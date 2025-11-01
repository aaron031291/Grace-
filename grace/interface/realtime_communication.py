"""
Real-Time Bidirectional Communication System

Grace can:
- Receive messages from you (WebSocket)
- Send proactive messages to you (notifications)
- Stream responses in real-time
- Notify when she needs help
- Alert about important events
- Continuous connection

You and Grace are always connected!
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages"""
    USER_MESSAGE = "user_message"
    GRACE_RESPONSE = "grace_response"
    GRACE_NOTIFICATION = "grace_notification"  # Grace proactively reaches out!
    GRACE_NEEDS_HELP = "grace_needs_help"  # Grace asks for human assistance
    SYSTEM_EVENT = "system_event"
    THINKING_UPDATE = "thinking_update"  # Grace's thought process
    KERNEL_UPDATE = "kernel_update"  # Updates from subsystems


@dataclass
class Message:
    """A message in the system"""
    message_id: str
    message_type: MessageType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    sender: str  # "user" or "grace" or "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender
        }


class GraceNotificationSystem:
    """
    Grace's proactive notification system.
    
    Grace can initiate contact when she:
    - Needs human help/guidance
    - Discovers something important
    - Completes a long-running task
    - Detects an anomaly
    - Has a question
    - Wants to share an insight
    """
    
    def __init__(self):
        self.notification_queue: List[Message] = []
        self.notification_handlers: Set[Callable] = set()
        
        logger.info("Grace Notification System initialized")
        logger.info("  Grace can now proactively reach out to you!")
    
    def register_handler(self, handler: Callable):
        """Register handler for Grace's notifications"""
        self.notification_handlers.add(handler)
        logger.info(f"Registered notification handler: {handler.__name__}")
    
    async def grace_notifies(
        self,
        content: str,
        priority: str = "normal",
        reason: str = "information",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Grace proactively sends a notification.
        
        This is Grace reaching out to YOU!
        """
        notification = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.GRACE_NOTIFICATION,
            content=content,
            metadata={
                "priority": priority,
                "reason": reason,
                **(metadata or {})
            },
            timestamp=datetime.utcnow(),
            sender="grace"
        )
        
        self.notification_queue.append(notification)
        
        logger.info(f"🔔 Grace sends notification: {content[:50]}...")
        logger.info(f"   Priority: {priority}, Reason: {reason}")
        
        # Notify all handlers
        for handler in self.notification_handlers:
            try:
                await handler(notification)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
    
    async def grace_needs_help(
        self,
        issue: str,
        context: Dict[str, Any],
        urgency: str = "normal"
    ):
        """
        Grace explicitly asks for human help.
        
        Grace knows when she's uncertain and proactively asks!
        """
        help_request = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.GRACE_NEEDS_HELP,
            content=f"I need your help: {issue}",
            metadata={
                "urgency": urgency,
                "context": context,
                "grace_state": "uncertain"
            },
            timestamp=datetime.utcnow(),
            sender="grace"
        )
        
        self.notification_queue.append(help_request)
        
        logger.info(f"🆘 Grace needs help: {issue}")
        logger.info(f"   Urgency: {urgency}")
        
        # Alert all handlers immediately
        for handler in self.notification_handlers:
            try:
                await handler(help_request)
            except Exception as e:
                logger.error(f"Help request handler error: {e}")


class RealTimeCommunicationChannel:
    """
    Real-time bidirectional communication channel.
    
    Persistent WebSocket connection between user and Grace.
    Messages flow both directions instantly.
    """
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}  # session_id -> websocket
        self.message_history: List[Message] = []
        self.notification_system = GraceNotificationSystem()
        
        logger.info("Real-Time Communication Channel initialized")
    
    async def connect(
        self,
        session_id: str,
        websocket: Any
    ):
        """Establish real-time connection"""
        self.connections[session_id] = websocket
        
        logger.info(f"✅ Real-time connection established: {session_id}")
        
        # Send welcome message from Grace
        await self.send_to_user(
            session_id,
            "I'm ready! You can speak to me or type. I'll respond in real-time.",
            message_type=MessageType.GRACE_RESPONSE
        )
    
    def disconnect(self, session_id: str):
        """Disconnect session"""
        if session_id in self.connections:
            del self.connections[session_id]
            logger.info(f"Disconnected: {session_id}")
    
    async def send_to_user(
        self,
        session_id: str,
        content: str,
        message_type: MessageType = MessageType.GRACE_RESPONSE,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Grace sends message to user.
        
        This can be:
        - Response to user's message
        - Proactive notification
        - Request for help
        - System update
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
            sender="grace"
        )
        
        self.message_history.append(message)
        
        # Send via WebSocket
        if session_id in self.connections:
            websocket = self.connections[session_id]
            try:
                await websocket.send_json(message.to_dict())
                logger.debug(f"Sent to {session_id}: {content[:50]}...")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
    
    async def receive_from_user(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Receive message from user"""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.USER_MESSAGE,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
            sender="user"
        )
        
        self.message_history.append(message)
        
        logger.debug(f"Received from {session_id}: {content[:50]}...")
        
        return message
    
    async def stream_thinking(
        self,
        session_id: str,
        thinking_content: str
    ):
        """
        Stream Grace's thinking process in real-time.
        
        User can see what Grace is thinking as she thinks!
        """
        await self.send_to_user(
            session_id,
            thinking_content,
            message_type=MessageType.THINKING_UPDATE,
            metadata={"streaming": True}
        )
    
    async def broadcast_kernel_update(
        self,
        kernel_name: str,
        update: Dict[str, Any]
    ):
        """
        Broadcast update from a kernel/subsystem.
        
        Users can see all of Grace's internals in real-time!
        """
        for session_id in self.connections.keys():
            await self.send_to_user(
                session_id,
                f"Kernel update: {kernel_name}",
                message_type=MessageType.KERNEL_UPDATE,
                metadata=update
            )


# Global communication channel
_comm_channel: Optional[RealTimeCommunicationChannel] = None


def get_communication_channel() -> RealTimeCommunicationChannel:
    """Get global communication channel"""
    global _comm_channel
    if _comm_channel is None:
        _comm_channel = RealTimeCommunicationChannel()
    return _comm_channel


if __name__ == "__main__":
    # Demo
    async def demo():
        print("📡 Real-Time Communication Demo\n")
        
        channel = RealTimeCommunicationChannel()
        
        # Mock websocket
        class MockWebSocket:
            async def send_json(self, data):
                print(f"  → Sent: {data['content'][:50]}...")
        
        # Connect
        await channel.connect("session_001", MockWebSocket())
        
        # Simulate bidirectional communication
        print("\n👤 User sends message...")
        await channel.receive_from_user(
            "session_001",
            "How do I build a REST API?"
        )
        
        print("\n🧠 Grace responds...")
        await channel.send_to_user(
            "session_001",
            "I'll help you build a REST API using FastAPI..."
        )
        
        print("\n🔔 Grace proactively notifies...")
        await channel.notification_system.grace_notifies(
            "I've completed analyzing your codebase and found 3 optimization opportunities!",
            priority="normal",
            reason="proactive_insight"
        )
        
        print("\n🆘 Grace asks for help...")
        await channel.notification_system.grace_needs_help(
            "I'm uncertain about the best approach for distributed transactions",
            context={"domain": "microservices"},
            urgency="normal"
        )
        
        print(f"\n📊 Stats:")
        print(f"  Messages: {len(channel.message_history)}")
        print(f"  Connections: {len(channel.connections)}")
    
    asyncio.run(demo())
