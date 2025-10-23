"""
Grace AI Communication Channel - Bi-directional communication with user
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class CommunicationChannel:
    """Enables bi-directional, unprompted communication between Grace and user."""
    
    def __init__(self):
        self.messages: list = []
    
    async def send_to_user(self, message: str, context: str = None, urgency: str = "info"):
        """Send a message to the user proactively."""
        msg = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context,
            "urgency": urgency,
            "direction": "grace_to_user"
        }
        
        self.messages.append(msg)
        
        print("\n" + "="*60)
        print(f">>> Message from Grace [{urgency.upper()}] <<<")
        print("-"*60)
        print(message)
        if context:
            print(f"\nContext: {context}")
        print("-"*60 + "\n")
        
        logger.info(f"Sent message to user: {message}")
    
    async def request_user_input(self, question: str, context: str = None) -> Optional[str]:
        """Request input from the user."""
        print("\n" + "="*60)
        print(">>> Grace is asking for your input <<<")
        print("-"*60)
        print(question)
        if context:
            print(f"\nContext: {context}")
        print("-"*60)
        
        response = input("Your response: ").strip()
        
        msg = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "context": context,
            "response": response,
            "direction": "user_to_grace"
        }
        
        self.messages.append(msg)
        
        logger.info(f"Received user input: {response}")
        return response
    
    def get_message_history(self) -> list:
        """Get message history."""
        return self.messages.copy()
