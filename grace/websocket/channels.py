"""
Channel-based message routing for WebSocket
"""

from typing import Dict, Set, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Predefined channel types"""
    SYSTEM = "system"
    COLLABORATION = "collaboration"
    TASKS = "tasks"
    POLICIES = "policies"
    DOCUMENTS = "documents"
    NOTIFICATIONS = "notifications"
    USER = "user"


class ChannelManager:
    """
    Manages channel subscriptions and permissions
    """
    
    def __init__(self):
        self.channel_permissions: Dict[str, Set[str]] = {}  # channel -> set of allowed user_ids
        logger.info("ChannelManager initialized")
    
    def create_channel(self, channel_name: str, allowed_users: Optional[Set[str]] = None):
        """
        Create a channel with optional access control
        
        Args:
            channel_name: Name of the channel
            allowed_users: Set of user IDs allowed to access (None = public)
        """
        if allowed_users is not None:
            self.channel_permissions[channel_name] = allowed_users
            logger.info(f"Created private channel: {channel_name} ({len(allowed_users)} users)")
        else:
            logger.info(f"Created public channel: {channel_name}")
    
    def can_access(self, user_id: str, channel_name: str) -> bool:
        """
        Check if user can access a channel
        
        Args:
            user_id: User ID
            channel_name: Channel name
            
        Returns:
            True if user can access, False otherwise
        """
        # If channel has no permissions set, it's public
        if channel_name not in self.channel_permissions:
            return True
        
        # Check if user is in allowed list
        return user_id in self.channel_permissions[channel_name]
    
    def grant_access(self, user_id: str, channel_name: str):
        """Grant user access to a channel"""
        if channel_name not in self.channel_permissions:
            self.channel_permissions[channel_name] = set()
        
        self.channel_permissions[channel_name].add(user_id)
        logger.info(f"Granted access to {user_id} for channel: {channel_name}")
    
    def revoke_access(self, user_id: str, channel_name: str):
        """Revoke user access from a channel"""
        if channel_name in self.channel_permissions:
            self.channel_permissions[channel_name].discard(user_id)
            logger.info(f"Revoked access from {user_id} for channel: {channel_name}")
    
    def get_user_channel(self, user_id: str) -> str:
        """Get user's private notification channel"""
        return f"{ChannelType.USER.value}:{user_id}"
    
    def get_session_channel(self, session_id: str) -> str:
        """Get collaboration session channel"""
        return f"{ChannelType.COLLABORATION.value}:{session_id}"
    
    def get_task_channel(self, task_id: str) -> str:
        """Get task updates channel"""
        return f"{ChannelType.TASKS.value}:{task_id}"
    
    def get_policy_channel(self, policy_id: str) -> str:
        """Get policy updates channel"""
        return f"{ChannelType.POLICIES.value}:{policy_id}"
    
    def get_document_channel(self, document_id: str) -> str:
        """Get document updates channel"""
        return f"{ChannelType.DOCUMENTS.value}:{document_id}"
    
    def delete_channel(self, channel_name: str):
        """Delete a channel and its permissions"""
        if channel_name in self.channel_permissions:
            del self.channel_permissions[channel_name]
            logger.info(f"Deleted channel: {channel_name}")
