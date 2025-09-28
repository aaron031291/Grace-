"""Notification service with multiple dispatch methods."""
import asyncio
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Callable
import logging
import uuid

from ..models import Notification, NotificationAction

logger = logging.getLogger(__name__)


class NotificationService:
    """Manages notifications with toasts, inbox, email, and webhooks."""
    
    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        self.user_notifications: Dict[str, List[str]] = {}  # user_id -> notif_ids
        self.dispatchers: Dict[str, Callable] = {}
    
    def create_notification(
        self, 
        user_id: str, 
        level: str, 
        message: str, 
        actions: Optional[List[Dict]] = None
    ) -> str:
        """Create and store a notification."""
        notif_id = str(uuid.uuid4())
        
        notification_actions = []
        if actions:
            notification_actions = [
                NotificationAction(label=action["label"], action=action["action"])
                for action in actions
            ]
        
        notification = Notification(
            notif_id=notif_id,
            level=level,
            message=message,
            actions=notification_actions,
            read=False,
            created_at=utc_now()
        )
        
        self.notifications[notif_id] = notification
        
        # Add to user's notification list
        if user_id not in self.user_notifications:
            self.user_notifications[user_id] = []
        self.user_notifications[user_id].append(notif_id)
        
        logger.info(f"Created {level} notification for user {user_id}: {message}")
        return notif_id
    
    async def dispatch_notification(self, user_id: str, notif_id: str, channels: List[str] = None):
        """Dispatch notification through specified channels."""
        if notif_id not in self.notifications:
            logger.error(f"Notification {notif_id} not found")
            return
        
        notification = self.notifications[notif_id]
        channels = channels or ["toast", "inbox"]
        
        dispatch_tasks = []
        for channel in channels:
            if channel in self.dispatchers:
                task = asyncio.create_task(
                    self.dispatchers[channel](user_id, notification)
                )
                dispatch_tasks.append(task)
        
        if dispatch_tasks:
            await asyncio.gather(*dispatch_tasks, return_exceptions=True)
    
    def mark_as_read(self, notif_id: str) -> bool:
        """Mark notification as read."""
        if notif_id not in self.notifications:
            return False
        
        self.notifications[notif_id].read = True
        return True
    
    def get_user_notifications(
        self, 
        user_id: str, 
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a user."""
        if user_id not in self.user_notifications:
            return []
        
        notif_ids = self.user_notifications[user_id]
        notifications = [
            self.notifications[notif_id] 
            for notif_id in notif_ids 
            if notif_id in self.notifications
        ]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        # Sort by created_at descending
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return notifications[:limit]
    
    def delete_notification(self, user_id: str, notif_id: str) -> bool:
        """Delete a notification."""
        if notif_id not in self.notifications:
            return False
        
        del self.notifications[notif_id]
        
        if user_id in self.user_notifications:
            if notif_id in self.user_notifications[user_id]:
                self.user_notifications[user_id].remove(notif_id)
        
        return True
    
    def register_dispatcher(self, channel: str, dispatcher_func: Callable):
        """Register a notification dispatcher for a channel."""
        self.dispatchers[channel] = dispatcher_func
        logger.info(f"Registered dispatcher for channel: {channel}")
    
    async def _toast_dispatcher(self, user_id: str, notification: Notification):
        """Default toast dispatcher (placeholder)."""
        logger.info(f"TOAST for {user_id}: {notification.message}")
    
    async def _inbox_dispatcher(self, user_id: str, notification: Notification):
        """Default inbox dispatcher (already handled by storage)."""
        logger.info(f"INBOX for {user_id}: {notification.message}")
    
    async def _email_dispatcher(self, user_id: str, notification: Notification):
        """Email dispatcher (placeholder implementation)."""
        logger.info(f"EMAIL for {user_id}: {notification.message}")
        # In real implementation: send email via SMTP or service
    
    async def _webhook_dispatcher(self, user_id: str, notification: Notification):
        """Webhook dispatcher (placeholder implementation)."""
        logger.info(f"WEBHOOK for {user_id}: {notification.message}")
        # In real implementation: POST to webhook URL
    
    def setup_default_dispatchers(self):
        """Setup default notification dispatchers."""
        self.register_dispatcher("toast", self._toast_dispatcher)
        self.register_dispatcher("inbox", self._inbox_dispatcher)
        self.register_dispatcher("email", self._email_dispatcher)
        self.register_dispatcher("webhook", self._webhook_dispatcher)
    
    def cleanup_old_notifications(self, days_to_keep: int = 30) -> int:
        """Clean up old notifications and return count removed."""
        cutoff_date = utc_now() - timedelta(days=days_to_keep)
        
        old_notif_ids = [
            notif_id for notif_id, notification in self.notifications.items()
            if notification.created_at < cutoff_date and notification.read
        ]
        
        # Remove from notifications
        for notif_id in old_notif_ids:
            del self.notifications[notif_id]
        
        # Remove from user notification lists
        for user_id, user_notif_ids in self.user_notifications.items():
            self.user_notifications[user_id] = [
                nid for nid in user_notif_ids if nid not in old_notif_ids
            ]
        
        logger.info(f"Cleaned up {len(old_notif_ids)} old notifications")
        return len(old_notif_ids)
    
    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications for user."""
        if user_id not in self.user_notifications:
            return 0
        
        unread_count = 0
        for notif_id in self.user_notifications[user_id]:
            if notif_id in self.notifications:
                if not self.notifications[notif_id].read:
                    unread_count += 1
        
        return unread_count
    
    def create_and_dispatch(
        self,
        user_id: str,
        level: str,
        message: str,
        actions: Optional[List[Dict]] = None,
        channels: List[str] = None
    ) -> str:
        """Create notification and dispatch immediately (convenience method)."""
        notif_id = self.create_notification(user_id, level, message, actions)
        
        # Schedule dispatch
        asyncio.create_task(
            self.dispatch_notification(user_id, notif_id, channels)
        )
        
        return notif_id
    
    def get_stats(self) -> Dict:
        """Get notification service statistics."""
        notifications = list(self.notifications.values())
        
        # Count by level
        level_counts = {}
        for notification in notifications:
            level_counts[notification.level] = level_counts.get(notification.level, 0) + 1
        
        # Count read vs unread
        read_count = sum(1 for n in notifications if n.read)
        unread_count = len(notifications) - read_count
        
        # Active users with notifications
        active_users = len([uid for uid, nids in self.user_notifications.items() if nids])
        
        return {
            "total_notifications": len(notifications),
            "level_distribution": level_counts,
            "read_notifications": read_count,
            "unread_notifications": unread_count,
            "active_users": active_users,
            "registered_dispatchers": list(self.dispatchers.keys())
        }