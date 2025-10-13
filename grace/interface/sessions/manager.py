"""Session management for Interface Kernel."""
import asyncio
import time
from datetime import datetime, timedelta
from grace.utils.time import now_utc
from typing import Dict, List, Optional
import logging

from ..models import UISession, UserIdentity, ClientInfo, generate_session_id

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages UI sessions with presence and multi-instance orchestration."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, UISession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_task = None
    
    def create(self, user: Dict, client: Dict) -> Dict:
        """Create a new UI session."""
        try:
            # Convert dict to UserIdentity
            user_identity = UserIdentity(**user)
            client_info = ClientInfo(**client)
            
            session_id = generate_session_id()
            
            session = UISession(
                session_id=session_id,
                user=user_identity,
                client=client_info,
                created_at=now_utc(),
                last_seen=now_utc()
            )
            
            self.sessions[session_id] = session
            
            logger.info(f"Created session {session_id} for user {user_identity.user_id}")
            
            return session.dict()
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def touch(self, session_id: str) -> None:
        """Update session last_seen timestamp."""
        if session_id in self.sessions:
            self.sessions[session_id].last_seen = now_utc()
    
    def get_session(self, session_id: str) -> Optional[UISession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def close(self, session_id: str) -> None:
        """Close a session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            logger.info(f"Closed session {session_id} for user {session.user.user_id}")
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[UISession]:
        """List sessions, optionally filtered by user."""
        sessions = list(self.sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user.user_id == user_id]
        
        return sessions
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        return len(self.sessions)
    
    def get_user_sessions(self, user_id: str) -> List[UISession]:
        """Get all sessions for a specific user."""
        return [s for s in self.sessions.values() if s.user.user_id == user_id]
    
    async def start_cleanup_task(self):
        """Start background task to clean up expired sessions."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
            
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_expired_sessions(self):
        """Background task to remove expired sessions."""
        while True:
            try:
                current_time = now_utc()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if session.last_seen:
                        time_since_seen = current_time - session.last_seen
                        if time_since_seen > self.session_timeout:
                            expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    self.close(session_id)
                    logger.info(f"Cleaned up expired session {session_id}")
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def get_stats(self) -> Dict:
        """Get session manager statistics."""
        current_time = now_utc()

        # Calculate session age distribution
        session_ages = []
        for session in self.sessions.values():
            if session.created_at:
                age = (current_time - session.created_at).total_seconds()
                session_ages.append(age)

        # Count by roles
        role_counts = {}
        for session in self.sessions.values():
            for role in getattr(session.user, "roles", []):
                role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total_sessions": len(self.sessions),
            "session_timeout_minutes": self.session_timeout.total_seconds() / 60,
            "role_distribution": role_counts,
            "avg_session_age_seconds": (sum(session_ages) / len(session_ages)) if session_ages else 0,
        }