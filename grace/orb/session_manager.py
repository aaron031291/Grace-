"""
Orb Session Manager - Manages multiple sessions with lifecycle
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import asyncio
import logging

from grace.orb.interface import OrbInterface

logger = logging.getLogger(__name__)


class OrbSessionManager:
    """
    Manages multiple Orb sessions with automatic cleanup and persistence
    """
    
    def __init__(
        self,
        embedding_service=None,
        vector_store=None,
        session_timeout: timedelta = timedelta(hours=1)
    ):
        """
        Initialize session manager
        
        Args:
            embedding_service: Embedding service
            vector_store: Vector store
            session_timeout: Inactivity timeout for sessions
        """
        self.orb = OrbInterface(embedding_service, vector_store)
        self.session_timeout = session_timeout
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        
        logger.info("Orb Session Manager initialized")
    
    def create_user_session(
        self,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create session for a user
        
        Args:
            user_id: User identifier
            metadata: Optional metadata
            
        Returns:
            Session ID
        """
        session_id = self.orb.create_session(user_id, metadata)
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        
        self.user_sessions[user_id].append(session_id)
        
        return session_id
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        return self.user_sessions.get(user_id, [])
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.orb.sessions)
    
    async def close_and_save_session(self, session_id: str) -> Dict:
        """
        Close session, save to storage, and return summary
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary with duration and topics
        """
        # Close session and get summary
        summary = self.orb.close_session(session_id)
        
        # Save to vector store
        await self.orb.save_session(session_id)
        
        return summary
    
    async def cleanup_inactive_sessions(self):
        """
        Clean up sessions that have been inactive for too long
        """
        now = datetime.now(timezone.utc)
        sessions_to_cleanup = []
        
        for session_id, session in self.orb.sessions.items():
            inactive_duration = now - session.last_activity
            
            if inactive_duration > self.session_timeout:
                sessions_to_cleanup.append(session_id)
        
        for session_id in sessions_to_cleanup:
            logger.info(f"Cleaning up inactive session {session_id}")
            
            # Save before cleanup
            await self.orb.save_session(session_id)
            
            # Remove from memory
            self.orb.cleanup_session(session_id)
        
        if sessions_to_cleanup:
            logger.info(f"Cleaned up {len(sessions_to_cleanup)} inactive sessions")
    
    def get_session_statistics(self) -> Dict:
        """
        Get statistics about all sessions
        
        Returns:
            Statistics dictionary
        """
        total_sessions = len(self.orb.sessions)
        total_users = len(self.user_sessions)
        
        # Calculate average duration
        durations = [
            session.get_duration_seconds()
            for session in self.orb.sessions.values()
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Get all topics across sessions
        all_topics = {}
        for session in self.orb.sessions.values():
            for topic, count in session.topics.items():
                all_topics[topic] = all_topics.get(topic, 0) + count
        
        # Top topics across all sessions
        top_topics = sorted(
            all_topics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_active_sessions": total_sessions,
            "total_users": total_users,
            "average_duration_seconds": avg_duration,
            "average_duration_formatted": self._format_duration(avg_duration),
            "top_topics_across_sessions": [
                {"topic": topic, "mentions": count}
                for topic, count in top_topics
            ]
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string"""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
