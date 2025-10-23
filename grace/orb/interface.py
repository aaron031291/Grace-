"""
Orb Interface implementation with session memory and topic extraction
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
import uuid

from grace.session.memory import SessionMemory

logger = logging.getLogger(__name__)


class OrbInterface:
    """
    Orb Interface - Grace's primary interaction layer
    
    Manages sessions, extracts topics, and maintains conversation context
    """
    
    def __init__(self, embedding_service=None, vector_store=None):
        """
        Initialize Orb Interface
        
        Args:
            embedding_service: Embedding service for semantic analysis
            vector_store: Vector store for session persistence
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.sessions: Dict[str, SessionMemory] = {}
        
        logger.info("Orb Interface initialized")
    
    def create_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: User identifier
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = SessionMemory(
            session_id=session_id,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )
        
        # Add initial metadata
        if metadata:
            session.add_message(
                role="system",
                content="Session started",
                metadata={**metadata, "user_id": user_id}
            )
        
        self.sessions[session_id] = session
        
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session_id
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add message to session with automatic topic extraction
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.sessions[session_id]
        session.add_message(role, content, metadata)
        
        return True
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """
        Close session and return summary with duration and topics
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary including duration and key topics
        """
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return {}
        
        session = self.sessions[session_id]
        session.close_session()
        
        # Get comprehensive summary
        summary = session.get_summary()
        
        logger.info(
            f"Session {session_id} closed. "
            f"Duration: {summary['duration_formatted']}, "
            f"Messages: {summary['total_messages']}, "
            f"Key topics: {[t['topic'] for t in summary['key_topics']]}"
        )
        
        return summary
    
    async def save_session(self, session_id: str) -> bool:
        """
        Save session to persistent storage with embeddings
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.sessions[session_id]
        
        # Save to vector store
        success = await session.save_to_vector_store()
        
        if success:
            logger.info(f"Session {session_id} saved to vector store")
        
        return success
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session information including duration and topics
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information
        """
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        return session.get_summary()
    
    def get_session_transcript(self, session_id: str) -> str:
        """
        Get full conversation transcript
        
        Args:
            session_id: Session identifier
            
        Returns:
            Complete transcript
        """
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        return session.get_chat_transcript()
    
    def extract_session_topics(self, session_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key topics from session
        
        Args:
            session_id: Session identifier
            top_n: Number of top topics to return
            
        Returns:
            List of topics with scores
        """
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        return session.get_key_topics(top_n)
    
    def cleanup_session(self, session_id: str):
        """
        Remove session from memory
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")
