"""
Session memory management with production implementation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    Manages session memory with key topics, entities, and conversation flow
    """
    
    def __init__(self, session_id: str, embedding_service=None, vector_store=None):
        """
        Initialize session memory
        
        Args:
            session_id: Unique session identifier
            embedding_service: Optional embedding service for semantic analysis
            vector_store: Optional vector store for long-term memory
        """
        self.session_id = session_id
        self.start_time = datetime.now(timezone.utc)
        self.end_time = None  # Set when session closes
        self.last_activity = self.start_time
        
        # Conversation tracking
        self.messages: List[Dict[str, Any]] = []
        self.turns = 0
        
        # Topic and entity tracking
        self.topics: Dict[str, int] = defaultdict(int)  # topic -> mention count
        self.entities: Dict[str, List[str]] = defaultdict(list)  # entity_type -> [entities]
        self.decisions: List[Dict[str, Any]] = []
        
        # Services
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
        # NLP for topic extraction
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP pipeline for topic extraction"""
        self.nlp_available = False
        self.topic_extractor = None
        
        # Try to load KeyBERT
        try:
            from keybert import KeyBERT
            self.topic_extractor = KeyBERT()
            self.nlp_available = True
            logger.info("KeyBERT initialized for topic extraction")
            return
        except ImportError:
            logger.debug("KeyBERT not available, trying spaCy...")
        
        # Try to load spaCy with RAKE
        try:
            import spacy
            from spacy.lang.en.stop_words import STOP_WORDS
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded, use blank
                self.nlp = spacy.blank("en")
                logger.warning("spaCy model not loaded, using blank pipeline")
            
            self.nlp_available = True
            logger.info("spaCy initialized for topic extraction")
            return
        except ImportError:
            logger.warning("No NLP library available for topic extraction, using keyword fallback")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a message to session memory
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        timestamp = datetime.now(timezone.utc)
        self.last_activity = timestamp
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        if role in ["user", "assistant"]:
            self.turns += 1
        
        # Extract topics and entities from content
        self._extract_topics_nlp(content)
        self._extract_entities(content)
        
        logger.debug(f"Added message to session {self.session_id}: {role}")
    
    def close_session(self):
        """Mark session as closed and set end time"""
        self.end_time = datetime.now(timezone.utc)
        logger.info(f"Session {self.session_id} closed at {self.end_time.isoformat()}")
    
    def get_session_duration(self) -> timedelta:
        """
        Calculate session duration
        
        Returns:
            timedelta object representing session duration
        """
        end = self.end_time if self.end_time else self.last_activity
        return end - self.start_time
    
    def get_duration_seconds(self) -> float:
        """Get session duration in seconds"""
        return self.get_session_duration().total_seconds()
    
    def get_duration_formatted(self) -> str:
        """Get human-readable duration string"""
        duration = self.get_session_duration()
        total_seconds = int(duration.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _extract_topics_nlp(self, text: str):
        """
        Extract topics from text using NLP pipeline
        
        Uses KeyBERT if available, falls back to spaCy, then simple keywords
        
        Args:
            text: Text to extract topics from
        """
        if not text or len(text.strip()) < 10:
            return
        
        # Try KeyBERT first (best results)
        if self.topic_extractor is not None:
            try:
                keywords = self.topic_extractor.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=5,
                    diversity=0.7
                )
                for keyword, score in keywords:
                    if score > 0.3:  # Only keep confident topics
                        self.topics[keyword] += 1
                return
            except Exception as e:
                logger.debug(f"KeyBERT extraction failed: {e}")
        
        # Try spaCy with custom RAKE-like approach
        if hasattr(self, 'nlp') and self.nlp is not None:
            try:
                doc = self.nlp(text.lower())
                
                # Extract noun phrases
                if doc.noun_chunks:
                    for chunk in doc.noun_chunks:
                        chunk_text = chunk.text.strip()
                        if len(chunk_text) > 3 and chunk_text not in ['the', 'this', 'that']:
                            self.topics[chunk_text] += 1
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        self.topics[ent.text.lower()] += 1
                
                return
            except Exception as e:
                logger.debug(f"spaCy extraction failed: {e}")
        
        # Fallback to simple keyword extraction
        self._extract_topics_keywords(text)
    
    def _extract_topics_keywords(self, text: str):
        """
        Fallback keyword-based topic extraction
        
        Args:
            text: Text to extract topics from
        """
        # Common important keywords in AI/tech domain
        keywords = [
            "authentication", "authorization", "database", "api", "security",
            "performance", "error", "bug", "feature", "deployment", "testing",
            "documentation", "configuration", "monitoring", "logging", "user",
            "data", "model", "training", "inference", "query", "search",
            "vector", "embedding", "policy", "task", "session", "websocket"
        ]
        
        text_lower = text.lower()
        
        # Extract single keywords
        for keyword in keywords:
            if keyword in text_lower:
                self.topics[keyword] += 1
        
        # Extract potential multi-word topics (bigrams)
        words = text_lower.split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            # Check if it's a meaningful bigram
            if any(kw in bigram for kw in keywords):
                self.topics[bigram] += 1
    
    def add_decision(self, decision: Dict[str, Any]):
        """Record a decision made during the session"""
        decision["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.decisions.append(decision)
        
        # Track decision as topic
        if "type" in decision:
            self.topics[f"decision:{decision['type']}"] += 1
    
    def _extract_entities(self, text: str):
        """
        Extract named entities from text
        
        Args:
            text: Text to extract entities from
        """
        import re
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            self.entities["email"].extend(emails)
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            self.entities["url"].extend(urls)
        
        # Extract UUIDs
        uuids = re.findall(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', text, re.IGNORECASE)
        if uuids:
            self.entities["uuid"].extend(uuids)
        
        # Extract file paths
        paths = re.findall(r'/[\w\-./]+', text)
        if paths:
            self.entities["path"].extend([p for p in paths if len(p) > 5])
    
    def get_key_topics(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most discussed topics in the session
        
        Args:
            top_n: Number of top topics to return
            
        Returns:
            List of topics with mention counts and relevance scores
        """
        sorted_topics = sorted(
            self.topics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {
                "topic": topic,
                "mentions": count,
                "relevance": self._calculate_topic_relevance(count),
                "confidence": min(1.0, count / 10.0)  # Normalize confidence
            }
            for topic, count in sorted_topics
        ]
    
    def _calculate_topic_relevance(self, mentions: int) -> str:
        """Calculate relevance score for a topic"""
        if mentions >= 10:
            return "critical"
        elif mentions >= 5:
            return "high"
        elif mentions >= 2:
            return "medium"
        else:
            return "low"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive session summary
        
        Returns:
            Dictionary with session statistics and insights
        """
        duration = self.get_session_duration()
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_activity": self.last_activity.isoformat(),
            "duration_seconds": self.get_duration_seconds(),
            "duration_formatted": self.get_duration_formatted(),
            "total_messages": len(self.messages),
            "conversation_turns": self.turns,
            "key_topics": self.get_key_topics(),
            "entities_found": {
                entity_type: len(set(entities))
                for entity_type, entities in self.entities.items()
            },
            "decisions_made": len(self.decisions),
            "activity_level": self._calculate_activity_level(),
            "is_active": self.end_time is None,
        }
    
    def _calculate_activity_level(self) -> str:
        """Calculate overall activity level of the session"""
        duration_minutes = self.get_duration_seconds() / 60
        if duration_minutes < 0.1:
            duration_minutes = 0.1  # Avoid division by zero
        
        messages_per_minute = len(self.messages) / duration_minutes
        
        if messages_per_minute > 5:
            return "very_high"
        elif messages_per_minute > 2:
            return "high"
        elif messages_per_minute > 1:
            return "medium"
        elif messages_per_minute > 0.5:
            return "low"
        else:
            return "very_low"
    
    def get_chat_transcript(self) -> str:
        """
        Get full chat transcript as a single text string
        
        Returns:
            Complete conversation transcript
        """
        transcript_parts = []
        
        for msg in self.messages:
            role = msg['role'].upper()
            content = msg['content']
            timestamp = msg['timestamp']
            transcript_parts.append(f"[{timestamp}] {role}: {content}")
        
        return "\n".join(transcript_parts)
    
    async def save_to_vector_store(self) -> bool:
        """
        Save session memory to vector store for long-term retrieval
        
        Returns:
            Success status
        """
        if not self.embedding_service or not self.vector_store:
            logger.warning("Cannot save to vector store: services not available")
            return False
        
        try:
            # Create comprehensive session document
            session_text = self._create_session_document()
            
            # Generate embedding
            embedding = self.embedding_service.embed_text(session_text)
            
            # Prepare metadata with duration and topics
            metadata = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": self.get_duration_seconds(),
                "duration_formatted": self.get_duration_formatted(),
                "message_count": len(self.messages),
                "key_topics": [topic["topic"] for topic in self.get_key_topics()],
                "topic_scores": {
                    topic["topic"]: topic["mentions"] 
                    for topic in self.get_key_topics()
                },
                "entities": {k: list(set(v)) for k, v in self.entities.items()},
                "activity_level": self._calculate_activity_level(),
                "type": "session_memory"
            }
            
            # Store in vector database
            self.vector_store.get_store().add_vectors(
                vectors=[embedding],
                metadata=[metadata],
                ids=[f"session:{self.session_id}"]
            )
            
            logger.info(f"Saved session memory to vector store: {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session memory to vector store: {e}")
            return False
    
    def _create_session_document(self) -> str:
        """Create a text document summarizing the session for vectorization"""
        parts = [
            f"Session: {self.session_id}",
            f"Duration: {self.get_duration_formatted()}",
            f"Messages: {len(self.messages)}",
            f"Activity: {self._calculate_activity_level()}",
            "",
            "Key Topics:",
        ]
        
        for topic_info in self.get_key_topics():
            parts.append(
                f"- {topic_info['topic']} "
                f"(mentioned {topic_info['mentions']} times, "
                f"relevance: {topic_info['relevance']})"
            )
        
        if self.entities:
            parts.append("")
            parts.append("Entities Found:")
            for entity_type, entities in self.entities.items():
                unique_entities = list(set(entities))
                if unique_entities:
                    parts.append(f"- {entity_type}: {len(unique_entities)}")
        
        parts.append("")
        parts.append("Conversation Summary:")
        
        # Add message content (limited to avoid token overflow)
        for msg in self.messages[-10:]:  # Last 10 messages
            parts.append(f"{msg['role']}: {msg['content'][:200]}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session memory to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_activity": self.last_activity.isoformat(),
            "duration": {
                "seconds": self.get_duration_seconds(),
                "formatted": self.get_duration_formatted()
            },
            "messages": self.messages,
            "topics": dict(self.topics),
            "key_topics": self.get_key_topics(),
            "entities": {k: list(set(v)) for k, v in self.entities.items()},
            "decisions": self.decisions,
            "summary": self.get_summary()
        }
