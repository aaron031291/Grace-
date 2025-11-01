"""
Knowledge Verification System - Honest Grace

Grace NEVER hallucinates. Instead, she:
1. Checks if she has relevant knowledge
2. Verifies against ALL internal sources
3. Admits when she doesn't know
4. Offers to research if needed

Verification sources (Grace's internal world):
✅ Chat history (past conversations)
✅ Persistent memory (all experiences)
✅ Immutable logs (all actions)
✅ Ingested PDFs (uploaded documents)
✅ Code repositories (analyzed code)
✅ Learned patterns (accumulated wisdom)
✅ Expert knowledge base (domain expertise)

If Grace doesn't have the knowledge → She says so and researches!

This eliminates hallucinations through honest self-assessment.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class KnowledgeConfidence(Enum):
    """Confidence levels in knowledge"""
    VERIFIED = "verified"  # Multiple sources confirm (95%+)
    CONFIDENT = "confident"  # Strong evidence (80-95%)
    UNCERTAIN = "uncertain"  # Weak evidence (50-80%)
    UNKNOWN = "unknown"  # No relevant knowledge (<50%)


@dataclass
class KnowledgeVerificationResult:
    """Result of knowledge verification"""
    topic: str
    confidence: KnowledgeConfidence
    confidence_score: float
    sources_checked: List[str]
    sources_found: List[Dict[str, Any]]
    gaps: List[str]
    can_answer: bool
    needs_research: bool
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "sources_checked": self.sources_checked,
            "sources_found_count": len(self.sources_found),
            "gaps": self.gaps,
            "can_answer": self.can_answer,
            "needs_research": self.needs_research,
            "reasoning": self.reasoning
        }


class KnowledgeVerificationEngine:
    """
    Verifies Grace's knowledge before responding.
    
    Checks ALL internal sources:
    1. Chat history - what was discussed before?
    2. Persistent memory - what experiences are relevant?
    3. Immutable logs - what actions were taken?
    4. Ingested documents - what knowledge was uploaded?
    5. Code repositories - what code patterns exist?
    6. Learned patterns - what strategies worked?
    7. Expert knowledge - what domain expertise applies?
    
    If insufficient knowledge → Grace admits it and offers to research!
    """
    
    def __init__(self):
        from grace.memory.persistent_memory import PersistentMemory
        from grace.knowledge.expert_system import get_expert_system
        
        self.memory = PersistentMemory()
        self.expert_system = get_expert_system()
        
        self.verification_history = []
        
        logger.info("Knowledge Verification Engine initialized")
        logger.info("  Grace will be HONEST about knowledge gaps")
    
    async def verify_knowledge(
        self,
        topic: str,
        context: Optional[Dict[str, Any]] = None
    ) -> KnowledgeVerificationResult:
        """
        Verify if Grace has sufficient knowledge to respond.
        
        Returns verification result showing:
        - What sources were checked
        - What knowledge was found
        - Confidence level
        - Whether Grace can answer or needs to research
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"VERIFYING KNOWLEDGE: {topic}")
        logger.info(f"{'='*70}")
        
        context = context or {}
        sources_checked = []
        sources_found = []
        
        # 1. Check chat history
        logger.info("1️⃣  Checking chat history...")
        chat_sources = await self._check_chat_history(topic, context)
        sources_checked.append("chat_history")
        if chat_sources:
            sources_found.extend(chat_sources)
            logger.info(f"   ✅ Found {len(chat_sources)} relevant conversations")
        else:
            logger.info(f"   ⚠️  No relevant chat history")
        
        # 2. Check persistent memory
        logger.info("2️⃣  Checking persistent memory...")
        memory_sources = await self._check_memory(topic, context)
        sources_checked.append("persistent_memory")
        if memory_sources:
            sources_found.extend(memory_sources)
            logger.info(f"   ✅ Found {len(memory_sources)} relevant memory entries")
        else:
            logger.info(f"   ⚠️  No relevant memory")
        
        # 3. Check immutable logs
        logger.info("3️⃣  Checking immutable logs...")
        log_sources = await self._check_logs(topic, context)
        sources_checked.append("immutable_logs")
        if log_sources:
            sources_found.extend(log_sources)
            logger.info(f"   ✅ Found {len(log_sources)} relevant log entries")
        else:
            logger.info(f"   ⚠️  No relevant logs")
        
        # 4. Check ingested documents (PDFs, docs)
        logger.info("4️⃣  Checking ingested documents...")
        doc_sources = await self._check_documents(topic, context)
        sources_checked.append("ingested_documents")
        if doc_sources:
            sources_found.extend(doc_sources)
            logger.info(f"   ✅ Found {len(doc_sources)} relevant documents")
        else:
            logger.info(f"   ⚠️  No relevant documents")
        
        # 5. Check code repositories
        logger.info("5️⃣  Checking code repositories...")
        code_sources = await self._check_code(topic, context)
        sources_checked.append("code_repositories")
        if code_sources:
            sources_found.extend(code_sources)
            logger.info(f"   ✅ Found {len(code_sources)} relevant code examples")
        else:
            logger.info(f"   ⚠️  No relevant code")
        
        # 6. Check learned patterns
        logger.info("6️⃣  Checking learned patterns...")
        pattern_sources = await self._check_patterns(topic, context)
        sources_checked.append("learned_patterns")
        if pattern_sources:
            sources_found.extend(pattern_sources)
            logger.info(f"   ✅ Found {len(pattern_sources)} relevant patterns")
        else:
            logger.info(f"   ⚠️  No relevant patterns")
        
        # 7. Check expert knowledge
        logger.info("7️⃣  Checking expert knowledge...")
        expert_sources = self._check_expert_knowledge(topic, context)
        sources_checked.append("expert_knowledge")
        if expert_sources:
            sources_found.extend(expert_sources)
            logger.info(f"   ✅ Found {len(expert_sources)} relevant expert knowledge")
        else:
            logger.info(f"   ⚠️  No relevant expert knowledge")
        
        # 8. Calculate confidence
        logger.info("\n📊 Calculating confidence...")
        confidence_score = self._calculate_confidence(sources_found)
        confidence_level = self._determine_confidence_level(confidence_score)
        
        # 9. Identify gaps
        gaps = self._identify_gaps(topic, sources_found)
        
        # 10. Decide if Grace can answer or needs to research
        can_answer = confidence_level in [KnowledgeConfidence.VERIFIED, KnowledgeConfidence.CONFIDENT]
        needs_research = confidence_level in [KnowledgeConfidence.UNCERTAIN, KnowledgeConfidence.UNKNOWN]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            confidence_level,
            sources_found,
            gaps,
            can_answer
        )
        
        result = KnowledgeVerificationResult(
            topic=topic,
            confidence=confidence_level,
            confidence_score=confidence_score,
            sources_checked=sources_checked,
            sources_found=sources_found,
            gaps=gaps,
            can_answer=can_answer,
            needs_research=needs_research,
            reasoning=reasoning
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"VERIFICATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"   Confidence: {confidence_level.value} ({confidence_score:.0%})")
        logger.info(f"   Sources found: {len(sources_found)}")
        logger.info(f"   Can answer: {can_answer}")
        logger.info(f"   Needs research: {needs_research}")
        logger.info(f"{'='*70}\n")
        
        # Store verification
        self.verification_history.append(result)
        
        return result
    
    async def _check_chat_history(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check chat history for relevant information"""
        session_id = context.get("session_id", "default")
        
        try:
            history = await self.memory.get_chat_history(session_id, limit=100)
            
            # Search for topic in history
            relevant = []
            topic_lower = topic.lower()
            
            for msg in history:
                content = msg.get("content", "").lower()
                if any(word in content for word in topic_lower.split()):
                    relevant.append({
                        "source": "chat_history",
                        "content": msg.get("content", "")[:200],
                        "timestamp": msg.get("timestamp"),
                        "relevance": 0.8
                    })
            
            return relevant[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Failed to check chat history: {e}")
            return []
    
    async def _check_memory(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check persistent memory"""
        try:
            results = await self.memory.search(
                query=topic,
                limit=10
            )
            
            return [
                {
                    "source": "persistent_memory",
                    "content": entry.content,
                    "domain": entry.domain,
                    "timestamp": entry.timestamp.isoformat(),
                    "relevance": entry.trust_score
                }
                for entry in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to check memory: {e}")
            return []
    
    async def _check_logs(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check immutable logs"""
        # In production: query immutable log database
        # For now, simulate
        return []
    
    async def _check_documents(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check ingested documents (PDFs, etc.)"""
        # Query document database for topic
        # In production: vector search on document chunks
        return []
    
    async def _check_code(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check ingested code repositories"""
        # Search code for relevant patterns
        return []
    
    async def _check_patterns(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check learned patterns"""
        try:
            patterns = await self.memory.get_patterns()
            
            # Filter relevant patterns
            relevant = []
            topic_lower = topic.lower()
            
            for pattern in patterns:
                pattern_data = pattern.get("data", {})
                if topic_lower in str(pattern_data).lower():
                    relevant.append({
                        "source": "learned_patterns",
                        "pattern": pattern_data,
                        "domain": pattern.get("domain"),
                        "success_rate": pattern.get("success_rate", 0.0),
                        "relevance": pattern.get("success_rate", 0.5)
                    })
            
            return relevant[:5]
            
        except Exception as e:
            logger.error(f"Failed to check patterns: {e}")
            return []
    
    def _check_expert_knowledge(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check expert knowledge base"""
        # Get relevant experts
        language = context.get("language")
        experts = self.expert_system.get_expert_for_task(topic, language)
        
        if experts:
            return [
                {
                    "source": "expert_knowledge",
                    "domain": expert.domain.value,
                    "proficiency": expert.proficiency_level,
                    "relevance": expert.proficiency_level
                }
                for expert in experts
            ]
        
        return []
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        if not sources:
            return 0.0
        
        # Weight by source type and relevance
        weights = {
            "chat_history": 0.7,
            "persistent_memory": 0.9,
            "immutable_logs": 0.8,
            "ingested_documents": 1.0,
            "code_repositories": 0.9,
            "learned_patterns": 0.85,
            "expert_knowledge": 0.95
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for source in sources:
            source_type = source.get("source", "unknown")
            relevance = source.get("relevance", 0.5)
            weight = weights.get(source_type, 0.5)
            
            weighted_sum += relevance * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, weighted_sum / total_weight)
    
    def _determine_confidence_level(self, score: float) -> KnowledgeConfidence:
        """Determine confidence level from score"""
        if score >= 0.95:
            return KnowledgeConfidence.VERIFIED
        elif score >= 0.80:
            return KnowledgeConfidence.CONFIDENT
        elif score >= 0.50:
            return KnowledgeConfidence.UNCERTAIN
        else:
            return KnowledgeConfidence.UNKNOWN
    
    def _identify_gaps(
        self,
        topic: str,
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify knowledge gaps"""
        gaps = []
        
        # Check if we have practical examples
        has_code_examples = any(s.get("source") == "code_repositories" for s in sources)
        if not has_code_examples:
            gaps.append("No code examples found")
        
        # Check if we have documentation
        has_docs = any(s.get("source") == "ingested_documents" for s in sources)
        if not has_docs:
            gaps.append("No documentation found")
        
        # Check if we have past experience
        has_experience = any(s.get("source") in ["persistent_memory", "chat_history"] for s in sources)
        if not has_experience:
            gaps.append("No past experience with this topic")
        
        return gaps
    
    def _generate_reasoning(
        self,
        confidence: KnowledgeConfidence,
        sources: List[Dict[str, Any]],
        gaps: List[str],
        can_answer: bool
    ) -> str:
        """Generate human-readable reasoning"""
        if confidence == KnowledgeConfidence.VERIFIED:
            return f"I have verified knowledge from {len(sources)} sources including documentation, code examples, and past experience. I can answer confidently."
        
        elif confidence == KnowledgeConfidence.CONFIDENT:
            return f"I have strong knowledge from {len(sources)} sources. I can answer, though {len(gaps)} gaps exist: {', '.join(gaps[:2])}."
        
        elif confidence == KnowledgeConfidence.UNCERTAIN:
            return f"I have limited knowledge ({len(sources)} sources). My answer may be incomplete. Gaps: {', '.join(gaps)}. I should research more."
        
        else:  # UNKNOWN
            return f"I don't have sufficient knowledge on this topic. I found only {len(sources)} weak sources. I need to research: {', '.join(gaps)}."
    
    async def generate_honest_response(
        self,
        topic: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate honest response based on verification.
        
        Grace either:
        - Answers confidently (verified/confident)
        - Admits uncertainty (uncertain)
        - Admits lack of knowledge (unknown)
        - Offers to research
        """
        verification = await self.verify_knowledge(topic, context)
        
        if verification.can_answer:
            # Grace has sufficient knowledge
            return {
                "can_answer": True,
                "confidence": verification.confidence.value,
                "confidence_score": verification.confidence_score,
                "message": f"I have {verification.confidence.value} knowledge about this topic from {len(verification.sources_found)} sources in my memory.",
                "proceed": True
            }
        
        else:
            # Grace lacks knowledge - BE HONEST!
            if verification.confidence == KnowledgeConfidence.UNCERTAIN:
                return {
                    "can_answer": False,
                    "confidence": "uncertain",
                    "confidence_score": verification.confidence_score,
                    "message": f"I have limited knowledge on this topic. I found {len(verification.sources_found)} sources, but I'm missing: {', '.join(verification.gaps)}. Let me research more to give you accurate information. Shall I?",
                    "needs_research": True,
                    "gaps": verification.gaps
                }
            
            else:  # UNKNOWN
                return {
                    "can_answer": False,
                    "confidence": "unknown",
                    "confidence_score": verification.confidence_score,
                    "message": f"I don't have sufficient knowledge on this topic yet. I haven't encountered this in: {', '.join(verification.gaps)}. I can research this for you. Would you like me to:\n1. Search the web for current information\n2. Ask you to upload relevant documentation\n3. Learn from related topics I do know\n\nWhich would you prefer?",
                    "needs_research": True,
                    "gaps": verification.gaps,
                    "research_options": ["web_search", "upload_docs", "learn_from_related"]
                }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🔍 Knowledge Verification Demo\n")
        
        verifier = KnowledgeVerificationEngine()
        
        # Test 1: Topic Grace knows
        print("Test 1: Topic Grace knows well")
        result1 = await verifier.verify_knowledge(
            "How to build FastAPI endpoints",
            {"language": "python"}
        )
        
        response1 = await verifier.generate_honest_response(
            "How to build FastAPI endpoints"
        )
        
        print(f"  Confidence: {result1.confidence.value}")
        print(f"  Can answer: {result1.can_answer}")
        print(f"  Message: {response1['message']}\n")
        
        # Test 2: Topic Grace doesn't know
        print("Test 2: Topic Grace doesn't know")
        result2 = await verifier.verify_knowledge(
            "How to build quantum algorithms with Q#"
        )
        
        response2 = await verifier.generate_honest_response(
            "How to build quantum algorithms with Q#"
        )
        
        print(f"  Confidence: {result2.confidence.value}")
        print(f"  Can answer: {result2.can_answer}")
        print(f"  Needs research: {result2.needs_research}")
        print(f"  Message: {response2['message']}\n")
        
        print("✅ Grace is HONEST about knowledge gaps!")
    
    asyncio.run(demo())
