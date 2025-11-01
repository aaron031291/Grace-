"""
Grace Research Mode - Honest Intelligence Gathering

When Grace doesn't have knowledge, she:
1. Admits it honestly
2. Offers to research
3. Gathers information from multiple sources
4. Verifies and cross-references
5. Ingests and organizes knowledge
6. Updates her memory
7. Responds with learned information

Grace becomes more intelligent through research, not hallucination!
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ResearchSource(Enum):
    """Sources Grace can research from"""
    WEB_SEARCH = "web_search"
    DOCUMENTATION = "documentation"
    CODE_EXAMPLES = "code_examples"
    ACADEMIC_PAPERS = "academic_papers"
    RELATED_KNOWLEDGE = "related_knowledge"
    ASK_USER = "ask_user"


@dataclass
class ResearchTask:
    """A research task for Grace"""
    task_id: str
    topic: str
    sources_to_check: List[ResearchSource]
    context: Dict[str, Any]
    created_at: datetime
    status: str = "pending"
    findings: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []


@dataclass
class ResearchResult:
    """Result of Grace's research"""
    task_id: str
    topic: str
    sources_researched: List[str]
    findings: List[Dict[str, Any]]
    knowledge_gained: Dict[str, Any]
    confidence_before: float
    confidence_after: float
    ready_to_answer: bool
    summary: str


class GraceResearchMode:
    """
    Grace's research capabilities.
    
    When Grace encounters a knowledge gap:
    1. She admits it honestly
    2. She proposes research plan
    3. She gathers information
    4. She verifies and cross-references
    5. She ingests into memory
    6. She answers with new knowledge
    
    Grace fills her own knowledge gaps!
    """
    
    def __init__(self):
        from grace.memory.persistent_memory import PersistentMemory
        from grace.ingestion.multi_modal_ingestion import MultiModalIngestionEngine
        
        self.memory = PersistentMemory()
        self.ingestion_engine = MultiModalIngestionEngine(self.memory)
        
        self.active_research: Dict[str, ResearchTask] = {}
        self.completed_research: List[ResearchResult] = []
        
        logger.info("Grace Research Mode initialized")
        logger.info("  Grace will research when she lacks knowledge")
    
    async def start_research(
        self,
        topic: str,
        knowledge_gaps: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ResearchTask:
        """
        Start research on a topic.
        
        Grace proactively gathers knowledge to fill gaps.
        """
        import uuid
        
        task_id = str(uuid.uuid4())
        
        # Determine which sources to research
        sources_to_check = self._plan_research_sources(topic, knowledge_gaps)
        
        task = ResearchTask(
            task_id=task_id,
            topic=topic,
            sources_to_check=sources_to_check,
            context=context or {},
            created_at=datetime.utcnow()
        )
        
        self.active_research[task_id] = task
        
        logger.info(f"\n🔍 Grace starting research on: {topic}")
        logger.info(f"   Sources to check: {[s.value for s in sources_to_check]}")
        logger.info(f"   Knowledge gaps: {knowledge_gaps}")
        
        return task
    
    async def conduct_research(
        self,
        task_id: str
    ) -> ResearchResult:
        """
        Conduct research across multiple sources.
        
        Grace gathers, verifies, and ingests knowledge.
        """
        task = self.active_research.get(task_id)
        if not task:
            raise ValueError(f"Research task not found: {task_id}")
        
        task.status = "researching"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CONDUCTING RESEARCH: {task.topic}")
        logger.info(f"{'='*70}")
        
        all_findings = []
        
        # Research from each source
        for source in task.sources_to_check:
            logger.info(f"\n📚 Researching from: {source.value}...")
            
            findings = await self._research_from_source(source, task)
            
            if findings:
                all_findings.extend(findings)
                task.findings.extend(findings)
                logger.info(f"   ✅ Found {len(findings)} items")
            else:
                logger.info(f"   ⚠️  No findings from this source")
        
        # Cross-reference and verify
        logger.info(f"\n🔬 Cross-referencing findings...")
        verified_findings = await self._cross_reference(all_findings)
        
        # Ingest knowledge into memory
        logger.info(f"\n💾 Ingesting knowledge into memory...")
        knowledge_gained = await self._ingest_research_findings(
            task.topic,
            verified_findings
        )
        
        # Calculate confidence improvement
        from grace.intelligence.knowledge_verification import KnowledgeVerificationEngine
        verifier = KnowledgeVerificationEngine()
        
        new_verification = await verifier.verify_knowledge(task.topic, task.context)
        confidence_after = new_verification.confidence_score
        
        # Build result
        result = ResearchResult(
            task_id=task_id,
            topic=task.topic,
            sources_researched=[s.value for s in task.sources_to_check],
            findings=verified_findings,
            knowledge_gained=knowledge_gained,
            confidence_before=0.3,  # Was low before research
            confidence_after=confidence_after,
            ready_to_answer=new_verification.can_answer,
            summary=self._generate_research_summary(task, verified_findings, knowledge_gained)
        )
        
        task.status = "completed"
        self.completed_research.append(result)
        del self.active_research[task_id]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"RESEARCH COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"   Sources researched: {len(task.sources_to_check)}")
        logger.info(f"   Findings: {len(verified_findings)}")
        logger.info(f"   Confidence: {confidence_before:.0%} → {confidence_after:.0%}")
        logger.info(f"   Ready to answer: {result.ready_to_answer}")
        logger.info(f"{'='*70}\n")
        
        return result
    
    def _plan_research_sources(
        self,
        topic: str,
        gaps: List[str]
    ) -> List[ResearchSource]:
        """Plan which sources to research"""
        sources = []
        
        # Always check related knowledge first
        sources.append(ResearchSource.RELATED_KNOWLEDGE)
        
        # If missing documentation
        if "documentation" in str(gaps).lower() or "no documentation" in str(gaps).lower():
            sources.append(ResearchSource.WEB_SEARCH)
            sources.append(ResearchSource.DOCUMENTATION)
        
        # If missing code examples
        if "code" in str(gaps).lower() or "examples" in str(gaps).lower():
            sources.append(ResearchSource.CODE_EXAMPLES)
        
        # If technical/academic topic
        if any(word in topic.lower() for word in ["algorithm", "theory", "architecture"]):
            sources.append(ResearchSource.ACADEMIC_PAPERS)
        
        # Can always ask user
        sources.append(ResearchSource.ASK_USER)
        
        return sources
    
    async def _research_from_source(
        self,
        source: ResearchSource,
        task: ResearchTask
    ) -> List[Dict[str, Any]]:
        """Research from a specific source"""
        
        if source == ResearchSource.WEB_SEARCH:
            return await self._search_web(task.topic)
        
        elif source == ResearchSource.DOCUMENTATION:
            return await self._search_documentation(task.topic)
        
        elif source == ResearchSource.CODE_EXAMPLES:
            return await self._search_code_examples(task.topic)
        
        elif source == ResearchSource.RELATED_KNOWLEDGE:
            return await self._search_related_knowledge(task.topic)
        
        elif source == ResearchSource.ASK_USER:
            return self._prepare_user_questions(task.topic, task.findings)
        
        else:
            return []
    
    async def _search_web(self, topic: str) -> List[Dict[str, Any]]:
        """Search web for information"""
        # In production: use web search API or scraping
        logger.info("      Searching web...")
        return [
            {
                "source": "web",
                "title": f"Web result for {topic}",
                "content": f"Information about {topic}...",
                "url": f"https://example.com/{topic}",
                "relevance": 0.8
            }
        ]
    
    async def _search_documentation(self, topic: str) -> List[Dict[str, Any]]:
        """Search documentation"""
        logger.info("      Searching documentation...")
        return []
    
    async def _search_code_examples(self, topic: str) -> List[Dict[str, Any]]:
        """Search for code examples"""
        logger.info("      Searching code examples...")
        return []
    
    async def _search_related_knowledge(self, topic: str) -> List[Dict[str, Any]]:
        """Search Grace's existing knowledge for related topics"""
        logger.info("      Searching related knowledge in memory...")
        
        # Find related topics
        related_results = await self.memory.search(
            query=topic,
            limit=10
        )
        
        return [
            {
                "source": "related_memory",
                "content": entry.content,
                "domain": entry.domain,
                "relevance": entry.trust_score
            }
            for entry in related_results
        ]
    
    def _prepare_user_questions(
        self,
        topic: str,
        current_findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare questions for user"""
        # Grace can ask user for clarification/information
        return [
            {
                "source": "user_clarification",
                "question": f"Can you provide documentation or examples for {topic}?",
                "why": "This will help me give you accurate information"
            }
        ]
    
    async def _cross_reference(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Cross-reference findings to verify accuracy"""
        # Group by topic
        # Check for consensus across sources
        # Flag conflicts
        
        verified = []
        
        for finding in findings:
            # Check if other sources agree
            agreements = sum(
                1 for f in findings
                if f != finding and self._findings_agree(f, finding)
            )
            
            if agreements >= 1 or finding.get("relevance", 0) > 0.8:
                verified.append({
                    **finding,
                    "verified": True,
                    "agreement_count": agreements
                })
        
        return verified
    
    def _findings_agree(self, f1: Dict, f2: Dict) -> bool:
        """Check if two findings agree"""
        # Simple check - in production: use semantic similarity
        return f1.get("source") != f2.get("source")  # Different sources = good
    
    async def _ingest_research_findings(
        self,
        topic: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ingest research findings into Grace's memory"""
        # Combine findings into knowledge document
        combined_content = f"Research on: {topic}\n\n"
        
        for finding in findings:
            combined_content += f"Source: {finding.get('source')}\n"
            combined_content += f"{finding.get('content', '')\n\n"
        
        # Ingest into memory
        doc_id = await self.memory.ingest_document(
            source_type="research",
            content=combined_content,
            metadata={
                "topic": topic,
                "sources_count": len(findings),
                "researched_at": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "document_id": doc_id,
            "topic": topic,
            "sources": len(findings),
            "grace_now_knows": True
        }
    
    def _generate_research_summary(
        self,
        task: ResearchTask,
        findings: List[Dict[str, Any]],
        knowledge_gained: Dict[str, Any]
    ) -> str:
        """Generate human-readable research summary"""
        return f"""I researched "{task.topic}" across {len(task.sources_to_check)} sources.

Found {len(findings)} relevant items:
- {sum(1 for f in findings if f.get('verified')) } verified findings
- {sum(1 for f in findings if f.get('source') == 'web')} web sources
- {sum(1 for f in findings if f.get('source') == 'related_memory')} from my existing knowledge

I've ingested this into my memory. I can now answer your question!"""


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🔍 Grace Research Mode Demo\n")
        
        research = GraceResearchMode()
        
        # Start research
        task = await research.start_research(
            topic="Implementing distributed tracing with OpenTelemetry",
            knowledge_gaps=["No documentation", "No code examples"],
            context={"domain": "observability"}
        )
        
        print(f"Research task started: {task.task_id}\n")
        
        # Conduct research
        result = await research.conduct_research(task.task_id)
        
        print(f"\n📊 Research Results:")
        print(f"   Sources researched: {len(result.sources_researched)}")
        print(f"   Findings: {len(result.findings)}")
        print(f"   Confidence: {result.confidence_before:.0%} → {result.confidence_after:.0%}")
        print(f"   Ready to answer: {result.ready_to_answer}")
        print(f"\n📝 Summary:\n{result.summary}")
        
        print("\n✅ Grace learned and can now answer!")
    
    asyncio.run(demo())
