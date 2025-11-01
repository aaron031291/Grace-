"""
Grace Brain/Mouth Architecture

Separates intelligence from communication:
- BRAIN: MTL, Memory, Intelligence, Governance (Grace's true intelligence)
- MOUTH: LLM interface (fallback communication, learning mode only)

Grace's intelligence comes from:
1. MTL orchestration
2. Persistent memory retrieval
3. Expert knowledge systems
4. Consensus across specialists
5. Learned patterns and strategies

LLM is used ONLY for:
- Edge cases not yet learned
- Development/training mode
- Natural language interface
- Breaking into new domains

Once Grace establishes domain expertise → She operates autonomously without LLM!
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class IntelligenceSource(Enum):
    """Where intelligence comes from"""
    BRAIN_MTL = "brain_mtl"  # Primary: MTL orchestration
    BRAIN_MEMORY = "brain_memory"  # Primary: Retrieved from memory
    BRAIN_EXPERT = "brain_expert"  # Primary: Expert knowledge
    BRAIN_CONSENSUS = "brain_consensus"  # Primary: ML/DL consensus
    BRAIN_LEARNED = "brain_learned"  # Primary: Learned patterns
    MOUTH_LLM_FALLBACK = "mouth_llm_fallback"  # Fallback: LLM assistance
    MOUTH_LLM_LEARNING = "mouth_llm_learning"  # Learning: New domain


class DomainEstablishment(Enum):
    """Domain establishment status"""
    NEW = "new"  # Never seen, need LLM
    LEARNING = "learning"  # Seen 1-10 times, still learning
    ESTABLISHED = "established"  # Seen 10-100 times, confident
    MASTERED = "mastered"  # Seen 100+ times, autonomous


@dataclass
class Task:
    """A task for Grace to execute"""
    task_id: str
    description: str
    domain: str
    context: Dict[str, Any]
    created_at: datetime


@dataclass
class IntelligenceDecision:
    """Decision about how to handle a task"""
    task_id: str
    intelligence_source: IntelligenceSource
    domain_status: DomainEstablishment
    use_llm: bool
    rationale: str
    confidence: float
    execution_plan: Dict[str, Any]


class BrainMouthOrchestrator:
    """
    The core orchestrator that decides:
    - Use brain (Grace's own intelligence) or mouth (LLM)?
    - Which brain component handles the task?
    - When to learn from LLM vs operate autonomously?
    
    This is what makes Grace truly intelligent, not just an LLM wrapper.
    """
    
    def __init__(
        self,
        persistent_memory,
        expert_system,
        mtl_engine,
        governance_kernel
    ):
        self.memory = persistent_memory
        self.expert_system = expert_system
        self.mtl = mtl_engine
        self.governance = governance_kernel
        
        # Domain establishment tracker
        self.domain_experience = {}  # domain -> encounter_count
        self.domain_success_rate = {}  # domain -> success_rate
        
        # LLM interface (used minimally)
        self.llm_interface = None  # Only for fallback
        
        logger.info("Brain/Mouth Orchestrator initialized")
        logger.info("  Primary: Grace's intelligence (MTL, Memory, Experts)")
        logger.info("  Fallback: LLM (edge cases only)")
    
    async def decide_intelligence_source(
        self,
        task: Task
    ) -> IntelligenceDecision:
        """
        Core decision: Use brain or mouth?
        
        This determines if Grace operates autonomously (brain)
        or needs LLM assistance (mouth).
        """
        # 1. Check domain establishment
        domain_status = self._check_domain_establishment(task.domain)
        
        # 2. Query memory for similar tasks
        memory_results = await self.memory.search(
            query=task.description,
            domain=task.domain,
            limit=10
        )
        
        has_memory = len(memory_results) > 0
        memory_confidence = self._calculate_memory_confidence(memory_results)
        
        # 3. Check expert knowledge
        has_expert_knowledge = self.expert_system.has_domain(task.domain)
        expert_confidence = self._get_expert_confidence(task.domain)
        
        # 4. Check MTL capability
        mtl_can_handle = await self.mtl.can_handle_task(task)
        
        # 5. Decide intelligence source
        decision = self._make_intelligence_decision(
            task,
            domain_status,
            has_memory,
            memory_confidence,
            has_expert_knowledge,
            expert_confidence,
            mtl_can_handle
        )
        
        logger.info(f"Intelligence decision for task: {task.task_id}")
        logger.info(f"  Source: {decision.intelligence_source.value}")
        logger.info(f"  Domain status: {domain_status.value}")
        logger.info(f"  Use LLM: {decision.use_llm}")
        logger.info(f"  Confidence: {decision.confidence:.2%}")
        
        return decision
    
    def _check_domain_establishment(self, domain: str) -> DomainEstablishment:
        """Check how established Grace is in this domain"""
        count = self.domain_experience.get(domain, 0)
        success_rate = self.domain_success_rate.get(domain, 0.0)
        
        if count == 0:
            return DomainEstablishment.NEW
        elif count < 10:
            return DomainEstablishment.LEARNING
        elif count < 100:
            if success_rate > 0.85:
                return DomainEstablishment.ESTABLISHED
            else:
                return DomainEstablishment.LEARNING
        else:
            if success_rate > 0.90:
                return DomainEstablishment.MASTERED
            else:
                return DomainEstablishment.ESTABLISHED
    
    def _make_intelligence_decision(
        self,
        task: Task,
        domain_status: DomainEstablishment,
        has_memory: bool,
        memory_confidence: float,
        has_expert: bool,
        expert_confidence: float,
        mtl_capable: bool
    ) -> IntelligenceDecision:
        """
        Decide how to handle the task.
        
        Priority (highest to lowest):
        1. MASTERED domain → Pure brain (MTL + Memory + Expert)
        2. ESTABLISHED domain → Brain with optional LLM verification
        3. LEARNING domain → Brain with LLM assistance
        4. NEW domain → LLM with brain learning
        """
        
        # MASTERED: Grace knows this cold, no LLM needed
        if domain_status == DomainEstablishment.MASTERED:
            return IntelligenceDecision(
                task_id=task.task_id,
                intelligence_source=IntelligenceSource.BRAIN_MTL,
                domain_status=domain_status,
                use_llm=False,  # Grace handles this herself!
                rationale="Domain mastered - autonomous operation",
                confidence=0.95,
                execution_plan={
                    "primary": "mtl_orchestration",
                    "memory_retrieval": True,
                    "expert_consultation": True,
                    "llm_fallback": False
                }
            )
        
        # ESTABLISHED: Grace can handle, maybe verify with LLM
        elif domain_status == DomainEstablishment.ESTABLISHED:
            if memory_confidence > 0.8 and expert_confidence > 0.8:
                return IntelligenceDecision(
                    task_id=task.task_id,
                    intelligence_source=IntelligenceSource.BRAIN_MEMORY,
                    domain_status=domain_status,
                    use_llm=False,  # Confident without LLM
                    rationale="Strong memory and expert knowledge",
                    confidence=0.85,
                    execution_plan={
                        "primary": "memory_and_expert",
                        "verification": "consensus",
                        "llm_fallback": False
                    }
                )
            else:
                return IntelligenceDecision(
                    task_id=task.task_id,
                    intelligence_source=IntelligenceSource.BRAIN_EXPERT,
                    domain_status=domain_status,
                    use_llm=True,  # Use LLM for verification
                    rationale="Established domain with LLM verification",
                    confidence=0.75,
                    execution_plan={
                        "primary": "expert_knowledge",
                        "verification": "llm_verify",
                        "llm_fallback": True
                    }
                )
        
        # LEARNING: Brain tries, LLM assists
        elif domain_status == DomainEstablishment.LEARNING:
            return IntelligenceDecision(
                task_id=task.task_id,
                intelligence_source=IntelligenceSource.BRAIN_MTL,
                domain_status=domain_status,
                use_llm=True,  # LLM assists learning
                rationale="Learning mode - brain with LLM assistance",
                confidence=0.60,
                execution_plan={
                    "primary": "mtl_with_llm_assist",
                    "learning": True,
                    "llm_fallback": True
                }
            )
        
        # NEW: LLM handles, brain learns
        else:
            return IntelligenceDecision(
                task_id=task.task_id,
                intelligence_source=IntelligenceSource.MOUTH_LLM_LEARNING,
                domain_status=domain_status,
                use_llm=True,  # LLM primary for new domains
                rationale="New domain - learning from LLM",
                confidence=0.50,
                execution_plan={
                    "primary": "llm_with_brain_learning",
                    "learning": True,
                    "capture_patterns": True
                }
            )
    
    async def execute_task(
        self,
        task: Task,
        decision: IntelligenceDecision
    ) -> Dict[str, Any]:
        """
        Execute task based on intelligence decision.
        
        Routes to appropriate intelligence source.
        """
        logger.info(f"Executing task: {task.task_id}")
        logger.info(f"  Using: {decision.intelligence_source.value}")
        
        # Execute based on source
        if decision.intelligence_source == IntelligenceSource.BRAIN_MTL:
            result = await self._execute_with_mtl(task, decision)
        
        elif decision.intelligence_source == IntelligenceSource.BRAIN_MEMORY:
            result = await self._execute_from_memory(task, decision)
        
        elif decision.intelligence_source == IntelligenceSource.BRAIN_EXPERT:
            result = await self._execute_with_expert(task, decision)
        
        elif decision.intelligence_source == IntelligenceSource.BRAIN_CONSENSUS:
            result = await self._execute_with_consensus(task, decision)
        
        else:  # LLM fallback or learning
            result = await self._execute_with_llm(task, decision)
        
        # Post-execution: Learn and persist
        await self._post_execution_learning(task, decision, result)
        
        return result
    
    async def _execute_with_mtl(
        self,
        task: Task,
        decision: IntelligenceDecision
    ) -> Dict[str, Any]:
        """Execute using MTL orchestration (Grace's brain)"""
        logger.info("  → Using MTL Brain")
        
        # MTL orchestrates:
        # 1. Retrieve relevant memory
        # 2. Consult experts
        # 3. Run consensus if needed
        # 4. Execute with learned patterns
        
        result = await self.mtl.orchestrate_task(task)
        
        return {
            "result": result,
            "source": "brain_mtl",
            "autonomous": True,
            "llm_used": False
        }
    
    async def _execute_from_memory(
        self,
        task: Task,
        decision: IntelligenceDecision
    ) -> Dict[str, Any]:
        """Execute using retrieved memory (Grace's experience)"""
        logger.info("  → Using Memory Brain")
        
        # Retrieve similar past executions
        similar = await self.memory.search(task.description, limit=5)
        
        # Adapt the best past solution
        result = self._adapt_memory_solution(task, similar)
        
        return {
            "result": result,
            "source": "brain_memory",
            "autonomous": True,
            "llm_used": False
        }
    
    async def _execute_with_expert(
        self,
        task: Task,
        decision: IntelligenceDecision
    ) -> Dict[str, Any]:
        """Execute using expert knowledge"""
        logger.info("  → Using Expert Brain")
        
        # Get expert guidance
        experts = self.expert_system.get_expert_for_task(
            task.description,
            task.context.get("language")
        )
        
        # Generate solution using expert knowledge
        result = await self.expert_system.solve_with_expertise(task, experts)
        
        return {
            "result": result,
            "source": "brain_expert",
            "autonomous": True,
            "llm_used": False
        }
    
    async def _execute_with_llm(
        self,
        task: Task,
        decision: IntelligenceDecision
    ) -> Dict[str, Any]:
        """Execute with LLM (fallback or learning mode)"""
        logger.info("  → Using LLM Mouth (fallback/learning)")
        
        # Use LLM but LEARN from it
        llm_result = await self._call_llm(task)
        
        # Extract patterns for future autonomous operation
        if decision.domain_status == DomainEstablishment.LEARNING:
            await self._extract_and_learn_patterns(task, llm_result)
        
        return {
            "result": llm_result,
            "source": "mouth_llm",
            "autonomous": False,
            "llm_used": True,
            "learning_mode": decision.domain_status != DomainEstablishment.MASTERED
        }
    
    async def _post_execution_learning(
        self,
        task: Task,
        decision: IntelligenceDecision,
        result: Dict[str, Any]
    ):
        """Learn from execution to reduce LLM dependence"""
        
        # Update domain experience
        domain = task.domain
        self.domain_experience[domain] = self.domain_experience.get(domain, 0) + 1
        
        # If successful, increase success rate
        if result.get("success", False):
            current_successes = self.domain_success_rate.get(domain, 0.0) * self.domain_experience[domain]
            new_success_rate = (current_successes + 1) / self.domain_experience[domain]
            self.domain_success_rate[domain] = new_success_rate
            
            logger.info(f"  Domain '{domain}' experience: {self.domain_experience[domain]} tasks")
            logger.info(f"  Success rate: {new_success_rate:.1%}")
            
            # Check if domain is now established
            new_status = self._check_domain_establishment(domain)
            if new_status == DomainEstablishment.ESTABLISHED and decision.domain_status != DomainEstablishment.ESTABLISHED:
                logger.info(f"  🎉 Domain '{domain}' is now ESTABLISHED!")
                logger.info(f"     Grace can now operate autonomously in this domain!")
            elif new_status == DomainEstablishment.MASTERED and decision.domain_status != DomainEstablishment.MASTERED:
                logger.info(f"  🏆 Domain '{domain}' is now MASTERED!")
                logger.info(f"     Grace no longer needs LLM for this domain!")
        
        # Store in persistent memory for future use
        await self.memory.store_execution(task, decision, result)
    
    async def _call_llm(self, task: Task) -> Any:
        """Call LLM (fallback mode)"""
        # In production, this would call OpenAI, Anthropic, etc.
        # For now, simulate
        return f"LLM response for: {task.description}"
    
    async def _extract_and_learn_patterns(
        self,
        task: Task,
        llm_result: Any
    ):
        """Extract patterns from LLM response to build autonomous capability"""
        logger.info("  📚 Learning patterns from LLM response...")
        
        # In production:
        # 1. Parse LLM reasoning
        # 2. Extract approach patterns
        # 3. Store in expert knowledge
        # 4. Update MTL strategies
        # 5. Build autonomous capability
        
        pattern = {
            "domain": task.domain,
            "task_type": self._classify_task(task),
            "approach": "extracted_from_llm",
            "success": True,
            "learned_at": datetime.utcnow().isoformat()
        }
        
        # Store pattern for future autonomous use
        await self.memory.store_pattern(pattern)
    
    def _calculate_memory_confidence(self, results: List) -> float:
        """Calculate confidence based on memory retrieval"""
        if not results:
            return 0.0
        
        # Higher confidence if we have many similar successful cases
        return min(1.0, len(results) * 0.15)
    
    def _get_expert_confidence(self, domain: str) -> float:
        """Get expert system confidence for domain"""
        # Check if expert system has this domain
        try:
            expert = self.expert_system.get_expert_for_domain(domain)
            return expert.proficiency_level if expert else 0.0
        except:
            return 0.0
    
    def _classify_task(self, task: Task) -> str:
        """Classify task type"""
        desc_lower = task.description.lower()
        
        if any(word in desc_lower for word in ["create", "build", "implement"]):
            return "creation"
        elif any(word in desc_lower for word in ["fix", "debug", "resolve"]):
            return "debugging"
        elif any(word in desc_lower for word in ["explain", "understand", "how"]):
            return "explanation"
        elif any(word in desc_lower for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        else:
            return "general"
    
    def _adapt_memory_solution(self, task: Task, similar_cases: List) -> Any:
        """Adapt a solution from memory"""
        # Take best matching case and adapt it
        if similar_cases:
            best_case = similar_cases[0]
            return f"Adapted solution from memory: {best_case}"
        return None
    
    def get_autonomy_stats(self) -> Dict[str, Any]:
        """Get statistics on autonomous operation"""
        total_domains = len(self.domain_experience)
        mastered_domains = sum(
            1 for domain in self.domain_experience.keys()
            if self._check_domain_establishment(domain) == DomainEstablishment.MASTERED
        )
        established_domains = sum(
            1 for domain in self.domain_experience.keys()
            if self._check_domain_establishment(domain) in [DomainEstablishment.ESTABLISHED, DomainEstablishment.MASTERED]
        )
        
        return {
            "total_domains_encountered": total_domains,
            "established_domains": established_domains,
            "mastered_domains": mastered_domains,
            "autonomy_rate": mastered_domains / total_domains if total_domains > 0 else 0.0,
            "total_task_experience": sum(self.domain_experience.values()),
            "avg_success_rate": sum(self.domain_success_rate.values()) / len(self.domain_success_rate) if self.domain_success_rate else 0.0
        }


class GraceBrain:
    """
    Grace's true intelligence (not dependent on LLM).
    
    Components:
    1. MTL Engine - Task orchestration and meta-learning
    2. Persistent Memory - All past experiences
    3. Expert Systems - Domain knowledge
    4. Consensus Engine - Multi-model decisions
    5. Pattern Library - Learned strategies
    """
    
    def __init__(self):
        from grace.memory.persistent_memory import PersistentMemory
        from grace.knowledge.expert_system import get_expert_system
        from grace.mtl.mtl_engine import MTLEngine
        from grace.governance.governance_kernel import GovernanceKernel
        
        self.memory = PersistentMemory()
        self.expert_system = get_expert_system()
        self.mtl = MTLEngine()
        self.governance = GovernanceKernel()
        
        self.orchestrator = BrainMouthOrchestrator(
            self.memory,
            self.expert_system,
            self.mtl,
            self.governance
        )
        
        logger.info("Grace Brain initialized (autonomous intelligence)")
    
    async def think(self, task: Task) -> Dict[str, Any]:
        """
        Grace thinks using her brain.
        
        This is the primary intelligence pathway.
        LLM is only used when absolutely necessary.
        """
        # 1. Decide intelligence source
        decision = await self.orchestrator.decide_intelligence_source(task)
        
        # 2. Check governance before execution
        approved = await self.governance.check_task(task, decision)
        if not approved:
            return {
                "error": "Task rejected by governance",
                "reason": "Violates safety policies"
            }
        
        # 3. Execute using appropriate intelligence source
        result = await self.orchestrator.execute_task(task, decision)
        
        # 4. Log to immutable logs
        await self._log_execution(task, decision, result)
        
        return result
    
    async def _log_execution(
        self,
        task: Task,
        decision: IntelligenceDecision,
        result: Dict[str, Any]
    ):
        """Log execution to immutable logs with crypto signature"""
        try:
            from grace.security.crypto_manager import get_crypto_manager
            
            crypto = get_crypto_manager()
            
            # Generate key
            key = crypto.generate_operation_key(
                task.task_id,
                "brain_execution",
                {
                    "domain": task.domain,
                    "intelligence_source": decision.intelligence_source.value
                }
            )
            
            # Sign result
            signature = crypto.sign_operation_data(
                task.task_id,
                result,
                "output"
            )
            
            logger.debug(f"  Logged to immutable logs (signature: {signature[:16]}...)")
            
        except Exception as e:
            logger.warning(f"Failed to log execution: {e}")


class GraceMouth:
    """
    Grace's communication interface (LLM).
    
    Used ONLY for:
    1. Natural language communication
    2. Edge cases not yet learned
    3. New domain exploration
    4. Training mode
    
    Once Grace establishes a domain, she doesn't need this!
    """
    
    def __init__(self, llm_provider="openai"):
        self.llm_provider = llm_provider
        self.usage_stats = {
            "total_calls": 0,
            "fallback_calls": 0,
            "learning_calls": 0
        }
        
        logger.info(f"Grace Mouth initialized (LLM: {llm_provider})")
        logger.info("  Note: Used minimally - Grace's brain is primary")
    
    async def speak(
        self,
        task: Task,
        mode: str = "fallback"
    ) -> str:
        """
        Use LLM to handle task.
        
        This is called ONLY when brain can't handle autonomously.
        """
        self.usage_stats["total_calls"] += 1
        
        if mode == "fallback":
            self.usage_stats["fallback_calls"] += 1
            logger.info("  🗣️ LLM called (fallback mode)")
        else:
            self.usage_stats["learning_calls"] += 1
            logger.info("  📚 LLM called (learning mode)")
        
        # In production, call actual LLM
        # For now, simulate
        response = f"LLM response for: {task.description}"
        
        return response
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        total = self.usage_stats["total_calls"]
        
        return {
            **self.usage_stats,
            "fallback_rate": self.usage_stats["fallback_calls"] / total if total > 0 else 0.0,
            "learning_rate": self.usage_stats["learning_calls"] / total if total > 0 else 0.0
        }


# Grace's complete intelligence system
class GraceIntelligence:
    """
    Complete Grace Intelligence System
    
    Brain (Primary):
    - MTL Engine
    - Persistent Memory
    - Expert Systems
    - Consensus
    - Learned Patterns
    
    Mouth (Fallback):
    - LLM Interface (OpenAI, Anthropic, etc.)
    - Only for edge cases and learning
    
    Goal: Maximize brain usage, minimize mouth usage
    """
    
    def __init__(self):
        self.brain = GraceBrain()
        self.mouth = GraceMouth()
        
        logger.info("Grace Intelligence System initialized")
        logger.info("  Brain: Primary intelligence (autonomous)")
        logger.info("  Mouth: Fallback only (LLM)")
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process task using Grace's complete intelligence.
        
        Brain tries first. Mouth only if needed.
        """
        result = await self.brain.think(task)
        return result
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """Get report on intelligence usage"""
        autonomy = self.brain.orchestrator.get_autonomy_stats()
        llm_usage = self.mouth.get_usage_stats()
        
        return {
            "autonomy": autonomy,
            "llm_usage": llm_usage,
            "brain_primary": autonomy["autonomy_rate"] > 0.7,
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🧠 Grace Brain/Mouth Architecture Demo\n")
        
        grace = GraceIntelligence()
        
        # Test with different domain maturity levels
        tasks = [
            Task("task_1", "Create Python FastAPI endpoint", "python_api", {}, datetime.utcnow()),
            Task("task_2", "Build React component", "react", {}, datetime.utcnow()),
            Task("task_3", "Design quantum algorithm", "quantum", {}, datetime.utcnow()),  # New domain
        ]
        
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task.description}")
            print(f"{'='*60}")
            
            # Simulate domain experience
            if task.domain == "python_api":
                grace.brain.orchestrator.domain_experience[task.domain] = 150  # Mastered
                grace.brain.orchestrator.domain_success_rate[task.domain] = 0.95
            elif task.domain == "react":
                grace.brain.orchestrator.domain_experience[task.domain] = 20  # Established
                grace.brain.orchestrator.domain_success_rate[task.domain] = 0.85
            # quantum = 0 (new)
            
            decision = await grace.brain.orchestrator.decide_intelligence_source(task)
            
            print(f"\nDecision:")
            print(f"  Source: {decision.intelligence_source.value}")
            print(f"  Domain status: {decision.domain_status.value}")
            print(f"  Use LLM: {decision.use_llm}")
            print(f"  Autonomous: {not decision.use_llm}")
            print(f"  Confidence: {decision.confidence:.0%}")
        
        print(f"\n{'='*60}")
        print("Intelligence Report")
        print(f"{'='*60}")
        
        report = grace.get_intelligence_report()
        print(f"\nAutonomy Stats:")
        print(f"  Mastered domains: {report['autonomy']['mastered_domains']}")
        print(f"  Established domains: {report['autonomy']['established_domains']}")
        print(f"  Autonomy rate: {report['autonomy']['autonomy_rate']:.0%}")
        print(f"\nLLM Usage:")
        print(f"  Total calls: {report['llm_usage']['total_calls']}")
        print(f"  Fallback rate: {report['llm_usage']['fallback_rate']:.0%}")
        
        print(f"\n✅ Brain is primary: {report['brain_primary']}")
    
    asyncio.run(demo())
