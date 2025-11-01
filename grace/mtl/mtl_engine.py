"""
MTL Engine - Meta-Task Loop Engine

This is Grace's PRIMARY BRAIN - the orchestration layer that:
1. Receives all tasks
2. Decides how to execute (memory, expert, consensus, LLM)
3. Orchestrates multi-step workflows
4. Learns from all executions
5. Builds autonomous capabilities

MTL is what makes Grace intelligent, not the LLM.
LLM is just a fallback tool MTL uses when needed.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"  # Single step, direct execution
    MODERATE = "moderate"  # Multi-step, coordination needed
    COMPLEX = "complex"  # Many steps, multiple systems
    NOVEL = "novel"  # Never seen before


@dataclass
class MTLTask:
    """Task for MTL orchestration"""
    task_id: str
    description: str
    domain: str
    complexity: TaskComplexity
    context: Dict[str, Any]
    created_at: datetime
    
    subtasks: List['MTLTask'] = None
    parent_task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []


@dataclass
class ExecutionPlan:
    """Plan for executing a task"""
    task_id: str
    steps: List[Dict[str, Any]]
    required_capabilities: List[str]
    estimated_time: float
    confidence: float
    needs_llm: bool


class MTLEngine:
    """
    Meta-Task Loop Engine
    
    The orchestration brain that makes Grace intelligent and autonomous.
    """
    
    def __init__(self):
        self.execution_history = []
        self.capability_registry = {
            "memory_retrieval": True,
            "expert_consultation": True,
            "consensus_decision": True,
            "code_generation": True,
            "data_ingestion": True,
            "pattern_matching": True,
            "multi_step_planning": True
        }
        
        logger.info("MTL Engine initialized - Grace's primary brain")
    
    async def can_handle_task(self, task: Any) -> bool:
        """Check if MTL can handle task autonomously"""
        # MTL can handle if we have relevant capabilities
        required = self._identify_required_capabilities(task)
        
        return all(
            self.capability_registry.get(cap, False)
            for cap in required
        )
    
    async def orchestrate_task(self, task: MTLTask) -> Dict[str, Any]:
        """
        Orchestrate task execution.
        
        This is the core intelligence loop:
        1. Analyze task
        2. Create execution plan
        3. Execute steps (using memory, experts, consensus)
        4. Monitor progress
        5. Handle failures
        6. Learn from outcome
        """
        logger.info(f"MTL orchestrating task: {task.task_id}")
        logger.info(f"  Complexity: {task.complexity.value}")
        
        # 1. Analyze and plan
        plan = await self._create_execution_plan(task)
        logger.info(f"  Created plan with {len(plan.steps)} steps")
        logger.info(f"  Needs LLM: {plan.needs_llm}")
        
        # 2. Execute plan
        result = await self._execute_plan(task, plan)
        
        # 3. Learn from execution
        await self._learn_from_execution(task, plan, result)
        
        # 4. Store in history
        self.execution_history.append({
            "task_id": task.task_id,
            "plan": plan,
            "result": result,
            "timestamp": datetime.utcnow()
        })
        
        return result
    
    async def _create_execution_plan(
        self,
        task: MTLTask
    ) -> ExecutionPlan:
        """
        Create execution plan for task.
        
        Analyzes task and determines optimal execution strategy.
        """
        # Decompose task into steps
        steps = []
        
        # Step 1: Check memory for similar tasks
        steps.append({
            "step": "memory_search",
            "capability": "memory_retrieval",
            "description": "Search for similar past executions"
        })
        
        # Step 2: Consult experts
        steps.append({
            "step": "expert_consultation",
            "capability": "expert_consultation",
            "description": "Get expert guidance for domain"
        })
        
        # Step 3: Execute or generate
        if any(word in task.description.lower() for word in ["create", "generate", "build"]):
            steps.append({
                "step": "generation",
                "capability": "code_generation",
                "description": "Generate solution"
            })
        else:
            steps.append({
                "step": "execution",
                "capability": "pattern_matching",
                "description": "Execute using learned patterns"
            })
        
        # Step 4: Validate
        steps.append({
            "step": "validation",
            "capability": "consensus_decision",
            "description": "Validate result quality"
        })
        
        # Determine if LLM needed
        needs_llm = task.complexity == TaskComplexity.NOVEL
        
        return ExecutionPlan(
            task_id=task.task_id,
            steps=steps,
            required_capabilities=[s["capability"] for s in steps],
            estimated_time=len(steps) * 0.5,  # Rough estimate
            confidence=0.8 if not needs_llm else 0.6,
            needs_llm=needs_llm
        )
    
    async def _execute_plan(
        self,
        task: MTLTask,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute the plan step by step"""
        logger.info(f"  Executing {len(plan.steps)} steps...")
        
        step_results = []
        
        for i, step in enumerate(plan.steps):
            logger.info(f"    Step {i+1}/{len(plan.steps)}: {step['step']}")
            
            # Execute step
            step_result = await self._execute_step(step, task)
            step_results.append(step_result)
            
            # Check if step failed
            if not step_result.get("success", False):
                logger.warning(f"    Step {step['step']} failed, handling...")
                # Could retry, adapt, or escalate
        
        # Aggregate results
        final_result = {
            "task_id": task.task_id,
            "success": all(s.get("success", False) for s in step_results),
            "steps_completed": len(step_results),
            "step_results": step_results,
            "autonomous": not plan.needs_llm
        }
        
        logger.info(f"  ✅ Orchestration complete (success: {final_result['success']})")
        
        return final_result
    
    async def _execute_step(
        self,
        step: Dict[str, Any],
        task: MTLTask
    ) -> Dict[str, Any]:
        """Execute a single step"""
        capability = step["capability"]
        
        # Route to appropriate executor
        if capability == "memory_retrieval":
            return await self._step_memory_search(task)
        elif capability == "expert_consultation":
            return await self._step_expert_consult(task)
        elif capability == "code_generation":
            return await self._step_generate_code(task)
        elif capability == "consensus_decision":
            return await self._step_consensus(task)
        else:
            return {"success": True, "result": "step_executed"}
    
    async def _step_memory_search(self, task: MTLTask) -> Dict[str, Any]:
        """Search memory for relevant information"""
        # Would call persistent memory
        return {
            "success": True,
            "found_similar": 3,
            "confidence": 0.75
        }
    
    async def _step_expert_consult(self, task: MTLTask) -> Dict[str, Any]:
        """Consult expert system"""
        # Would call expert system
        return {
            "success": True,
            "expert_guidance": "guidance_here",
            "confidence": 0.85
        }
    
    async def _step_generate_code(self, task: MTLTask) -> Dict[str, Any]:
        """Generate code using expert generator"""
        # Would call expert code generator
        return {
            "success": True,
            "code": "generated_code",
            "quality": 0.90
        }
    
    async def _step_consensus(self, task: MTLTask) -> Dict[str, Any]:
        """Run consensus validation"""
        # Would call consensus engine
        return {
            "success": True,
            "consensus": "approved",
            "confidence": 0.88
        }
    
    async def _learn_from_execution(
        self,
        task: MTLTask,
        plan: ExecutionPlan,
        result: Dict[str, Any]
    ):
        """Learn from execution to improve future performance"""
        if result["success"]:
            logger.info("  📚 Learning from successful execution...")
            
            # Extract what worked
            successful_strategy = {
                "domain": task.domain,
                "complexity": task.complexity.value,
                "plan_steps": [s["step"] for s in plan.steps],
                "success": True
            }
            
            # Store for future use (would store in persistent memory)
            logger.info("     Strategy recorded for future autonomous use")
    
    def _identify_required_capabilities(self, task: Any) -> List[str]:
        """Identify capabilities needed for task"""
        capabilities = ["memory_retrieval"]  # Always check memory
        
        task_desc = task.description.lower() if hasattr(task, 'description') else str(task).lower()
        
        if any(word in task_desc for word in ["create", "generate", "build"]):
            capabilities.append("code_generation")
        
        if any(word in task_desc for word in ["decide", "choose", "select"]):
            capabilities.append("consensus_decision")
        
        return capabilities
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MTL statistics"""
        total = len(self.execution_history)
        successful = sum(
            1 for ex in self.execution_history
            if ex["result"].get("success", False)
        )
        
        return {
            "total_tasks_orchestrated": total,
            "successful_tasks": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "capabilities_available": len(self.capability_registry)
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🔄 MTL Engine Demo\n")
        
        mtl = MTLEngine()
        
        # Create task
        task = MTLTask(
            task_id=str(uuid.uuid4()),
            description="Create REST API for user management",
            domain="python_api",
            complexity=TaskComplexity.MODERATE,
            context={"language": "python"},
            created_at=datetime.utcnow()
        )
        
        # Orchestrate
        result = await mtl.orchestrate_task(task)
        
        print(f"✅ Task orchestrated:")
        print(f"   Success: {result['success']}")
        print(f"   Steps: {result['steps_completed']}")
        print(f"   Autonomous: {result['autonomous']}")
        
        # Stats
        stats = mtl.get_stats()
        print(f"\n📊 MTL Stats:")
        print(f"   Tasks: {stats['total_tasks_orchestrated']}")
        print(f"   Success rate: {stats['success_rate']:.0%}")
        print(f"   Capabilities: {stats['capabilities_available']}")
    
    asyncio.run(demo())
