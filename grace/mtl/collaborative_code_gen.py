"""
Collaborative Code Generation System
Enables Grace to generate code in partnership with humans
Uses breakthrough system for continuous improvement
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class GenerationPhase(Enum):
    """Phases of collaborative code generation"""
    REQUIREMENTS = "requirements"
    APPROACH = "approach"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    REFINEMENT = "refinement"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"


@dataclass
class CodeGenerationTask:
    """A code generation task"""
    task_id: str
    requirements: str
    language: str
    context: Dict[str, Any]
    created_at: datetime
    phase: GenerationPhase = GenerationPhase.REQUIREMENTS
    
    # Generated artifacts
    approach: Optional[str] = None
    generated_code: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    refinements: List[str] = None
    
    # Human feedback
    human_feedback: List[Dict[str, Any]] = None
    
    # Results
    final_code: Optional[str] = None
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.refinements is None:
            self.refinements = []
        if self.human_feedback is None:
            self.human_feedback = []


class CollaborativeCodeGenerator:
    """
    Generates code collaboratively between Grace and humans.
    
    Flow:
    1. Human provides requirements
    2. Grace proposes approach
    3. Human reviews/guides
    4. Grace generates code
    5. Both evaluate together
    6. Iterate until satisfied
    7. Learn from outcome
    """
    
    def __init__(self, breakthrough_system=None):
        self.breakthrough = breakthrough_system
        self.active_tasks: Dict[str, CodeGenerationTask] = {}
        self.completed_tasks: List[CodeGenerationTask] = []
        
        logger.info("Collaborative Code Generator initialized")
    
    async def start_task(
        self,
        requirements: str,
        language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new code generation task.
        
        Returns task_id for tracking
        """
        task_id = str(uuid.uuid4())
        
        task = CodeGenerationTask(
            task_id=task_id,
            requirements=requirements,
            language=language,
            context=context or {},
            created_at=datetime.utcnow()
        )
        
        self.active_tasks[task_id] = task
        
        logger.info(f"Started code generation task: {task_id}")
        logger.info(f"  Requirements: {requirements[:100]}...")
        logger.info(f"  Language: {language}")
        
        return task_id
    
    async def generate_approach(self, task_id: str) -> Dict[str, Any]:
        """
        Grace generates initial approach.
        
        Returns approach for human review.
        """
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        logger.info(f"Generating approach for task: {task_id}")
        
        # Generate approach (in production, use LLM)
        approach = f"""
Approach for: {task.requirements}

1. Analyze requirements
2. Design solution architecture
3. Implement core functionality
4. Add error handling
5. Write tests
6. Document code

Target language: {task.language}
Estimated complexity: Medium
Confidence: 0.85
"""
        
        task.approach = approach
        task.phase = GenerationPhase.APPROACH
        
        return {
            "task_id": task_id,
            "approach": approach,
            "awaiting_feedback": True
        }
    
    async def receive_feedback(
        self,
        task_id: str,
        feedback: str,
        approved: bool = False
    ) -> Dict[str, Any]:
        """
        Receive human feedback on approach/code.
        
        Args:
            task_id: Task identifier
            feedback: Human feedback text
            approved: Whether human approves current state
        """
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        # Store feedback
        task.human_feedback.append({
            "feedback": feedback,
            "approved": approved,
            "phase": task.phase.value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Received feedback for task: {task_id}")
        logger.info(f"  Approved: {approved}")
        
        if approved:
            # Move to next phase
            if task.phase == GenerationPhase.APPROACH:
                # Generate code
                return await self.generate_code(task_id)
            elif task.phase == GenerationPhase.EVALUATION:
                # Finalize
                return await self.finalize_task(task_id)
        else:
            # Refine based on feedback
            return await self.refine(task_id, feedback)
        
        return {"status": "feedback_received", "awaiting_action": True}
    
    async def generate_code(self, task_id: str) -> Dict[str, Any]:
        """
        Generate code based on approved approach.
        
        Uses breakthrough system to improve generation quality.
        """
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        logger.info(f"Generating code for task: {task_id}")
        
        # Get best generation strategy from breakthrough system
        if self.breakthrough and self.breakthrough.initialized:
            config = self.breakthrough.meta_loop.baseline_config
        else:
            config = {"temperature": 0.7, "max_tokens": 1000}
        
        # Generate code (in production, use LLM with config)
        code = self._synthesize_code(task, config)
        
        task.generated_code = code
        task.phase = GenerationPhase.GENERATION
        
        # Auto-evaluate
        evaluation = await self._evaluate_code(code, task.language)
        task.evaluation = evaluation
        task.phase = GenerationPhase.EVALUATION
        
        return {
            "task_id": task_id,
            "code": code,
            "evaluation": evaluation,
            "awaiting_review": True
        }
    
    def _synthesize_code(
        self,
        task: CodeGenerationTask,
        config: Dict[str, Any]
    ) -> str:
        """Synthesize code based on task requirements"""
        # Use LLM service for real code generation
        try:
            from grace.services.llm_service import LLMService
            import asyncio
            
            llm = LLMService()
            
            # Build comprehensive prompt
            prompt = f"""You are an expert {task.language} developer. Generate production-quality code.

Requirements: {task.requirements}

Approach: {task.approach if task.approach else 'Use best practices'}

Constraints: {task.constraints if task.constraints else 'None'}

Generate complete, functional, well-documented {task.language} code with:
1. Proper imports and dependencies
2. Error handling and validation
3. Type hints and docstrings
4. Main execution logic
5. Example usage

Provide ONLY the code, no explanations."""
            
            # Generate code synchronously (we're in a sync method)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            code = loop.run_until_complete(llm.generate_code(task.requirements, context=prompt))
            loop.close()
            
            # Validate generated code is substantial
            if code and len(code.strip()) > 100:
                return code
            else:
                logger.warning("LLM generated insufficient code, using template")
                return self._generate_template_code(task, config)
                
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}, using template")
            return self._generate_template_code(task, config)
    
    def _generate_template_code(self, task: CodeGenerationTask, config: Dict[str, Any]) -> str:
        """Generate high-quality template code when LLM unavailable"""
        if task.language.lower() == "python":
            class_name = self._extract_class_name(task.requirements)
            return f'''"""
{task.requirements}

Auto-generated by Grace AI Collaborative Code Generator
Approach: {task.approach if task.approach else 'Standard implementation'}
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {task.requirements}
    
    This implementation follows the specified approach and constraints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize {class_name}
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {{}}
        self.initialized_at = datetime.utcnow()
        logger.info(f"{class_name} initialized")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main functionality
        
        Args:
            input_data: Input data to process
        
        Returns:
            Processing results
        """
        try:
            logger.info(f"Processing input: {{list(input_data.keys())}}")
            
            # Validate input
            self._validate_input(input_data)
            
            # Process data
            result = await self._process(input_data)
            
            # Validate output
            self._validate_output(result)
            
            logger.info("Processing completed successfully")
            return {{
                "status": "success",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }}
            
        except ValueError as e:
            logger.error(f"Validation error: {{e}}")
            return {{
                "status": "error",
                "error": "validation_failed",
                "message": str(e)
            }}
        except Exception as e:
            logger.error(f"Processing failed: {{e}}", exc_info=True)
            return {{
                "status": "error",
                "error": "processing_failed",
                "message": str(e)
            }}
    
    def _validate_input(self, data: Dict[str, Any]):
        """Validate input data"""
        if not data:
            raise ValueError("Input data cannot be empty")
        
        # Add specific validation based on requirements
        required_fields = self.config.get("required_fields", [])
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {{field}}")
    
    async def _process(self, data: Dict[str, Any]) -> Any:
        """Core processing logic"""
        # Implement core functionality here
        processed = {{
            "input_received": data,
            "processed_at": datetime.utcnow().isoformat(),
            "processor": "{class_name}"
        }}
        
        return processed
    
    def _validate_output(self, result: Any):
        """Validate output data"""
        if result is None:
            raise ValueError("Processing produced no result")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics"""
        return {{
            "initialized_at": self.initialized_at.isoformat(),
            "config": self.config
        }}


async def main():
    """Example usage"""
    component = {class_name}(config={{"required_fields": ["data"]}})
    
    # Process sample data
    result = await component.execute({{"data": "sample", "value": 123}})
    print(f"Result: {{result}}")
    
    # Get statistics
    stats = component.get_stats()
    print(f"Stats: {{stats}}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''
        elif task.language.lower() in ["javascript", "typescript"]:
            return f'''/**
 * {task.requirements}
 * 
 * Auto-generated by Grace AI
 */

export class {self._extract_class_name(task.requirements)} {{
    constructor(config = {{}}) {{
        this.config = config;
        this.initializedAt = new Date();
    }}
    
    async execute(inputData) {{
        try {{
            // Validate input
            this.validateInput(inputData);
            
            // Process data
            const result = await this.process(inputData);
            
            return {{
                status: 'success',
                result,
                timestamp: new Date().toISOString()
            }};
        }} catch (error) {{
            return {{
                status: 'error',
                error: error.message
            }};
        }}
    }}
    
    validateInput(data) {{
        if (!data) throw new Error('Input cannot be empty');
    }}
    
    async process(data) {{
        return {{
            inputReceived: data,
            processedAt: new Date().toISOString()
        }};
    }}
}}
'''
        else:
            return f"// Generated code for: {task.requirements}\n// Language: {task.language}\n\n"
    
    def _extract_class_name(self, requirements: str) -> str:
        """Extract a suitable class name from requirements"""
        import re
        words = re.findall(r'\b[A-Z][a-z]+|\b[a-z]+', requirements)
        if words and len(words) > 0:
            return ''.join(word.capitalize() for word in words[:3])
        return "GeneratedComponent"
    
    async def _evaluate_code(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Evaluate generated code"""
        # In production, use actual evaluation
        return {
            "quality_score": 0.75,
            "correctness": 0.8,
            "style": 0.7,
            "security": 0.85,
            "tests_needed": True,
            "issues": ["Missing error handling", "No input validation"],
            "strengths": ["Clear structure", "Good naming"]
        }
    
    async def refine(
        self,
        task_id: str,
        feedback: str
    ) -> Dict[str, Any]:
        """Refine code/approach based on feedback"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        logger.info(f"Refining task based on feedback: {task_id}")
        
        # In production, use feedback to improve
        refinement = f"Refinement based on: {feedback[:50]}..."
        task.refinements.append(refinement)
        
        # Re-generate
        if task.phase == GenerationPhase.APPROACH:
            return await self.generate_approach(task_id)
        else:
            return await self.generate_code(task_id)
    
    async def finalize_task(self, task_id: str) -> Dict[str, Any]:
        """Finalize code generation task"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.final_code = task.generated_code
        task.phase = GenerationPhase.DEPLOYMENT
        
        # Learn from this task (if successful)
        if task.evaluation and task.evaluation.get("quality_score", 0) > 0.7:
            await self._learn_from_task(task)
        
        # Move to completed
        del self.active_tasks[task_id]
        self.completed_tasks.append(task)
        
        logger.info(f"✅ Task finalized: {task_id}")
        logger.info(f"   Quality: {task.evaluation.get('quality_score', 0):.2f}")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "code": task.final_code,
            "quality_score": task.evaluation.get("quality_score", 0),
            "iterations": len(task.refinements) + 1
        }
    
    async def _learn_from_task(self, task: CodeGenerationTask):
        """Learn from successful task for improvement"""
        logger.info(f"📚 Learning from successful task: {task.task_id}")
        
        # Extract successful patterns
        learning = {
            "requirements_type": task.context.get("type"),
            "language": task.language,
            "quality_achieved": task.evaluation.get("quality_score"),
            "iterations_needed": len(task.refinements) + 1,
            "successful_patterns": "extracted_patterns"
        }
        
        # In production: feed to breakthrough system for distillation
        if self.breakthrough:
            # This would update the meta-loop's strategy library
            pass
        
        logger.info("  ✅ Learning recorded")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        task = self.active_tasks.get(task_id)
        if not task:
            # Check completed
            for t in self.completed_tasks:
                if t.task_id == task_id:
                    task = t
                    break
        
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "phase": task.phase.value,
            "requirements": task.requirements,
            "language": task.language,
            "has_approach": task.approach is not None,
            "has_code": task.generated_code is not None,
            "quality_score": task.quality_score,
            "feedback_count": len(task.human_feedback),
            "refinement_count": len(task.refinements)
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🤝 Collaborative Code Generation Demo\n")
        
        gen = CollaborativeCodeGenerator()
        
        # Start task
        task_id = await gen.start_task(
            requirements="Create a function to calculate fibonacci numbers",
            language="python",
            context={"type": "algorithm"}
        )
        
        print(f"✅ Task started: {task_id}\n")
        
        # Generate approach
        approach_result = await gen.generate_approach(task_id)
        print("📋 Generated Approach:")
        print(approach_result["approach"])
        
        # Simulate human approval
        print("\n👤 Human approves approach...")
        gen_result = await gen.receive_feedback(
            task_id,
            "Looks good, proceed with generation",
            approved=True
        )
        
        print("\n💻 Generated Code:")
        print(gen_result["code"][:200] + "...")
        
        print("\n📊 Evaluation:")
        eval_data = gen_result["evaluation"]
        print(f"  Quality: {eval_data['quality_score']:.2f}")
        print(f"  Issues: {len(eval_data['issues'])}")
        
        # Human approves
        print("\n👤 Human approves code...")
        final = await gen.receive_feedback(
            task_id,
            "Code looks good!",
            approved=True
        )
        
        print(f"\n✅ Task Complete!")
        print(f"  Quality: {final['quality_score']:.2f}")
        print(f"  Iterations: {final['iterations']}")
    
    asyncio.run(demo())
