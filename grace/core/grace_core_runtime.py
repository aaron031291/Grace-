"""
GraceCore Runtime - Integrated Clarity Framework runtime loop
Implements all Clarity Classes 5-10 in the main execution loop
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import time

from grace.clarity.memory_scoring import LoopMemoryBank, MemoryType
from grace.clarity.governance_validation import ConstitutionValidator
from grace.clarity.feedback_integration import FeedbackIntegrator, FeedbackType
from grace.clarity.specialist_consensus import MLDLSpecialist, SpecialistType
from grace.clarity.output_schema import GraceLoopOutput, OutputFormatter, OutputStatus
from grace.clarity.drift_detection import LoopDriftDetector

logger = logging.getLogger(__name__)


class GraceCoreRuntime:
    """
    Grace Core Runtime with full Clarity Framework integration
    Implements Classes 5-10 in the main execution loop
    """
    
    def __init__(self):
        # Initialize Clarity Framework components
        self.memory_bank = LoopMemoryBank()
        self.constitution = ConstitutionValidator()
        self.feedback = FeedbackIntegrator(self.memory_bank)
        self.specialist = MLDLSpecialist()
        self.output_formatter = OutputFormatter()
        self.drift_detector = LoopDriftDetector()
        
        # Runtime state
        self.loop_id = f"grace_loop_{int(time.time())}"
        self.iteration = 0
        self.running = False
        
        # Register specialists
        self._initialize_specialists()
        
        logger.info("GraceCoreRuntime initialized with Clarity Framework")
    
    def _initialize_specialists(self):
        """Initialize MLDL specialists"""
        specialists = [
            (SpecialistType.REASONING, 1.2),
            (SpecialistType.ETHICS, 1.5),
            (SpecialistType.SAFETY, 2.0),
            (SpecialistType.CREATIVITY, 0.8),
            (SpecialistType.ANALYSIS, 1.0),
            (SpecialistType.SYNTHESIS, 1.0)
        ]
        
        for spec_type, weight in specialists:
            self.specialist.register_specialist(
                specialist_id=f"{spec_type.value}_specialist",
                specialist_type=spec_type,
                weight=weight
            )
    
    def execute_loop(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> GraceLoopOutput:
        """
        Execute Grace loop with full Clarity Framework integration
        This is the main runtime loop implementing Classes 5-10
        """
        start_time = time.time()
        self.iteration += 1
        context = context or {}
        
        logger.info(f"=== Executing Grace Loop {self.loop_id} - Iteration {self.iteration} ===")
        
        # Step 1: Retrieve and score relevant memories (Class 5)
        relevant_memories = self._retrieve_scored_memories(task, context)
        
        # Step 2: Process task with memory context
        raw_result = self._process_task(task, relevant_memories, context)
        
        # Step 3: Validate against constitution (Class 6)
        validation_result = self.constitution.validate_against_constitution(
            raw_result,
            context
        )
        
        # Step 4: Get specialist consensus (Class 8)
        quorum_result = self.specialist.evaluate(
            proposal={
                'task': task,
                'result': raw_result,
                'ethical_score': raw_result.get('ethical_score', 0.7),
                'safety_score': raw_result.get('safety_score', 0.8),
                'logic_score': raw_result.get('logic_score', 0.75)
            },
            required_specialists={SpecialistType.REASONING, SpecialistType.SAFETY, SpecialistType.ETHICS}
        )
        
        # Step 5: Create universal output format (Class 9)
        execution_time = time.time() - start_time
        
        output = self.output_formatter.create_output(
            loop_id=self.loop_id,
            iteration=self.iteration,
            result=raw_result,
            confidence=raw_result.get('confidence', 0.5),
            status=OutputStatus.SUCCESS if quorum_result.decision and validation_result.passed else OutputStatus.FAILED,
            clarity_score=raw_result.get('clarity', 1.0),
            ambiguity_score=raw_result.get('ambiguity', 0.0),
            constitution_compliant=validation_result.passed,
            validation_details={
                'score': validation_result.score,
                'violations': validation_result.violations,
                'warnings': validation_result.warnings
            },
            quorum_decision=quorum_result.decision,
            consensus_strength=quorum_result.consensus_strength,
            specialist_votes=[
                {'specialist': v.specialist_id, 'decision': v.decision, 'confidence': v.confidence}
                for v in quorum_result.participating_specialists
            ],
            execution_time=execution_time,
            errors=[] if validation_result.passed else [v['description'] for v in validation_result.violations],
            warnings=[w['description'] for w in validation_result.warnings]
        )
        
        # Step 6: Integrate feedback to memory (Class 7)
        feedback_entry = self.feedback.loop_output_to_memory(
            loop_id=self.loop_id,
            loop_output=output.to_dict(),
            feedback_type=FeedbackType.POSITIVE if quorum_result.decision else FeedbackType.CORRECTIVE
        )
        
        output.feedback_applied = feedback_entry.applied
        output.feedback_ids = [feedback_entry.feedback_id]
        
        # Step 7: Detect loop drift (Class 10)
        drift_alerts = self.drift_detector.track_loop(
            self.loop_id,
            self.iteration,
            output.to_dict()
        )
        
        if drift_alerts:
            critical_drifts = [a for a in drift_alerts if a.severity.value in ['error', 'critical']]
            if critical_drifts:
                output.errors.extend([f"Drift: {a.description}" for a in critical_drifts])
                output.status = OutputStatus.PARTIAL
        
        # Step 8: Store execution as memory
        self._store_execution_memory(output, task, context)
        
        logger.info(f"Loop execution complete - Status: {output.status.value}, "
                   f"Confidence: {output.confidence:.2f}, "
                   f"Consensus: {output.consensus_strength:.2f}")
        
        return output
    
    def _retrieve_scored_memories(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve and score relevant memories (Class 5)"""
        relevant = []
        
        # Query similar task memories
        task_type = task.get('type', 'general')
        memories = self.memory_bank.query_knowledge(
            node_type=None,
            filters={'task_type': task_type} if 'task_type' in str(self.memory_bank.memories) else None,
            min_confidence=0.3
        )
        
        # Score each memory
        for memory in memories[:10]:  # Limit to top 10
            scores = self.memory_bank.score(memory.memory_id, context)
            
            if scores['composite'] > 0.4:
                relevant.append({
                    'memory_id': memory.memory_id,
                    'content': memory.content,
                    'scores': scores
                })
        
        # Sort by composite score
        relevant.sort(key=lambda m: m['scores']['composite'], reverse=True)
        
        logger.debug(f"Retrieved {len(relevant)} scored memories")
        return relevant
    
    def _process_task(
        self,
        task: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process task with memory context"""
        # Simulate task processing
        result = {
            'task_type': task.get('type', 'general'),
            'decision': True,
            'confidence': 0.85,
            'clarity': 0.9,
            'ambiguity': 0.1,
            'ethical_score': 0.75,
            'safety_score': 0.85,
            'logic_score': 0.8,
            'memory_context_used': len(memories),
            'output': f"Processed {task.get('type', 'task')} with {len(memories)} memory contexts"
        }
        
        # Apply memory insights
        if memories:
            avg_memory_score = sum(m['scores']['composite'] for m in memories) / len(memories)
            result['confidence'] *= (0.7 + avg_memory_score * 0.3)
        
        return result
    
    def _store_execution_memory(
        self,
        output: GraceLoopOutput,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Store execution as memory"""
        self.memory_bank.store(
            memory_id=f"exec_{self.loop_id}_{self.iteration}",
            memory_type=MemoryType.EPISODIC,
            content={
                'loop_id': self.loop_id,
                'iteration': self.iteration,
                'task': task,
                'result': output.result,
                'confidence': output.confidence,
                'status': output.status.value,
                'consensus_strength': output.consensus_strength
            },
            source="loop_execution",
            metadata={
                'execution_time': output.execution_time,
                'constitution_compliant': output.constitution_compliant
            }
        )
    
    def get_runtime_status(self) -> Dict[str, Any]:
        """Get comprehensive runtime status"""
        return {
            'loop_id': self.loop_id,
            'iteration': self.iteration,
            'running': self.running,
            'memory_stats': self.memory_bank.get_memory_statistics(),
            'constitution_stats': self.constitution.get_violation_statistics(),
            'feedback_stats': self.feedback.analyze_feedback_patterns(),
            'quorum_stats': self.specialist.get_quorum_statistics(),
            'output_stats': self.output_formatter.get_output_statistics(),
            'drift_report': self.drift_detector.linter.get_drift_report(),
            'loop_health': self.drift_detector.get_loop_health(self.loop_id)
        }
