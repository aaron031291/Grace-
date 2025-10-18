"""
Unified Output Schema - Canonical loop output format (Class 9)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Individual step in reasoning chain"""
    step_number: int
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    timestamp: str


@dataclass
class TrustMetrics:
    """Trust and confidence metrics"""
    overall_trust: float
    component_trust: Dict[str, float]
    consensus_confidence: float
    governance_passed: bool
    memory_quality: float
    feedback_score: Optional[float] = None


@dataclass
class GraceLoopOutput:
    """
    Canonical output format for Grace loops
    
    Complete, structured output with all metadata
    """
    # Identification
    output_id: str
    loop_id: str
    timestamp: str
    
    # Input summary
    input_summary: str
    input_context: Dict[str, Any]
    
    # Decision
    decision: Any
    decision_type: str
    
    # Reasoning
    reasoning_chain: List[ReasoningStep]
    explanation: str
    
    # Trust metrics
    trust_metrics: TrustMetrics
    
    # Consensus
    consensus_id: Optional[str]
    specialist_agreement: Optional[float]
    
    # Governance
    governance_validation: Dict[str, Any]
    
    # Additional metadata
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class UnifiedOutputGenerator:
    """
    Generates canonical GraceLoopOutput objects
    """
    
    def __init__(self):
        self.output_history: List[GraceLoopOutput] = []
        logger.info("UnifiedOutputGenerator initialized")
    
    def generate_output(
        self,
        loop_id: str,
        input_data: Dict[str, Any],
        decision: Any,
        reasoning_steps: List[Dict[str, Any]],
        trust_metrics: Dict[str, Any],
        consensus_data: Optional[Dict[str, Any]] = None,
        governance_result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraceLoopOutput:
        """
        Generate canonical loop output
        
        Args:
            loop_id: Loop identifier
            input_data: Input data and context
            decision: The decision made
            reasoning_steps: Steps in reasoning process
            trust_metrics: Trust and confidence metrics
            consensus_data: Specialist consensus data
            governance_result: Governance validation result
            metadata: Additional metadata
            
        Returns:
            GraceLoopOutput object
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Generate output ID
        output_id = self._generate_output_id(loop_id, timestamp)
        
        # Create reasoning chain
        reasoning_chain = [
            ReasoningStep(
                step_number=i + 1,
                description=step.get("description", ""),
                input_data=step.get("input", {}),
                output_data=step.get("output", {}),
                confidence=step.get("confidence", 1.0),
                timestamp=step.get("timestamp", timestamp)
            )
            for i, step in enumerate(reasoning_steps)
        ]
        
        # Create trust metrics
        trust_metrics_obj = TrustMetrics(
            overall_trust=trust_metrics.get("overall_trust", 0.5),
            component_trust=trust_metrics.get("component_trust", {}),
            consensus_confidence=trust_metrics.get("consensus_confidence", 0.5),
            governance_passed=trust_metrics.get("governance_passed", True),
            memory_quality=trust_metrics.get("memory_quality", 0.5),
            feedback_score=trust_metrics.get("feedback_score")
        )
        
        # Create output
        output = GraceLoopOutput(
            output_id=output_id,
            loop_id=loop_id,
            timestamp=timestamp,
            input_summary=self._summarize_input(input_data),
            input_context=input_data.get("context", {}),
            decision=decision,
            decision_type=type(decision).__name__,
            reasoning_chain=reasoning_chain,
            explanation=self._generate_explanation(reasoning_chain, decision),
            trust_metrics=trust_metrics_obj,
            consensus_id=consensus_data.get("consensus_id") if consensus_data else None,
            specialist_agreement=consensus_data.get("agreement") if consensus_data else None,
            governance_validation=governance_result or {"passed": True, "violations": []},
            metadata=metadata or {}
        )
        
        # Store in history
        self.output_history.append(output)
        
        logger.info(f"Generated unified output: {output_id} for loop {loop_id}")
        
        return output
    
    def _generate_output_id(self, loop_id: str, timestamp: str) -> str:
        """Generate unique output ID"""
        import hashlib
        data = f"{loop_id}:{timestamp}"
        hash_val = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"output_{hash_val}"
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> str:
        """Generate input summary"""
        summary_parts = []
        
        if "query" in input_data:
            summary_parts.append(f"Query: {input_data['query']}")
        
        if "task" in input_data:
            summary_parts.append(f"Task: {input_data['task']}")
        
        if "context" in input_data:
            context_keys = list(input_data["context"].keys())[:3]
            summary_parts.append(f"Context: {', '.join(context_keys)}")
        
        return "; ".join(summary_parts) if summary_parts else "No input summary"
    
    def _generate_explanation(
        self,
        reasoning_chain: List[ReasoningStep],
        decision: Any
    ) -> str:
        """Generate human-readable explanation"""
        if not reasoning_chain:
            return f"Decision: {decision}"
        
        explanation = f"Decision reached through {len(reasoning_chain)} reasoning steps:\n"
        
        for step in reasoning_chain[:3]:  # Show first 3 steps
            explanation += f"  {step.step_number}. {step.description}\n"
        
        if len(reasoning_chain) > 3:
            explanation += f"  ... and {len(reasoning_chain) - 3} more steps\n"
        
        explanation += f"\nFinal decision: {decision}"
        
        return explanation
    
    def get_output_by_id(self, output_id: str) -> Optional[GraceLoopOutput]:
        """Retrieve output by ID"""
        for output in self.output_history:
            if output.output_id == output_id:
                return output
        return None
    
    def get_outputs_for_loop(self, loop_id: str) -> List[GraceLoopOutput]:
        """Get all outputs for a specific loop"""
        return [o for o in self.output_history if o.loop_id == loop_id]
