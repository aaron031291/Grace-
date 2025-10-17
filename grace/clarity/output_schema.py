"""
Class 9: Output Format - Universal GraceLoopOutput schema
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class OutputStatus(Enum):
    """Status of loop output"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class GraceLoopOutput:
    """
    Universal output schema for Grace loops
    Core implementation of Class 9
    """
    loop_id: str
    iteration: int
    status: OutputStatus
    result: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Clarity metrics
    clarity_score: float = 1.0
    ambiguity_score: float = 0.0
    
    # Validation
    constitution_compliant: bool = True
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Feedback
    feedback_applied: bool = False
    feedback_ids: List[str] = field(default_factory=list)
    
    # Consensus
    quorum_decision: Optional[bool] = None
    consensus_strength: float = 0.0
    
    # Metadata
    execution_time: float = 0.0
    memory_used: int = 0
    specialist_votes: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraceLoopOutput':
        """Create from dictionary"""
        if isinstance(data.get('status'), str):
            data['status'] = OutputStatus(data['status'])
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class OutputFormatter:
    """
    Formats and validates GraceLoopOutput
    Ensures system-wide consistency
    """
    
    def __init__(self):
        self.required_fields = {
            'loop_id', 'iteration', 'status', 'result', 'confidence'
        }
        self.output_history: List[GraceLoopOutput] = []
        logger.info("OutputFormatter initialized")
    
    def create_output(
        self,
        loop_id: str,
        iteration: int,
        result: Dict[str, Any],
        confidence: float,
        status: OutputStatus = OutputStatus.SUCCESS,
        **kwargs
    ) -> GraceLoopOutput:
        """Create standardized loop output"""
        output = GraceLoopOutput(
            loop_id=loop_id,
            iteration=iteration,
            status=status,
            result=result,
            confidence=confidence,
            **kwargs
        )
        
        # Validate
        if not self.validate(output):
            logger.warning(f"Output validation failed for loop {loop_id}")
        
        # Store
        self.output_history.append(output)
        
        return output
    
    def validate(self, output: GraceLoopOutput) -> bool:
        """Validate output schema"""
        # Check required fields
        for field in self.required_fields:
            if not hasattr(output, field):
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate confidence range
        if not 0.0 <= output.confidence <= 1.0:
            logger.error(f"Invalid confidence: {output.confidence}")
            return False
        
        # Validate clarity scores
        if not 0.0 <= output.clarity_score <= 1.0:
            logger.error(f"Invalid clarity_score: {output.clarity_score}")
            return False
        
        if not 0.0 <= output.ambiguity_score <= 1.0:
            logger.error(f"Invalid ambiguity_score: {output.ambiguity_score}")
            return False
        
        return True
    
    def format_for_display(self, output: GraceLoopOutput) -> str:
        """Format output for human-readable display"""
        lines = [
            f"=== Grace Loop Output ===",
            f"Loop ID: {output.loop_id}",
            f"Iteration: {output.iteration}",
            f"Status: {output.status.value}",
            f"Confidence: {output.confidence:.2%}",
            f"Clarity: {output.clarity_score:.2%}",
            f"Ambiguity: {output.ambiguity_score:.2%}",
            f"Constitution Compliant: {output.constitution_compliant}",
            f"Quorum Decision: {output.quorum_decision}",
            f"Consensus Strength: {output.consensus_strength:.2%}",
            f"Execution Time: {output.execution_time:.3f}s",
            f"Timestamp: {output.timestamp.isoformat()}",
            f"\nResult:",
            json.dumps(output.result, indent=2)
        ]
        
        if output.errors:
            lines.append(f"\nErrors:")
            for error in output.errors:
                lines.append(f"  - {error}")
        
        if output.warnings:
            lines.append(f"\nWarnings:")
            for warning in output.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)
    
    def get_output_statistics(self) -> Dict[str, Any]:
        """Get output statistics"""
        if not self.output_history:
            return {'total_outputs': 0}
        
        total = len(self.output_history)
        
        by_status = {}
        for output in self.output_history:
            status = output.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        avg_confidence = sum(o.confidence for o in self.output_history) / total
        avg_clarity = sum(o.clarity_score for o in self.output_history) / total
        avg_ambiguity = sum(o.ambiguity_score for o in self.output_history) / total
        
        compliant_count = sum(1 for o in self.output_history if o.constitution_compliant)
        
        return {
            'total_outputs': total,
            'by_status': by_status,
            'avg_confidence': avg_confidence,
            'avg_clarity': avg_clarity,
            'avg_ambiguity': avg_ambiguity,
            'constitution_compliance_rate': compliant_count / total,
            'avg_execution_time': sum(o.execution_time for o in self.output_history) / total
        }
