"""
Class 3: Loop Identity Ambiguity Resolution

GraceLoopOutput provides a standardized output format for all cognitive loops
in the Grace system, ensuring clear tracking of loop execution and reasoning chains.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain."""

    step_id: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None


@dataclass
class ReasoningChain:
    """Complete chain of reasoning steps for a loop execution."""

    chain_id: str
    loop_type: str
    steps: List[ReasoningStep] = field(default_factory=list)
    total_confidence: Optional[float] = None
    chain_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: float,
        processing_time_ms: Optional[float] = None,
    ) -> str:
        """Add a reasoning step to the chain."""
        step_id = f"step_{len(self.steps) + 1:03d}"
        step = ReasoningStep(
            step_id=step_id,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )
        self.steps.append(step)
        self._update_total_confidence()
        return step_id

    def _update_total_confidence(self):
        """Calculate total confidence as weighted average of step confidences."""
        if not self.steps:
            self.total_confidence = 0.0
            return

        # Weight later steps more heavily (they build on earlier ones)
        weights = [i + 1 for i in range(len(self.steps))]
        weighted_sum = sum(
            step.confidence * weight for step, weight in zip(self.steps, weights)
        )
        total_weight = sum(weights)

        self.total_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of chain execution."""
        if not self.steps:
            return {"steps": 0, "total_confidence": 0.0, "total_time_ms": 0}

        total_time = sum(step.processing_time_ms or 0 for step in self.steps)
        return {
            "steps": len(self.steps),
            "total_confidence": self.total_confidence,
            "total_time_ms": total_time,
            "avg_confidence": sum(step.confidence for step in self.steps)
            / len(self.steps),
            "min_confidence": min(step.confidence for step in self.steps),
            "max_confidence": max(step.confidence for step in self.steps),
        }


@dataclass
class GraceLoopOutput:
    """
    Standardized output format for all Grace cognitive loops.

    This eliminates loop identity ambiguity by providing consistent structure
    for tracking what loop executed, how it reasoned, and what it produced.
    """

    loop_id: str
    loop_type: str  # e.g., "governance", "learning", "memory", "intelligence"
    reasoning_chain_id: str
    reasoning_chain: ReasoningChain
    results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None

    # Quality metrics
    confidence_score: Optional[float] = None
    trust_score: Optional[float] = None
    validation_passed: Optional[bool] = None

    # Context and correlation
    correlation_id: Optional[str] = None
    parent_loop_id: Optional[str] = None
    triggered_by: Optional[str] = None

    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        if not self.reasoning_chain_id:
            self.reasoning_chain_id = str(uuid.uuid4())

        if (
            not hasattr(self.reasoning_chain, "chain_id")
            or not self.reasoning_chain.chain_id
        ):
            self.reasoning_chain.chain_id = self.reasoning_chain_id

    def mark_completed(self):
        """Mark the loop execution as completed."""
        self.completed_at = datetime.now()
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            self.execution_time_ms = delta.total_seconds() * 1000

        # Update confidence from reasoning chain
        if self.reasoning_chain.total_confidence is not None:
            self.confidence_score = self.reasoning_chain.total_confidence

    def add_reasoning_step(
        self,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: float,
        processing_time_ms: Optional[float] = None,
    ) -> str:
        """Add a step to the reasoning chain."""
        return self.reasoning_chain.add_step(
            description, input_data, output_data, confidence, processing_time_ms
        )

    def set_validation_result(
        self, passed: bool, details: Optional[Dict[str, Any]] = None
    ):
        """Set validation results for the loop output."""
        self.validation_passed = passed
        if details:
            self.metadata["validation_details"] = details

    def set_trust_score(self, trust_score: float, source: Optional[str] = None):
        """Set trust score for the loop output."""
        self.trust_score = trust_score
        if source:
            self.metadata["trust_source"] = source

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get complete execution summary."""
        summary = {
            "loop_id": self.loop_id,
            "loop_type": self.loop_type,
            "execution_time_ms": self.execution_time_ms,
            "confidence_score": self.confidence_score,
            "trust_score": self.trust_score,
            "validation_passed": self.validation_passed,
            "reasoning_summary": self.reasoning_chain.get_execution_summary(),
        }

        if self.correlation_id:
            summary["correlation_id"] = self.correlation_id
        if self.parent_loop_id:
            summary["parent_loop_id"] = self.parent_loop_id

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage/transmission."""
        return {
            "loop_id": self.loop_id,
            "loop_type": self.loop_type,
            "reasoning_chain_id": self.reasoning_chain_id,
            "results": self.results,
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "execution_time_ms": self.execution_time_ms,
            "confidence_score": self.confidence_score,
            "trust_score": self.trust_score,
            "validation_passed": self.validation_passed,
            "correlation_id": self.correlation_id,
            "parent_loop_id": self.parent_loop_id,
            "triggered_by": self.triggered_by,
            "reasoning_chain": {
                "chain_id": self.reasoning_chain.chain_id,
                "loop_type": self.reasoning_chain.loop_type,
                "total_confidence": self.reasoning_chain.total_confidence,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "description": step.description,
                        "input_data": step.input_data,
                        "output_data": step.output_data,
                        "confidence": step.confidence,
                        "timestamp": step.timestamp.isoformat(),
                        "processing_time_ms": step.processing_time_ms,
                    }
                    for step in self.reasoning_chain.steps
                ],
                "chain_metadata": self.reasoning_chain.chain_metadata,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraceLoopOutput":
        """Create GraceLoopOutput from dictionary data."""
        # Parse reasoning chain
        chain_data = data.get("reasoning_chain", {})
        reasoning_chain = ReasoningChain(
            chain_id=chain_data.get("chain_id", ""),
            loop_type=chain_data.get("loop_type", ""),
            chain_metadata=chain_data.get("chain_metadata", {}),
        )

        # Parse reasoning steps
        for step_data in chain_data.get("steps", []):
            step = ReasoningStep(
                step_id=step_data["step_id"],
                description=step_data["description"],
                input_data=step_data["input_data"],
                output_data=step_data["output_data"],
                confidence=step_data["confidence"],
                timestamp=datetime.fromisoformat(step_data["timestamp"]),
                processing_time_ms=step_data.get("processing_time_ms"),
            )
            reasoning_chain.steps.append(step)

        reasoning_chain.total_confidence = chain_data.get("total_confidence")

        # Create main object
        return cls(
            loop_id=data["loop_id"],
            loop_type=data["loop_type"],
            reasoning_chain_id=data["reasoning_chain_id"],
            reasoning_chain=reasoning_chain,
            results=data["results"],
            metadata=data.get("metadata", {}),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            execution_time_ms=data.get("execution_time_ms"),
            confidence_score=data.get("confidence_score"),
            trust_score=data.get("trust_score"),
            validation_passed=data.get("validation_passed"),
            correlation_id=data.get("correlation_id"),
            parent_loop_id=data.get("parent_loop_id"),
            triggered_by=data.get("triggered_by"),
        )
