"""
Unified Logic - Complete implementation with decision synthesis and conflict resolution
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class DecisionSource(Enum):
    """Sources of decisions in the system"""
    CLARITY = "clarity"
    TRANSCENDENCE = "transcendence"
    SWARM = "swarm"
    MEMORY = "memory"
    GOVERNANCE = "governance"
    CONSCIOUSNESS = "consciousness"


@dataclass
class Decision:
    """Represents a decision from a layer"""
    source: DecisionSource
    decision: Any
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictResolution:
    """Result of conflict resolution"""
    resolved_decision: Any
    resolution_method: str
    confidence: float
    conflicting_sources: List[DecisionSource]
    arbitration_reason: str


class CrossLayerDecisionSynthesizer:
    """
    Synthesizes decisions across all Grace layers
    Creates integration map and unified decisions
    """
    
    def __init__(self):
        self.decision_history: List[Dict[str, Any]] = []
        self.integration_map: Dict[str, List[DecisionSource]] = {}
        logger.info("CrossLayerDecisionSynthesizer initialized")
    
    def synthesize_decision(
        self,
        decisions: List[Decision],
        context: Dict[str, Any]
    ) -> Decision:
        """
        Synthesize multiple layer decisions into unified decision
        """
        if not decisions:
            raise ValueError("No decisions to synthesize")
        
        # Single decision - no synthesis needed
        if len(decisions) == 1:
            return decisions[0]
        
        # Check for conflicts
        if self._has_conflicts(decisions):
            logger.warning("Conflicts detected in decisions")
            # Will be handled by ConflictResolver
            return decisions[0]  # Temporary
        
        # Weighted synthesis
        synthesized = self._weighted_synthesis(decisions, context)
        
        # Record in integration map
        self._update_integration_map(decisions)
        
        # Store in history
        self.decision_history.append({
            'synthesized': synthesized,
            'sources': [d.source.value for d in decisions],
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Synthesized decision from {len(decisions)} sources")
        
        return synthesized
    
    def _has_conflicts(self, decisions: List[Decision]) -> bool:
        """Check if decisions conflict"""
        if len(decisions) < 2:
            return False
        
        # Check boolean decisions
        bool_decisions = [d for d in decisions if isinstance(d.decision, bool)]
        if len(bool_decisions) >= 2:
            values = [d.decision for d in bool_decisions]
            if not all(v == values[0] for v in values):
                return True
        
        # Check numeric decisions (>20% variance)
        numeric_decisions = [d for d in decisions if isinstance(d.decision, (int, float))]
        if len(numeric_decisions) >= 2:
            values = [d.decision for d in numeric_decisions]
            avg = sum(values) / len(values)
            variance = max(abs(v - avg) / avg for v in values if avg != 0)
            if variance > 0.2:
                return True
        
        return False
    
    def _weighted_synthesis(
        self,
        decisions: List[Decision],
        context: Dict[str, Any]
    ) -> Decision:
        """Perform weighted synthesis of decisions"""
        
        # Calculate weights based on confidence and source priority
        weights = []
        for decision in decisions:
            base_weight = decision.confidence
            
            # Priority weights by source
            source_weights = {
                DecisionSource.GOVERNANCE: 1.5,  # Highest priority
                DecisionSource.CONSCIOUSNESS: 1.3,
                DecisionSource.CLARITY: 1.2,
                DecisionSource.TRANSCENDENCE: 1.1,
                DecisionSource.SWARM: 1.0,
                DecisionSource.MEMORY: 0.9
            }
            
            weight = base_weight * source_weights.get(decision.source, 1.0)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Find highest weighted decision
        max_idx = normalized_weights.index(max(normalized_weights))
        primary_decision = decisions[max_idx]
        
        # Create synthesized decision
        synthesized = Decision(
            source=DecisionSource.CLARITY,  # Unified source
            decision=primary_decision.decision,
            confidence=sum(d.confidence * w for d, w in zip(decisions, normalized_weights)),
            rationale=f"Synthesized from {len(decisions)} sources: {primary_decision.rationale}",
            metadata={
                'sources': [d.source.value for d in decisions],
                'weights': normalized_weights,
                'primary_source': primary_decision.source.value
            }
        )
        
        return synthesized
    
    def _update_integration_map(self, decisions: List[Decision]):
        """Update integration map with decision sources"""
        timestamp = datetime.now().isoformat()
        
        if timestamp not in self.integration_map:
            self.integration_map[timestamp] = []
        
        self.integration_map[timestamp].extend([d.source for d in decisions])
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics on decision integration"""
        if not self.decision_history:
            return {'total_syntheses': 0}
        
        source_counts = {}
        for record in self.decision_history:
            for source in record['sources']:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_syntheses': len(self.decision_history),
            'source_usage': source_counts,
            'avg_sources_per_synthesis': sum(len(r['sources']) for r in self.decision_history) / len(self.decision_history)
        }


class ConflictResolver:
    """
    Arbitration system for conflicting decisions
    Resolves conflicts using multiple strategies
    """
    
    def __init__(self, governance_engine=None):
        self.governance = governance_engine
        self.resolution_history: List[ConflictResolution] = []
        logger.info("ConflictResolver initialized")
    
    def resolve_conflict(
        self,
        conflicting_decisions: List[Decision],
        context: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve conflicts between decisions using arbitration
        """
        if len(conflicting_decisions) < 2:
            raise ValueError("Need at least 2 decisions to resolve conflict")
        
        # Try governance-based resolution first
        if self.governance:
            resolution = self._governance_arbitration(conflicting_decisions, context)
            if resolution:
                self.resolution_history.append(resolution)
                logger.info(f"Resolved conflict via governance: {resolution.resolution_method}")
                return resolution
        
        # Try confidence-based resolution
        resolution = self._confidence_arbitration(conflicting_decisions)
        self.resolution_history.append(resolution)
        
        logger.info(f"Resolved conflict via {resolution.resolution_method}")
        return resolution
    
    def _governance_arbitration(
        self,
        decisions: List[Decision],
        context: Dict[str, Any]
    ) -> Optional[ConflictResolution]:
        """Resolve using governance rules"""
        if not self.governance:
            return None
        
        # Governance has highest priority
        governance_decisions = [d for d in decisions if d.source == DecisionSource.GOVERNANCE]
        
        if governance_decisions:
            chosen = governance_decisions[0]
            
            return ConflictResolution(
                resolved_decision=chosen.decision,
                resolution_method="governance_priority",
                confidence=chosen.confidence,
                conflicting_sources=[d.source for d in decisions],
                arbitration_reason="Governance decision takes precedence"
            )
        
        # Check constitutional compliance
        for decision in sorted(decisions, key=lambda d: d.confidence, reverse=True):
            # Validate against constitution
            validation = self.governance.validate_against_constitution(
                {'decision': decision.decision},
                context
            )
            
            if validation.passed:
                return ConflictResolution(
                    resolved_decision=decision.decision,
                    resolution_method="constitutional_compliance",
                    confidence=decision.confidence * validation.score,
                    conflicting_sources=[d.source for d in decisions],
                    arbitration_reason=f"Constitutionally compliant with score {validation.score:.2f}"
                )
        
        return None
    
    def _confidence_arbitration(self, decisions: List[Decision]) -> ConflictResolution:
        """Resolve using confidence scores"""
        # Sort by confidence
        sorted_decisions = sorted(decisions, key=lambda d: d.confidence, reverse=True)
        chosen = sorted_decisions[0]
        
        return ConflictResolution(
            resolved_decision=chosen.decision,
            resolution_method="highest_confidence",
            confidence=chosen.confidence,
            conflicting_sources=[d.source for d in decisions],
            arbitration_reason=f"Highest confidence: {chosen.confidence:.2f} from {chosen.source.value}"
        )
    
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get conflict resolution statistics"""
        if not self.resolution_history:
            return {'total_conflicts': 0}
        
        method_counts = {}
        for resolution in self.resolution_history:
            method = resolution.resolution_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'total_conflicts': len(self.resolution_history),
            'resolution_methods': method_counts,
            'avg_confidence': sum(r.confidence for r in self.resolution_history) / len(self.resolution_history)
        }


class ResponseOrchestrator:
    """
    Orchestrates responses with event emission to TriggerMesh
    """
    
    def __init__(self, event_bus=None, trigger_mesh=None):
        self.event_bus = event_bus
        self.trigger_mesh = trigger_mesh
        self.response_history: List[Dict[str, Any]] = []
        logger.info("ResponseOrchestrator initialized")
    
    async def orchestrate_response(
        self,
        decision: Decision,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrate response with event emission
        """
        # Emit to EventBus
        if self.event_bus:
            self.event_bus.publish(
                event_type="decision.made",
                payload={
                    'source': decision.source.value,
                    'decision': decision.decision,
                    'confidence': decision.confidence
                },
                source="unified_logic"
            )
        
        # Emit to TriggerMesh
        if self.trigger_mesh:
            await self.trigger_mesh.trigger_event(
                event_type="decision_response",
                payload={'decision': decision.decision},
                priority="normal"
            )
        
        # Create response
        response = {
            'decision': decision.decision,
            'source': decision.source.value,
            'confidence': decision.confidence,
            'rationale': decision.rationale,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        self.response_history.append(response)
        
        logger.info(f"Orchestrated response for decision from {decision.source.value}")
        
        return response
    
    def get_response_metrics(self) -> Dict[str, Any]:
        """Get response orchestration metrics"""
        return {
            'total_responses': len(self.response_history),
            'event_bus_connected': self.event_bus is not None,
            'trigger_mesh_connected': self.trigger_mesh is not None
        }


class UnifiedLogic:
    """
    Complete Unified Logic system with synthesis, conflict resolution, and orchestration
    """
    
    def __init__(self, governance_engine=None, event_bus=None, trigger_mesh=None):
        self.synthesizer = CrossLayerDecisionSynthesizer()
        self.resolver = ConflictResolver(governance_engine)
        self.orchestrator = ResponseOrchestrator(event_bus, trigger_mesh)
        self.governance = governance_engine
        
        # Telemetry
        self.telemetry_enabled = True
        self.metrics = {
            'total_decisions': 0,
            'conflicts_resolved': 0,
            'responses_orchestrated': 0
        }
        
        logger.info("UnifiedLogic fully initialized")
    
    async def process_decision(
        self,
        decisions: List[Decision],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete decision processing pipeline
        """
        self.metrics['total_decisions'] += 1
        
        # Check for conflicts
        if self.synthesizer._has_conflicts(decisions):
            # Resolve conflict
            resolution = self.resolver.resolve_conflict(decisions, context)
            self.metrics['conflicts_resolved'] += 1
            
            final_decision = Decision(
                source=DecisionSource.CLARITY,
                decision=resolution.resolved_decision,
                confidence=resolution.confidence,
                rationale=resolution.arbitration_reason
            )
        else:
            # Synthesize
            final_decision = self.synthesizer.synthesize_decision(decisions, context)
        
        # Validate with governance if available
        if self.governance:
            validation = self.governance.validate_against_constitution(
                {'decision': final_decision.decision},
                context
            )
            
            if not validation.passed:
                logger.warning("Decision failed governance validation")
                final_decision.metadata['governance_violations'] = validation.violations
        
        # Orchestrate response
        response = await self.orchestrator.orchestrate_response(final_decision, context)
        self.metrics['responses_orchestrated'] += 1
        
        # Emit telemetry
        if self.telemetry_enabled:
            self._emit_telemetry(final_decision, response)
        
        return response
    
    def _emit_telemetry(self, decision: Decision, response: Dict[str, Any]):
        """Emit telemetry data"""
        # Hook for telemetry system
        logger.debug(f"Telemetry: Decision from {decision.source.value}, confidence {decision.confidence:.2f}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            **self.metrics,
            'synthesizer_stats': self.synthesizer.get_integration_statistics(),
            'resolver_stats': self.resolver.get_conflict_statistics(),
            'orchestrator_metrics': self.orchestrator.get_response_metrics()
        }
