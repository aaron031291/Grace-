"""
Scientific Discovery Accelerator - Hypothesis-driven scientific reasoning
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Status of scientific hypothesis"""
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    REFUTED = "refuted"
    REFINED = "refined"


class ExperimentType(Enum):
    """Types of scientific experiments"""
    OBSERVATION = "observation"
    CONTROLLED = "controlled"
    SIMULATION = "simulation"
    META_ANALYSIS = "meta_analysis"


@dataclass
class Hypothesis:
    """Scientific hypothesis structure"""
    id: str
    statement: str
    domain: str
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence_for: List[Dict] = field(default_factory=list)
    evidence_against: List[Dict] = field(default_factory=list)
    related_hypotheses: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Scientific experiment design"""
    id: str
    hypothesis_id: str
    experiment_type: ExperimentType
    design: Dict[str, Any]
    predicted_outcome: Any
    actual_outcome: Optional[Any] = None
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryInsight:
    """Scientific discovery or insight"""
    id: str
    insight_type: str
    description: str
    supporting_hypotheses: List[str]
    novelty_score: float
    impact_score: float
    domains: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScientificDiscoveryAccelerator:
    """
    Accelerates scientific discovery through hypothesis generation,
    experiment design, and insight synthesis
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.discoveries: Dict[str, DiscoveryInsight] = {}
        self.knowledge_graph: Dict[str, Set[str]] = {}
        logger.info("ScientificDiscoveryAccelerator initialized")
    
    def generate_hypothesis(
        self,
        observations: List[Dict[str, Any]],
        domain: str,
        context: Optional[Dict] = None
    ) -> Hypothesis:
        """Generate hypothesis from observations"""
        # Analyze patterns in observations
        patterns = self._identify_patterns(observations)
        
        # Generate hypothesis statement
        statement = self._synthesize_hypothesis_statement(patterns, domain, context)
        
        # Create hypothesis
        hypothesis = Hypothesis(
            id=f"hyp_{len(self.hypotheses)}_{domain}",
            statement=statement,
            domain=domain,
            confidence=self._estimate_initial_confidence(patterns),
            metadata={
                'observations_count': len(observations),
                'patterns': patterns,
                'context': context or {}
            }
        )
        
        self.hypotheses[hypothesis.id] = hypothesis
        logger.info(f"Generated hypothesis: {hypothesis.id}")
        
        return hypothesis
    
    def design_experiment(
        self,
        hypothesis: Hypothesis,
        experiment_type: ExperimentType,
        constraints: Optional[Dict] = None
    ) -> Experiment:
        """Design experiment to test hypothesis"""
        design = self._create_experiment_design(
            hypothesis,
            experiment_type,
            constraints or {}
        )
        
        predicted_outcome = self._predict_outcome(hypothesis, design)
        
        experiment = Experiment(
            id=f"exp_{len(self.experiments)}_{hypothesis.id}",
            hypothesis_id=hypothesis.id,
            experiment_type=experiment_type,
            design=design,
            predicted_outcome=predicted_outcome,
            metadata={
                'constraints': constraints or {},
                'designed_at': datetime.now().isoformat()
            }
        )
        
        self.experiments[experiment.id] = experiment
        hypothesis.status = HypothesisStatus.TESTING
        
        logger.info(f"Designed experiment: {experiment.id}")
        return experiment
    
    def evaluate_evidence(
        self,
        hypothesis_id: str,
        evidence: Dict[str, Any],
        supports: bool
    ) -> Hypothesis:
        """Evaluate new evidence for hypothesis"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        # Add evidence
        evidence_entry = {
            **evidence,
            'timestamp': datetime.now().isoformat(),
            'strength': self._assess_evidence_strength(evidence)
        }
        
        if supports:
            hypothesis.evidence_for.append(evidence_entry)
        else:
            hypothesis.evidence_against.append(evidence_entry)
        
        # Update confidence
        hypothesis.confidence = self._calculate_bayesian_confidence(hypothesis)
        
        # Update status
        if hypothesis.confidence > 0.9:
            hypothesis.status = HypothesisStatus.VALIDATED
        elif hypothesis.confidence < 0.1:
            hypothesis.status = HypothesisStatus.REFUTED
        
        hypothesis.updated_at = datetime.now()
        
        logger.info(f"Evaluated evidence for {hypothesis_id}, new confidence: {hypothesis.confidence:.2f}")
        return hypothesis
    
    def synthesize_discovery(
        self,
        hypothesis_ids: List[str],
        integration_context: Optional[Dict] = None
    ) -> DiscoveryInsight:
        """Synthesize discovery from validated hypotheses"""
        hypotheses = [self.hypotheses[hid] for hid in hypothesis_ids if hid in self.hypotheses]
        
        if not hypotheses:
            raise ValueError("No valid hypotheses provided")
        
        # Synthesize insight
        insight_description = self._synthesize_insight(hypotheses, integration_context)
        
        # Calculate scores
        novelty = self._calculate_novelty_score(hypotheses)
        impact = self._calculate_impact_score(hypotheses)
        
        discovery = DiscoveryInsight(
            id=f"disc_{len(self.discoveries)}",
            insight_type=self._classify_insight_type(hypotheses),
            description=insight_description,
            supporting_hypotheses=hypothesis_ids,
            novelty_score=novelty,
            impact_score=impact,
            domains=list(set(h.domain for h in hypotheses)),
            metadata={
                'integration_context': integration_context or {},
                'synthesized_at': datetime.now().isoformat()
            }
        )
        
        self.discoveries[discovery.id] = discovery
        
        # Update knowledge graph
        self._update_knowledge_graph(discovery, hypotheses)
        
        logger.info(f"Synthesized discovery: {discovery.id} (novelty: {novelty:.2f}, impact: {impact:.2f})")
        return discovery
    
    def find_research_gaps(self, domain: str) -> List[Dict[str, Any]]:
        """Identify research gaps and opportunities"""
        domain_hypotheses = [
            h for h in self.hypotheses.values()
            if h.domain == domain
        ]
        
        gaps = []
        
        # Identify untested areas
        tested_concepts = set()
        for h in domain_hypotheses:
            tested_concepts.update(self._extract_concepts(h.statement))
        
        # Find concept combinations not yet explored
        all_concepts = list(tested_concepts)
        for i, c1 in enumerate(all_concepts):
            for c2 in all_concepts[i+1:]:
                combo = f"{c1}-{c2} relationship"
                if not any(combo in h.statement for h in domain_hypotheses):
                    gaps.append({
                        'type': 'untested_relationship',
                        'concepts': [c1, c2],
                        'priority': self._calculate_gap_priority(c1, c2, domain),
                        'suggested_hypothesis': f"Investigate relationship between {c1} and {c2}"
                    })
        
        # Find contradictions requiring resolution
        for h1 in domain_hypotheses:
            for h2 in domain_hypotheses:
                if h1.id != h2.id and self._are_contradictory(h1, h2):
                    gaps.append({
                        'type': 'contradiction',
                        'hypotheses': [h1.id, h2.id],
                        'priority': 0.8,
                        'suggested_hypothesis': f"Resolve contradiction between {h1.id} and {h2.id}"
                    })
        
        return sorted(gaps, key=lambda x: x['priority'], reverse=True)
    
    def _identify_patterns(self, observations: List[Dict]) -> List[Dict]:
        """Identify patterns in observations"""
        patterns = []
        
        if not observations:
            return patterns
        
        # Simple pattern detection
        for key in observations[0].keys():
            values = [obs.get(key) for obs in observations if key in obs]
            
            # Check for correlation, trends, clusters
            if len(set(values)) < len(values) * 0.3:  # Clustering
                patterns.append({
                    'type': 'clustering',
                    'variable': key,
                    'dominant_values': list(set(values))
                })
        
        return patterns
    
    def _synthesize_hypothesis_statement(
        self,
        patterns: List[Dict],
        domain: str,
        context: Optional[Dict]
    ) -> str:
        """Synthesize hypothesis statement from patterns"""
        if not patterns:
            return f"In {domain}, there exists an unexplored relationship"
        
        pattern = patterns[0]
        return f"In {domain}, {pattern['variable']} exhibits {pattern['type']} behavior"
    
    def _estimate_initial_confidence(self, patterns: List[Dict]) -> float:
        """Estimate initial confidence based on pattern strength"""
        if not patterns:
            return 0.3
        
        return min(0.5 + len(patterns) * 0.1, 0.7)
    
    def _create_experiment_design(
        self,
        hypothesis: Hypothesis,
        experiment_type: ExperimentType,
        constraints: Dict
    ) -> Dict[str, Any]:
        """Create experiment design"""
        return {
            'type': experiment_type.value,
            'hypothesis': hypothesis.statement,
            'variables': self._extract_concepts(hypothesis.statement),
            'controls': constraints.get('controls', []),
            'sample_size': constraints.get('sample_size', 100),
            'duration': constraints.get('duration', '1 week'),
            'methodology': self._select_methodology(experiment_type, hypothesis)
        }
    
    def _predict_outcome(self, hypothesis: Hypothesis, design: Dict) -> Any:
        """Predict experiment outcome"""
        return {
            'expected': 'supports_hypothesis' if hypothesis.confidence > 0.5 else 'refutes_hypothesis',
            'confidence': hypothesis.confidence
        }
    
    def _assess_evidence_strength(self, evidence: Dict) -> float:
        """Assess strength of evidence"""
        strength = 0.5
        
        if evidence.get('statistical_significance'):
            strength += 0.2
        if evidence.get('reproducible'):
            strength += 0.2
        if evidence.get('sample_size', 0) > 100:
            strength += 0.1
        
        return min(strength, 1.0)
    
    def _calculate_bayesian_confidence(self, hypothesis: Hypothesis) -> float:
        """Calculate Bayesian confidence update"""
        prior = hypothesis.confidence
        
        # Weight evidence
        support_weight = sum(e['strength'] for e in hypothesis.evidence_for)
        refute_weight = sum(e['strength'] for e in hypothesis.evidence_against)
        
        if support_weight + refute_weight == 0:
            return prior
        
        # Bayesian update (simplified)
        posterior = (prior * support_weight + (1 - prior) * refute_weight) / (support_weight + refute_weight)
        
        return max(0.0, min(1.0, posterior))
    
    def _synthesize_insight(self, hypotheses: List[Hypothesis], context: Optional[Dict]) -> str:
        """Synthesize insight from hypotheses"""
        domains = set(h.domain for h in hypotheses)
        return f"Cross-domain insight connecting {', '.join(domains)}"
    
    def _classify_insight_type(self, hypotheses: List[Hypothesis]) -> str:
        """Classify type of insight"""
        if len(set(h.domain for h in hypotheses)) > 1:
            return "cross_domain_integration"
        return "domain_advancement"
    
    def _calculate_novelty_score(self, hypotheses: List[Hypothesis]) -> float:
        """Calculate novelty score"""
        avg_confidence = sum(h.confidence for h in hypotheses) / len(hypotheses)
        cross_domain = len(set(h.domain for h in hypotheses)) > 1
        
        return min(avg_confidence * (1.5 if cross_domain else 1.0), 1.0)
    
    def _calculate_impact_score(self, hypotheses: List[Hypothesis]) -> float:
        """Calculate potential impact score"""
        evidence_count = sum(len(h.evidence_for) + len(h.evidence_against) for h in hypotheses)
        return min(evidence_count / 20.0, 1.0)
    
    def _update_knowledge_graph(self, discovery: DiscoveryInsight, hypotheses: List[Hypothesis]):
        """Update knowledge graph with discovery"""
        for h in hypotheses:
            if discovery.id not in self.knowledge_graph:
                self.knowledge_graph[discovery.id] = set()
            self.knowledge_graph[discovery.id].add(h.id)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        return [w for w in words if w not in stopwords and len(w) > 3][:5]
    
    def _select_methodology(self, experiment_type: ExperimentType, hypothesis: Hypothesis) -> str:
        """Select appropriate methodology"""
        methodologies = {
            ExperimentType.OBSERVATION: "longitudinal_observation",
            ExperimentType.CONTROLLED: "randomized_controlled_trial",
            ExperimentType.SIMULATION: "computational_simulation",
            ExperimentType.META_ANALYSIS: "systematic_review"
        }
        return methodologies.get(experiment_type, "standard_protocol")
    
    def _calculate_gap_priority(self, concept1: str, concept2: str, domain: str) -> float:
        """Calculate priority for research gap"""
        # Higher priority for concepts with more existing research
        related_count = sum(
            1 for h in self.hypotheses.values()
            if h.domain == domain and (concept1 in h.statement or concept2 in h.statement)
        )
        return min(related_count / 10.0, 1.0)
    
    def _are_contradictory(self, h1: Hypothesis, h2: Hypothesis) -> bool:
        """Check if hypotheses are contradictory"""
        # Simple contradiction check
        if h1.domain != h2.domain:
            return False
        
        # Check for opposing evidence
        return (h1.confidence > 0.7 and h2.confidence > 0.7 and 
                h1.status == HypothesisStatus.VALIDATED and 
                h2.status == HypothesisStatus.VALIDATED and
                len(set(self._extract_concepts(h1.statement)) & 
                    set(self._extract_concepts(h2.statement))) > 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        return {
            'total_hypotheses': len(self.hypotheses),
            'validated_hypotheses': sum(1 for h in self.hypotheses.values() 
                                       if h.status == HypothesisStatus.VALIDATED),
            'active_experiments': sum(1 for e in self.experiments.values() if not e.completed),
            'discoveries': len(self.discoveries),
            'domains': list(set(h.domain for h in self.hypotheses.values())),
            'avg_confidence': sum(h.confidence for h in self.hypotheses.values()) / len(self.hypotheses) 
                            if self.hypotheses else 0
        }
