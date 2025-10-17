"""
Consciousness Awareness Index - Tracks system self-awareness and capabilities
"""

from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CapabilityEntry:
    """Entry for a system capability"""
    name: str
    category: str
    description: str
    confidence: float
    last_used: Optional[datetime] = None
    usage_count: int = 0
    effectiveness: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class AwarenessIndex:
    """
    Maintains index of system capabilities and self-awareness
    Integrates with transcendence layer for advanced reasoning awareness
    """
    
    def __init__(self):
        self.capabilities: Dict[str, CapabilityEntry] = {}
        self.awareness_level: float = 0.5
        self.self_knowledge: Dict[str, Any] = {}
        self._initialize_core_awareness()
        self._register_transcendence_awareness()
        logger.info("AwarenessIndex initialized")
    
    def _initialize_core_awareness(self):
        """Initialize core system awareness"""
        self.self_knowledge = {
            'system_name': 'Grace',
            'primary_purpose': 'Advanced AI reasoning and decision support',
            'architecture': 'Multi-layered consciousness and reasoning system',
            'initialized_at': datetime.now().isoformat()
        }
        
        # Register core capabilities
        core_capabilities = [
            CapabilityEntry(
                name="logical_reasoning",
                category="core",
                description="Deductive and inductive reasoning",
                confidence=0.9
            ),
            CapabilityEntry(
                name="pattern_recognition",
                category="core",
                description="Identify patterns in data and behavior",
                confidence=0.85
            ),
            CapabilityEntry(
                name="meta_cognition",
                category="consciousness",
                description="Self-awareness and reflection on own processes",
                confidence=0.75
            )
        ]
        
        for cap in core_capabilities:
            self.capabilities[cap.name] = cap
    
    def _register_transcendence_awareness(self):
        """Register transcendence layer capabilities in awareness index"""
        transcendence_capabilities = [
            CapabilityEntry(
                name="quantum_superposition_reasoning",
                category="transcendence",
                description="Explore multiple solution states simultaneously using quantum-inspired algorithms",
                confidence=0.7,
                metadata={
                    'component': 'quantum_algorithms',
                    'methods': [
                        'initialize_superposition',
                        'amplitude_amplification',
                        'quantum_annealing'
                    ],
                    'best_for': [
                        'optimization_problems',
                        'search_in_large_spaces',
                        'probabilistic_reasoning'
                    ]
                }
            ),
            CapabilityEntry(
                name="quantum_probabilistic_reasoning",
                category="transcendence",
                description="Advanced probabilistic reasoning using quantum probability principles",
                confidence=0.75,
                metadata={
                    'component': 'quantum_algorithms',
                    'methods': ['probabilistic_reasoning'],
                    'best_for': [
                        'hypothesis_evaluation',
                        'uncertainty_quantification',
                        'multi-hypothesis_analysis'
                    ]
                }
            ),
            CapabilityEntry(
                name="hypothesis_generation",
                category="transcendence",
                description="Generate scientific hypotheses from observations",
                confidence=0.8,
                metadata={
                    'component': 'scientific_discovery',
                    'methods': ['generate_hypothesis', 'design_experiment'],
                    'best_for': [
                        'scientific_research',
                        'knowledge_discovery',
                        'causal_inference'
                    ]
                }
            ),
            CapabilityEntry(
                name="research_gap_analysis",
                category="transcendence",
                description="Identify unexplored areas and research opportunities",
                confidence=0.7,
                metadata={
                    'component': 'scientific_discovery',
                    'methods': ['find_research_gaps'],
                    'best_for': [
                        'research_planning',
                        'knowledge_mapping',
                        'innovation_discovery'
                    ]
                }
            ),
            CapabilityEntry(
                name="discovery_synthesis",
                category="transcendence",
                description="Synthesize insights from validated hypotheses",
                confidence=0.75,
                metadata={
                    'component': 'scientific_discovery',
                    'methods': ['synthesize_discovery'],
                    'best_for': [
                        'cross_domain_integration',
                        'theory_building',
                        'knowledge_synthesis'
                    ]
                }
            ),
            CapabilityEntry(
                name="societal_impact_assessment",
                category="transcendence",
                description="Evaluate societal, ethical, and policy implications",
                confidence=0.8,
                metadata={
                    'component': 'societal_impact',
                    'methods': ['assess_impact', 'compare_alternatives'],
                    'best_for': [
                        'policy_evaluation',
                        'decision_making',
                        'risk_assessment'
                    ]
                }
            ),
            CapabilityEntry(
                name="ethical_reasoning",
                category="transcendence",
                description="Analyze ethical dilemmas using multiple philosophical frameworks",
                confidence=0.85,
                metadata={
                    'component': 'societal_impact',
                    'methods': ['analyze_ethical_dilemma'],
                    'frameworks': [
                        'utilitarianism',
                        'deontology',
                        'virtue_ethics',
                        'care_ethics',
                        'justice_as_fairness'
                    ],
                    'best_for': [
                        'ethical_decision_making',
                        'value_alignment',
                        'moral_reasoning'
                    ]
                }
            ),
            CapabilityEntry(
                name="policy_foresight",
                category="transcendence",
                description="Simulate policy implementation and predict outcomes",
                confidence=0.7,
                metadata={
                    'component': 'societal_impact',
                    'methods': ['simulate_policy'],
                    'best_for': [
                        'policy_planning',
                        'scenario_analysis',
                        'impact_prediction'
                    ]
                }
            )
        ]
        
        for cap in transcendence_capabilities:
            self.capabilities[cap.name] = cap
            logger.info(f"Registered transcendence capability: {cap.name}")
        
        # Update awareness level with transcendence integration
        self.awareness_level = min(self.awareness_level + 0.2, 1.0)
        logger.info(f"Awareness level updated to: {self.awareness_level:.2f}")
    
    def record_capability_use(
        self,
        capability_name: str,
        effectiveness: Optional[float] = None
    ):
        """Record usage of a capability"""
        if capability_name not in self.capabilities:
            logger.warning(f"Unknown capability: {capability_name}")
            return
        
        cap = self.capabilities[capability_name]
        cap.last_used = datetime.now()
        cap.usage_count += 1
        
        if effectiveness is not None:
            # Update effectiveness with moving average
            cap.effectiveness = (cap.effectiveness * 0.8) + (effectiveness * 0.2)
    
    def get_best_capability_for(self, task_type: str) -> Optional[CapabilityEntry]:
        """Get best capability for a specific task type"""
        candidates = [
            cap for cap in self.capabilities.values()
            if task_type in cap.metadata.get('best_for', [])
        ]
        
        if not candidates:
            return None
        
        # Sort by confidence and effectiveness
        return max(
            candidates,
            key=lambda c: (c.confidence * 0.6 + c.effectiveness * 0.4)
        )
    
    def get_capabilities_by_category(self, category: str) -> List[CapabilityEntry]:
        """Get all capabilities in a category"""
        return [
            cap for cap in self.capabilities.values()
            if cap.category == category
        ]
    
    def get_transcendence_capabilities(self) -> List[CapabilityEntry]:
        """Get all transcendence layer capabilities"""
        return self.get_capabilities_by_category("transcendence")
    
    def introspect(self) -> Dict[str, Any]:
        """Perform system introspection"""
        return {
            'awareness_level': self.awareness_level,
            'total_capabilities': len(self.capabilities),
            'transcendence_capabilities': len(self.get_transcendence_capabilities()),
            'most_used_capabilities': sorted(
                self.capabilities.values(),
                key=lambda c: c.usage_count,
                reverse=True
            )[:5],
            'highest_confidence': max(
                self.capabilities.values(),
                key=lambda c: c.confidence
            ).name,
            'self_knowledge': self.self_knowledge,
            'categories': list(set(c.category for c in self.capabilities.values()))
        }
    
    def recommend_capability(self, problem_description: str) -> Dict[str, Any]:
        """Recommend best capability for a problem"""
        # Simple keyword matching
        recommendations = []
        
        problem_lower = problem_description.lower()
        
        for cap in self.capabilities.values():
            relevance = 0
            
            # Check description
            if any(word in problem_lower for word in cap.description.lower().split()):
                relevance += 0.3
            
            # Check best_for metadata
            for use_case in cap.metadata.get('best_for', []):
                if use_case.replace('_', ' ') in problem_lower:
                    relevance += 0.5
            
            if relevance > 0:
                recommendations.append({
                    'capability': cap.name,
                    'relevance': relevance,
                    'confidence': cap.confidence,
                    'effectiveness': cap.effectiveness,
                    'score': relevance * cap.confidence * cap.effectiveness
                })
        
        recommendations.sort(key=lambda r: r['score'], reverse=True)
        
        return {
            'top_recommendation': recommendations[0] if recommendations else None,
            'all_recommendations': recommendations[:5],
            'problem': problem_description
        }
