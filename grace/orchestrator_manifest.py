"""
Orchestrator Manifest - Registry of all Grace system components
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of system components"""
    CORE = "core"
    CONSCIOUSNESS = "consciousness"
    TRANSCENDENCE = "transcendence"
    REASONING = "reasoning"
    INTEGRATION = "integration"
    UTILITY = "utility"


@dataclass
class ComponentManifest:
    """Manifest entry for a system component"""
    name: str
    component_type: ComponentType
    module_path: str
    class_name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestratorManifest:
    """
    Central registry for all Grace system components
    Manages component discovery, initialization, and orchestration
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentManifest] = {}
        self._initialize_core_components()
        self._register_transcendence_layer()
        logger.info("OrchestratorManifest initialized")
    
    def _initialize_core_components(self):
        """Register core system components"""
        core_components = [
            ComponentManifest(
                name="unified_logic",
                component_type=ComponentType.CORE,
                module_path="grace.core.unified_logic",
                class_name="UnifiedLogic",
                description="Core reasoning and logic engine",
                dependencies=[]
            ),
            ComponentManifest(
                name="consciousness_layer",
                component_type=ComponentType.CONSCIOUSNESS,
                module_path="grace.consciousness.layer",
                class_name="ConsciousnessLayer",
                description="Self-awareness and meta-cognition",
                dependencies=["unified_logic"]
            )
        ]
        
        for comp in core_components:
            self.components[comp.name] = comp
    
    def _register_transcendence_layer(self):
        """Register transcendence layer components"""
        transcendence_components = [
            ComponentManifest(
                name="quantum_algorithms",
                component_type=ComponentType.TRANSCENDENCE,
                module_path="grace.transcendent.quantum_algorithms",
                class_name="QuantumAlgorithmLibrary",
                description="Quantum-inspired computation and probabilistic reasoning",
                dependencies=[],
                optional=True,
                metadata={
                    "capabilities": [
                        "superposition_search",
                        "quantum_annealing",
                        "amplitude_amplification",
                        "probabilistic_reasoning"
                    ],
                    "use_cases": [
                        "optimization_problems",
                        "hypothesis_exploration",
                        "multi-state_reasoning"
                    ]
                }
            ),
            ComponentManifest(
                name="scientific_discovery",
                component_type=ComponentType.TRANSCENDENCE,
                module_path="grace.transcendent.scientific_discovery",
                class_name="ScientificDiscoveryAccelerator",
                description="Hypothesis-driven scientific reasoning and discovery",
                dependencies=[],
                optional=True,
                metadata={
                    "capabilities": [
                        "hypothesis_generation",
                        "experiment_design",
                        "evidence_evaluation",
                        "insight_synthesis",
                        "research_gap_identification"
                    ],
                    "use_cases": [
                        "scientific_research",
                        "knowledge_discovery",
                        "theory_building"
                    ]
                }
            ),
            ComponentManifest(
                name="societal_impact",
                component_type=ComponentType.TRANSCENDENCE,
                module_path="grace.transcendent.societal_impact",
                class_name="SocietalImpactEvaluator",
                description="Ethics and policy foresight simulation",
                dependencies=[],
                optional=True,
                metadata={
                    "capabilities": [
                        "impact_assessment",
                        "ethical_analysis",
                        "policy_simulation",
                        "stakeholder_analysis",
                        "risk_evaluation"
                    ],
                    "use_cases": [
                        "policy_evaluation",
                        "ethical_decision_making",
                        "social_impact_analysis"
                    ]
                }
            ),
            ComponentManifest(
                name="transcendence_orchestrator",
                component_type=ComponentType.TRANSCENDENCE,
                module_path="grace.transcendent.orchestrator",
                class_name="TranscendenceOrchestrator",
                description="Coordinates all transcendence layer capabilities",
                dependencies=[
                    "quantum_algorithms",
                    "scientific_discovery",
                    "societal_impact"
                ],
                optional=True
            )
        ]
        
        for comp in transcendence_components:
            self.components[comp.name] = comp
            logger.info(f"Registered transcendence component: {comp.name}")
    
    def register_component(self, manifest: ComponentManifest):
        """Register a new component"""
        if manifest.name in self.components:
            logger.warning(f"Component {manifest.name} already registered, updating...")
        
        self.components[manifest.name] = manifest
        logger.info(f"Registered component: {manifest.name}")
    
    def get_component(self, name: str) -> Optional[ComponentManifest]:
        """Get component manifest by name"""
        return self.components.get(name)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[ComponentManifest]:
        """Get all components of a specific type"""
        return [
            comp for comp in self.components.values()
            if comp.component_type == component_type
        ]
    
    def get_enabled_components(self) -> List[ComponentManifest]:
        """Get all enabled components"""
        return [comp for comp in self.components.values() if comp.enabled]
    
    def resolve_dependencies(self, component_name: str) -> List[str]:
        """Resolve component dependencies in order"""
        if component_name not in self.components:
            return []
        
        component = self.components[component_name]
        resolved = []
        
        for dep in component.dependencies:
            if dep not in self.components:
                logger.warning(f"Dependency {dep} not found for {component_name}")
                continue
            
            # Recursive dependency resolution
            resolved.extend(self.resolve_dependencies(dep))
            
            if dep not in resolved:
                resolved.append(dep)
        
        return resolved
    
    def get_initialization_order(self) -> List[str]:
        """Get components in dependency-resolved initialization order"""
        initialized = []
        
        for comp_name in self.components:
            if comp_name in initialized:
                continue
            
            # Add dependencies first
            deps = self.resolve_dependencies(comp_name)
            for dep in deps:
                if dep not in initialized:
                    initialized.append(dep)
            
            # Add component itself
            if comp_name not in initialized:
                initialized.append(comp_name)
        
        return initialized
    
    def get_manifest_summary(self) -> Dict[str, Any]:
        """Get summary of all registered components"""
        return {
            'total_components': len(self.components),
            'enabled_components': len(self.get_enabled_components()),
            'by_type': {
                component_type.value: len(self.get_components_by_type(component_type))
                for component_type in ComponentType
            },
            'transcendence_components': [
                comp.name for comp in self.get_components_by_type(ComponentType.TRANSCENDENCE)
            ],
            'optional_components': [
                comp.name for comp in self.components.values() if comp.optional
            ]
        }


# Global manifest instance
_manifest = OrchestratorManifest()


def get_manifest() -> OrchestratorManifest:
    """Get global manifest instance"""
    return _manifest
