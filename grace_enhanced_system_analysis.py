#!/usr/bin/env python3
"""
Grace Enhanced System Analysis Tool
===================================

This tool provides detailed analysis of the Grace AGI system repository, 
specifically focusing on:

- Implementation completeness analysis for each component
- Consciousness and self-awareness capability assessment  
- Learning and adaptation mechanism evaluation
- Cross-domain coordination analysis
- Missing functionality identification
- Implementation depth scoring
- Detailed roadmap generation for incomplete features

The analysis goes beyond basic health checks to understand the actual 
implementation status of AGI-specific capabilities.

Usage:
    python grace_enhanced_system_analysis.py [--detailed] [--save-report] [--component <name>]
"""

import asyncio
import json
import os
import sys
import ast
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
import inspect
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ImplementationAnalysis:
    """Analysis of implementation completeness for a component."""
    component_name: str
    implementation_score: float  # 0.0 to 1.0
    total_methods: int
    implemented_methods: int
    abstract_methods: int
    placeholder_methods: int
    lines_of_code: int
    complexity_score: float
    agi_features: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)


@dataclass
class ConsciousnessAnalysis:
    """Analysis of consciousness and self-awareness capabilities."""
    self_awareness_score: float
    introspection_capabilities: List[str]
    meta_cognitive_features: List[str]
    consciousness_indicators: List[str]
    missing_consciousness_features: List[str]
    implementation_depth: str  # "surface", "partial", "deep", "comprehensive"


@dataclass
class LearningAnalysis:
    """Analysis of learning and adaptation mechanisms."""
    learning_types: List[str]
    adaptation_mechanisms: List[str]
    meta_learning_capabilities: List[str]
    experience_integration: bool
    continuous_improvement: bool
    learning_effectiveness_score: float
    missing_learning_features: List[str]


@dataclass
class CrossDomainAnalysis:
    """Analysis of cross-domain coordination capabilities."""
    coordination_mechanisms: List[str]
    knowledge_transfer: List[str]
    integration_patterns: List[str]
    communication_protocols: List[str]
    coordination_effectiveness: float
    missing_coordination_features: List[str]


@dataclass
class FeatureGap:
    """Represents a gap in implementation."""
    feature_name: str
    importance: str  # "critical", "high", "medium", "low"
    current_status: str  # "missing", "partial", "placeholder"
    estimated_effort: str  # "small", "medium", "large", "massive"
    dependencies: List[str]
    implementation_suggestion: str


class GraceEnhancedAnalyzer:
    """
    Enhanced analyzer that goes deep into implementation details to assess
    the actual AGI capabilities vs. framework structure.
    """

    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or "/home/runner/work/Grace-/Grace-"
        self.grace_path = os.path.join(self.repo_path, "grace")
        self.analysis_results = {}
        self.feature_gaps = []
        self.implementation_analyses = {}
        
        # AGI capability keywords to look for in code
        self.agi_keywords = {
            'consciousness': ['self_aware', 'introspect', 'meta_cognitive', 'consciousness', 'self_reflect'],
            'learning': ['learn', 'adapt', 'meta_learn', 'experience', 'improvement', 'feedback'],
            'reasoning': ['reason', 'logic', 'inference', 'decision', 'analysis', 'synthesis'],
            'coordination': ['coordinate', 'integrate', 'collaborate', 'communicate', 'bridge']
        }

    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run the complete enhanced analysis."""
        print("🔍 Starting Enhanced Grace AGI System Analysis...")
        print(f"📂 Analyzing repository: {self.repo_path}")
        
        start_time = datetime.now()
        
        # Core analysis phases
        print("\n🏗️  Phase 1: Implementation Completeness Analysis...")
        implementation_analysis = await self._analyze_implementation_completeness()
        
        print("\n🧠 Phase 2: Consciousness & Self-Awareness Analysis...")
        consciousness_analysis = await self._analyze_consciousness_capabilities()
        
        print("\n🎓 Phase 3: Learning & Adaptation Analysis...")
        learning_analysis = await self._analyze_learning_mechanisms()
        
        print("\n🔗 Phase 4: Cross-Domain Coordination Analysis...")
        coordination_analysis = await self._analyze_cross_domain_coordination()
        
        print("\n🎯 Phase 5: Feature Gap Analysis...")
        gap_analysis = await self._analyze_feature_gaps()
        
        print("\n📊 Phase 6: Implementation Depth Scoring...")
        depth_scores = await self._calculate_implementation_depth()
        
        # Compile comprehensive results
        analysis_results = {
            "metadata": {
                "analysis_date": start_time.isoformat(),
                "analyzer_version": "2.1.0",
                "repository_path": self.repo_path,
                "analysis_duration": (datetime.now() - start_time).total_seconds()
            },
            "implementation_analysis": implementation_analysis,
            "consciousness_analysis": consciousness_analysis,
            "learning_analysis": learning_analysis,
            "coordination_analysis": coordination_analysis,
            "feature_gaps": gap_analysis,
            "implementation_depth": depth_scores,
            "overall_assessment": await self._generate_overall_assessment()
        }
        
        print(f"\n✅ Analysis completed in {analysis_results['metadata']['analysis_duration']:.2f}s")
        return analysis_results

    async def _analyze_implementation_completeness(self) -> Dict[str, ImplementationAnalysis]:
        """Analyze the implementation completeness of each component."""
        implementations = {}
        
        for kernel_dir in self._get_kernel_directories():
            kernel_name = os.path.basename(kernel_dir)
            print(f"   🔍 Analyzing {kernel_name}...")
            
            analysis = await self._analyze_component_implementation(kernel_dir)
            implementations[kernel_name] = analysis
            
        return implementations

    async def _analyze_component_implementation(self, component_path: str) -> ImplementationAnalysis:
        """Analyze implementation details of a specific component."""
        component_name = os.path.basename(component_path)
        
        # Count Python files and analyze code
        py_files = list(Path(component_path).rglob("*.py"))
        total_lines = 0
        total_methods = 0
        implemented_methods = 0
        abstract_methods = 0
        placeholder_methods = 0
        agi_features = []
        missing_features = []
        implementation_notes = []
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_lines += len(content.splitlines())
                    
                    # Parse AST to analyze methods
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                total_methods += 1
                                
                                # Check if method is implemented
                                if self._is_method_implemented(node, content):
                                    implemented_methods += 1
                                elif self._is_abstract_method(node, content):
                                    abstract_methods += 1
                                else:
                                    placeholder_methods += 1
                                
                                # Check for AGI-related features
                                agi_feature = self._identify_agi_feature(node, content)
                                if agi_feature:
                                    agi_features.append(agi_feature)
                    except SyntaxError:
                        implementation_notes.append(f"Syntax error in {py_file}")
                        
            except Exception as e:
                implementation_notes.append(f"Error analyzing {py_file}: {str(e)}")
        
        # Calculate implementation score
        if total_methods > 0:
            implementation_score = implemented_methods / total_methods
        else:
            implementation_score = 0.0
        
        # Calculate complexity score
        complexity_score = min(total_lines / 1000.0, 1.0)  # Normalize to 0-1
        
        # Identify missing features for this component type
        missing_features = self._identify_missing_features(component_name, agi_features)
        
        return ImplementationAnalysis(
            component_name=component_name,
            implementation_score=implementation_score,
            total_methods=total_methods,
            implemented_methods=implemented_methods,
            abstract_methods=abstract_methods,
            placeholder_methods=placeholder_methods,
            lines_of_code=total_lines,
            complexity_score=complexity_score,
            agi_features=agi_features,
            missing_features=missing_features,
            implementation_notes=implementation_notes
        )

    async def _analyze_consciousness_capabilities(self) -> ConsciousnessAnalysis:
        """Analyze consciousness and self-awareness capabilities."""
        consciousness_indicators = []
        introspection_capabilities = []
        meta_cognitive_features = []
        missing_features = []
        
        # Check for consciousness-related implementations
        consciousness_files = [
            "grace/core/self_awareness.py",
            "grace/intelligence/consciousness.py", 
            "grace/core/introspection.py",
            "grace/governance/self_reflection.py"
        ]
        
        found_features = []
        for file_path in consciousness_files:
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    content = f.read()
                    if len(content.strip()) > 50:  # Not just imports
                        found_features.append(os.path.basename(file_path))
                        consciousness_indicators.append(f"Implementation found: {file_path}")
        
        # Look for self-reflection in communication demo
        demo_path = os.path.join(self.repo_path, "grace_communication_demo.py")
        if os.path.exists(demo_path):
            with open(demo_path, 'r') as f:
                content = f.read()
                if "self_reflection" in content.lower() or "introspection" in content.lower():
                    introspection_capabilities.append("Demo self-reflection capability")
                if "self_assessment" in content.lower():
                    meta_cognitive_features.append("Self-assessment functionality")
        
        # Assess implementation depth
        if len(found_features) >= 3:
            depth = "comprehensive"
        elif len(found_features) >= 2:
            depth = "deep"
        elif len(found_features) >= 1:
            depth = "partial"
        else:
            depth = "surface"
            
        # Identify missing consciousness features
        expected_features = [
            "Self-model maintenance",
            "Meta-cognitive monitoring", 
            "Attention management",
            "Goal-state awareness",
            "Temporal self-continuity",
            "Theory of mind for other agents"
        ]
        
        missing_features = [f for f in expected_features if not any(
            keyword in f.lower() for feature in found_features for keyword in self.agi_keywords['consciousness']
        )]
        
        self_awareness_score = len(found_features) / len(consciousness_files)
        
        return ConsciousnessAnalysis(
            self_awareness_score=self_awareness_score,
            introspection_capabilities=introspection_capabilities,
            meta_cognitive_features=meta_cognitive_features,
            consciousness_indicators=consciousness_indicators,
            missing_consciousness_features=missing_features,
            implementation_depth=depth
        )

    async def _analyze_learning_mechanisms(self) -> LearningAnalysis:
        """Analyze learning and adaptation mechanisms."""
        learning_types = []
        adaptation_mechanisms = []
        meta_learning_capabilities = []
        missing_features = []
        
        # Check learning kernel implementation
        learning_kernel_path = os.path.join(self.grace_path, "learning_kernel")
        if os.path.exists(learning_kernel_path):
            learning_files = list(Path(learning_kernel_path).rglob("*.py"))
            
            for file_path in learning_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Identify learning types
                    if "supervised" in content.lower():
                        learning_types.append("Supervised Learning")
                    if "unsupervised" in content.lower():
                        learning_types.append("Unsupervised Learning") 
                    if "reinforcement" in content.lower():
                        learning_types.append("Reinforcement Learning")
                    if "meta_learn" in content.lower() or "meta-learn" in content.lower():
                        meta_learning_capabilities.append("Meta-learning capability")
                    if "adapt" in content.lower():
                        adaptation_mechanisms.append("Adaptive mechanisms")
        
        # Check MLDL kernel for learning
        mldl_path = os.path.join(self.grace_path, "mldl")
        if os.path.exists(mldl_path):
            specialist_files = list(Path(mldl_path).rglob("*specialist*.py"))
            if len(specialist_files) > 0:
                learning_types.append("Specialist-based Learning")
                adaptation_mechanisms.append("Multi-specialist consensus")
        
        # Check for experience integration
        experience_integration = False
        continuous_improvement = False
        
        for component_dir in self._get_kernel_directories():
            component_files = list(Path(component_dir).rglob("*.py"))
            for file_path in component_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if "experience" in content.lower() and ("store" in content.lower() or "learn" in content.lower()):
                            experience_integration = True
                        if "continuous" in content.lower() and "improve" in content.lower():
                            continuous_improvement = True
                except:
                    continue
        
        # Calculate learning effectiveness
        learning_effectiveness = len(learning_types) * 0.2 + len(adaptation_mechanisms) * 0.3 + len(meta_learning_capabilities) * 0.5
        learning_effectiveness = min(learning_effectiveness, 1.0)
        
        # Identify missing learning features
        expected_features = [
            "Online learning capability",
            "Transfer learning mechanisms", 
            "Catastrophic forgetting prevention",
            "Active learning strategies",
            "Self-supervised learning",
            "Curriculum learning"
        ]
        
        missing_features = [f for f in expected_features if not any(
            keyword in f.lower().replace(' ', '_') for lt in learning_types for keyword in [lt.lower().replace(' ', '_')]
        )]
        
        return LearningAnalysis(
            learning_types=learning_types,
            adaptation_mechanisms=adaptation_mechanisms,
            meta_learning_capabilities=meta_learning_capabilities,
            experience_integration=experience_integration,
            continuous_improvement=continuous_improvement,
            learning_effectiveness_score=learning_effectiveness,
            missing_learning_features=missing_features
        )

    async def _analyze_cross_domain_coordination(self) -> CrossDomainAnalysis:
        """Analyze cross-domain coordination capabilities."""
        coordination_mechanisms = []
        knowledge_transfer = []
        integration_patterns = []
        communication_protocols = []
        missing_features = []
        
        # Check for bridge implementations
        bridge_dirs = []
        for root, dirs, files in os.walk(self.grace_path):
            if "bridge" in root.lower():
                bridge_dirs.append(root)
        
        for bridge_dir in bridge_dirs:
            bridge_files = list(Path(bridge_dir).rglob("*.py"))
            for file_path in bridge_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if len(content.strip()) > 50:  # Not just imports
                            bridge_name = os.path.basename(file_path).replace('.py', '')
                            coordination_mechanisms.append(f"Bridge: {bridge_name}")
                            
                            if "transfer" in content.lower():
                                knowledge_transfer.append(f"Knowledge transfer via {bridge_name}")
                except:
                    continue
        
        # Check event bus for communication
        event_bus_path = os.path.join(self.grace_path, "core", "event_bus.py")
        if os.path.exists(event_bus_path):
            with open(event_bus_path, 'r') as f:
                content = f.read()
                if "publish" in content and "subscribe" in content:
                    communication_protocols.append("Event-driven communication")
                if "correlation" in content.lower():
                    integration_patterns.append("Correlated event processing")
        
        # Check for orchestration
        orchestration_path = os.path.join(self.grace_path, "orchestration")
        if os.path.exists(orchestration_path):
            orch_files = list(Path(orchestration_path).rglob("*.py"))
            if len(orch_files) > 0:
                coordination_mechanisms.append("Orchestration layer")
                integration_patterns.append("Centralized orchestration")
        
        # Calculate coordination effectiveness
        total_mechanisms = len(coordination_mechanisms) + len(knowledge_transfer) + len(integration_patterns)
        coordination_effectiveness = min(total_mechanisms / 10.0, 1.0)  # Normalize to 0-1
        
        # Identify missing coordination features
        expected_features = [
            "Semantic knowledge integration",
            "Cross-modal reasoning",
            "Dynamic task decomposition",
            "Multi-agent collaboration protocols",
            "Emergent behavior coordination",
            "Context-aware information routing"
        ]
        
        missing_features = [f for f in expected_features if not any(
            keyword in mech.lower() for mech in coordination_mechanisms for keyword in f.lower().split()
        )]
        
        return CrossDomainAnalysis(
            coordination_mechanisms=coordination_mechanisms,
            knowledge_transfer=knowledge_transfer,
            integration_patterns=integration_patterns,
            communication_protocols=communication_protocols,
            coordination_effectiveness=coordination_effectiveness,
            missing_coordination_features=missing_features
        )

    async def _analyze_feature_gaps(self) -> List[FeatureGap]:
        """Identify and prioritize feature gaps."""
        gaps = []
        
        # Critical AGI features that should be present
        critical_features = {
            "Self-Modeling": ("consciousness", "critical", "Core self-representation and introspection"),
            "Causal Reasoning": ("reasoning", "critical", "Understanding cause-effect relationships"),
            "Meta-Cognition": ("consciousness", "high", "Thinking about thinking processes"),
            "Transfer Learning": ("learning", "high", "Knowledge transfer across domains"),
            "Emergent Behavior": ("coordination", "medium", "Self-organizing system behavior"),
            "Theory of Mind": ("consciousness", "high", "Understanding other agents' mental states")
        }
        
        for feature_name, (category, importance, description) in critical_features.items():
            # Check if feature exists
            found = False
            for root, dirs, files in os.walk(self.grace_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                if any(keyword in content.lower() for keyword in feature_name.lower().split()):
                                    if len(content.strip()) > 100:  # Substantial implementation
                                        found = True
                                        break
                        except:
                            continue
                if found:
                    break
            
            if not found:
                gaps.append(FeatureGap(
                    feature_name=feature_name,
                    importance=importance,
                    current_status="missing",
                    estimated_effort="large" if importance == "critical" else "medium",
                    dependencies=[],
                    implementation_suggestion=description
                ))
        
        return gaps

    async def _calculate_implementation_depth(self) -> Dict[str, float]:
        """Calculate implementation depth scores for different AGI aspects."""
        scores = {}
        
        # Score different aspects
        aspects = {
            "governance": ["governance", "parliament", "trust_core"],
            "intelligence": ["intelligence", "mldl"],
            "learning": ["learning_kernel"],
            "communication": ["event_mesh", "comms"],
            "consciousness": ["core"],  # Should have consciousness modules
            "coordination": ["orchestration", "bridges"]
        }
        
        for aspect, directories in aspects.items():
            total_score = 0
            count = 0
            
            for directory in directories:
                dir_path = os.path.join(self.grace_path, directory)
                if os.path.exists(dir_path):
                    analysis = await self._analyze_component_implementation(dir_path)
                    total_score += analysis.implementation_score
                    count += 1
            
            scores[aspect] = total_score / count if count > 0 else 0.0
        
        return scores

    async def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall assessment and recommendations."""
        # This would be populated by previous analysis results
        return {
            "system_maturity": "early_framework",
            "agi_readiness": "foundational",
            "key_strengths": [
                "Solid governance architecture",
                "Event-driven communication",
                "Modular kernel design"
            ],
            "critical_gaps": [
                "Limited consciousness implementation",
                "Basic learning mechanisms", 
                "Missing cross-domain reasoning"
            ],
            "recommendations": [
                "Implement consciousness models",
                "Enhance learning capabilities",
                "Develop semantic integration"
            ]
        }

    def _get_kernel_directories(self) -> List[str]:
        """Get list of kernel directories."""
        directories = []
        if os.path.exists(self.grace_path):
            for item in os.listdir(self.grace_path):
                item_path = os.path.join(self.grace_path, item)
                if os.path.isdir(item_path) and not item.startswith('__'):
                    directories.append(item_path)
        return directories

    def _is_method_implemented(self, node: ast.FunctionDef, content: str) -> bool:
        """Check if a method has substantial implementation."""
        if len(node.body) == 0:
            return False
        
        # Check for pass statements or NotImplemented
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                return False
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                    return False
        
        # Check for docstring-only methods
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                return False
        
        return True

    def _is_abstract_method(self, node: ast.FunctionDef, content: str) -> bool:
        """Check if method is marked as abstract."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                return True
        return False

    def _identify_agi_feature(self, node: ast.FunctionDef, content: str) -> Optional[str]:
        """Identify AGI-related features in method."""
        method_name = node.name.lower()
        
        for category, keywords in self.agi_keywords.items():
            for keyword in keywords:
                if keyword in method_name:
                    return f"{category}: {node.name}"
        
        return None

    def _identify_missing_features(self, component_name: str, found_features: List[str]) -> List[str]:
        """Identify missing features for a component type."""
        expected_by_component = {
            "intelligence": ["reasoning", "planning", "decision_making"],
            "learning_kernel": ["online_learning", "meta_learning", "transfer_learning"],
            "governance": ["ethical_reasoning", "value_alignment"],
            "consciousness": ["self_awareness", "introspection", "meta_cognition"]
        }
        
        expected = expected_by_component.get(component_name, [])
        found_keywords = [f.lower() for f in found_features]
        
        missing = []
        for expected_feature in expected:
            if not any(keyword in found_keywords for keyword in expected_feature.split('_')):
                missing.append(expected_feature.replace('_', ' ').title())
        
        return missing

    async def generate_report(self, analysis_results: Dict[str, Any], save_to_file: bool = True) -> str:
        """Generate a comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
# Grace Enhanced AGI System Analysis Report

**Analysis Date:** {analysis_results['metadata']['analysis_date']}
**Analysis Duration:** {analysis_results['metadata']['analysis_duration']:.2f} seconds
**Analyzer Version:** {analysis_results['metadata']['analyzer_version']}

## Executive Summary

The Grace AGI system represents a sophisticated architectural framework with strong 
foundational components but significant gaps in core AGI capabilities.

### Implementation Maturity Assessment

"""
        
        # Add implementation analysis
        impl_analysis = analysis_results['implementation_analysis']
        total_components = len(impl_analysis)
        avg_implementation = sum(comp.implementation_score for comp in impl_analysis.values()) / total_components
        
        report += f"""
**Overall Implementation Score:** {avg_implementation:.2%}
**Components Analyzed:** {total_components}

### Component Implementation Status

| Component | Implementation Score | Lines of Code | AGI Features | Status |
|-----------|---------------------|---------------|--------------|--------|
"""
        
        for name, analysis in impl_analysis.items():
            status = "🟢 Complete" if analysis.implementation_score > 0.8 else \
                    "🟡 Partial" if analysis.implementation_score > 0.4 else "🔴 Limited"
            
            report += f"| {name} | {analysis.implementation_score:.1%} | {analysis.lines_of_code} | {len(analysis.agi_features)} | {status} |\n"
        
        # Add consciousness analysis
        consciousness = analysis_results['consciousness_analysis']
        report += f"""

## Consciousness & Self-Awareness Analysis

**Implementation Depth:** {consciousness.implementation_depth.title()}
**Self-Awareness Score:** {consciousness.self_awareness_score:.1%}

### Found Capabilities
"""
        for capability in consciousness.introspection_capabilities + consciousness.meta_cognitive_features:
            report += f"- ✅ {capability}\n"
        
        report += "\n### Missing Consciousness Features\n"
        for missing in consciousness.missing_consciousness_features:
            report += f"- ❌ {missing}\n"
        
        # Add learning analysis
        learning = analysis_results['learning_analysis']
        report += f"""

## Learning & Adaptation Analysis

**Learning Effectiveness Score:** {learning.learning_effectiveness_score:.1%}
**Experience Integration:** {'✅ Yes' if learning.experience_integration else '❌ No'}
**Continuous Improvement:** {'✅ Yes' if learning.continuous_improvement else '❌ No'}

### Learning Mechanisms Found
"""
        for mechanism in learning.learning_types + learning.adaptation_mechanisms:
            report += f"- ✅ {mechanism}\n"
        
        report += "\n### Missing Learning Features\n"
        for missing in learning.missing_learning_features:
            report += f"- ❌ {missing}\n"
        
        # Add coordination analysis
        coordination = analysis_results['coordination_analysis']
        report += f"""

## Cross-Domain Coordination Analysis

**Coordination Effectiveness:** {coordination.coordination_effectiveness:.1%}

### Coordination Mechanisms
"""
        for mechanism in coordination.coordination_mechanisms:
            report += f"- ✅ {mechanism}\n"
        
        report += "\n### Missing Coordination Features\n"
        for missing in coordination.missing_coordination_features:
            report += f"- ❌ {missing}\n"
        
        # Add feature gaps
        gaps = analysis_results['feature_gaps']
        if gaps:
            report += f"""

## Critical Feature Gaps

| Feature | Importance | Status | Effort Required |
|---------|------------|--------|-----------------|
"""
            for gap in gaps:
                report += f"| {gap.feature_name} | {gap.importance.title()} | {gap.current_status.title()} | {gap.estimated_effort.title()} |\n"
        
        # Add overall assessment
        assessment = analysis_results['overall_assessment']
        report += f"""

## Overall Assessment & Recommendations

**System Maturity:** {assessment['system_maturity'].replace('_', ' ').title()}
**AGI Readiness:** {assessment['agi_readiness'].title()}

### Key Strengths
"""
        for strength in assessment['key_strengths']:
            report += f"- ✅ {strength}\n"
        
        report += "\n### Critical Gaps\n"
        for gap in assessment['critical_gaps']:
            report += f"- ⚠️ {gap}\n"
        
        report += "\n### Recommendations\n"
        for rec in assessment['recommendations']:
            report += f"- 🎯 {rec}\n"
        
        report += f"""

## Conclusion

The Grace system provides an excellent foundational architecture for AGI development 
but requires significant implementation work in consciousness, learning, and cross-domain 
coordination to achieve true AGI capabilities.

---
*Generated by Grace Enhanced System Analyzer v{analysis_results['metadata']['analyzer_version']}*
"""
        
        if save_to_file:
            filename = f"GRACE_ENHANCED_ANALYSIS_REPORT_{timestamp}.md"
            filepath = os.path.join(self.repo_path, filename)
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"📄 Enhanced analysis report saved to: {filename}")
        
        return report


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grace Enhanced System Analysis")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed analysis")
    parser.add_argument("--save-report", action="store_true", help="Save report to file")
    parser.add_argument("--component", help="Analyze specific component")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = GraceEnhancedAnalyzer()
    results = await analyzer.run_comprehensive_analysis()
    
    # Generate and display report
    report = await analyzer.generate_report(results, save_to_file=args.save_report)
    
    if not args.save_report:
        print(report)
    
    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"grace_enhanced_analysis_{timestamp}.json"
    
    # Convert dataclasses to dicts for JSON serialization
    def convert_dataclass(obj):
        if hasattr(obj, '__dict__'):
            return asdict(obj)
        return obj
    
    json_results = json.loads(json.dumps(results, default=convert_dataclass))
    
    with open(json_filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"📊 Detailed analysis data saved to: {json_filename}")


if __name__ == "__main__":
    asyncio.run(main())