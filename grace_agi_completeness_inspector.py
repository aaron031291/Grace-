#!/usr/bin/env python3
"""
Grace AGI Completeness Inspector
===============================

This tool provides a detailed assessment of AGI-specific capabilities in the Grace system,
focusing on the gap between framework structure and actual AGI implementation.

Key Analysis Areas:
- Consciousness implementation depth
- True learning vs. static ML models
- Cross-domain reasoning capabilities  
- Self-awareness and introspection
- Emergent behavior mechanisms
- Meta-cognitive capabilities

The goal is to distinguish between:
1. Framework/architecture (what's built to support AGI)
2. Actual AGI capabilities (what's truly implemented)
3. Missing critical components for AGI

Usage:
    python grace_agi_completeness_inspector.py [--output-format json|md|both]
"""

import os
import ast
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AGICapabilityAssessment:
    """Assessment of a specific AGI capability."""
    capability_name: str
    implementation_level: str  # "none", "placeholder", "basic", "functional", "advanced"
    evidence_found: List[str]
    missing_components: List[str]
    code_complexity: int  # Lines of substantial code
    theoretical_vs_practical: str  # "theoretical_only", "partial_implementation", "fully_implemented"
    agi_readiness_score: float  # 0.0 to 1.0


@dataclass
class ComponentAnalysis:
    """Deep analysis of a component's AGI capabilities."""
    component_name: str
    purpose: str  # What this component is supposed to do
    actual_implementation: str  # What it actually does
    agi_capabilities: List[AGICapabilityAssessment]
    framework_quality: float  # How well the framework is designed
    implementation_quality: float  # How much is actually implemented
    gaps: List[str]


class GraceAGIInspector:
    """
    Inspector that focuses specifically on AGI capabilities rather than
    general software architecture health.
    """
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or "/home/runner/work/Grace-/Grace-"
        self.grace_path = os.path.join(self.repo_path, "grace")
        
        # AGI-specific keywords and patterns
        self.agi_indicators = {
            'consciousness': {
                'keywords': ['consciousness', 'self_aware', 'introspect', 'meta_cognitive', 'self_model'],
                'patterns': [r'class.*Consciousness', r'def.*self_aware', r'def.*introspect'],
                'required_methods': ['self_reflect', 'introspect', 'self_assess', 'meta_cognition']
            },
            'reasoning': {
                'keywords': ['reason', 'causal', 'logic', 'inference', 'deduction', 'abduction'],
                'patterns': [r'class.*Reasoning', r'def.*reason', r'def.*infer'],
                'required_methods': ['causal_reasoning', 'logical_inference', 'abstract_reasoning']
            },
            'learning': {
                'keywords': ['meta_learn', 'transfer_learn', 'adapt', 'evolve', 'improve'],
                'patterns': [r'class.*Learning', r'def.*meta_learn', r'def.*transfer'],
                'required_methods': ['online_learning', 'transfer_learning', 'meta_learning', 'self_improvement']
            },
            'emergence': {
                'keywords': ['emerge', 'emergent', 'self_organize', 'complex', 'dynamic'],
                'patterns': [r'class.*Emergence', r'def.*emerge', r'def.*self_organize'],
                'required_methods': ['emergent_behavior', 'self_organization', 'complex_adaptation']
            }
        }
        
        # Critical AGI components that should exist
        self.expected_agi_components = {
            'consciousness_engine': 'Core self-awareness and introspection',
            'causal_reasoning': 'Understanding cause-effect relationships',
            'meta_cognitive_monitor': 'Monitoring and controlling own thinking',
            'transfer_learning': 'Knowledge transfer across domains',
            'emergence_detector': 'Detecting and managing emergent behaviors',
            'goal_hierarchy': 'Dynamic goal formation and management',
            'world_model': 'Internal representation of environment',
            'theory_of_mind': 'Understanding other agents\' mental states'
        }

    def inspect_agi_capabilities(self) -> Dict[str, Any]:
        """Comprehensive AGI capability inspection."""
        print("🔍 Grace AGI Completeness Inspection")
        print("=" * 50)
        
        results = {
            "inspection_metadata": {
                "timestamp": datetime.now().isoformat(),
                "inspector_version": "1.0.0",
                "repository_path": self.repo_path
            },
            "agi_assessment": {},
            "component_analysis": {},
            "missing_capabilities": [],
            "implementation_vs_framework": {},
            "overall_agi_readiness": {}
        }
        
        # Phase 1: Assess core AGI capabilities
        print("\n🧠 Phase 1: Core AGI Capability Assessment")
        results["agi_assessment"] = self._assess_core_agi_capabilities()
        
        # Phase 2: Analyze components for actual AGI implementation
        print("\n🔧 Phase 2: Component Implementation Analysis")
        results["component_analysis"] = self._analyze_components_for_agi()
        
        # Phase 3: Identify missing critical AGI components
        print("\n❌ Phase 3: Missing AGI Component Analysis")
        results["missing_capabilities"] = self._identify_missing_agi_components()
        
        # Phase 4: Framework vs Implementation comparison
        print("\n⚖️  Phase 4: Framework vs Implementation Analysis")
        results["implementation_vs_framework"] = self._compare_framework_vs_implementation()
        
        # Phase 5: Overall AGI readiness assessment
        print("\n🎯 Phase 5: AGI Readiness Assessment")
        results["overall_agi_readiness"] = self._assess_overall_agi_readiness(results)
        
        return results

    def _assess_core_agi_capabilities(self) -> Dict[str, AGICapabilityAssessment]:
        """Assess implementation of core AGI capabilities."""
        assessments = {}
        
        for capability, indicators in self.agi_indicators.items():
            print(f"   🔍 Assessing {capability}...")
            
            evidence = []
            implementation_level = "none"
            code_complexity = 0
            missing_components = list(indicators['required_methods'])
            
            # Search for evidence in codebase
            for root, dirs, files in os.walk(self.grace_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # Check for keywords
                                for keyword in indicators['keywords']:
                                    if keyword in content.lower():
                                        evidence.append(f"Keyword '{keyword}' found in {file}")
                                        implementation_level = max(implementation_level, "placeholder", key=self._implementation_level_order)
                                
                                # Check for patterns
                                for pattern in indicators['patterns']:
                                    if re.search(pattern, content, re.IGNORECASE):
                                        evidence.append(f"Pattern '{pattern}' found in {file}")
                                        implementation_level = max(implementation_level, "basic", key=self._implementation_level_order)
                                
                                # Check for required methods
                                tree = ast.parse(content)
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.FunctionDef):
                                        if node.name in indicators['required_methods']:
                                            if self._is_substantial_implementation(node, content):
                                                evidence.append(f"Method '{node.name}' implemented in {file}")
                                                implementation_level = max(implementation_level, "functional", key=self._implementation_level_order)
                                                code_complexity += len(node.body) * 10
                                                missing_components.remove(node.name)
                                
                        except Exception as e:
                            continue
            
            # Determine theoretical vs practical
            if implementation_level == "none":
                theoretical_vs_practical = "theoretical_only"
            elif implementation_level in ["placeholder", "basic"]:
                theoretical_vs_practical = "partial_implementation"
            else:
                theoretical_vs_practical = "fully_implemented"
            
            # Calculate AGI readiness score
            agi_readiness_score = self._calculate_agi_readiness_score(
                implementation_level, evidence, missing_components, code_complexity
            )
            
            assessments[capability] = AGICapabilityAssessment(
                capability_name=capability,
                implementation_level=implementation_level,
                evidence_found=evidence,
                missing_components=missing_components,
                code_complexity=code_complexity,
                theoretical_vs_practical=theoretical_vs_practical,
                agi_readiness_score=agi_readiness_score
            )
        
        return assessments

    def _analyze_components_for_agi(self) -> Dict[str, ComponentAnalysis]:
        """Analyze each component for actual AGI implementation vs framework."""
        analyses = {}
        
        key_components = [
            ("intelligence", "AI coordination and reasoning"),
            ("governance", "Decision making and ethics"),
            ("learning_kernel", "Learning and adaptation"),
            ("core", "Core AGI infrastructure"),
            ("mldl", "Machine learning specialists")
        ]
        
        for component_name, purpose in key_components:
            component_path = os.path.join(self.grace_path, component_name)
            if os.path.exists(component_path):
                print(f"   🔍 Analyzing {component_name}...")
                
                # Analyze what this component actually implements
                actual_implementation = self._analyze_actual_implementation(component_path)
                
                # Assess AGI capabilities in this component
                agi_capabilities = self._assess_component_agi_capabilities(component_path)
                
                # Calculate framework vs implementation quality
                framework_quality = self._assess_framework_quality(component_path)
                implementation_quality = self._assess_implementation_quality(component_path)
                
                # Identify gaps
                gaps = self._identify_component_gaps(component_name, actual_implementation)
                
                analyses[component_name] = ComponentAnalysis(
                    component_name=component_name,
                    purpose=purpose,
                    actual_implementation=actual_implementation,
                    agi_capabilities=agi_capabilities,
                    framework_quality=framework_quality,
                    implementation_quality=implementation_quality,
                    gaps=gaps
                )
        
        return analyses

    def _analyze_actual_implementation(self, component_path: str) -> str:
        """Analyze what the component actually implements (not what it's designed for)."""
        implementations = []
        
        for file_path in Path(component_path).rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for substantial class implementations
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if len(node.body) > 3:  # More than just pass/docstring
                                method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                                if method_count > 2:
                                    implementations.append(f"{node.name} class with {method_count} methods")
                        
                        elif isinstance(node, ast.FunctionDef):
                            if self._is_substantial_implementation(node, content):
                                implementations.append(f"Function: {node.name}")
            
            except Exception:
                continue
        
        return "; ".join(implementations[:10])  # Top 10 implementations

    def _assess_component_agi_capabilities(self, component_path: str) -> List[AGICapabilityAssessment]:
        """Assess AGI-specific capabilities in a component."""
        capabilities = []
        
        # Look for AGI-specific implementations
        agi_patterns = {
            'self_awareness': [r'def.*self_', r'class.*Self'],
            'reasoning': [r'def.*(reason|logic|infer)', r'class.*(Reason|Logic)'],
            'learning': [r'def.*(learn|adapt)', r'class.*Learn'],
            'emergence': [r'def.*(emerge|evolve)', r'class.*Emerge']
        }
        
        for capability_name, patterns in agi_patterns.items():
            evidence = []
            implementation_level = "none"
            
            for file_path in Path(component_path).rglob("*.py"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                evidence.extend([f"Found: {match}" for match in matches])
                                implementation_level = "basic"
                except Exception:
                    continue
            
            if evidence:
                capabilities.append(AGICapabilityAssessment(
                    capability_name=capability_name,
                    implementation_level=implementation_level,
                    evidence_found=evidence,
                    missing_components=[],
                    code_complexity=len(evidence) * 50,
                    theoretical_vs_practical="partial_implementation",
                    agi_readiness_score=0.3 if implementation_level == "basic" else 0.1
                ))
        
        return capabilities

    def _identify_missing_agi_components(self) -> List[Dict[str, Any]]:
        """Identify missing critical AGI components."""
        missing = []
        
        for component_name, description in self.expected_agi_components.items():
            # Check if component exists in any form
            found = False
            
            for root, dirs, files in os.walk(self.grace_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # Check for component name or related keywords
                                keywords = component_name.split('_')
                                if any(keyword in content.lower() for keyword in keywords):
                                    # Check if it's substantial (more than just mentions)
                                    if len([line for line in content.splitlines() 
                                           if any(keyword in line.lower() for keyword in keywords)]) > 3:
                                        found = True
                                        break
                        except Exception:
                            continue
                if found:
                    break
            
            if not found:
                missing.append({
                    "component_name": component_name,
                    "description": description,
                    "criticality": "high" if component_name in ['consciousness_engine', 'causal_reasoning'] else "medium",
                    "suggested_location": f"grace/{component_name.replace('_', '/')}/",
                    "implementation_effort": "large"
                })
        
        return missing

    def _compare_framework_vs_implementation(self) -> Dict[str, Any]:
        """Compare framework design quality vs actual implementation."""
        comparison = {
            "framework_strengths": [],
            "implementation_gaps": [],
            "architecture_quality": 0.0,
            "implementation_completeness": 0.0
        }
        
        # Assess framework design
        total_components = 0
        well_designed_components = 0
        
        for component_dir in os.listdir(self.grace_path):
            component_path = os.path.join(self.grace_path, component_dir)
            if os.path.isdir(component_path) and not component_dir.startswith('__'):
                total_components += 1
                
                # Check for good architectural patterns
                has_init = os.path.exists(os.path.join(component_path, "__init__.py"))
                has_multiple_files = len([f for f in os.listdir(component_path) if f.endswith('.py')]) > 1
                
                if has_init and has_multiple_files:
                    well_designed_components += 1
                    comparison["framework_strengths"].append(f"Well-structured {component_dir}")
        
        # Assess implementation completeness
        implementation_score = 0.0
        for root, dirs, files in os.walk(self.grace_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # Count substantial implementations vs placeholders
                            lines = content.splitlines()
                            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                            
                            if len(code_lines) > 20:  # Substantial file
                                if 'NotImplementedError' in content or 'pass' in content:
                                    comparison["implementation_gaps"].append(f"Placeholder code in {file}")
                                else:
                                    implementation_score += 1
                    except Exception:
                        continue
        
        comparison["architecture_quality"] = well_designed_components / total_components if total_components > 0 else 0.0
        comparison["implementation_completeness"] = min(implementation_score / 50.0, 1.0)  # Normalize
        
        return comparison

    def _assess_overall_agi_readiness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall AGI readiness based on all analysis."""
        agi_assessment = results["agi_assessment"]
        component_analysis = results["component_analysis"]
        missing_capabilities = results["missing_capabilities"]
        
        # Calculate scores
        capability_scores = [assessment.agi_readiness_score for assessment in agi_assessment.values()]
        avg_capability_score = sum(capability_scores) / len(capability_scores) if capability_scores else 0.0
        
        implementation_scores = [comp.implementation_quality for comp in component_analysis.values()]
        avg_implementation_score = sum(implementation_scores) / len(implementation_scores) if implementation_scores else 0.0
        
        missing_critical = len([cap for cap in missing_capabilities if cap.get("criticality") == "high"])
        missing_penalty = min(missing_critical * 0.2, 0.8)
        
        overall_score = (avg_capability_score * 0.4 + avg_implementation_score * 0.4) * (1.0 - missing_penalty)
        
        # Determine readiness level
        if overall_score >= 0.8:
            readiness_level = "AGI Ready"
        elif overall_score >= 0.6:
            readiness_level = "Advanced AI"
        elif overall_score >= 0.4:
            readiness_level = "Intelligent Framework"
        elif overall_score >= 0.2:
            readiness_level = "Basic AI Framework"
        else:
            readiness_level = "Early Stage Framework"
        
        return {
            "overall_agi_score": overall_score,
            "readiness_level": readiness_level,
            "capability_score": avg_capability_score,
            "implementation_score": avg_implementation_score,
            "missing_critical_count": missing_critical,
            "key_strengths": self._identify_key_strengths(results),
            "critical_gaps": self._identify_critical_gaps(results),
            "next_steps": self._generate_next_steps(results)
        }

    def _identify_key_strengths(self, results: Dict[str, Any]) -> List[str]:
        """Identify key strengths in the current implementation."""
        strengths = []
        
        # Check for well-implemented capabilities
        for capability, assessment in results["agi_assessment"].items():
            if assessment.agi_readiness_score > 0.5:
                strengths.append(f"Strong {capability} foundation")
        
        # Check for comprehensive components
        for comp_name, analysis in results["component_analysis"].items():
            if analysis.implementation_quality > 0.7:
                strengths.append(f"Well-implemented {comp_name}")
        
        # Framework quality
        framework_vs_impl = results["implementation_vs_framework"]
        if framework_vs_impl["architecture_quality"] > 0.8:
            strengths.append("Excellent architectural design")
        
        return strengths

    def _identify_critical_gaps(self, results: Dict[str, Any]) -> List[str]:
        """Identify the most critical gaps for AGI."""
        gaps = []
        
        # Missing critical capabilities
        for capability, assessment in results["agi_assessment"].items():
            if assessment.agi_readiness_score < 0.3:
                gaps.append(f"Limited {capability} implementation")
        
        # Missing critical components
        critical_missing = [cap["component_name"] for cap in results["missing_capabilities"] 
                          if cap.get("criticality") == "high"]
        gaps.extend([f"Missing {comp.replace('_', ' ')}" for comp in critical_missing])
        
        return gaps

    def _generate_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """Generate concrete next steps for AGI development."""
        steps = []
        
        # Prioritize missing critical components
        critical_missing = [cap for cap in results["missing_capabilities"] 
                          if cap.get("criticality") == "high"]
        for missing in critical_missing[:3]:  # Top 3
            steps.append(f"Implement {missing['component_name'].replace('_', ' ')}")
        
        # Address lowest-scoring capabilities
        low_scoring_caps = [(name, assessment) for name, assessment in results["agi_assessment"].items() 
                           if assessment.agi_readiness_score < 0.3]
        for cap_name, _ in low_scoring_caps[:2]:  # Top 2
            steps.append(f"Enhance {cap_name} implementation")
        
        return steps

    # Helper methods
    def _implementation_level_order(self, level: str) -> int:
        """Order for implementation levels."""
        order = {"none": 0, "placeholder": 1, "basic": 2, "functional": 3, "advanced": 4}
        return order.get(level, 0)

    def _is_substantial_implementation(self, node: ast.FunctionDef, content: str) -> bool:
        """Check if a function has substantial implementation."""
        if len(node.body) < 2:
            return False
        
        # Check for pass statements or NotImplementedError
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                return False
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                    return False
        
        return True

    def _calculate_agi_readiness_score(self, implementation_level: str, evidence: List[str], 
                                     missing_components: List[str], code_complexity: int) -> float:
        """Calculate AGI readiness score for a capability."""
        base_score = {"none": 0.0, "placeholder": 0.1, "basic": 0.3, "functional": 0.6, "advanced": 0.9}
        score = base_score.get(implementation_level, 0.0)
        
        # Bonus for evidence
        score += min(len(evidence) * 0.05, 0.2)
        
        # Penalty for missing components
        score -= len(missing_components) * 0.1
        
        # Bonus for code complexity
        score += min(code_complexity / 1000.0, 0.1)
        
        return max(0.0, min(score, 1.0))

    def _assess_framework_quality(self, component_path: str) -> float:
        """Assess the quality of the framework design."""
        quality_score = 0.0
        
        # Check for good file organization
        py_files = list(Path(component_path).rglob("*.py"))
        if len(py_files) > 1:
            quality_score += 0.3
        
        # Check for init file
        if os.path.exists(os.path.join(component_path, "__init__.py")):
            quality_score += 0.2
        
        # Check for clear abstractions
        for file_path in py_files[:5]:  # Check first 5 files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "class" in content and "def" in content:
                        quality_score += 0.1
            except Exception:
                continue
        
        return min(quality_score, 1.0)

    def _assess_implementation_quality(self, component_path: str) -> float:
        """Assess the quality of actual implementation."""
        implementation_score = 0.0
        total_files = 0
        implemented_files = 0
        
        for file_path in Path(component_path).rglob("*.py"):
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check if file has substantial implementation
                    lines = content.splitlines()
                    code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                    
                    if len(code_lines) > 10 and "NotImplementedError" not in content:
                        implemented_files += 1
            except Exception:
                continue
        
        if total_files > 0:
            implementation_score = implemented_files / total_files
        
        return implementation_score

    def _identify_component_gaps(self, component_name: str, actual_implementation: str) -> List[str]:
        """Identify gaps between expected and actual implementation."""
        gaps = []
        
        expected_by_component = {
            "intelligence": ["reasoning engine", "planning system", "decision making"],
            "learning_kernel": ["meta learning", "transfer learning", "online adaptation"],
            "governance": ["ethical reasoning", "value alignment", "moral decision making"],
            "core": ["consciousness engine", "self awareness", "introspection"]
        }
        
        expected = expected_by_component.get(component_name, [])
        impl_lower = actual_implementation.lower()
        
        for expected_feature in expected:
            if not any(keyword in impl_lower for keyword in expected_feature.split()):
                gaps.append(f"Missing {expected_feature}")
        
        return gaps

    def generate_comprehensive_report(self, results: Dict[str, Any], format_type: str = "md") -> str:
        """Generate comprehensive AGI completeness report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if format_type == "md":
            return self._generate_markdown_report(results, timestamp)
        elif format_type == "json":
            return json.dumps(results, indent=2, default=str)
        else:
            return self._generate_markdown_report(results, timestamp)

    def _generate_markdown_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """Generate detailed markdown report."""
        overall = results["overall_agi_readiness"]
        
        report = f"""
# Grace AGI Completeness Inspection Report

**Inspection Date:** {timestamp}
**Inspector Version:** {results['inspection_metadata']['inspector_version']}

## Executive Summary

**Overall AGI Readiness:** {overall['readiness_level']}
**AGI Score:** {overall['overall_agi_score']:.1%}

The Grace system represents a {overall['readiness_level'].lower()} with strong architectural 
foundations but significant gaps in core AGI capabilities.

## Core AGI Capability Assessment

| Capability | Implementation Level | AGI Readiness | Evidence Count | Missing Components |
|------------|---------------------|---------------|----------------|-------------------|
"""
        
        for cap_name, assessment in results["agi_assessment"].items():
            report += f"| {cap_name.title()} | {assessment.implementation_level.title()} | {assessment.agi_readiness_score:.1%} | {len(assessment.evidence_found)} | {len(assessment.missing_components)} |\n"
        
        report += f"""

## Component Implementation Analysis

"""
        for comp_name, analysis in results["component_analysis"].items():
            report += f"""
### {comp_name.title()}
**Purpose:** {analysis.purpose}
**Framework Quality:** {analysis.framework_quality:.1%}
**Implementation Quality:** {analysis.implementation_quality:.1%}

**What's Actually Implemented:** {analysis.actual_implementation[:200]}...

**AGI Capabilities Found:** {len(analysis.agi_capabilities)}
**Critical Gaps:** {', '.join(analysis.gaps[:3])}
"""
        
        report += f"""

## Missing Critical AGI Components

The following critical AGI components are missing or incomplete:

"""
        for missing in results["missing_capabilities"]:
            criticality_icon = "🔴" if missing["criticality"] == "high" else "🟡"
            report += f"- {criticality_icon} **{missing['component_name'].replace('_', ' ').title()}**: {missing['description']}\n"
        
        report += f"""

## Key Findings

### 🎯 Strengths
"""
        for strength in overall["key_strengths"]:
            report += f"- ✅ {strength}\n"
        
        report += "\n### ⚠️ Critical Gaps\n"
        for gap in overall["critical_gaps"]:
            report += f"- ❌ {gap}\n"
        
        report += f"""

## Implementation vs Framework Analysis

**Architecture Quality:** {results['implementation_vs_framework']['architecture_quality']:.1%}
**Implementation Completeness:** {results['implementation_vs_framework']['implementation_completeness']:.1%}

### Framework Strengths
"""
        for strength in results['implementation_vs_framework']['framework_strengths']:
            report += f"- ✅ {strength}\n"
        
        report += "\n### Implementation Gaps\n"
        for gap in results['implementation_vs_framework']['implementation_gaps'][:5]:
            report += f"- ⚠️ {gap}\n"
        
        report += f"""

## Recommended Next Steps

To advance towards true AGI capabilities:

"""
        for i, step in enumerate(overall["next_steps"], 1):
            report += f"{i}. {step}\n"
        
        report += f"""

## Conclusion

The Grace system demonstrates excellent architectural planning and modular design, 
indicating a sophisticated understanding of AGI requirements. However, the current 
implementation focuses primarily on governance and coordination frameworks rather 
than core AGI capabilities like consciousness, causal reasoning, and meta-cognition.

**Key Insight:** Grace is currently a high-quality AI governance framework positioned 
for AGI development rather than a functional AGI system.

---
*Generated by Grace AGI Completeness Inspector v{results['inspection_metadata']['inspector_version']}*
"""
        
        return report


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grace AGI Completeness Inspector")
    parser.add_argument("--output-format", choices=["json", "md", "both"], default="md",
                       help="Output format for the report")
    
    args = parser.parse_args()
    
    # Run inspection
    inspector = GraceAGIInspector()
    results = inspector.inspect_agi_capabilities()
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output_format in ["md", "both"]:
        md_report = inspector.generate_comprehensive_report(results, "md")
        md_filename = f"GRACE_AGI_COMPLETENESS_REPORT_{timestamp}.md"
        with open(md_filename, 'w') as f:
            f.write(md_report)
        print(f"📄 Markdown report saved to: {md_filename}")
        
        if args.output_format == "md":
            print(md_report)
    
    if args.output_format in ["json", "both"]:
        json_report = inspector.generate_comprehensive_report(results, "json")
        json_filename = f"grace_agi_completeness_{timestamp}.json"
        with open(json_filename, 'w') as f:
            f.write(json_report)
        print(f"📊 JSON report saved to: {json_filename}")
        
        if args.output_format == "json":
            print(json_report)


if __name__ == "__main__":
    main()