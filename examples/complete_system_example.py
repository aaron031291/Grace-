"""
Complete Grace System Example - All Components Working Together
"""

import asyncio
from datetime import datetime, timezone

async def main():
    print("=" * 80)
    print("Grace Complete System Example")
    print("=" * 80)
    
    from grace.integration.swarm_transcendence_integration import SwarmTranscendenceIntegration
    
    # Initialize Grace node with all features
    grace_node = SwarmTranscendenceIntegration(
        node_id="node_main",
        enable_swarm=True,
        enable_quantum=True,
        enable_discovery=True,
        enable_impact=True
    )
    
    await grace_node.start()
    
    print("\n✅ Grace node initialized with all advanced features")
    
    # Example 1: Multi-node decision with quantum enhancement
    print("\n" + "=" * 80)
    print("Example 1: Policy Decision with Full Enhancement")
    print("=" * 80)
    
    policy_context = {
        "policy_id": "education_ai_2024",
        "description": "Integrate AI tools in education",
        "target_groups": ["students", "teachers"],
        "options": [
            {
                "name": "Gradual rollout",
                "cost": 1000000,
                "benefit": 2000000,
                "feasibility": 0.9,
                "impact": 0.7
            },
            {
                "name": "Pilot program",
                "cost": 200000,
                "benefit": 500000,
                "feasibility": 1.0,
                "impact": 0.5
            },
            {
                "name": "Full deployment",
                "cost": 5000000,
                "benefit": 8000000,
                "feasibility": 0.6,
                "impact": 0.9
            }
        ],
        "societal_context": {
            "total_population": 10000000,
            "students_fraction": 0.15,
            "teachers_fraction": 0.02
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    local_decision = {
        "selected_option": "Gradual rollout",
        "economic_impact": 0.3,
        "welfare_impact": 0.5,
        "employment_impact": 0.1,
        "health_impact": 0.05,
        "environmental_impact": 0.0
    }
    
    # Get enhanced decision
    enhanced = await grace_node.enhanced_decision_making(
        policy_context,
        local_decision,
        local_confidence=0.75
    )
    
    print("\n📊 Enhanced Decision Result:")
    print(f"  Local Decision: {enhanced['local_decision']['selected_option']}")
    print(f"  Local Confidence: {enhanced['local_confidence']:.3f}")
    print(f"  Enhancements Applied: {', '.join(enhanced['enhancements'])}")
    
    if "quantum_enhancement" in enhanced:
        quantum = enhanced["quantum_enhancement"]
        print(f"\n  🔮 Quantum Enhancement:")
        print(f"    Recommended: {quantum.get('quantum_recommended', {}).get('strategy', 'N/A')}")
        print(f"    Method: {quantum.get('method')}")
    
    if "impact_assessment" in enhanced:
        impact = enhanced["impact_assessment"]
        print(f"\n  🌍 Impact Assessment:")
        print(f"    Projected Outcomes: {len(impact['projected_outcomes'])} metrics")
        print(f"    Risks: {len(impact['risks'])}")
        print(f"    Benefits: {len(impact['benefits'])}")
        print(f"    Confidence: {impact['confidence']:.3f}")
    
    print(f"\n  ✨ Final Synthesized Decision:")
    final = enhanced["final_decision"]
    print(f"    Decision: {final['decision']['selected_option']}")
    print(f"    Confidence: {final['confidence']:.3f}")
    print(f"    Synthesis Method: {final['synthesis_method']}")
    
    # Validate with governance
    print("\n  ⚖️  Governance Validation:")
    validation = await grace_node.validate_with_governance(
        final["decision"],
        policy_context
    )
    print(f"    Passed: {validation['passed']}")
    print(f"    Violations: {len(validation['violations'])}")
    print(f"    Governance Confidence: {validation['confidence']:.3f}")
    
    # Create unified output
    print("\n  📝 Creating Unified Output...")
    unified_output = grace_node.create_unified_output(
        loop_id="policy_decision_loop",
        enhanced_result=enhanced,
        context=policy_context
    )
    
    print(f"    Output ID: {unified_output.output_id}")
    print(f"    Loop ID: {unified_output.loop_id}")
    print(f"    Reasoning Steps: {len(unified_output.reasoning_chain)}")
    print(f"    Overall Trust: {unified_output.trust_metrics.overall_trust:.3f}")
    
    # Example 2: Scientific Discovery
    print("\n" + "=" * 80)
    print("Example 2: Scientific Discovery with Data Analysis")
    print("=" * 80)
    
    import numpy as np
    
    # Generate research data
    research_data = []
    for i in range(100):
        temperature = np.random.uniform(20, 35)
        humidity = np.random.uniform(40, 80)
        growth_rate = 0.5 * temperature + 0.3 * humidity + np.random.normal(0, 2)
        
        research_data.append({
            "temperature": temperature,
            "humidity": humidity,
            "growth_rate": growth_rate,
            "experiment_id": i
        })
    
    scientific_context = {
        "data": research_data,
        "target": "growth_rate",
        "domain_knowledge": {
            "field": "agriculture",
            "known_factors": ["temperature", "humidity"]
        },
        "hypothesis_testing": True
    }
    
    local_hypothesis = {
        "hypothesis": "Temperature and humidity affect growth rate",
        "confidence": 0.7
    }
    
    enhanced_scientific = await grace_node.enhanced_decision_making(
        scientific_context,
        local_hypothesis,
        local_confidence=0.7
    )
    
    print("\n🔬 Scientific Discovery Results:")
    
    if "hypotheses" in enhanced_scientific:
        hypotheses = enhanced_scientific["hypotheses"]
        print(f"  Generated Hypotheses: {len(hypotheses)}")
        
        for i, hyp in enumerate(hypotheses[:3], 1):
            print(f"\n  Hypothesis {i}:")
            print(f"    Statement: {hyp['hypothesis']}")
            print(f"    Confidence: {hyp['confidence']:.3f}")
            print(f"    Testable: {hyp['testable']}")
            print(f"    Variables: {', '.join(hyp['variables'])}")
    
    # Example 3: Optimization Problem
    print("\n" + "=" * 80)
    print("Example 3: Quantum-Inspired Optimization")
    print("=" * 80)
    
    optimization_context = {
        "optimization_problem": {
            "objective": lambda params: (params['x'] - 5)**2 + (params['y'] + 3)**2,
            "variables": ['x', 'y'],
            "bounds": {'x': (-10, 10), 'y': (-10, 10)}
        },
        "description": "Minimize distance from point (5, -3)"
    }
    
    local_optimization = {"x": 0, "y": 0}
    
    enhanced_opt = await grace_node.enhanced_decision_making(
        optimization_context,
        local_optimization,
        local_confidence=0.5
    )
    
    print("\n⚛️  Quantum Optimization Results:")
    
    if "quantum_enhancement" in enhanced_opt:
        quantum_result = enhanced_opt["quantum_enhancement"]
        optimal = quantum_result.get("quantum_optimal", {})
        
        print(f"  Optimal x: {optimal.get('x', 0):.3f} (target: 5.0)")
        print(f"  Optimal y: {optimal.get('y', 0):.3f} (target: -3.0)")
        print(f"  Method: {quantum_result.get('method')}")
    
    # Cleanup
    await grace_node.stop()
    
    print("\n" + "=" * 80)
    print("✅ Complete System Example Finished Successfully!")
    print("=" * 80)
    
    print("\n📚 Summary of Demonstrated Features:")
    print("  ✓ Multi-node swarm coordination")
    print("  ✓ Quantum-inspired reasoning and optimization")
    print("  ✓ Scientific hypothesis generation")
    print("  ✓ Societal impact evaluation")
    print("  ✓ Governance validation")
    print("  ✓ Unified output generation")
    print("  ✓ Decision synthesis from multiple sources")
    print("  ✓ Full integration of all Grace components")

if __name__ == "__main__":
    asyncio.run(main())
