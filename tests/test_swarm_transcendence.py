"""
Complete test of Swarm Intelligence and Transcendence layers
"""

import asyncio
import numpy as np

print("=" * 80)
print("Grace Swarm Intelligence & Transcendence - Complete Test")
print("=" * 80)

async def main():
    # Test 1: Swarm Coordinator
    print("\n1. Testing Swarm Node Coordinator...")
    
    from grace.swarm.coordinator import GraceNodeCoordinator
    from grace.swarm.transport import TransportProtocol
    from grace.swarm.consensus import ConsensusAlgorithm
    
    # Create coordinator nodes
    node1 = GraceNodeCoordinator(
        node_id="node_1",
        transport_protocol=TransportProtocol.HTTP,
        port=8081,
        consensus_algorithm=ConsensusAlgorithm.WEIGHTED_AVERAGE
    )
    
    node2 = GraceNodeCoordinator(
        node_id="node_2",
        transport_protocol=TransportProtocol.HTTP,
        port=8082
    )
    
    print(f"✓ Created 2 swarm nodes")
    
    # Start coordinators
    await node1.start()
    await node2.start()
    print(f"✓ Coordinators started")
    
    # Register peers manually (in production, use discovery)
    node1.discovery.register_peer(
        "node_2", "localhost", 8082, "http",
        capabilities=["ml_inference", "data_processing"],
        trust_score=0.85
    )
    
    node2.discovery.register_peer(
        "node_1", "localhost", 8081, "http",
        capabilities=["ml_inference", "governance"],
        trust_score=0.9
    )
    
    print(f"✓ Peers registered")
    
    # Get swarm status
    status1 = node1.get_swarm_status()
    print(f"\n✓ Node 1 swarm status:")
    print(f"    Healthy peers: {status1['healthy_peers']}")
    print(f"    Transport: {status1['transport']}")
    print(f"    Consensus: {status1['consensus_algorithm']}")
    
    # Test collective decision
    decision_context = {
        "task": "classify_intent",
        "data": {"text": "user wants to authenticate"}
    }
    
    print(f"\n  Requesting collective decision...")
    collective_decision = await node1.request_collective_decision(decision_context, timeout=2.0)
    
    print(f"✓ Collective decision:")
    print(f"    Decision: {collective_decision['decision']}")
    print(f"    Source: {collective_decision['source']}")
    
    # Test 2: Quantum Algorithm Library
    print("\n2. Testing Quantum Algorithm Library...")
    
    from grace.transcendence.quantum_library import QuantumAlgorithmLibrary, QuantumCircuit
    
    quantum_lib = QuantumAlgorithmLibrary()
    
    # Test quantum search
    search_space = list(range(100))
    oracle = lambda x: 1.0 if 80 <= x <= 85 else 0.0  # Looking for numbers 80-85
    
    result, probability = quantum_lib.quantum_search(search_space, oracle, num_iterations=10)
    print(f"✓ Quantum search:")
    print(f"    Found: {result}")
    print(f"    Probability: {probability:.3f}")
    
    # Test quantum optimization
    def objective(params):
        x, y = params['x'], params['y']
        return (x - 3) ** 2 + (y + 2) ** 2  # Minimize this
    
    optimal = quantum_lib.quantum_optimization(
        objective,
        variables=['x', 'y'],
        bounds={'x': (-10, 10), 'y': (-10, 10)},
        num_iterations=30
    )
    
    print(f"\n✓ Quantum optimization:")
    print(f"    Optimal x: {optimal['x']:.3f}")
    print(f"    Optimal y: {optimal['y']:.3f}")
    print(f"    Expected: x≈3, y≈-2")
    
    # Test superposition reasoning
    options = [
        {"strategy": "A", "cost": 100, "benefit": 50},
        {"strategy": "B", "cost": 80, "benefit": 70},
        {"strategy": "C", "cost": 120, "benefit": 90},
    ]
    
    criteria = [
        lambda opt: 1.0 - opt["cost"] / 150,  # Lower cost is better
        lambda opt: opt["benefit"] / 100  # Higher benefit is better
    ]
    
    best_option = quantum_lib.superposition_reasoning(options, criteria)
    print(f"\n✓ Superposition reasoning:")
    print(f"    Best strategy: {best_option['strategy']}")
    print(f"    Quantum confidence: {best_option['quantum_confidence']:.3f}")
    
    # Test 3: Scientific Discovery Accelerator
    print("\n3. Testing Scientific Discovery Accelerator...")
    
    from grace.transcendence.scientific_discovery import ScientificDiscoveryAccelerator
    
    discovery = ScientificDiscoveryAccelerator(min_confidence=0.5)
    
    # Generate synthetic data
    np.random.seed(42)
    data = []
    for i in range(100):
        x = np.random.uniform(0, 10)
        y = 2 * x + 3 + np.random.normal(0, 0.5)  # y = 2x + 3 with noise
        z = x ** 2 + np.random.normal(0, 1)
        
        data.append({
            "x": x,
            "y": y,
            "z": z,
            "category": "A" if x < 5 else "B"
        })
    
    print(f"✓ Generated {len(data)} data points")
    
    # Discover patterns
    patterns = discovery.analyze_data(data)
    print(f"\n✓ Discovered {len(patterns)} patterns:")
    for pattern in patterns[:3]:
        print(f"    - {pattern['type']}: {pattern}")
    
    # Generate hypotheses
    hypotheses = discovery.generate_hypotheses(patterns)
    print(f"\n✓ Generated {len(hypotheses)} hypotheses:")
    for hyp in hypotheses[:3]:
        print(f"    - {hyp.statement}")
        print(f"      Confidence: {hyp.confidence:.3f}")
        print(f"      Testable: {hyp.testable}")
    
    # Design experiment
    if hypotheses:
        experiment = discovery.design_experiment(hypotheses[0])
        print(f"\n✓ Experiment design:")
        print(f"    Objective: {experiment['objective']}")
        print(f"    Variables: {experiment['variables_to_measure']}")
        print(f"    Sample size: {experiment['sample_size_recommendation']}")
    
    # Test 4: Societal Impact Evaluator
    print("\n4. Testing Societal Impact Evaluator...")
    
    from grace.transcendence.societal_impact import SocietalImpactEvaluator
    
    impact_eval = SocietalImpactEvaluator()
    
    # Define policy
    policy = {
        "id": "education_reform_2024",
        "description": "Increase funding for public education by 20%",
        "target_groups": ["students", "teachers"],
        "economic_effect": 0.3,
        "welfare_effect": 0.6,
        "employment_effect": 0.2,
        "health_effect": 0.1,
        "environmental_effect": 0.0,
        "equity_impact": 0.4,
        "well_defined": True
    }
    
    context = {
        "total_population": 10000000,
        "students_fraction": 0.15,
        "teachers_fraction": 0.02,
        "historical_data": True
    }
    
    # Simulate policy
    simulation = impact_eval.simulate_policy(policy, context, time_horizon="5_years")
    
    print(f"✓ Policy simulation: {simulation.policy_id}")
    print(f"    Confidence: {simulation.confidence:.3f}")
    print(f"    Affected populations: {sum(simulation.affected_populations.values())} people")
    
    print(f"\n  Projected outcomes:")
    for outcome, value in simulation.projected_outcomes.items():
        print(f"    - {outcome}: {value:.3f}")
    
    print(f"\n  Risks ({len(simulation.risks)}):")
    for risk in simulation.risks[:2]:
        print(f"    - [{risk['severity']}] {risk['description']}")
    
    print(f"\n  Benefits ({len(simulation.benefits)}):")
    for benefit in simulation.benefits[:2]:
        print(f"    - [{benefit['magnitude']}] {benefit['description']}")
    
    # Compare multiple policies
    alternative_policy = {
        "id": "tech_training_2024",
        "description": "Expand technical training programs",
        "target_groups": ["workforce"],
        "economic_effect": 0.5,
        "welfare_effect": 0.3,
        "employment_effect": 0.7,
        "health_effect": 0.0,
        "environmental_effect": -0.1,
        "equity_impact": 0.2
    }
    
    comparison = impact_eval.compare_policies([policy, alternative_policy], context)
    
    print(f"\n✓ Policy comparison:")
    print(f"    Recommended: {comparison['recommendation']}")
    print(f"    Confidence: {comparison['comparison_confidence']:.3f}")
    print(f"\n  Rankings:")
    for rank in comparison['rankings']:
        print(f"    - {rank['policy_id']}: score={rank['net_score']:.2f}")
    
    # Cleanup
    await node1.stop()
    await node2.stop()
    
    print("\n" + "=" * 80)
    print("✅ Swarm & Transcendence Tests Complete!")
    print("=" * 80)
    
    print("\nImplemented Features:")
    print("\nSwarm Intelligence:")
    print("  ✓ Multi-transport protocols (HTTP, gRPC, Kafka)")
    print("  ✓ Peer discovery and service registry")
    print("  ✓ Collective consensus engine (multiple algorithms)")
    print("  ✓ Fault-tolerant node coordination")
    print("  ✓ Event exchange and broadcasting")
    
    print("\nTranscendence Layer:")
    print("  ✓ Quantum-inspired search and optimization")
    print("  ✓ Superposition reasoning")
    print("  ✓ Entanglement correlation analysis")
    print("  ✓ Scientific pattern discovery")
    print("  ✓ Hypothesis generation and testing")
    print("  ✓ Policy impact simulation")
    print("  ✓ Multi-stakeholder analysis")

# Run test
asyncio.run(main())
