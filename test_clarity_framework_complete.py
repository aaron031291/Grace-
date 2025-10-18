"""
Complete test of Clarity Framework Classes 5-10
"""

import asyncio
from datetime import datetime, timezone

print("=" * 80)
print("Grace Clarity Framework - Complete Test (Classes 5-10)")
print("=" * 80)

async def main():
    from grace.clarity.memory_bank import LoopMemoryBank
    from grace.clarity.governance_validator import GovernanceValidator
    from grace.clarity.feedback_integrator import FeedbackIntegrator
    from grace.clarity.specialist_consensus import SpecialistConsensus
    from grace.clarity.unified_output import UnifiedOutputGenerator, ReasoningStep
    from grace.clarity.drift_detector import DriftDetector
    from grace.mldl.quorum_aggregator import SpecialistOutput
    
    # Test 1: Memory Scoring & Trust (Class 5)
    print("\n1. Testing Loop Memory Bank...")
    
    memory_bank = LoopMemoryBank(
        min_confidence=0.6,
        min_relevance=0.5
    )
    
    # Add memory fragments
    memory_bank.add_fragment(
        "mem_auth_1",
        {"pattern": "user_auth", "result": "success"},
        source="auth_system",
        confidence=0.9,
        metadata={"category": "authentication"}
    )
    
    memory_bank.add_fragment(
        "mem_db_1",
        {"pattern": "db_query", "latency": 50},
        source="database",
        confidence=0.8
    )
    
    memory_bank.add_fragment(
        "mem_model_1",
        {"prediction": 0.75, "accuracy": 0.85},
        source="ml_model",
        confidence=0.7
    )
    
    print(f"✓ Added 3 memory fragments")
    
    # Retrieve memories with context
    query_context = {"query": "authentication patterns", "task": "analyze"}
    memories = memory_bank.retrieve_memories(query_context, max_results=5)
    
    print(f"✓ Retrieved {len(memories)} memories:")
    for mem, score in memories[:3]:
        print(f"    - {mem.fragment_id}: score={score:.3f}, conf={mem.confidence:.3f}")
    
    # Update consensus
    memory_bank.update_consensus("mem_auth_1", agreement=True)
    memory_bank.update_consensus("mem_auth_1", agreement=True)
    
    stats = memory_bank.get_statistics()
    print(f"✓ Memory bank stats:")
    print(f"    Total fragments: {stats['total_fragments']}")
    print(f"    Avg confidence: {stats['avg_confidence']:.3f}")
    print(f"    Avg trust: {stats['avg_trust_score']:.3f}")
    
    # Test 2: Governance Validation (Class 6)
    print("\n2. Testing Governance Validator...")
    
    validator = GovernanceValidator()
    
    # Test decision without reasoning (should trigger amendment)
    decision_no_reasoning = {
        "action": "grant_access",
        "user_id": "user123",
        "resource": "sensitive_data"
    }
    
    result = validator.validate_decision(
        decision_no_reasoning,
        context={"user_consent": False}
    )
    
    print(f"✓ Validation result:")
    print(f"    Passed: {result.passed}")
    print(f"    Violations: {len(result.violations)}")
    print(f"    Amendments: {len(result.amendments)}")
    print(f"    Confidence: {result.confidence:.3f}")
    
    if result.violations:
        for violation in result.violations[:2]:
            print(f"    - {violation}")
    
    # Test 3: Feedback Integration (Class 7)
    print("\n3. Testing Feedback Integrator...")
    
    feedback_integrator = FeedbackIntegrator()
    
    # Record various feedback types
    feedback_integrator.record_feedback(
        loop_id="loop_1",
        decision_id="dec_1",
        feedback_type="human_correction",
        rating=0.8,
        corrections={"accuracy": "high"},
        reviewer_id="reviewer_1",
        metadata={"components": ["ml_model", "auth_system"]}
    )
    
    feedback_integrator.record_feedback(
        loop_id="loop_1",
        decision_id="dec_1",
        feedback_type="reviewer_vote",
        rating=0.6,
        metadata={"components": ["ml_model"]}
    )
    
    feedback_integrator.record_feedback(
        loop_id="loop_2",
        decision_id="dec_2",
        feedback_type="auto_eval",
        rating=-0.3,
        metadata={"components": ["database"]}
    )
    
    print(f"✓ Recorded 3 feedback entries")
    
    aggregated = feedback_integrator.get_aggregated_feedback()
    print(f"✓ Aggregated feedback:")
    print(f"    Total: {aggregated['total']}")
    print(f"    Avg rating: {aggregated['avg_rating']:.3f}")
    print(f"    Positive: {aggregated['positive']}, Negative: {aggregated['negative']}")
    print(f"    By type: {list(aggregated['by_type'].keys())}")
    
    # Test 4: Specialist Consensus (Class 8)
    print("\n4. Testing Specialist Consensus...")
    
    consensus_gen = SpecialistConsensus()
    
    # Create specialist outputs
    specialist_outputs = [
        SpecialistOutput(
            specialist_id="lstm_model",
            specialist_type="time_series",
            prediction=0.78,
            confidence=0.9,
            uncertainty={"lower": 0.72, "upper": 0.84}
        ),
        SpecialistOutput(
            specialist_id="transformer_model",
            specialist_type="nlp",
            prediction=0.75,
            confidence=0.85,
            uncertainty={"lower": 0.68, "upper": 0.82}
        ),
        SpecialistOutput(
            specialist_id="rf_model",
            specialist_type="tabular",
            prediction=0.80,
            confidence=0.8,
            uncertainty={"lower": 0.73, "upper": 0.87}
        )
    ]
    
    context = {"task": "prediction", "timestamp": datetime.now(timezone.utc).isoformat()}
    consensus = consensus_gen.generate_consensus(specialist_outputs, context)
    
    print(f"✓ Generated consensus:")
    print(f"    ID: {consensus['consensus_id']}")
    print(f"    Prediction: {consensus['prediction']:.3f}")
    print(f"    Confidence: {consensus['confidence']:.3f}")
    print(f"    Agreement: {consensus['agreement']:.3f}")
    print(f"    Method: {consensus['method']}")
    print(f"    Specialists: {len(consensus['specialists'])}")
    
    # Feed to unified logic
    unified_input = consensus_gen.feed_to_unified_logic(consensus)
    print(f"✓ Prepared for Unified Logic:")
    print(f"    Keys: {list(unified_input.keys())}")
    
    # Test 5: Unified Output (Class 9)
    print("\n5. Testing Unified Output Generator...")
    
    output_gen = UnifiedOutputGenerator()
    
    # Generate unified output
    reasoning_steps = [
        {
            "description": "Analyzed input data",
            "input": {"data": "raw"},
            "output": {"processed": True},
            "confidence": 0.9,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "description": "Applied ML models",
            "input": {"processed": True},
            "output": {"prediction": 0.75},
            "confidence": 0.85,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "description": "Validated with governance",
            "input": {"prediction": 0.75},
            "output": {"approved": True},
            "confidence": 0.95,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    trust_metrics = {
        "overall_trust": 0.85,
        "component_trust": {"ml_model": 0.9, "governance": 0.95},
        "consensus_confidence": 0.87,
        "governance_passed": True,
        "memory_quality": 0.8,
        "feedback_score": 0.7
    }
    
    unified_output = output_gen.generate_output(
        loop_id="main_loop",
        input_data={"query": "predict user intent", "context": {"user_id": "123"}},
        decision={"action": "recommend", "items": [1, 2, 3]},
        reasoning_steps=reasoning_steps,
        trust_metrics=trust_metrics,
        consensus_data=consensus,
        governance_result=result.to_dict() if hasattr(result, 'to_dict') else {},
        metadata={"version": "1.0"}
    )
    
    print(f"✓ Generated unified output:")
    print(f"    ID: {unified_output.output_id}")
    print(f"    Loop: {unified_output.loop_id}")
    print(f"    Decision type: {unified_output.decision_type}")
    print(f"    Reasoning steps: {len(unified_output.reasoning_chain)}")
    print(f"    Overall trust: {unified_output.trust_metrics.overall_trust:.3f}")
    print(f"    Consensus ID: {unified_output.consensus_id}")
    
    print(f"\n✓ Explanation preview:")
    print(f"    {unified_output.explanation[:150]}...")
    
    # Test 6: Drift Detection (Class 10)
    print("\n6. Testing Loop Drift Detector...")
    
    drift_detector = DriftDetector(drift_threshold=0.3)
    
    # Create historical outputs
    historical_outputs = []
    for i in range(15):
        hist_output = output_gen.generate_output(
            loop_id="main_loop",
            input_data={"query": f"query_{i}"},
            decision={"action": "recommend", "score": 0.75 + (i * 0.01)},
            reasoning_steps=[
                {"description": f"Step {j}", "input": {}, "output": {}, "confidence": 0.8}
                for j in range(3)
            ],
            trust_metrics={
                "overall_trust": 0.8 + (i * 0.005),
                "component_trust": {},
                "consensus_confidence": 0.85,
                "governance_passed": True,
                "memory_quality": 0.8
            }
        )
        historical_outputs.append(hist_output)
    
    print(f"✓ Created {len(historical_outputs)} historical outputs")
    
    # Test normal output (no drift)
    normal_output = output_gen.generate_output(
        loop_id="main_loop",
        input_data={"query": "normal"},
        decision={"action": "recommend", "score": 0.78},
        reasoning_steps=[
            {"description": f"Step {j}", "input": {}, "output": {}, "confidence": 0.8}
            for j in range(3)
        ],
        trust_metrics={
            "overall_trust": 0.82,
            "component_trust": {},
            "consensus_confidence": 0.85,
            "governance_passed": True,
            "memory_quality": 0.8
        }
    )
    
    drift_result_normal = drift_detector.detect_drift(normal_output, historical_outputs)
    print(f"\n✓ Normal output drift check:")
    print(f"    Drift detected: {drift_result_normal['drift_detected']}")
    print(f"    Drift score: {drift_result_normal['drift_score']:.3f}")
    print(f"    Requires review: {drift_result_normal['requires_review']}")
    
    # Test anomalous output (drift)
    anomalous_output = output_gen.generate_output(
        loop_id="main_loop",
        input_data={"query": "anomalous"},
        decision={"action": "block", "score": 0.3},  # Different action, low score
        reasoning_steps=[
            {"description": f"Step {j}", "input": {}, "output": {}, "confidence": 0.5}
            for j in range(10)  # More steps than usual
        ],
        trust_metrics={
            "overall_trust": 0.4,  # Much lower
            "component_trust": {},
            "consensus_confidence": 0.5,
            "governance_passed": False,
            "memory_quality": 0.5
        }
    )
    
    drift_result_anomalous = drift_detector.detect_drift(anomalous_output, historical_outputs)
    print(f"\n✓ Anomalous output drift check:")
    print(f"    Drift detected: {drift_result_anomalous['drift_detected']}")
    print(f"    Drift score: {drift_result_anomalous['drift_score']:.3f}")
    print(f"    Requires review: {drift_result_anomalous['requires_review']}")
    print(f"    Anomaly details: {drift_result_anomalous['anomaly_details']}")
    
    # Get drift statistics
    drift_stats = drift_detector.get_drift_statistics()
    print(f"\n✓ Drift statistics:")
    print(f"    Total alerts: {drift_stats['total_alerts']}")
    print(f"    Drift rate: {drift_stats['drift_rate']:.3f}")
    print(f"    Review required: {drift_stats['review_required']}")
    
    print("\n" + "=" * 80)
    print("✅ Clarity Framework Complete Test Finished!")
    print("=" * 80)
    
    print("\nImplemented Classes:")
    print("  ✓ Class 5: Memory Scoring & Trust (LoopMemoryBank)")
    print("  ✓ Class 6: Governance Validation (GovernanceValidator)")
    print("  ✓ Class 7: Feedback Integration (FeedbackIntegrator)")
    print("  ✓ Class 8: Specialist Consensus (SpecialistConsensus)")
    print("  ✓ Class 9: Unified Output (GraceLoopOutput)")
    print("  ✓ Class 10: Loop Drift Detection (DriftDetector)")
    
    print("\nKey Features:")
    print("  • Memory fragments with source credibility, recency, consensus")
    print("  • Constitutional constraint checking with auto-amendments")
    print("  • Feedback capture with weight/trust adjustments")
    print("  • Consensus generation with unique IDs")
    print("  • Canonical output format with complete metadata")
    print("  • Statistical drift detection with anomaly flagging")

# Run test
asyncio.run(main())
