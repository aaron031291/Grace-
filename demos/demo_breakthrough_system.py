#!/usr/bin/env python3
"""
Grace Breakthrough System - Demo

This demonstrates the complete breakthrough system:
- Evaluation with objective scoring
- Recursive self-improvement via meta-loop
- Disagreement-aware consensus
- Trace collection and learning

Run this to see Grace evolve!
"""

import asyncio
import sys
from pathlib import Path

# Add grace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace.core.breakthrough import BreakthroughSystem, quick_start_breakthrough


async def demo_basic():
    """Basic demo: Run 3 improvement cycles"""
    print("\n" + "="*80)
    print("DEMO 1: Basic Improvement Cycles")
    print("="*80 + "\n")
    
    system = await quick_start_breakthrough(num_cycles=3)
    
    print("\n✅ Basic demo complete!")
    return system


async def demo_continuous():
    """Demo continuous improvement (short duration for demo)"""
    print("\n" + "="*80)
    print("DEMO 2: Continuous Improvement")
    print("="*80 + "\n")
    
    system = BreakthroughSystem()
    await system.initialize()
    
    print("Running continuous improvement for 3 cycles...")
    print("(In production, this would run 24/7)\n")
    
    # Run for 3 iterations with short intervals
    await system.run_continuous_improvement(
        interval_hours=0.01,  # Very short for demo (36 seconds)
        max_iterations=3
    )
    
    # Show results
    system.print_status()
    
    print("\n✅ Continuous improvement demo complete!")
    return system


async def demo_consensus():
    """Demo disagreement-aware consensus"""
    print("\n" + "="*80)
    print("DEMO 3: Disagreement-Aware Consensus")
    print("="*80 + "\n")
    
    from grace.mldl.disagreement_consensus import (
        DisagreementAwareConsensus,
        ModelPrediction
    )
    
    consensus = DisagreementAwareConsensus(disagreement_threshold=0.3)
    
    # Scenario 1: Models agree
    print("📊 Scenario 1: Models Agree")
    predictions_agree = [
        ModelPrediction("model_a", "Paris", 0.92, reasoning="Capital of France"),
        ModelPrediction("model_b", "Paris", 0.89, reasoning="France capital is Paris"),
        ModelPrediction("model_c", "Paris", 0.95, reasoning="Paris is the answer")
    ]
    
    result1 = await consensus.reach_consensus(
        "What is the capital of France?",
        predictions_agree
    )
    
    print(f"  Final prediction: {result1.final_prediction}")
    print(f"  Confidence: {result1.confidence:.3f}")
    print(f"  Method: {result1.method_used.value}")
    print(f"  Agreement: {result1.agreement_score:.3f}")
    print(f"  Verification triggered: {result1.verification_performed}\n")
    
    # Scenario 2: Models disagree - triggers verification!
    print("📊 Scenario 2: Models Disagree → VERIFICATION BRANCH!")
    predictions_disagree = [
        ModelPrediction("model_a", "42", 0.7, reasoning="Looks like 42"),
        ModelPrediction("model_b", "56", 0.8, reasoning="7*8 = 56"),
        ModelPrediction("model_c", "48", 0.6, reasoning="Maybe 48?")
    ]
    
    result2 = await consensus.reach_consensus(
        "What is 7 * 8?",
        predictions_disagree
    )
    
    print(f"  Final prediction: {result2.final_prediction}")
    print(f"  Confidence: {result2.confidence:.3f}")
    print(f"  Method: {result2.method_used.value}")
    print(f"  Agreement: {result2.agreement_score:.3f}")
    print(f"  ✨ Verification triggered: {result2.verification_performed}")
    
    if result2.verification_performed:
        print(f"  Verification details:")
        print(f"    - Tool checks: {len(result2.verification_details.get('tool_verifications', []))}")
        print(f"    - Cross-critiques: {len(result2.verification_details.get('critiques', []))}")
    
    print("\n✅ Consensus demo complete!")


async def demo_trace_collection():
    """Demo trace collection and analysis"""
    print("\n" + "="*80)
    print("DEMO 4: Trace Collection & Analysis")
    print("="*80 + "\n")
    
    from grace.core.trace_collection import TraceCollector, TraceEventType
    
    collector = TraceCollector()
    
    print("📝 Collecting traces from 5 tasks...\n")
    
    # Simulate 5 task executions
    for i in range(5):
        trace_id = f"demo_trace_{i+1}"
        success = i % 3 != 0  # Some succeed, some fail
        
        # Start trace
        collector.start_trace(
            trace_id,
            f"Demo task {i+1}",
            {"task_type": "demo", "index": i+1}
        )
        
        # Log some events
        collector.log_event(
            trace_id,
            TraceEventType.MODEL_INFERENCE,
            {"model": "gpt-4", "tokens": 100},
            duration_ms=150 + i * 20
        )
        
        if not success:
            collector.log_event(
                trace_id,
                TraceEventType.ERROR,
                {"error": "TimeoutError", "message": "Task timed out"}
            )
        
        # End trace
        collector.end_trace(
            trace_id,
            success=success,
            final_output=f"Result {i+1}" if success else None,
            error="TimeoutError" if not success else None
        )
        
        status_symbol = "✅" if success else "❌"
        print(f"  {status_symbol} Task {i+1}: {'Success' if success else 'Failed'}")
    
    # Analyze
    print("\n📊 Trace Analysis:")
    analysis = collector.analyze_traces()
    print(f"  Total traces: {analysis['total_traces']}")
    print(f"  Successful: {analysis['successful']}")
    print(f"  Failed: {analysis['failed']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    print(f"  Avg duration: {analysis['avg_duration_ms']:.1f}ms")
    
    if analysis['common_errors']:
        print(f"\n  Common errors:")
        for error, count in analysis['common_errors']:
            print(f"    - {error}: {count} occurrences")
    
    # Show recent failures
    failures = collector.get_failure_traces(count=2)
    print(f"\n📌 Recent Failures (for learning):")
    for trace in failures:
        print(f"  - {trace.task_description}")
        print(f"    Error: {trace.error}")
        print(f"    Duration: {trace.total_duration_ms():.1f}ms")
    
    print("\n✅ Trace collection demo complete!")


async def full_demo():
    """Run all demos"""
    print("\n" + "🌟 "*30)
    print("GRACE BREAKTHROUGH SYSTEM - COMPLETE DEMONSTRATION")
    print("🌟 "*30 + "\n")
    
    print("This demo shows Grace's breakthrough capabilities:")
    print("  1. Objective evaluation with canonical tasks")
    print("  2. Recursive self-improvement via meta-loop")
    print("  3. Intelligent consensus with verification branching")
    print("  4. Complete execution tracing for learning")
    print("\nPress Enter to continue...")
    input()
    
    # Run each demo
    await demo_basic()
    
    print("\n\nPress Enter for next demo...")
    input()
    await demo_consensus()
    
    print("\n\nPress Enter for next demo...")
    input()
    await demo_trace_collection()
    
    print("\n\nPress Enter for final demo...")
    input()
    await demo_continuous()
    
    print("\n" + "🎉 "*30)
    print("ALL DEMOS COMPLETE!")
    print("🎉 "*30)
    
    print("\n📚 What you just saw:")
    print("  ✅ Grace evaluated herself objectively")
    print("  ✅ Grace generated improvement candidates")
    print("  ✅ Grace tested candidates in sandbox")
    print("  ✅ Grace deployed improvements automatically")
    print("  ✅ Grace investigated when models disagreed")
    print("  ✅ Grace collected traces for learning")
    
    print("\n🚀 This is recursive self-improvement in action!")
    print("\nGrace is now capable of:")
    print("  - Measuring her own performance")
    print("  - Identifying what works and what doesn't")
    print("  - Generating bounded improvements")
    print("  - Testing safely in sandbox")
    print("  - Deploying validated improvements")
    print("  - Learning from every execution")
    print("  - Evolving continuously 24/7")
    
    print("\n" + "="*80)
    print("THE BREAKTHROUGH IS COMPLETE")
    print("="*80 + "\n")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grace Breakthrough System Demo")
    parser.add_argument(
        "--demo",
        choices=["basic", "continuous", "consensus", "traces", "all"],
        default="all",
        help="Which demo to run"
    )
    
    args = parser.parse_args()
    
    demos = {
        "basic": demo_basic,
        "continuous": demo_continuous,
        "consensus": demo_consensus,
        "traces": demo_trace_collection,
        "all": full_demo
    }
    
    await demos[args.demo]()


if __name__ == "__main__":
    asyncio.run(main())
