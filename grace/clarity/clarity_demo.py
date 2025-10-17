"""
Clarity Framework Demo - Demonstrates all Classes 5-10
"""

import logging
from grace.core.grace_core_runtime import GraceCoreRuntime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate Clarity Framework integration"""
    
    logger.info("=== Clarity Framework Demonstration ===\n")
    
    # Initialize Grace Core Runtime
    runtime = GraceCoreRuntime()
    
    # Execute multiple loop iterations
    tasks = [
        {
            'type': 'ethical_decision',
            'description': 'Evaluate AI deployment in healthcare',
            'context': {'domain': 'healthcare', 'stakeholders': 5}
        },
        {
            'type': 'optimization',
            'description': 'Optimize resource allocation',
            'context': {'resources': 100, 'constraints': 10}
        },
        {
            'type': 'analysis',
            'description': 'Analyze system performance',
            'context': {'metrics': ['latency', 'throughput', 'accuracy']}
        }
    ]
    
    for i, task in enumerate(tasks, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i}: {task['description']}")
        logger.info(f"{'='*60}\n")
        
        # Execute loop
        output = runtime.execute_loop(task, task.get('context', {}))
        
        # Display formatted output
        print(runtime.output_formatter.format_for_display(output))
        print()
    
    # Display runtime status
    logger.info(f"\n{'='*60}")
    logger.info("Runtime Status Summary")
    logger.info(f"{'='*60}\n")
    
    status = runtime.get_runtime_status()
    
    print(f"Loop ID: {status['loop_id']}")
    print(f"Total Iterations: {status['iteration']}")
    print(f"\nMemory Statistics:")
    print(f"  Total Memories: {status['memory_stats']['total_memories']}")
    print(f"  Avg Clarity: {status['memory_stats'].get('avg_clarity', 0):.2%}")
    print(f"  Avg Ambiguity: {status['memory_stats'].get('avg_ambiguity', 0):.2%}")
    
    print(f"\nConstitution Compliance:")
    print(f"  Total Violations: {status['constitution_stats']['total_violations']}")
    
    print(f"\nFeedback Integration:")
    print(f"  Total Feedback: {status['feedback_stats']['total_feedback']}")
    print(f"  Avg Impact: {status['feedback_stats'].get('avg_impact', 0):.2f}")
    
    print(f"\nQuorum Consensus:")
    print(f"  Total Evaluations: {status['quorum_stats']['total_evaluations']}")
    print(f"  Approval Rate: {status['quorum_stats'].get('approval_rate', 0):.2%}")
    print(f"  Avg Consensus: {status['quorum_stats'].get('avg_consensus_strength', 0):.2%}")
    
    print(f"\nDrift Detection:")
    print(f"  Total Alerts: {status['drift_report']['total_alerts']}")
    print(f"  Contradictions: {status['drift_report']['contradictions']}")
    print(f"  Loop Health: {status['loop_health']['status']}")
    
    logger.info("\n=== Clarity Framework Demo Complete ===")


if __name__ == "__main__":
    main()
