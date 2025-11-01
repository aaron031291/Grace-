"""
Grace Breakthrough System - Integration & Bootstrap

This module wires together all breakthrough components:
1. Evaluation Harness
2. Meta-Loop Optimizer
3. Disagreement-Aware Consensus
4. Trace Collection

This is the main entry point for activating recursive self-improvement.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .evaluation_harness import EvaluationHarness
from .meta_loop import MetaLoopOptimizer
from .trace_collection import TraceCollector
from ..mldl.disagreement_consensus import DisagreementAwareConsensus

logger = logging.getLogger(__name__)


class BreakthroughSystem:
    """
    The complete breakthrough system.
    
    Integrates all components for recursive self-improvement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        logger.info("🚀 Initializing Grace Breakthrough System...")
        
        # Initialize components
        self.trace_collector = TraceCollector()
        self.evaluation_harness = EvaluationHarness()
        self.consensus_engine = DisagreementAwareConsensus(
            disagreement_threshold=self.config.get("disagreement_threshold", 0.3)
        )
        self.meta_loop = MetaLoopOptimizer(
            evaluation_harness=self.evaluation_harness,
            improvement_threshold=self.config.get("improvement_threshold", 0.02)
        )
        
        self.initialized = False
        self.running = False
        
        logger.info("✅ Breakthrough System components initialized")
    
    async def initialize(self, baseline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system with a baseline configuration.
        
        This evaluates the starting point before any improvements.
        """
        logger.info("📊 Establishing baseline performance...")
        
        await self.meta_loop.initialize_baseline(baseline_config)
        
        self.initialized = True
        logger.info("✅ Baseline established. System ready for improvement!")
        
        # Print baseline stats
        baseline_score = self.meta_loop.baseline_score
        logger.info(f"   Baseline reward: {baseline_score:.4f}")
    
    async def run_single_improvement_cycle(self) -> Dict[str, Any]:
        """
        Run a single improvement cycle.
        
        Returns:
            Result dictionary with improvement details
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info("\n" + "="*70)
        logger.info("Starting Improvement Cycle")
        logger.info("="*70 + "\n")
        
        # Run meta-loop cycle
        result = await self.meta_loop.improvement_cycle()
        
        # Summarize
        summary = {
            "cycle_complete": True,
            "status": result.status.value,
            "improvement": result.improvement,
            "confidence": result.confidence,
            "deployed": result.status.value == "deployed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("\n" + "="*70)
        logger.info(f"Cycle Complete: {result.status.value}")
        logger.info(f"Improvement: {result.improvement:+.4f}")
        logger.info("="*70 + "\n")
        
        return summary
    
    async def run_continuous_improvement(
        self,
        interval_hours: float = 24.0,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous improvement loop.
        
        This is the "set it and forget it" mode - Grace continuously
        improves herself in the background.
        
        Args:
            interval_hours: Hours between improvement cycles
            max_iterations: Max cycles to run (None = infinite)
        """
        if not self.initialized:
            await self.initialize()
        
        self.running = True
        
        logger.info("🔄 Starting Continuous Improvement Mode")
        logger.info(f"   Interval: {interval_hours} hours")
        logger.info(f"   Max iterations: {max_iterations or 'infinite'}")
        logger.info("")
        
        try:
            await self.meta_loop.continuous_improvement(
                interval_hours=interval_hours,
                max_iterations=max_iterations
            )
        finally:
            self.running = False
    
    def stop(self):
        """Stop continuous improvement"""
        logger.info("⏹️  Stopping continuous improvement...")
        self.meta_loop.stop()
        self.running = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        
        # Meta-loop stats
        improvement_summary = self.meta_loop.get_improvement_summary()
        
        # Consensus stats
        consensus_stats = self.consensus_engine.get_stats()
        
        # Trace stats
        trace_analysis = self.trace_collector.analyze_traces(
            self.trace_collector.get_recent_traces(100)
        )
        
        return {
            "initialized": self.initialized,
            "running": self.running,
            "improvement": improvement_summary,
            "consensus": consensus_stats,
            "traces": trace_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def print_status(self):
        """Print human-readable status"""
        status = self.get_system_status()
        
        print("\n" + "="*70)
        print("GRACE BREAKTHROUGH SYSTEM STATUS")
        print("="*70)
        
        print(f"\n🔧 System State:")
        print(f"   Initialized: {status['initialized']}")
        print(f"   Running: {status['running']}")
        
        if status['initialized']:
            print(f"\n📈 Improvement Progress:")
            imp = status['improvement']
            print(f"   Candidates Generated: {imp['total_candidates_generated']}")
            print(f"   Successful Deployments: {imp['total_deployments']}")
            print(f"   Current Reward: {imp['current_reward']:.4f}")
            print(f"   Total Improvement: {imp['total_improvement']:+.4f} ({imp['improvement_percentage']:+.1f}%)")
            
            print(f"\n🧠 Consensus Intelligence:")
            cons = status['consensus']
            if cons.get('total_consensus', 0) > 0:
                print(f"   Total Decisions: {cons['total_consensus']}")
                print(f"   Verification Triggered: {cons['verification_triggered']} ({cons['verification_rate']:.1%})")
                print(f"   Avg Confidence: {cons['avg_confidence']:.3f}")
                print(f"   Avg Agreement: {cons['avg_agreement']:.3f}")
            else:
                print(f"   No consensus decisions yet")
            
            print(f"\n📝 Execution Traces:")
            traces = status['traces']
            if traces.get('total_traces', 0) > 0:
                print(f"   Total Traces: {traces['total_traces']}")
                print(f"   Success Rate: {traces['success_rate']:.1%}")
                print(f"   Avg Duration: {traces['avg_duration_ms']:.1f}ms")
            else:
                print(f"   No traces collected yet")
        
        print("\n" + "="*70 + "\n")


# Convenience function for quick start
async def quick_start_breakthrough(
    baseline_config: Optional[Dict[str, Any]] = None,
    num_cycles: int = 3
) -> BreakthroughSystem:
    """
    Quick start the breakthrough system and run a few cycles.
    
    Perfect for testing and demos.
    
    Args:
        baseline_config: Starting configuration
        num_cycles: Number of improvement cycles to run
    
    Returns:
        Initialized BreakthroughSystem
    """
    print("\n" + "🚀 "*25)
    print("GRACE BREAKTHROUGH SYSTEM - QUICK START")
    print("🚀 "*25 + "\n")
    
    # Create system
    system = BreakthroughSystem()
    
    # Initialize
    await system.initialize(baseline_config)
    
    # Run cycles
    print(f"\n🔄 Running {num_cycles} improvement cycles...\n")
    
    for i in range(num_cycles):
        print(f"\n{'─'*70}")
        print(f"CYCLE {i+1}/{num_cycles}")
        print(f"{'─'*70}\n")
        
        result = await system.run_single_improvement_cycle()
        
        if result['deployed']:
            print("✅ IMPROVEMENT DEPLOYED!")
        else:
            print(f"❌ Not deployed: {result['status']}")
        
        await asyncio.sleep(0.5)  # Small delay for readability
    
    # Show final status
    system.print_status()
    
    return system


if __name__ == "__main__":
    # Demo: Quick start with 3 improvement cycles
    async def demo():
        system = await quick_start_breakthrough(num_cycles=3)
        
        print("\n✨ Demo complete!")
        print("\nTo run continuous improvement:")
        print("  await system.run_continuous_improvement(interval_hours=24)")
        print("\nTo check status:")
        print("  system.print_status()")
    
    asyncio.run(demo())
