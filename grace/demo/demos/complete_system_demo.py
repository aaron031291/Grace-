"""
Complete System Demonstration
Shows all implemented Grace components working together
"""

import logging
import asyncio
from datetime import datetime

# Import all completed components
from grace.core.unified_logic import UnifiedLogic, Decision, DecisionSource
from grace.clarity.memory_scoring import LoopMemoryBank
from grace.clarity.governance_validation import ConstitutionValidator
from grace.mtl.immutable_logs import ImmutableLogs
from grace.mtl.human_readable import HumanReadableFormatter
from grace.trust.trust_score import TrustScoreManager
from grace.swarm import SwarmOrchestrator
from grace.transcendent import TranscendenceOrchestrator
from grace.memory import EnhancedMemoryCore
from grace.integration import EventBus, QuorumIntegration, AVNReporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run complete system demonstration"""
    
    print("=" * 70)
    print("GRACE AI SYSTEM - COMPLETE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize all systems
    print("🚀 Initializing Grace Systems...")
    print()
    
    # Core systems
    event_bus = EventBus()
    avn = AVNReporter()
    quorum = QuorumIntegration()
    
    # Trust and audit
    trust_manager = TrustScoreManager()
    immutable_logs = ImmutableLogs()
    formatter = HumanReadableFormatter()
    
    # Memory and governance
    memory_bank = LoopMemoryBank()
    constitution = ConstitutionValidator()
    
    # Unified Logic
    unified_logic = UnifiedLogic(
        governance_engine=constitution,
        event_bus=event_bus,
        trigger_mesh=None  # Will add later
    )
    
    # Swarm and Transcendence
    swarm = SwarmOrchestrator()
    swarm.connect_event_bus(event_bus)
    swarm.connect_quorum(quorum)
    
    transcendence = TranscendenceOrchestrator()
    
    print("✅ All systems initialized")
    print()
    
    # Demo 1: Trust Score Management
    print("=" * 70)
    print("DEMO 1: Trust Score Management")
    print("=" * 70)
    print()
    
    # Initialize trust for components
    trust_manager.initialize_trust("grace_core", "component", initial_score=0.8)
    trust_manager.initialize_trust("swarm_node_1", "node", initial_score=0.6)
    
    # Record successes
    trust_manager.record_success("grace_core", weight=1.0)
    trust_manager.record_success("grace_core", weight=0.8)
    trust_manager.record_failure("swarm_node_1", severity=0.5)
    
    # Get trust scores
    core_trust = trust_manager.get_trust_score("grace_core")
    print(f"Grace Core Trust: {core_trust.score:.2%} ({core_trust.level.name})")
    print(f"  Successes: {core_trust.successes}, Failures: {core_trust.failures}")
    print()
    
    # Demo 2: Immutable Logs with Human-Readable Output
    print("=" * 70)
    print("DEMO 2: Immutable Audit Logs")
    print("=" * 70)
    print()
    
    # Log constitutional operations
    log_entry = immutable_logs.log_constitutional_operation(
        actor="grace_core",
        action="decision_made",
        data={
            'decision': 'approve_deployment',
            'confidence': 0.95,
            'rationale': 'All safety checks passed'
        },
        constitutional_check=True,
        metadata={'criticality': 'high'}
    )
    
    # Format as human-readable
    readable_log = formatter.format_log_entry({
        'timestamp': log_entry.timestamp,
        'actor': log_entry.actor,
        'action': log_entry.action,
        'constitutional_check': log_entry.constitutional_check,
        'current_hash': log_entry.current_hash,
        'data': log_entry.data
    })
    
    print(readable_log)
    print()
    
    # Verify immutability
    is_immutable = immutable_logs.ensure_audit_immutability()
    print(f"🔒 Chain Immutability: {'✅ VERIFIED' if is_immutable else '❌ COMPROMISED'}")
    print()
    
    # Demo 3: Unified Logic Decision Making
    print("=" * 70)
    print("DEMO 3: Unified Logic with Conflict Resolution")
    print("=" * 70)
    print()
    
    # Create conflicting decisions
    decisions = [
        Decision(
            source=DecisionSource.GOVERNANCE,
            decision=True,
            confidence=0.9,
            rationale="Constitutional compliance verified"
        ),
        Decision(
            source=DecisionSource.SWARM,
            decision=False,
            confidence=0.7,
            rationale="Resource constraints detected"
        )
    ]
    
    # Process through unified logic
    result = await unified_logic.process_decision(
        decisions,
        context={'action': 'deploy_model', 'environment': 'production'}
    )
    
    print(f"Final Decision: {result['decision']}")
    print(f"Source: {result['source']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Rationale: {result['rationale']}")
    print()
    
    # Demo 4: Memory with Clarity Scoring
    print("=" * 70)
    print("DEMO 4: Memory System with Clarity Scoring")
    print("=" * 70)
    print()
    
    # Store memory
    memory_id = "mem_demo_001"
    memory_bank.store(
        memory_id=memory_id,
        memory_type=memory_bank.MemoryType.EPISODIC,
        content={
            'event': 'successful_deployment',
            'confidence': 0.95,
            'outcome': 'positive'
        },
        source="demo",
        metadata={'demo': True}
    )
    
    # Score memory
    scores = memory_bank.score(memory_id, context={'relevant': True})
    
    print(f"Memory Scores:")
    print(f"  Clarity: {scores['clarity']:.2%}")
    print(f"  Relevance: {scores['relevance']:.2%}")
    print(f"  Ambiguity: {scores['ambiguity']:.2%}")
    print(f"  Composite: {scores['composite']:.2%}")
    print()
    
    # Demo 5: Complete System Metrics
    print("=" * 70)
    print("DEMO 5: Complete System Metrics")
    print("=" * 70)
    print()
    
    # Unified Logic metrics
    ul_metrics = unified_logic.get_system_metrics()
    print("📊 Unified Logic:")
    print(f"  Total Decisions: {ul_metrics['total_decisions']}")
    print(f"  Conflicts Resolved: {ul_metrics['conflicts_resolved']}")
    print(f"  Responses Orchestrated: {ul_metrics['responses_orchestrated']}")
    print()
    
    # Trust metrics
    trust_stats = trust_manager.get_trust_statistics()
    print("🔐 Trust System:")
    print(f"  Total Entities: {trust_stats['total_entities']}")
    print(f"  Average Score: {trust_stats['avg_score']:.2%}")
    print(f"  Success Rate: {trust_stats['success_rate']:.2%}")
    print()
    
    # Audit log metrics
    audit_stats = immutable_logs.get_audit_statistics()
    print("📝 Immutable Logs:")
    print(f"  Total Entries: {audit_stats['total_entries']}")
    print(f"  Immutable: {audit_stats['immutable']}")
    print(f"  Constitutional Compliant: {audit_stats['constitutional_compliant']}")
    print()
    
    # Memory metrics
    memory_stats = memory_bank.get_memory_statistics()
    print("💾 Memory System:")
    print(f"  Total Memories: {memory_stats['total_memories']}")
    print(f"  Average Clarity: {memory_stats.get('avg_clarity', 0):.2%}")
    print(f"  High Ambiguity Count: {memory_stats.get('high_ambiguity_count', 0)}")
    print()
    
    # Event Bus metrics
    event_stats = event_bus.get_statistics()
    print("📡 Event Bus:")
    print(f"  Total Events: {event_stats['total_events']}")
    print(f"  Event Types: {event_stats['event_types']}")
    print()
    
    # AVN metrics
    avn_health = avn.get_system_health()
    print("🏥 System Health:")
    print(f"  Status: {avn_health['status']}")
    print(f"  Components: {avn_health['components']}")
    print()
    
    # Demo 6: Human-Readable Export
    print("=" * 70)
    print("DEMO 6: Human-Readable Comprehensive Report")
    print("=" * 70)
    print()
    
    # Generate comprehensive report
    report_data = {
        'logs': [
            {
                'timestamp': log_entry.timestamp,
                'actor': log_entry.actor,
                'action': log_entry.action,
                'constitutional_check': log_entry.constitutional_check,
                'current_hash': log_entry.current_hash,
                'data': log_entry.data
            }
        ],
        'decisions': [
            {
                'timestamp': datetime.now(),
                'source': result['source'],
                'decision': result['decision'],
                'confidence': result['confidence'],
                'rationale': result['rationale'],
                'metadata': {}
            }
        ],
        'health': avn_health
    }
    
    narrative_report = formatter.export_narrative_report(report_data)
    print(narrative_report)
    print()
    
    print("=" * 70)
    print("🎉 DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✅ Trust Score Management")
    print("  ✅ Immutable Audit Logs")
    print("  ✅ Unified Logic Decision Making")
    print("  ✅ Memory with Clarity Scoring")
    print("  ✅ System-Wide Metrics")
    print("  ✅ Human-Readable Reports")
    print()
    print("All core systems operational! 🚀")


if __name__ == "__main__":
    asyncio.run(main())
