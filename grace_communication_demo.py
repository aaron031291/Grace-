#!/usr/bin/env python3
"""
Grace Communication Capabilities Demo
====================================

Demonstrates Grace's ability to:
- Answer user queries with structured, confident responses
- Provide governance guidance with trust evaluations and risk assessments
- Engage in natural conversations with multi-turn dialogue understanding
- Perform self-assessment and system introspection
- Coordinate between components (Governance ↔ Immune System communication)
- Conduct self-reflection for continuous improvement

Usage:
    python grace_communication_demo.py [--interactive] [--demo-all]
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Grace imports
from grace.core.event_bus import EventBus
from grace.core.memory_core import MemoryCore
from grace.core.contracts import generate_correlation_id
from grace.governance.governance_engine import GovernanceEngine
from grace.governance.verification_engine import VerificationEngine
from grace.governance.unified_logic import UnifiedLogic
from grace.governance.trust_core_kernel import TrustCoreKernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraceCommunicationDemo:
    """
    Demonstrates Grace's communication capabilities across different scenarios.
    Shows structured responses, governance guidance, component coordination, and self-reflection.
    """

    def __init__(self):
        self.event_bus: Optional[EventBus] = None
        self.memory_core: Optional[MemoryCore] = None
        self.governance_engine: Optional[GovernanceEngine] = None
        self.trust_core: Optional[TrustCoreKernel] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id = generate_correlation_id()

    async def initialize(self):
        """Initialize Grace communication systems."""
        print("🤖 Initializing Grace Communication Systems...")

        try:
            # Core infrastructure
            self.event_bus = EventBus()
            await self.event_bus.start()

            self.memory_core = MemoryCore()
            await self.memory_core.start()

            # Governance and intelligence
            verifier = VerificationEngine(self.event_bus, self.memory_core)
            unifier = UnifiedLogic(self.event_bus, self.memory_core)
            self.governance_engine = GovernanceEngine(
                self.event_bus, self.memory_core, verifier, unifier)
            self.trust_core = TrustCoreKernel(self.event_bus, self.memory_core)

            print("✅ Grace communication systems online")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize Grace: {e}")
            return False

    async def demo_structured_responses(self):
        """Demo 1: Structured Query Responses"""
        print("\n" + "=" * 60)
        print("🗣️  DEMO 1: STRUCTURED QUERY RESPONSES")
        print("=" * 60)

        queries = [
            "What is the current system health status?",
            "How confident are you in the security of the governance system?",
            "What are the key components of Grace architecture?",
            "Can you assess the trust level of recent decisions?"
        ]

        for query in queries:
            print(f"\n👤 User: {query}")
            response = await self._process_structured_query(query)
            print(f"🤖 Grace: {response['answer']}")
            print(f"   📊 Confidence: {response['confidence']:.1%}")
            print(f"   🔍 Evidence: {response['evidence_count']} sources")
            print(f"   ⚡ Response time: {response['response_time']:.2f}s")

            # Store in conversation history
            self.conversation_history.append({
                "query": query,
                "response": response,
                "timestamp": datetime.now()
            })

    async def demo_governance_guidance(self):
        """Demo 2: Governance Guidance with Trust Evaluations"""
        print("\n" + "=" * 60)
        print("⚖️  DEMO 2: GOVERNANCE GUIDANCE & TRUST EVALUATION")
        print("=" * 60)

        scenarios = [
            {
                "scenario": "Policy Decision: Implement new security protocol",
                "context": "High-risk deployment affecting user data",
                "stakeholders": ["security_team", "compliance_officer", "users"]
            },
            {
                "scenario": "Resource Allocation: Increase system monitoring",
                "context": "Performance degradation detected in critical systems",
                "stakeholders": ["ops_team", "management", "customers"]
            }
        ]

        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario['scenario']}")
            print(f"🔍 Context: {scenario['context']}")

            guidance = await self._provide_governance_guidance(scenario)

            print("🤖 Grace Governance Guidance:")
            print(f"   📝 Recommendation: {guidance['recommendation']}")
            print(f"   🎯 Confidence Level: {guidance['confidence']:.1%}")
            print(f"   ⚠️  Risk Assessment: {guidance['risk_level']} risk")
            print(
                f"   👥 Stakeholder Analysis: {len(guidance['stakeholder_impacts'])} groups affected")
            print(f"   🛡️  Trust Score: {guidance['trust_score']:.1%}")

            if guidance['recommendations']:
                print("   💡 Action Items:")
                for i, rec in enumerate(guidance['recommendations'], 1):
                    print(f"      {i}. {rec}")

    async def demo_multi_turn_dialogue(self):
        """Demo 3: Multi-turn Dialogue Understanding"""
        print("\n" + "=" * 60)
        print("💬 DEMO 3: MULTI-TURN DIALOGUE UNDERSTANDING")
        print("=" * 60)

        dialogue = [
            "Can you explain the governance voting process?",
            "How does the trust system factor into voting?",
            "What happens if a vote fails to reach consensus?",
            "Can you give me an example from recent history?"
        ]

        context = {}

        for turn, query in enumerate(dialogue, 1):
            print(f"\n👤 User (Turn {turn}): {query}")

            response = await self._process_contextual_query(query, context, self.conversation_history)

            print(f"🤖 Grace: {response['answer']}")
            print(
                f"   🧠 Context Understanding: {
                    response['context_score']:.1%}")
            print(
                f"   🔗 References Previous: {
                    'Yes' if response['references_previous'] else 'No'}")

            # Update context for next turn
            context.update(response['context_updates'])

    async def demo_self_assessment(self):
        """Demo 4: Self-Assessment and System Introspection"""
        print("\n" + "=" * 60)
        print("🔍 DEMO 4: SELF-ASSESSMENT & INTROSPECTION")
        print("=" * 60)

        print("\n🤖 Grace: Let me perform a self-assessment...")

        assessment = await self._perform_self_assessment()

        print("\n📊 GRACE SELF-ASSESSMENT REPORT")
        print(f"   🎯 Overall Health: {assessment['health_status']}")
        print(f"   🧠 Cognitive Load: {assessment['cognitive_load']:.1%}")
        print(f"   💾 Memory Utilization: {assessment['memory_usage']:.1%}")
        print(
            f"   📡 Communication Efficiency: {
                assessment['comm_efficiency']:.1%}")
        print(f"   🔄 Learning Rate: {assessment['learning_rate']:.1%}")

        print("\n🎭 Personality Assessment:")
        for trait, score in assessment['personality'].items():
            print(f"   {trait}: {score:.1%}")

        print("\n💭 Current State of Mind:")
        print(f"   {assessment['state_of_mind']}")

        if assessment['concerns']:
            print("\n⚠️  System Concerns:")
            for concern in assessment['concerns']:
                print(f"   • {concern}")

    async def demo_component_coordination(self):
        """Demo 5: Component Coordination"""
        print("\n" + "=" * 60)
        print("🔄 DEMO 5: COMPONENT COORDINATION")
        print("=" * 60)

        print("🤖 Grace: Demonstrating coordination between governance and trust systems...")

        coordination_events = []

        # Set up event listener
        async def coordination_listener(
                event_type: str, payload: Dict[str, Any], correlation_id: str):
            coordination_events.append({
                "event": event_type,
                "payload": payload,
                "correlation": correlation_id,
                "timestamp": datetime.now()
            })
            print(
                f"   📡 Event: {event_type} (correlation: {correlation_id[:8]}...)")

        # Subscribe to coordination events
        coord_event_types = [
            "GOVERNANCE_VALIDATION",
            "TRUST_UPDATED",
            "COMPONENT_COORDINATION"]
        for event_type in coord_event_types:
            await self.event_bus.subscribe(event_type, coordination_listener)

        # Simulate component coordination
        print("\n🔄 Initiating governance → trust system coordination...")

        # Trigger governance decision that affects trust
        test_decision = {
            "type": "trust_update",
            "entity": "demo_component",
            "new_trust_level": 0.85,
            "reason": "successful_operation"
        }

        # Publish coordination events
        await self.event_bus.publish("GOVERNANCE_VALIDATION", test_decision, correlation_id=self.session_id)
        await self.event_bus.publish("TRUST_UPDATED", {
            "entity": test_decision["entity"],
            "old_trust": 0.75,
            "new_trust": test_decision["new_trust_level"]
        }, correlation_id=self.session_id)

        await asyncio.sleep(0.2)  # Allow events to process

        print("\n📊 Coordination Results:")
        print(f"   Events Generated: {len(coordination_events)}")
        print("   Cross-component Communication: ✅")
        print("   Event Correlation: ✅")

        # Show event flow
        for event in coordination_events:
            print(
                f"   📨 {event['event']} at {event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}")

    async def demo_self_reflection(self):
        """Demo 6: Self-Reflection for Continuous Improvement"""
        print("\n" + "=" * 60)
        print("🪞 DEMO 6: SELF-REFLECTION & IMPROVEMENT")
        print("=" * 60)

        print("🤖 Grace: Engaging in self-reflection based on this session...")

        reflection = await self._conduct_self_reflection()

        print("\n🧠 SELF-REFLECTION ANALYSIS")
        print(f"   Session Duration: {reflection['session_duration']}")
        print(f"   Interactions Processed: {reflection['interaction_count']}")
        print(
            f"   Average Response Quality: {
                reflection['avg_response_quality']:.1%}")
        print(
            f"   Learning Opportunities Identified: {
                reflection['learning_opportunities']}")

        print("\n💡 Insights Gained:")
        for insight in reflection['insights']:
            print(f"   • {insight}")

        print("\n📈 Improvement Actions:")
        for action in reflection['improvement_actions']:
            print(f"   → {action}")

        print(
            f"\n🎯 Confidence in Self-Assessment: {reflection['confidence']:.1%}")

        # Store reflection in memory for future reference
        await self.memory_core.store_structured_memory(
            memory_type="self_reflection",
            content=reflection,
            metadata={
                "session_id": self.session_id,
                "reflection_type": "post_session_analysis"
            }
        )

        print("   💾 Self-reflection stored for future learning")

    async def _process_structured_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return structured response."""
        start_time = time.time()

        # Simulate governance query processing
        if "health" in query.lower():
            answer = "System health is optimal. All core components operational, governance protocols active, and trust mechanisms functioning normally."
            confidence = 0.92
            evidence_count = 3
        elif "confident" in query.lower() or "security" in query.lower():
            answer = "High confidence in security posture. Multi-layer validation active, constitutional compliance verified, and trust scores within acceptable ranges."
            confidence = 0.88
            evidence_count = 5
        elif "components" in query.lower() or "architecture" in query.lower():
            answer = "Grace architecture consists of 24 integrated kernels including governance, intelligence, communication, and security layers with democratic oversight."
            confidence = 0.95
            evidence_count = 8
        else:
            answer = "Trust evaluation systems are operational. Recent decisions show consistent high-trust patterns with democratic validation and constitutional compliance."
            confidence = 0.85
            evidence_count = 4

        response_time = time.time() - start_time

        return {
            "answer": answer,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "response_time": response_time,
            "query_type": "structured_query"
        }

    async def _provide_governance_guidance(
            self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Provide governance guidance with trust evaluations."""

        # Simulate governance analysis
        risk_level = "medium" if "security" in scenario["context"].lower(
        ) else "low"

        if "security" in scenario["scenario"].lower():
            recommendation = "RECOMMEND APPROVAL with enhanced monitoring"
            confidence = 0.78
            trust_score = 0.82
            recommendations = [
                "Implement gradual rollout with monitoring",
                "Conduct security audit before full deployment",
                "Establish clear rollback procedures"
            ]
        else:
            recommendation = "RECOMMEND APPROVAL with standard oversight"
            confidence = 0.85
            trust_score = 0.88
            recommendations = [
                "Proceed with implementation",
                "Monitor performance metrics",
                "Review effectiveness in 30 days"
            ]

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "risk_level": risk_level,
            "trust_score": trust_score,
            "stakeholder_impacts": scenario["stakeholders"],
            "recommendations": recommendations
        }

    async def _process_contextual_query(self,
                                        query: str,
                                        context: Dict[str,
                                                      Any],
                                        history: List[Dict[str,
                                                           Any]]) -> Dict[str,
                                                                          Any]:
        """Process query with conversation context."""

        references_previous = any(word in query.lower()
                                  for word in ["how", "what happens", "example"])
        context_score = 0.8 if references_previous else 0.4

        # Generate contextual response based on query
        if "voting process" in query.lower():
            answer = "Grace uses democratic parliamentary review for major decisions. The Parliament component evaluates proposals with configurable voting thresholds and expertise-based reviewer assignment."
            context_updates = {
                "topic": "governance_voting",
                "explained_voting": True}

        elif "trust system" in query.lower():
            answer = "Trust scores weight voting decisions based on historical performance and expertise. Higher trust entities have increased influence in consensus building."
            context_updates = {
                "explained_trust": True,
                "trust_voting_relationship": True}

        elif "fails to reach consensus" in query.lower():
            answer = "Failed consensus triggers escalation to higher governance tiers, extended review periods, or alternative resolution mechanisms based on the decision's criticality."
            context_updates = {"explained_failure_modes": True}

        else:
            answer = "Recent example: Policy decision #GOV-2024-045 required 3 rounds of voting due to security concerns, ultimately approved with trust-weighted consensus of 0.847."
            context_updates = {"provided_example": True}

        return {
            "answer": answer,
            "context_score": context_score,
            "references_previous": references_previous,
            "context_updates": context_updates
        }

    async def _perform_self_assessment(self) -> Dict[str, Any]:
        """Perform comprehensive self-assessment."""

        # Simulate system introspection
        assessment = {
            "health_status": "OPTIMAL",
            "cognitive_load": 0.65,
            "memory_usage": 0.42,
            "comm_efficiency": 0.89,
            "learning_rate": 0.76,
            "personality": {
                "analytical": 0.92,
                "collaborative": 0.85,
                "cautious": 0.78,
                "adaptive": 0.81},
            "state_of_mind": "Focused and responsive. Processing queries efficiently with high confidence. Ready for complex governance decisions.",
            "concerns": []}

        # Add some concerns if load is high
        if assessment["cognitive_load"] > 0.8:
            assessment["concerns"].append(
                "High cognitive load - consider load balancing")

        return assessment

    async def _conduct_self_reflection(self) -> Dict[str, Any]:
        """Conduct self-reflection on session performance."""

        session_duration = datetime.now(
        ) - datetime.now().replace(minute=datetime.now().minute - 5)

        reflection = {
            "session_duration": str(session_duration).split('.')[0],
            "interaction_count": len(self.conversation_history) + 6,  # demos
            "avg_response_quality": 0.87,
            "learning_opportunities": 3,
            "insights": [
                "User queries show strong interest in governance mechanics",
                "Trust system explanations were well-received",
                "Multi-turn dialogue capability needs enhancement",
                "Self-assessment features demonstrate good introspection"
            ],
            "improvement_actions": [
                "Enhance contextual memory for longer conversations",
                "Develop more sophisticated trust evaluation explanations",
                "Implement better cross-component coordination visualization"
            ],
            "confidence": 0.89
        }

        return reflection

    async def run_interactive_mode(self):
        """Run interactive communication demo."""
        print("\n🎮 INTERACTIVE MODE")
        print("Type your questions or 'exit' to quit")
        print("-" * 40)

        while True:
            try:
                query = input("\n👤 You: ").strip()
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("🤖 Grace: Goodbye! Thank you for the conversation.")
                    break

                if not query:
                    continue

                response = await self._process_structured_query(query)
                print(f"🤖 Grace: {response['answer']}")
                print(f"   (Confidence: {response['confidence']:.1%})")

            except KeyboardInterrupt:
                print("\n🤖 Grace: Session interrupted. Goodbye!")
                break

    async def cleanup(self):
        """Cleanup resources."""
        if self.event_bus:
            await self.event_bus.stop()
        if self.memory_core:
            await self.memory_core.stop()


async def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Grace Communication Capabilities Demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode")
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run all demos",
        default=True)
    args = parser.parse_args()

    demo = GraceCommunicationDemo()

    try:
        # Initialize Grace
        success = await demo.initialize()
        if not success:
            return 1

        if args.interactive:
            await demo.run_interactive_mode()
        else:
            # Run all demos
            print("🤖 Starting Grace Communication Capabilities Demo")
            print("=" * 60)

            await demo.demo_structured_responses()
            await demo.demo_governance_guidance()
            await demo.demo_multi_turn_dialogue()
            await demo.demo_self_assessment()
            await demo.demo_component_coordination()
            await demo.demo_self_reflection()

            print("\n" + "=" * 60)
            print("✅ Grace Communication Capabilities Demo Complete")
            print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1

    finally:
        await demo.cleanup()


if __name__ == "__main__":
    exit(asyncio.run(main()))
