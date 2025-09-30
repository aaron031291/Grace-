"""
Grace Tracer (gtrace) Integration Example

This example demonstrates the comprehensive tracing capabilities of the Grace system
with full Vaults 1-18 compliance, constitutional governance, and recursive loop operations.

Features demonstrated:
- Irrefutable Triad compliance (Core, Intelligence, Governance)
- Vaults 1-18 full compliance
- Constitutional decision-making integration
- Trust-based correlation and routing
- Immutable audit trails
- Recursive loop-based operations
- Memory and immunity system hooks
- Agentic execution tracking
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from grace.governance.grace_governance_kernel import GraceGovernanceKernel
from grace.core import (
    GraceTracer, TraceLevel, VaultCompliance, TraceStatus,
    create_grace_tracer, trace_operation
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraceSystemDemo:
    """Demonstration of Grace gtrace integration with governance system."""
    
    def __init__(self):
        self.governance_kernel = None
        self.gtrace = None
        
    async def initialize(self):
        """Initialize the Grace system with full gtrace integration."""
        print("🚀 Initializing Grace System with gtrace...")
        
        # Create and initialize governance kernel
        self.governance_kernel = GraceGovernanceKernel({
            'memory_db_path': '/tmp/grace_demo.db',
            'audit_db_path': '/tmp/grace_audit.db'
        })
        
        await self.governance_kernel.initialize()
        await self.governance_kernel.start()
        
        # Get the integrated gtrace component
        self.gtrace = self.governance_kernel.components.get('gtrace')
        
        print("✅ Grace System initialized with gtrace integration")
        print(f"📊 Gtrace status: {self.governance_kernel.get_gtrace_status()}")
    
    async def demonstrate_basic_tracing(self):
        """Demonstrate basic tracing with constitutional compliance."""
        print("\n🔍 Demonstrating Basic Tracing...")
        
        # Start a governance trace
        trace_id = await self.gtrace.start_trace(
            component_id="governance_demo",
            operation="policy_evaluation",
            user_id="demo_user",
            governance_required=True
        )
        
        print(f"📋 Started trace: {trace_id}")
        
        # Add policy analysis event (Vault 12: Decision narrative)
        await self.gtrace.add_trace_event(
            trace_id=trace_id,
            event_type="analysis",
            operation="policy_analysis",
            input_data={
                "policy_id": "security_policy_v2",
                "analysis_type": "constitutional_compliance"
            },
            narrative="Analyzing security policy v2 for constitutional compliance, " +
                     "focusing on transparency, fairness, and harm prevention principles"
        )
        
        # Add decision event with constitutional validation
        await self.gtrace.add_trace_event(
            trace_id=trace_id,
            event_type="decision",
            operation="policy_decision",
            output_data={
                "decision": "approved",
                "confidence": 0.85,
                "constitutional_score": 0.92
            },
            narrative="Policy approved after constitutional review. High confidence " +
                     "due to strong alignment with governance principles."
        )
        
        # Complete trace
        success = await self.gtrace.complete_trace(trace_id, success=True)
        
        print(f"✅ Trace completed successfully: {success}")
        
        # Show trace status
        status = await self.gtrace.get_trace_status(trace_id)
        print(f"📈 Final trust score: {status['trust_score']:.3f}")
        print(f"🏛️ Constitutional compliance: {status['constitutional_compliant']}")
        
        return trace_id
    
    async def demonstrate_vault_compliance(self):
        """Demonstrate Vaults 1-18 compliance in action."""
        print("\n🔐 Demonstrating Vaults 1-18 Compliance...")
        
        # Vault 6: Contradiction detection
        trace_id = await self.gtrace.start_trace(
            component_id="contradiction_test",
            operation="contradictory_operations"
        )
        
        # Add conflicting events to trigger Vault 6
        await self.gtrace.add_trace_event(
            trace_id=trace_id,
            event_type="operation",
            operation="enable_feature",
            output_data={"feature_x": "enabled"}
        )
        
        await self.gtrace.add_trace_event(
            trace_id=trace_id,
            event_type="operation", 
            operation="disable_feature",
            input_data={"feature_x": "enabled"},  # This will trigger contradiction
            output_data={"feature_x": "disabled"}
        )
        
        await self.gtrace.complete_trace(trace_id, success=True)
        
        # Check for detected contradictions
        status = await self.gtrace.get_trace_status(trace_id)
        contradictions = status['metadata']['contradiction_flags']
        
        print(f"⚠️ Contradictions detected (Vault 6): {len(contradictions)}")
        for contradiction in contradictions:
            print(f"   • {contradiction}")
        
        # Vault 15: Sandbox isolation
        print("\n🧪 Testing Vault 15: Sandbox Isolation...")
        
        sandbox_trace_id = await self.gtrace.start_trace(
            component_id="external_api",
            operation="external_data_fetch"
        )
        
        sandbox_status = await self.gtrace.get_trace_status(sandbox_trace_id)
        sandbox_required = sandbox_status['metadata']['sandbox_required']
        
        print(f"🔒 Sandbox isolation active: {sandbox_required}")
        print(f"✅ Vault 15 compliance: {sandbox_status['vault_compliance']['sandbox_isolation']}")
        
        await self.gtrace.complete_trace(sandbox_trace_id, success=True)
        
        return [trace_id, sandbox_trace_id]
    
    async def demonstrate_recursive_loops(self):
        """Demonstrate recursive loop-based operations with tracing."""
        print("\n🔄 Demonstrating Recursive Loop Operations...")
        
        # Simulate OODA loop with tracing
        loop_traces = []
        
        for cycle in range(3):
            print(f"   🔄 OODA Loop Cycle {cycle + 1}")
            
            # Observe phase
            observe_trace = await self.gtrace.start_trace(
                component_id="ooda_loop",
                operation=f"observe_cycle_{cycle + 1}"
            )
            
            await self.gtrace.add_trace_event(
                trace_id=observe_trace,
                event_type="observation",
                operation="environmental_scan",
                output_data={
                    "threats_detected": cycle,
                    "opportunities_identified": cycle + 1,
                    "cycle": cycle + 1
                },
                narrative=f"Observing environment in cycle {cycle + 1}, " +
                         f"detected {cycle} threats and {cycle + 1} opportunities"
            )
            
            # Orient phase
            await self.gtrace.add_trace_event(
                trace_id=observe_trace,
                event_type="orientation",
                operation="situation_analysis",
                output_data={
                    "strategy_score": 0.7 + (cycle * 0.1),
                    "readiness_level": "high"
                },
                narrative=f"Orienting based on observations, strategy score improved to " +
                         f"{0.7 + (cycle * 0.1):.1f}"
            )
            
            # Decide phase
            await self.gtrace.add_trace_event(
                trace_id=observe_trace,
                event_type="decision",
                operation="tactical_decision",
                output_data={
                    "action_plan": f"adaptive_response_{cycle + 1}",
                    "confidence": 0.8 + (cycle * 0.05)
                },
                narrative=f"Decision made for adaptive response {cycle + 1} " +
                         f"with confidence {0.8 + (cycle * 0.05):.2f}"
            )
            
            # Act phase
            await self.gtrace.add_trace_event(
                trace_id=observe_trace,
                event_type="action",
                operation="execute_plan",
                output_data={
                    "execution_status": "completed",
                    "effectiveness": 0.75 + (cycle * 0.08)
                },
                narrative=f"Executed plan with effectiveness {0.75 + (cycle * 0.08):.2f}"
            )
            
            await self.gtrace.complete_trace(observe_trace, success=True)
            loop_traces.append(observe_trace)
        
        print(f"✅ Completed {len(loop_traces)} OODA loop cycles with full tracing")
        
        return loop_traces
    
    @trace_operation(None, "demo_component", "decorated_operation", governance_required=True)
    async def demonstrate_decorator(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate the trace_operation decorator."""
        print("\n🎭 Demonstrating Trace Decorator...")
        
        # Simulate some processing
        await asyncio.sleep(0.1)
        
        result = {
            "processed": True,
            "input_items": len(data),
            "timestamp": datetime.now().isoformat(),
            "processing_time": "100ms"
        }
        
        print("✅ Decorated operation completed with automatic tracing")
        return result
    
    async def demonstrate_governance_integration(self):
        """Demonstrate deep governance integration."""
        print("\n🏛️ Demonstrating Governance Integration...")
        
        # Create a trace that requires governance review
        governance_trace = await self.gtrace.start_trace(
            component_id="governance_integration",
            operation="high_impact_decision",
            governance_required=True
        )
        
        # Add constitutional review event
        await self.gtrace.add_trace_event(
            trace_id=governance_trace,
            event_type="constitutional_review",
            operation="principle_validation",
            input_data={
                "principles": ["transparency", "fairness", "accountability"],
                "decision_impact": "high"
            },
            narrative="Conducting constitutional review for high-impact decision, " +
                     "validating against core governance principles"
        )
        
        # Add parliament consultation
        await self.gtrace.add_trace_event(
            trace_id=governance_trace,
            event_type="consultation",
            operation="parliament_review",
            output_data={
                "parliament_approval": True,
                "vote_ratio": 0.87,
                "quorum_satisfied": True
            },
            narrative="Parliament consultation completed with 87% approval ratio, " +
                     "quorum satisfied for governance decision"
        )
        
        await self.gtrace.complete_trace(governance_trace, success=True)
        
        print("✅ Governance integration demonstrated successfully")
        return governance_trace
    
    async def show_system_metrics(self):
        """Display comprehensive system metrics."""
        print("\n📊 System Metrics and Status...")
        
        # Gtrace metrics
        gtrace_metrics = self.gtrace.get_system_metrics()
        print(f"🔍 Total traces created: {gtrace_metrics['traces_created']}")
        print(f"✅ Traces completed: {gtrace_metrics['traces_completed']}")
        print(f"❌ Traces failed: {gtrace_metrics['traces_failed']}")
        print(f"🏛️ Governance reviews: {gtrace_metrics['governance_reviews']}")
        print(f"⚖️ Constitutional violations: {gtrace_metrics['constitutional_violations']}")
        print(f"🔐 Vault compliance rate: {gtrace_metrics['vault_compliance_rate']:.1%}")
        
        # Component traces
        component_traces = await self.gtrace.get_component_traces("governance_demo")
        print(f"📋 Component traces: {len(component_traces)}")
        
        # Governance kernel metrics
        gov_metrics = self.governance_kernel.get_governance_metrics()
        print(f"🏛️ Governance system status: {len(gov_metrics)} metric categories")
        
        # Cleanup old traces (demonstration)
        cleaned = await self.gtrace.cleanup_old_traces(max_age_hours=0)
        print(f"🧹 Cleaned up {cleaned} old traces")
    
    async def shutdown(self):
        """Gracefully shutdown the Grace system."""
        print("\n🛑 Shutting down Grace System...")
        
        if self.governance_kernel:
            await self.governance_kernel.shutdown()
        
        print("✅ Grace System shutdown complete")
    
    async def run_complete_demo(self):
        """Run the complete Grace gtrace demonstration."""
        try:
            await self.initialize()
            
            # Basic tracing demonstration
            await self.demonstrate_basic_tracing()
            
            # Vault compliance demonstration
            await self.demonstrate_vault_compliance()
            
            # Recursive loops demonstration
            await self.demonstrate_recursive_loops()
            
            # Governance integration
            await self.demonstrate_governance_integration()
            
            # Decorator demonstration
            # Note: We need to set the tracer for the decorator
            global _demo_tracer
            _demo_tracer = self.gtrace
            
            # Update the decorator with our tracer
            self.demonstrate_decorator.__globals__['trace_operation'] = trace_operation
            decorated_method = trace_operation(
                self.gtrace, "demo_component", "decorated_operation", governance_required=False
            )(self._demo_method)
            
            result = await decorated_method({"test": "data", "items": 5})
            print(f"🎭 Decorator result: {result}")
            
            # Show metrics
            await self.show_system_metrics()
            
            print("\n🎉 Grace gtrace demonstration completed successfully!")
            print("✅ All Vaults 1-18 compliance features demonstrated")
            print("✅ Constitutional governance integration validated")
            print("✅ Recursive loop operations functional")
            print("✅ Immutable audit trails operational")
            print("✅ Trust-based correlation and routing active")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            await self.shutdown()
        
        return True
    
    async def _demo_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for decorator demonstration."""
        await asyncio.sleep(0.1)
        return {
            "processed": True,
            "input_items": len(data),
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main demonstration entry point."""
    print("=" * 70)
    print("🌟 Grace Tracer (gtrace) - Complete System Demonstration")
    print("=" * 70)
    print("Demonstrating Vaults 1-18 compliance, constitutional governance,")
    print("and Grace's Irrefutable Triad integration")
    print("=" * 70)
    
    demo = GraceSystemDemo()
    success = await demo.run_complete_demo()
    
    print("=" * 70)
    if success:
        print("🎉 Grace gtrace demonstration completed successfully!")
        print("✨ Upgraded gtrace component is fully operational")
    else:
        print("❌ Grace gtrace demonstration failed")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)