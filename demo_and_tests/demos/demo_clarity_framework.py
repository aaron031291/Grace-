"""
Demo script showing the Grace Clarity Framework in action.

This demonstrates all 4 clarity classes working together:
1. BaseComponent - Standardized component lifecycle
2. EnhancedEventBus - Declarative event routing
3. GraceLoopOutput - Standardized loop output format
4. GraceOrchestrator - Component manifest and lifecycle management
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import clarity framework components
from grace.clarity_framework import (
    BaseComponent, ComponentStatus, GraceLoopOutput, ReasoningChain,
    GraceComponentManifest, ComponentRole, TrustLevel, ActivationState,
    EnhancedEventBus, GraceOrchestrator
)


class DemoGovernanceComponent(BaseComponent):
    """Demo component showing BaseComponent usage."""
    
    def __init__(self):
        super().__init__("governance", "1.0.0")
        self.decision_count = 0
    
    async def activate(self) -> bool:
        """Activate the governance component."""
        self.logger.info("Activating governance component...")
        await asyncio.sleep(0.1)  # Simulate initialization
        return True
    
    async def deactivate(self) -> bool:
        """Deactivate the governance component."""
        self.logger.info("Deactivating governance component...")
        return True
    
    async def health_check(self) -> dict:
        """Check component health."""
        return {
            "status": "healthy",
            "decision_count": self.decision_count,
            "memory_usage": "low",
            "last_check": datetime.now().isoformat()
        }
    
    async def get_metrics(self) -> dict:
        """Get component metrics."""
        return {
            "decisions_processed": self.decision_count,
            "avg_decision_time_ms": 250.0,
            "success_rate": 0.95
        }
    
    async def process_governance_decision(self, payload: dict) -> GraceLoopOutput:
        """Process a governance decision and return standardized output."""
        loop_id = f"governance_{self.decision_count:04d}"
        
        # Create reasoning chain
        reasoning_chain = ReasoningChain(
            chain_id=f"chain_{loop_id}",
            loop_type="governance"
        )
        
        # Create loop output
        loop_output = GraceLoopOutput(
            loop_id=loop_id,
            loop_type="governance",
            reasoning_chain_id=reasoning_chain.chain_id,
            reasoning_chain=reasoning_chain,
            results={}
        )
        
        # Simulate reasoning process
        loop_output.add_reasoning_step(
            "Analyze constitutional compliance",
            input_data={"claim": payload.get("claim", "")},
            output_data={"compliance_score": 0.85},
            confidence=0.9
        )
        
        loop_output.add_reasoning_step(
            "Check policy alignment", 
            input_data={"compliance_score": 0.85},
            output_data={"policy_aligned": True},
            confidence=0.8
        )
        
        loop_output.add_reasoning_step(
            "Generate final decision",
            input_data={"compliance_score": 0.85, "policy_aligned": True},
            output_data={"decision": "APPROVED", "rationale": "Complies with constitution"},
            confidence=0.9
        )
        
        # Set final results
        loop_output.results = {
            "decision": "APPROVED",
            "confidence": 0.85,
            "rationale": "Decision approved based on constitutional compliance"
        }
        
        loop_output.mark_completed()
        self.decision_count += 1
        
        return loop_output


class DemoLearningComponent(BaseComponent):
    """Demo learning component."""
    
    def __init__(self):
        super().__init__("learning", "1.0.0")
        self.patterns_learned = 0
    
    async def activate(self) -> bool:
        await asyncio.sleep(0.1)
        return True
    
    async def deactivate(self) -> bool:
        return True
    
    async def health_check(self) -> dict:
        return {
            "status": "healthy",
            "patterns_learned": self.patterns_learned
        }
    
    async def get_metrics(self) -> dict:
        return {
            "patterns_learned": self.patterns_learned,
            "learning_rate": 0.02
        }


async def event_handler(event: dict):
    """Demo event handler."""
    print(f"📥 Received event: {event['type']} (correlation: {event['correlation_id']})")


async def demo_clarity_framework():
    """Demonstrate the complete clarity framework."""
    print("🚀 Grace Clarity Framework Demo")
    print("=" * 50)
    
    # 1. Create enhanced event bus with routing config
    print("\n1️⃣  Setting up Enhanced Event Bus...")
    config_path = "/tmp/demo_routing_config.yaml"
    event_bus = EnhancedEventBus(config_path)
    
    # Subscribe to events
    await event_bus.subscribe("demo_handler", "GOVERNANCE_*", event_handler)
    await event_bus.subscribe("demo_handler", "COMPONENT_*", event_handler)
    print("✅ Event bus configured with declarative routing")
    
    # 2. Create orchestrator
    print("\n2️⃣  Setting up Grace Orchestrator...")
    orchestrator = GraceOrchestrator(event_bus)
    await orchestrator.start_orchestration()
    print("✅ Orchestrator started")
    
    # 3. Create demo components
    print("\n3️⃣  Creating demo components...")
    
    # Governance component
    gov_component = DemoGovernanceComponent()
    gov_manifest = GraceComponentManifest(
        component_id=gov_component.component_id,
        component_type="governance",
        component_name="Demo Governance Component",
        version="1.0.0",
        role=ComponentRole.CORE_GOVERNANCE,
        trust_level=TrustLevel.HIGH_TRUST,
        trust_score=0.9
    )
    
    # Add capabilities
    gov_manifest.add_capability(
        "constitutional_validation",
        "Validate decisions against constitutional principles",
        ["claim", "policy"], ["decision", "rationale"],
        confidence_level=0.9
    )
    
    # Learning component  
    learning_component = DemoLearningComponent()
    learning_manifest = GraceComponentManifest(
        component_id=learning_component.component_id,
        component_type="learning",
        component_name="Demo Learning Component", 
        version="1.0.0",
        role=ComponentRole.KERNEL_LEARNING,
        trust_level=TrustLevel.MEDIUM_TRUST,
        trust_score=0.7
    )
    
    # Add dependency (learning depends on governance)
    learning_manifest.add_dependency(
        gov_component.component_id,
        "governance",
        ["constitutional_validation"],
        is_hard=False,
        fallback_strategy="skip_validation"
    )
    
    print("✅ Components created with manifests")
    
    # 4. Register components with orchestrator
    print("\n4️⃣  Registering components...")
    success1 = orchestrator.register_component(gov_component, gov_manifest)
    success2 = orchestrator.register_component(learning_component, learning_manifest)
    
    if success1 and success2:
        print("✅ Components registered successfully")
    else:
        print("❌ Component registration failed")
        return
    
    # 5. Activate components in dependency order
    print("\n5️⃣  Activating components...")
    activation_order = orchestrator.get_activation_order()
    print(f"Activation order: {activation_order}")
    
    for comp_id in activation_order:
        success = await orchestrator.activate_component(comp_id)
        if success:
            print(f"✅ Activated: {comp_id}")
        else:
            print(f"❌ Failed to activate: {comp_id}")
    
    # 6. Demonstrate loop output standardization
    print("\n6️⃣  Demonstrating standardized loop output...")
    
    governance_result = await gov_component.process_governance_decision({
        "claim": "This is a test governance decision",
        "context": "demo_run"
    })
    
    print(f"Loop ID: {governance_result.loop_id}")
    print(f"Loop Type: {governance_result.loop_type}")
    print(f"Confidence Score: {governance_result.confidence_score}")
    print(f"Reasoning Steps: {len(governance_result.reasoning_chain.steps)}")
    print(f"Execution Time: {governance_result.execution_time_ms}ms")
    
    # 7. Publish governance event 
    print("\n7️⃣  Publishing governance event...")
    correlation_id = await event_bus.publish(
        "GOVERNANCE_DECISION_COMPLETED",
        {
            "loop_output": governance_result.get_execution_summary(),
            "component_id": gov_component.component_id
        }
    )
    print(f"✅ Published event with correlation ID: {correlation_id}")
    
    # Wait a moment for event processing
    await asyncio.sleep(0.2)
    
    # 8. Check orchestration status
    print("\n8️⃣  Orchestration Status:")
    status = orchestrator.get_orchestration_status()
    print(f"Total Registered: {status['total_registered']}")
    print(f"Active Components: {status['active_components']}")
    print(f"Activation Order: {status['activation_order']}")
    
    # 9. Display routing metrics
    print("\n9️⃣  Event Bus Metrics:")
    metrics = event_bus.get_routing_metrics()
    print(f"Total Routing Rules: {metrics['total_routing_rules']}")
    print(f"Registered Components: {metrics['registered_components']}")
    print(f"Active Subscriptions: {metrics['active_subscriptions']}")
    
    # 10. Show component health
    print("\n🔟  Component Health Checks:")
    for comp_id in activation_order:
        component = orchestrator.registered_components[comp_id]
        health = await component.health_check()
        print(f"  {comp_id}: {health.get('status', 'unknown')}")
    
    # 11. Cleanup
    print("\n🧹 Cleaning up...")
    await orchestrator.stop_orchestration()
    print("✅ Orchestration stopped")
    
    print("\n🎉 Demo completed successfully!")
    print("\n" + "=" * 50)
    print("Key Features Demonstrated:")
    print("✨ BaseComponent - Standardized lifecycle management")
    print("✨ EnhancedEventBus - Declarative routing with YAML config")
    print("✨ GraceLoopOutput - Standardized reasoning and results")
    print("✨ GraceOrchestrator - Component manifest and dependency tracking")
    print("✨ Full integration - All components working together")
    

if __name__ == "__main__":
    asyncio.run(demo_clarity_framework())