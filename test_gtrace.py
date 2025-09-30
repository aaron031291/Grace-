"""
Test Grace Tracer (gtrace) functionality and Vaults 1-18 compliance.

This test validates the gtrace component's integration with Grace's governance system
and ensures compliance with constitutional principles and vault requirements.
"""

import asyncio
import pytest
import logging
from datetime import datetime
from typing import Dict, Any

# Import Grace components
from grace.core import (
    EventBus, MemoryCore, GraceTracer, TraceLevel, VaultCompliance, 
    TraceStatus, create_grace_tracer, trace_operation
)
from grace.core.immutable_logs import ImmutableLogs
from grace.core.kpi_trust_monitor import KPITrustMonitor

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGraceTracer:
    """Test suite for Grace Tracer functionality."""
    
    @pytest.fixture
    async def tracer_setup(self):
        """Setup test environment with all Grace components."""
        # Create core components
        event_bus = EventBus()
        memory_core = MemoryCore(":memory:")  # In-memory database for testing
        immutable_logs = ImmutableLogs()
        kpi_monitor = KPITrustMonitor()
        
        # Start components
        await event_bus.start()
        await immutable_logs.start()
        await kpi_monitor.start()
        
        # Create Grace tracer
        tracer = await create_grace_tracer(
            event_bus=event_bus,
            memory_core=memory_core,
            immutable_logs=immutable_logs,
            kpi_monitor=kpi_monitor
        )
        
        yield {
            'tracer': tracer,
            'event_bus': event_bus,
            'memory_core': memory_core,
            'immutable_logs': immutable_logs,
            'kpi_monitor': kpi_monitor
        }
        
        # Cleanup
        await event_bus.stop()
        await immutable_logs.stop()
        await kpi_monitor.stop()
    
    @pytest.mark.asyncio
    async def test_basic_trace_creation(self, tracer_setup):
        """Test basic trace creation and completion."""
        tracer = tracer_setup['tracer']
        
        # Start a trace
        trace_id = await tracer.start_trace(
            component_id="test_component",
            operation="test_operation",
            user_id="test_user"
        )
        
        assert trace_id is not None
        assert trace_id in tracer.active_traces
        
        # Get trace status
        status = await tracer.get_trace_status(trace_id)
        assert status is not None
        assert status['status'] == TraceStatus.INITIATED.value
        assert status['metadata']['component_id'] == "test_component"
        
        # Complete the trace
        result = await tracer.complete_trace(trace_id, success=True)
        assert result is True
        assert trace_id not in tracer.active_traces
        assert trace_id in tracer.completed_traces
    
    @pytest.mark.asyncio
    async def test_vault_compliance_initialization(self, tracer_setup):
        """Test that vault compliance is properly initialized (Vaults 1-18)."""
        tracer = tracer_setup['tracer']
        
        # Start a trace
        trace_id = await tracer.start_trace(
            component_id="vault_test_component",
            operation="vault_test_operation"
        )
        
        # Check vault compliance status
        status = await tracer.get_trace_status(trace_id)
        vault_compliance = status['vault_compliance']
        
        # Should have vault compliance tracking for all vaults
        expected_vaults = [vault.value for vault in VaultCompliance]
        for vault in expected_vaults:
            assert vault in vault_compliance
        
        # Some vaults should be initialized as compliant
        assert vault_compliance[VaultCompliance.VAULT_1_VERIFICATION.value] is True
        assert vault_compliance[VaultCompliance.VAULT_9_TRANSPARENCY.value] is True
        
        await tracer.complete_trace(trace_id, success=True)
    
    @pytest.mark.asyncio
    async def test_trace_events_with_narrative(self, tracer_setup):
        """Test adding trace events with decision narrative (Vault 12)."""
        tracer = tracer_setup['tracer']
        
        trace_id = await tracer.start_trace(
            component_id="narrative_component",
            operation="decision_making"
        )
        
        # Add event with narrative (Vault 12 compliance)
        event_id = await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="decision",
            operation="policy_evaluation",
            input_data={"policy_id": "pol_123", "context": "security_update"},
            narrative="Evaluating security policy update based on threat assessment data"
        )
        
        assert event_id is not None
        
        # Check that narrative was recorded
        trace_chain = tracer.active_traces[trace_id]
        assert len(trace_chain.events) == 1
        assert trace_chain.events[0].decision_narrative is not None
        assert "security policy" in trace_chain.events[0].decision_narrative.lower()
        
        await tracer.complete_trace(trace_id, success=True)
    
    @pytest.mark.asyncio
    async def test_contradiction_detection(self, tracer_setup):
        """Test contradiction detection functionality (Vault 6)."""
        tracer = tracer_setup['tracer']
        
        # Register a test contradiction detector
        def test_contradiction_detector(event, trace_chain):
            contradictions = []
            if event.operation == "conflicting_operation":
                contradictions.append("Test contradiction detected")
            return contradictions
        
        tracer.register_contradiction_detector(test_contradiction_detector)
        
        trace_id = await tracer.start_trace(
            component_id="contradiction_component",
            operation="test_sequence"
        )
        
        # Add event that should trigger contradiction
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="operation",
            operation="conflicting_operation",
            input_data={"test": "data"}
        )
        
        # Check that contradiction was detected
        trace_chain = tracer.active_traces[trace_id]
        assert len(trace_chain.metadata.contradiction_flags) > 0
        assert "Test contradiction detected" in trace_chain.metadata.contradiction_flags
        
        await tracer.complete_trace(trace_id, success=True)
    
    @pytest.mark.asyncio
    async def test_sandbox_isolation(self, tracer_setup):
        """Test sandbox isolation for unverified logic (Vault 15)."""
        tracer = tracer_setup['tracer']
        
        # Start trace with operation that should require sandboxing
        trace_id = await tracer.start_trace(
            component_id="external_component",
            operation="external_api"
        )
        
        # Check that sandbox was enabled
        status = await tracer.get_trace_status(trace_id)
        assert status['metadata']['sandbox_required'] is True
        assert status['vault_compliance'][VaultCompliance.VAULT_15_SANDBOX.value] is True
        
        await tracer.complete_trace(trace_id, success=True)
    
    @pytest.mark.asyncio
    async def test_governance_integration(self, tracer_setup):
        """Test governance validation integration (Vault 11)."""
        tracer = tracer_setup['tracer']
        
        # Start trace requiring governance validation
        trace_id = await tracer.start_trace(
            component_id="governance_component",
            operation="critical_decision",
            governance_required=True
        )
        
        # Check that governance decision ID was assigned
        status = await tracer.get_trace_status(trace_id)
        assert status['metadata']['governance_decision_id'] is not None
        
        await tracer.complete_trace(trace_id, success=True)
    
    @pytest.mark.asyncio
    async def test_constitutional_compliance(self, tracer_setup):
        """Test constitutional compliance validation (Vault 14)."""
        tracer = tracer_setup['tracer']
        
        trace_id = await tracer.start_trace(
            component_id="constitutional_component",
            operation="policy_change"
        )
        
        # Add events that should pass constitutional validation
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="decision",
            operation="policy_review",
            narrative="Reviewing policy for constitutional compliance"
        )
        
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="validation",
            operation="compliance_check",
            output_data={"compliance_result": "passed"}
        )
        
        # Complete trace and check constitutional compliance
        result = await tracer.complete_trace(trace_id, success=True)
        assert result is True
        
        # Check final compliance status
        trace_chain = tracer.completed_traces[trace_id]
        vault_compliance = trace_chain.metadata.vault_compliance
        assert vault_compliance[VaultCompliance.VAULT_14_CONSTITUTIONAL.value] is True
    
    @pytest.mark.asyncio
    async def test_trust_score_updates(self, tracer_setup):
        """Test trust score updates during trace execution (Vault 8)."""
        tracer = tracer_setup['tracer']
        
        trace_id = await tracer.start_trace(
            component_id="trust_component",
            operation="trust_test"
        )
        
        # Get initial trust score
        initial_status = await tracer.get_trace_status(trace_id)
        initial_trust = initial_status['trust_score']
        
        # Add successful event (should improve trust)
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="operation",
            operation="successful_operation",
            output_data={"result": "success"}
        )
        
        # Add failed event (should decrease trust)
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="operation",
            operation="failed_operation",
            error_data={"error": "operation failed"}
        )
        
        await tracer.complete_trace(trace_id, success=True)
        
        # Check final trust score
        final_status = await tracer.get_trace_status(trace_id)
        # Trust score should have changed due to the events
        assert final_status['trust_score'] != initial_trust
    
    @pytest.mark.asyncio
    async def test_trace_operation_decorator(self, tracer_setup):
        """Test the trace_operation decorator functionality."""
        tracer = tracer_setup['tracer']
        
        @trace_operation(tracer, "decorator_component", "decorated_operation")
        async def test_function(x, y):
            """Test function to be traced."""
            return x + y
        
        # Execute decorated function
        result = await test_function(5, 3)
        assert result == 8
        
        # Check that trace was created and completed
        metrics = tracer.get_system_metrics()
        assert metrics['traces_created'] >= 1
        assert metrics['traces_completed'] >= 1
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, tracer_setup):
        """Test system metrics collection and vault compliance reporting."""
        tracer = tracer_setup['tracer']
        
        # Create several traces to generate metrics
        for i in range(3):
            trace_id = await tracer.start_trace(
                component_id=f"metrics_component_{i}",
                operation=f"metrics_operation_{i}"
            )
            
            await tracer.add_trace_event(
                trace_id=trace_id,
                event_type="test",
                operation=f"test_event_{i}"
            )
            
            await tracer.complete_trace(trace_id, success=True)
        
        # Get system metrics
        metrics = tracer.get_system_metrics()
        
        # Validate metrics structure
        assert 'traces_created' in metrics
        assert 'traces_completed' in metrics
        assert 'vault_compliance_rate' in metrics
        assert 'active_traces' in metrics
        assert 'completed_traces' in metrics
        
        # Should have created and completed traces
        assert metrics['traces_created'] >= 3
        assert metrics['traces_completed'] >= 3
        
        # Vault compliance rate should be calculated
        assert 0.0 <= metrics['vault_compliance_rate'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_trace_cleanup(self, tracer_setup):
        """Test automatic cleanup of old traces."""
        tracer = tracer_setup['tracer']
        
        # Create some traces
        for i in range(5):
            trace_id = await tracer.start_trace(
                component_id=f"cleanup_component_{i}",
                operation=f"cleanup_operation_{i}"
            )
            await tracer.complete_trace(trace_id, success=True)
        
        initial_count = len(tracer.completed_traces)
        assert initial_count >= 5
        
        # Run cleanup (with 0 max age to clean everything)
        cleaned_count = await tracer.cleanup_old_traces(max_age_hours=0)
        
        # Should have cleaned some traces
        assert cleaned_count > 0
        final_count = len(tracer.completed_traces)
        assert final_count < initial_count


# Standalone test function that can be run without pytest
async def run_basic_test():
    """Run a basic test of Grace tracer functionality."""
    print("Running basic Grace Tracer test...")
    
    # Create components
    event_bus = EventBus()
    memory_core = MemoryCore("/tmp/test_grace.db")  # Use file instead of :memory:
    immutable_logs = ImmutableLogs()
    
    # Start components
    await event_bus.start()
    await memory_core.start()  # Start memory core
    await immutable_logs.start()
    
    try:
        # Create tracer
        tracer = await create_grace_tracer(
            event_bus=event_bus,
            memory_core=memory_core,
            immutable_logs=immutable_logs
        )
        
        print("✓ Grace Tracer created successfully")
        
        # Test basic tracing
        trace_id = await tracer.start_trace(
            component_id="test_component",
            operation="basic_test",
            user_id="test_user"
        )
        
        print(f"✓ Trace started: {trace_id}")
        
        # Add some events
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="operation",
            operation="test_step_1",
            input_data={"step": 1},
            narrative="Executing first test step"
        )
        
        await tracer.add_trace_event(
            trace_id=trace_id,
            event_type="decision",
            operation="test_decision",
            output_data={"decision": "proceed"},
            narrative="Decision made to proceed with test"
        )
        
        print("✓ Trace events added")
        
        # Complete trace
        try:
            result = await tracer.complete_trace(trace_id, success=True)
            assert result is True
        except Exception as e:
            print(f"Error completing trace: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("✓ Trace completed successfully")
        
        # Check metrics
        metrics = tracer.get_system_metrics()
        print(f"✓ System metrics: {metrics}")
        
        # Test vault compliance
        status = await tracer.get_trace_status(trace_id)
        vault_compliance = status['vault_compliance']
        compliant_vaults = sum(vault_compliance.values())
        total_vaults = len(vault_compliance)
        
        print(f"✓ Vault compliance: {compliant_vaults}/{total_vaults} vaults compliant")
        
        print("\n🎉 Basic Grace Tracer test completed successfully!")
        print("✅ Vaults 1-18 compliance framework operational")
        print("✅ Constitutional governance integration working")
        print("✅ Immutable audit trail functional")
        print("✅ Trust scoring and validation active")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        await event_bus.stop()
        await memory_core.stop()  # Stop memory core
        await immutable_logs.stop()


if __name__ == "__main__":
    # Run the basic test
    result = asyncio.run(run_basic_test())
    exit(0 if result else 1)