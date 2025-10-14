#!/usr/bin/env python3
"""Test script for Grace Event Mesh and Message Envelope."""

import asyncio
import sys
import os

# Add grace to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from grace.contracts.message_envelope import (
        GraceMessageEnvelope,
        EventTypes,
        GMEHeaders,
    )
    from grace.layer_02_event_mesh.grace_event_bus import (
        GraceEventBus,
        RetryConfig,
        BackpressureConfig,
    )

    print("✅ Successfully imported event mesh components")

    async def test_event_mesh():
        """Test basic event mesh functionality."""
        print("🧪 Testing Grace Event Mesh...")

        # Create event bus
        event_bus = GraceEventBus(
            retry_config=RetryConfig(max_retries=2),
            backpressure_config=BackpressureConfig(max_queue_size=1000),
        )

        # Start event bus
        await event_bus.start()

        # Test message creation
        gme = GraceMessageEnvelope.create_event(
            event_type=EventTypes.GOVERNANCE_VALIDATION,
            payload={"test": "data", "priority": "high"},
            source="test_script",
            priority="high",
        )

        print(f"Created GME: {gme.msg_id}")
        print(f"Event type: {gme.headers.event_type}")
        print(f"Hash: {gme.compute_hash()}")

        # Test event handler
        received_events = []

        async def test_handler(message):
            received_events.append(message)
            print(f"Received event: {message.headers.event_type}")

        # Subscribe to events
        subscription_id = event_bus.subscribe(
            EventTypes.GOVERNANCE_VALIDATION, test_handler
        )

        # Publish event
        await event_bus.publish_gme(gme)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check results
        assert len(received_events) == 1
        assert received_events[0].msg_id == gme.msg_id

        # Get stats
        stats = event_bus.get_stats()
        print(f"Processed messages: {stats['processing_stats']['messages_processed']}")

        # Cleanup
        event_bus.unsubscribe(EventTypes.GOVERNANCE_VALIDATION, subscription_id)
        await event_bus.stop()

        print("✅ Event mesh test passed!")
        return

    def test_message_envelope():
        """Test GME functionality."""
        print("🧪 Testing Grace Message Envelope...")

        # Create message envelope
        gme = GraceMessageEnvelope.create_event(
            event_type=EventTypes.MLDL_TRAINING_STARTED,
            payload={"model_id": "test-model", "dataset": "training-data"},
            source="mldl_kernel",
        )

        # Test serialization
        data = gme.to_dict()
        gme2 = GraceMessageEnvelope.from_dict(data)

        assert gme.msg_id == gme2.msg_id
        assert gme.headers.event_type == gme2.headers.event_type
        assert gme.payload == gme2.payload

        # Test expiration
        assert not gme.is_expired()

        # Test retry
        initial_count = gme.retry_count
        gme.increment_retry()
        assert gme.retry_count == initial_count + 1

        print("✅ Message envelope test passed!")
        return

    def run_tests():
        """Run all tests."""
        print("🚀 Running Grace Event Mesh Tests...\n")

        tests_passed = 0
        tests_total = 0

        # Test message envelope
        try:
            tests_total += 1
            if test_message_envelope():
                tests_passed += 1
        except Exception as e:
            print(f"❌ Message envelope test failed: {e}")

        print()

        # Test event mesh
        try:
            tests_total += 1
            if asyncio.run(test_event_mesh()):
                tests_passed += 1
        except Exception as e:
            print(f"❌ Event mesh test failed: {e}")

        print(f"\n📊 Results: {tests_passed}/{tests_total} tests passed")
        if tests_passed == tests_total:
            return
        else:
            assert False, f"{tests_total - tests_passed} event mesh tests failed"

    if __name__ == "__main__":
        success = run_tests()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Dependencies not available. Skipping event mesh tests.")
    sys.exit(0)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)
