#!/usr/bin/env python3
"""
Test script for Grace Communications Schema validation.
"""

import sys
import json
from pathlib import Path

# Add the Grace root to Python path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from grace.comms import (
    create_envelope,
    MessageKind,
    Priority,
    QoSClass,
    validate_envelope,
)


def test_basic_envelope_creation():
    """Test basic envelope creation."""
    print("🧪 Testing basic envelope creation...")

    envelope = create_envelope(
        kind=MessageKind.EVENT,
        domain="intelligence",
        name="INTEL_INFER_COMPLETED",
        payload={
            "request_id": "req_abc123",
            "result": {
                "confidence": 0.95,
                "answer": "This document appears to be technical documentation.",
            },
        },
        priority=Priority.P0,
        qos=QoSClass.REALTIME,
    )

    print(f"✅ Created envelope with msg_id: {envelope.msg_id}")
    print(f"   - Domain: {envelope.domain}")
    print(f"   - Name: {envelope.name}")
    print(f"   - Priority: {envelope.headers.priority}")
    print(f"   - QoS: {envelope.headers.qos}")
    print(f"   - Correlation ID: {envelope.headers.correlation_id}")
    return envelope


def test_envelope_validation():
    """Test envelope validation."""
    print("\n🧪 Testing envelope validation...")

    # Create a test envelope
    envelope = create_envelope(
        kind=MessageKind.COMMAND,
        domain="mldl",
        name="MLDL_DEPLOY",
        payload={
            "model_key": "classification.xgb",
            "version": "1.2.3",
            "environment": "production",
        },
        priority=Priority.P0,
        idempotency_key="deploy_xgb_123",
    )

    # Validate it
    result = validate_envelope(envelope.model_dump())

    if result.passed:
        print("✅ Envelope validation passed")
    else:
        print("❌ Envelope validation failed:")
        for error in result.errors:
            print(f"   - {error}")

    if result.warnings:
        print("⚠️  Validation warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")

    return result.passed


def test_large_payload_warning():
    """Test large payload warning."""
    print("\n🧪 Testing large payload handling...")

    # Create envelope with large payload
    large_payload = {"data": "x" * 300000}  # 300KB payload

    envelope = create_envelope(
        kind=MessageKind.EVENT,
        domain="ingress",
        name="ING_CAPTURED_RAW",
        payload=large_payload,
    )

    result = validate_envelope(envelope.model_dump())

    has_size_warning = any("Large payload" in warning for warning in result.warnings)
    if has_size_warning:
        print("✅ Large payload warning triggered correctly")
    else:
        print("❌ Large payload warning not triggered")

    return has_size_warning


def test_schema_files_exist():
    """Test that schema files exist."""
    print("\n🧪 Testing schema files exist...")

    schemas_dir = Path(__file__).parent / "contracts" / "comms"
    expected_files = [
        "envelope.schema.json",
        "rpc.schema.json",
        "errors.schema.json",
        "topics.yaml",
        "events.master.yaml",
        "registry.yaml",
        "efficiency.yaml",
        "qos.yaml",
        "security.yaml",
        "observability.yaml",
        "transports.yaml",
        "bindings.yaml",
        "experience.schema.json",
        "snapshot.schema.json",
        "defaults.yaml",
    ]

    missing_files = []
    for filename in expected_files:
        filepath = schemas_dir / filename
        if not filepath.exists():
            missing_files.append(filename)

    if not missing_files:
        print(f"✅ All {len(expected_files)} schema files present")
        return True
    else:
        print(f"❌ Missing {len(missing_files)} schema files:")
        for filename in missing_files:
            print(f"   - {filename}")
        return False


def test_envelope_json_schema():
    """Test that envelope JSON schema is valid."""
    print("\n🧪 Testing envelope JSON schema validity...")

    schema_path = Path(__file__).parent / "contracts" / "comms" / "envelope.schema.json"

    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Basic schema validation
        required_keys = ["$id", "$schema", "title", "type", "properties"]
        missing_keys = [key for key in required_keys if key not in schema]

        if missing_keys:
            print(f"❌ Schema missing keys: {missing_keys}")
            return False

        # Check required properties
        required_props = ["msg_id", "kind", "domain", "name", "ts", "headers"]
        schema_required = schema.get("required", [])
        missing_required = [
            prop for prop in required_props if prop not in schema_required
        ]

        if missing_required:
            print(f"❌ Schema missing required properties: {missing_required}")
            return False

        print("✅ Envelope JSON schema is valid")
        return True

    except Exception as e:
        print(f"❌ Error validating schema: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running Grace Communications Schema Tests\n")

    tests = [
        ("Schema Files Exist", test_schema_files_exist),
        ("Envelope JSON Schema", test_envelope_json_schema),
        ("Basic Envelope Creation", test_basic_envelope_creation),
        ("Envelope Validation", test_envelope_validation),
        ("Large Payload Warning", test_large_payload_warning),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            if result or result is None:  # None counts as passed
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
