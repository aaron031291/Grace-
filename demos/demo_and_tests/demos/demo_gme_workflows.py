#!/usr/bin/env python3
"""
Grace Communications Schema Examples - Demonstrating GME usage across kernels.
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone

# Add Grace to Python path
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
from grace.core.event_bus import EventBus


async def example_intelligence_workflow():
    """Example: Intelligence kernel workflow using GME."""
    print("📊 Intelligence Kernel GME Workflow")
    print("=" * 50)

    # 1. Intelligence request
    request_envelope = create_envelope(
        kind=MessageKind.QUERY,
        domain="intelligence",
        name="INTEL_REQUESTED",
        payload={
            "query": "Analyze this document for policy compliance",
            "context": {
                "document_id": "doc_abc123",
                "requester": "governance_kernel",
                "urgency": "high",
            },
        },
        priority=Priority.P0,
        qos=QoSClass.REALTIME,
        rbac=["intel.request", "intel.read"],
        governance_label="restricted",
    )

    print(f"✅ Created intelligence request: {request_envelope.msg_id}")
    print(f"   - Correlation ID: {request_envelope.headers.correlation_id}")
    print(f"   - Priority: {request_envelope.headers.priority}")
    print(f"   - RBAC: {request_envelope.headers.rbac}")

    # 2. Intelligence response
    response_envelope = create_envelope(
        kind=MessageKind.REPLY,
        domain="intelligence",
        name="INTEL_INFER_COMPLETED",
        payload={
            "request_id": "doc_abc123",
            "result": {
                "compliance_score": 0.95,
                "violations": [],
                "confidence": 0.98,
                "reasoning": "Document complies with all governance policies",
            },
            "processing_time_ms": 250,
        },
        priority=Priority.P0,
        qos=QoSClass.REALTIME,
        correlation_id=request_envelope.headers.correlation_id,  # Link to original request
        partition_key=request_envelope.headers.partition_key,  # Same partition for ordering
    )

    print(f"✅ Created intelligence response: {response_envelope.msg_id}")
    print(f"   - Linked to request via correlation_id")
    print(f"   - Same partition key for ordering")

    # Validate both envelopes
    req_validation = validate_envelope(request_envelope.model_dump())
    resp_validation = validate_envelope(response_envelope.model_dump())

    print(f"✅ Request validation: {'PASSED' if req_validation.passed else 'FAILED'}")
    print(f"✅ Response validation: {'PASSED' if resp_validation.passed else 'FAILED'}")


async def example_mldl_deployment():
    """Example: MLDL deployment workflow requiring governance approval."""
    print("\n🤖 MLDL Deployment GME Workflow")
    print("=" * 50)

    # 1. Deployment request
    deploy_request = create_envelope(
        kind=MessageKind.COMMAND,
        domain="mldl",
        name="MLDL_DEPLOYMENT_REQUESTED",
        payload={
            "model_key": "sentiment.classifier.bert",
            "version": "2.1.0",
            "environment": "production",
            "deployment_config": {
                "replicas": 3,
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "auto_scaling": True,
            },
        },
        priority=Priority.P0,  # High priority for production deployments
        qos=QoSClass.STANDARD,
        idempotency_key="deploy_sentiment_2.1.0_prod",  # Prevent duplicate deployments
        rbac=["mldl.deploy", "prod.write"],
        governance_label="internal",
    )

    print(f"✅ Created deployment request: {deploy_request.msg_id}")
    print(f"   - Idempotency key: {deploy_request.headers.idempotency_key}")
    print(f"   - Governance label: {deploy_request.headers.governance_label}")

    # 2. Governance approval (using same correlation_id)
    approval = create_envelope(
        kind=MessageKind.EVENT,
        domain="governance",
        name="GOVERNANCE_APPROVED",
        payload={
            "request_id": deploy_request.msg_id,
            "approved": True,
            "conditions": [
                "Require canary deployment first",
                "Monitor error rates for 24h",
            ],
            "approver": "governance_kernel",
            "reasoning": "Model version approved after policy review",
        },
        priority=Priority.P0,
        correlation_id=deploy_request.headers.correlation_id,
        partition_key=deploy_request.headers.partition_key,
        rbac=["gov.approve"],
        governance_label="internal",
    )

    print(f"✅ Created governance approval: {approval.msg_id}")

    # 3. Actual deployment event
    deployment = create_envelope(
        kind=MessageKind.EVENT,
        domain="mldl",
        name="MLDL_DEPLOYMENT_STARTED",
        payload={
            "deployment_id": "deploy_12345",
            "model_key": "sentiment.classifier.bert",
            "version": "2.1.0",
            "environment": "production",
            "strategy": "canary",
            "expected_completion": "2025-09-28T15:30:00Z",
        },
        priority=Priority.P1,
        correlation_id=deploy_request.headers.correlation_id,
        partition_key=deploy_request.headers.partition_key,
        causation_id=approval.msg_id,  # Shows this was caused by approval
    )

    print(f"✅ Created deployment event: {deployment.msg_id}")
    print(f"   - Causation chain: request -> approval -> deployment")


async def example_experience_collection():
    """Example: MLT experience collection across kernels."""
    print("\n📈 MLT Experience Collection GME Workflow")
    print("=" * 50)

    # Experience from Multi-OS kernel
    mos_experience = create_envelope(
        kind=MessageKind.EVENT,
        domain="mlt",
        name="MLT_EXPERIENCE",
        payload={
            "source": "multi_os_kernel",
            "task": "task_scheduling",
            "context": {
                "host_count": 5,
                "os_distribution": {"linux": 3, "windows": 1, "macos": 1},
                "resource_constraints": {"cpu": "high", "memory": "normal"},
            },
            "metrics": {
                "placement_success_rate": 0.95,
                "avg_placement_time_ms": 150,
                "resource_utilization": 0.78,
            },
            "outcome": "success",
            "timestamp": datetime.utcnow().isoformat(),
        },
        priority=Priority.P3,  # Low priority background data
        qos=QoSClass.BULK,
        partition_key="mos_learning",
        rbac=["mlt.experience.write"],
    )

    print(f"✅ Created Multi-OS experience: {mos_experience.msg_id}")

    # Experience from Intelligence kernel
    intel_experience = create_envelope(
        kind=MessageKind.EVENT,
        domain="mlt",
        name="MLT_EXPERIENCE",
        payload={
            "source": "intelligence_kernel",
            "task": "document_analysis",
            "context": {
                "document_type": "policy",
                "specialist_count": 4,
                "complexity": "high",
            },
            "metrics": {
                "avg_confidence": 0.92,
                "processing_time_ms": 850,
                "consensus_rate": 0.88,
            },
            "outcome": "success",
            "timestamp": datetime.utcnow().isoformat(),
        },
        priority=Priority.P3,
        qos=QoSClass.BULK,
        partition_key="intel_learning",
        rbac=["mlt.experience.write"],
    )

    print(f"✅ Created Intelligence experience: {intel_experience.msg_id}")

    # MLT insight generated from experiences
    insight = create_envelope(
        kind=MessageKind.EVENT,
        domain="mlt",
        name="MLT_INSIGHT_READY",
        payload={
            "insight_id": "insight_12345",
            "type": "performance_optimization",
            "scope": "multi_os_scheduling",
            "evidence": {
                "sample_size": 150,
                "confidence_interval": 0.95,
                "key_factors": ["cpu_availability", "os_type", "task_complexity"],
            },
            "recommendation": {
                "action": "adjust_placement_weights",
                "parameters": {
                    "cpu_weight": 0.45,
                    "os_affinity_weight": 0.25,
                    "latency_weight": 0.30,
                },
            },
            "confidence": 0.87,
        },
        priority=Priority.P1,
        qos=QoSClass.STANDARD,
        partition_key="mlt_insights",
        rbac=["mlt.insight.read"],
    )

    print(f"✅ Created MLT insight: {insight.msg_id}")
    print(f"   - Confidence: {insight.payload['confidence']}")


async def example_error_handling():
    """Example: Error handling and DLQ scenarios."""
    print("\n⚠️  Error Handling & Dead Letter Queue")
    print("=" * 50)

    # Failed message that needs retry
    failed_message = create_envelope(
        kind=MessageKind.COMMAND,
        domain="ingress",
        name="ING_PROCESS_SOURCE",
        payload={
            "source_id": "src_unreachable_123",
            "url": "https://unreachable.example.com/data",
            "max_retries": 3,
        },
        priority=Priority.P2,
        qos=QoSClass.STANDARD,
        headers_kwargs={
            "redelivery_count": 3,  # This message has already been retried
            "retry_policy": {
                "strategy": "exp",
                "max_attempts": 5,
                "base_ms": 1000,
                "jitter_ms": 500,
            },
        },
    )

    print(f"✅ Created failed message: {failed_message.msg_id}")
    print(f"   - Redelivery count: {failed_message.headers.redelivery_count}")
    print(f"   - Retry strategy: {failed_message.headers.retry_policy}")

    # DLQ message (simulated)
    dlq_message = create_envelope(
        kind=MessageKind.EVENT,
        domain="comms",
        name="MESSAGE_SENT_TO_DLQ",
        payload={
            "original_msg_id": failed_message.msg_id,
            "reason": "max_retries_exceeded",
            "error_details": {
                "error_code": "E.UNAVAILABLE",
                "error_message": "Source endpoint unreachable after 3 retries",
                "last_attempt": datetime.utcnow().isoformat(),
            },
            "dlq_topic": "grace.dlq",
            "retention_days": 7,
        },
        priority=Priority.P2,
        correlation_id=failed_message.headers.correlation_id,
        rbac=["dlq.write", "audit.write"],
    )

    print(f"✅ Created DLQ message: {dlq_message.msg_id}")
    print(f"   - Original message: {dlq_message.payload['original_msg_id']}")
    print(f"   - Error code: {dlq_message.payload['error_details']['error_code']}")


async def demonstrate_event_bus_integration():
    """Demonstrate GME integration with existing event bus."""
    print("\n🔌 Event Bus Integration")
    print("=" * 50)

    event_bus = EventBus()

    # Create a handler that processes GME messages
    async def gme_handler(event):
        """Handle GME-formatted events."""
        print(f"📨 Received GME event: {event.get('type', 'unknown')}")

        # If it's a GME envelope, validate it
        if "msg_id" in event.get("payload", {}):
            validation = validate_envelope(event["payload"])
            status = "✅ VALID" if validation.passed else "❌ INVALID"
            print(f"   Validation: {status}")

    # Subscribe to GME events
    await event_bus.subscribe("GME_MESSAGE", gme_handler)

    # Create and publish a GME message through the event bus
    sample_envelope = create_envelope(
        kind=MessageKind.EVENT,
        domain="demo",
        name="DEMO_EVENT",
        payload={"message": "Hello from GME!"},
        priority=Priority.P2,
    )

    # Publish GME envelope via existing event bus
    correlation_id = await event_bus.publish(
        "GME_MESSAGE", {"envelope": sample_envelope.model_dump()}
    )

    print(f"✅ Published GME message via event bus")
    print(f"   - Message ID: {sample_envelope.msg_id}")
    print(f"   - Bus Correlation ID: {correlation_id}")

    # Give handler time to process
    await asyncio.sleep(0.1)


async def main():
    """Run all Grace Communications examples."""
    print("🚀 Grace Communications Schema Examples")
    print("=" * 60)
    print("Demonstrating GME (Grace Message Envelope) usage across kernels")
    print("=" * 60)

    await example_intelligence_workflow()
    await example_mldl_deployment()
    await example_experience_collection()
    await example_error_handling()
    await demonstrate_event_bus_integration()

    print("\n🎉 Grace Communications Schema Examples Complete!")
    print("\nKey Benefits Demonstrated:")
    print("• Unified message format across all kernels")
    print("• Consistent routing and priority handling")
    print("• Built-in governance and security controls")
    print("• Traceability through correlation IDs")
    print("• Error handling and retry policies")
    print("• Integration with existing event bus")


if __name__ == "__main__":
    asyncio.run(main())
