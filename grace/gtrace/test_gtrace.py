#!/usr/bin/env python3
"""
Test Grace Trace (gtrace) implementation.
"""

import asyncio
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.gtrace import get_tracer, MemoryTracer, GTraceCollector
from grace.gtrace.tracer import SpanKind, SpanStatus


async def test_basic_tracing_async():
    """Test basic tracing functionality."""
    print("🔍 Testing basic Grace tracing...")

    tracer = get_tracer()
    print(f"✅ Tracer initialized: {tracer.service_name}")

    # Test basic span creation
    with tracer.span("test_operation", tags={"test": "basic"}) as span:
        span.log("test_event", {"message": "Testing basic span"})
        span.set_tag("result", "success")

    # Test async span
    async with tracer.async_span(
        "async_test_operation", kind=SpanKind.INTERNAL
    ) as span:
        span.log("async_test", {"message": "Testing async span"})
        span.set_tag("async", True)

    # Test error handling
    try:
        with tracer.span("error_test") as span:
            raise Exception("Test error")
    except Exception:
        pass  # Expected

    # Get stats
    stats = tracer.get_stats()
    print(f"✅ Basic tracing test completed. Stats: {stats}")
    return True


async def test_memory_tracing_async():
    """Test memory-specific tracing."""
    print("🧠 Testing memory tracing...")

    tracer = get_tracer()
    memory_tracer = MemoryTracer(tracer)

    # Test file upload tracing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write("Test content for memory tracing")
        tmp_path = tmp_file.name

    try:
        # Test file upload trace
        span = memory_tracer.trace_file_upload(
            tmp_path, "test_user", "test_session_123"
        )

        # Add metadata
        memory_tracer.add_file_metadata(span, 1024, "text/plain", "abc123")
        memory_tracer.add_processing_result(span, 5, 5, 384)

        tracer.finish_span(span)

        # Test memory search tracing
        search_span = memory_tracer.trace_memory_search(
            "test query", "test_user", "test_session_123"
        )

        memory_tracer.add_search_result(search_span, 3, 45.2, 0.8)
        tracer.finish_span(search_span)

        # Test copilot operation
        copilot_span = memory_tracer.trace_copilot_operation(
            "document_analysis",
            {"document_type": "pdf", "pages": 10, "confidence": 0.95},
        )

        tracer.finish_span(copilot_span)

        # Get memory operation stats
        stats = memory_tracer.get_memory_operation_stats()
        print(f"✅ Memory operations: {stats['total_memory_operations']}")
        print(f"✅ Operation types: {list(stats['operations_by_type'].keys())}")

    finally:
        # Clean up
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return True


async def test_trace_collector_async():
    """Test trace collection and querying."""
    print("📊 Testing trace collector...")

    tracer = get_tracer()
    collector = GTraceCollector(max_traces=100, retention_hours=1)

    # Create some test traces
    for i in range(10):
        with tracer.span(f"test_operation_{i}", tags={"iteration": i}) as span:
            span.log("iteration_event", {"iteration": i})

            # Create child span
            with tracer.span(
                f"child_operation_{i}", parent_context=span.context
            ) as child_span:
                child_span.set_tag("parent_operation", f"test_operation_{i}")

            # Collect the spans
            collector.collect_span(child_span)

        collector.collect_span(span)

    # Test querying
    all_spans = collector.get_spans(limit=50)
    print(f"✅ Collected {len(all_spans)} spans")

    # Test filtering
    test_spans = collector.get_spans(operation_name="test_operation_5")
    print(f"✅ Found {len(test_spans)} spans for specific operation")

    # Test trace retrieval
    if all_spans:
        trace_id = all_spans[0].context.trace_id
        trace_spans = collector.get_trace(trace_id)
        print(f"✅ Trace {trace_id} has {len(trace_spans)} spans")

        # Test trace summary
        summary = collector.get_trace_summary(trace_id)
        print(
            f"✅ Trace summary: {summary['span_count']} spans, {summary['duration_ms']:.2f}ms"
        )

    # Test collector stats
    stats = collector.get_stats()
    print(f"✅ Collector stats: {stats['current_stats']['total_spans']} total spans")

    return True


async def test_error_tracing_async():
    """Test error handling in traces."""
    print("❌ Testing error tracing...")

    tracer = get_tracer()
    memory_tracer = MemoryTracer(tracer)

    # Test error in memory operation
    span = memory_tracer.trace_file_upload("nonexistent_file.txt", "test_user")

    try:
        # Simulate an error
        raise FileNotFoundError("File not found")
    except Exception as e:
        memory_tracer.add_error_context(
            span, e, "file_reading", {"file_path": "nonexistent_file.txt"}
        )

    tracer.finish_span(span)

    # Verify error was recorded
    if span.status == SpanStatus.ERROR:
        print("✅ Error properly recorded in span")
        print(f"✅ Error type: {span.tags.get('error.type')}")
        print(f"✅ Error message: {span.tags.get('error.message')}")
    else:
        print("❌ Error not properly recorded")
        return False

    return True


async def test_trace_correlation_async():
    """Test trace correlation across operations."""
    print("🔗 Testing trace correlation...")

    tracer = get_tracer()
    memory_tracer = MemoryTracer(tracer)

    # Start a parent operation
    async with tracer.async_span("document_processing_pipeline") as parent_span:
        parent_trace_id = parent_span.context.trace_id

        # Create child operations that would typically happen in different components
        upload_span = memory_tracer.trace_file_upload(
            "test_document.pdf", "user123", parent_context=parent_span.context
        )
        tracer.finish_span(upload_span)

        extraction_span = memory_tracer.trace_text_extraction(
            "test_document.pdf", "pdf_extractor", parent_context=parent_span.context
        )
        tracer.finish_span(extraction_span)

        chunking_span = memory_tracer.trace_text_chunking(
            10000, "recursive_chunker", parent_context=parent_span.context
        )
        tracer.finish_span(chunking_span)

        embedding_span = memory_tracer.trace_embedding_generation(
            5, "sentence-transformers", parent_context=parent_span.context
        )
        tracer.finish_span(embedding_span)

        vector_span = memory_tracer.trace_vector_storage(
            5, 384, parent_context=parent_span.context
        )
        tracer.finish_span(vector_span)

    # Verify all spans share the same trace ID
    all_spans = tracer.get_finished_spans()
    correlated_spans = [s for s in all_spans if s.context.trace_id == parent_trace_id]

    print(f"✅ Found {len(correlated_spans)} spans with trace ID {parent_trace_id}")

    # Verify parent-child relationships
    parent_spans = [s for s in correlated_spans if s.context.parent_span_id is None]
    child_spans = [s for s in correlated_spans if s.context.parent_span_id is not None]

    print(f"✅ {len(parent_spans)} parent spans, {len(child_spans)} child spans")

    return (
        len(correlated_spans) >= 6
    )  # Should have at least 6 spans (parent + 5 children)


# Synchronous wrappers so pytest can run these without pytest-asyncio installed
def test_basic_tracing():
    assert asyncio.run(test_basic_tracing_async())


def test_memory_tracing():
    assert asyncio.run(test_memory_tracing_async())


def test_trace_collector():
    assert asyncio.run(test_trace_collector_async())


def test_error_tracing():
    assert asyncio.run(test_error_tracing_async())


def test_trace_correlation():
    assert asyncio.run(test_trace_correlation_async())


async def run_all_tests():
    """Run all GTrace tests."""
    print("🚀 Starting Grace Trace (gtrace) Tests...")
    print("=" * 60)

    tests = [
        ("Basic Tracing", test_basic_tracing),
        ("Memory Tracing", test_memory_tracing),
        ("Trace Collector", test_trace_collector),
        ("Error Tracing", test_error_tracing),
        ("Trace Correlation", test_trace_correlation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = await test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Grace Trace is working correctly.")
        return True
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_tests())
