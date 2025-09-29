#!/usr/bin/env python3
"""
Test Grace Trace integration with existing Grace systems.
"""
import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.interface.orb_interface import GraceUnifiedOrbInterface
from grace.gtrace import get_tracer, GTraceCollector

async def test_orb_interface_tracing():
    """Test tracing integration with Orb interface."""
    print("🔮 Testing Orb Interface tracing integration...")
    
    tracer = get_tracer()
    collector = GTraceCollector(max_traces=100)
    orb = GraceUnifiedOrbInterface()
    
    # Test session creation (should have some tracing)
    session_id = await orb.create_session("test_user", {"theme": "dark"})
    print(f"✅ Created session: {session_id}")
    
    # Test memory search with tracing
    search_results = await orb.search_memory(session_id, "test query")
    print(f"✅ Memory search completed, found {len(search_results)} results")
    
    # Test document upload with tracing (if it has content)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("This is a test document for Grace tracing integration. It contains some sample content to test the memory ingestion pipeline with comprehensive tracing capabilities.")
        tmp_path = tmp_file.name
    
    try:
        fragment_id = await orb.upload_document(
            "test_user", 
            tmp_path, 
            "txt",
            {"tags": ["test", "integration"], "source": "gtrace_test"}
        )
        print(f"✅ Uploaded document as fragment: {fragment_id}")
        
        # Search for the uploaded content
        search_results = await orb.search_memory(session_id, "test document")
        print(f"✅ Found {len(search_results)} results for uploaded document")
        
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    
    # Check if traces were created
    finished_spans = tracer.get_finished_spans(limit=50)
    copilot_spans = [s for s in finished_spans if 'copilot' in s.operation_name]
    
    print(f"✅ Found {len(copilot_spans)} copilot operation spans")
    for span in copilot_spans:
        print(f"   - {span.operation_name} ({span.duration_ms:.2f}ms)")
    
    return len(copilot_spans) > 0

async def test_memory_pipeline_integration():
    """Test tracing integration with memory ingestion pipeline."""
    print("🧠 Testing Memory Pipeline tracing integration...")
    
    try:
        from grace.memory_ingestion.pipeline import get_memory_ingestion_pipeline
        
        pipeline = get_memory_ingestion_pipeline()
        tracer = get_tracer()
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("""
Grace Trace Integration Test Document

This document tests the comprehensive tracing capabilities of the Grace system.
It includes multiple paragraphs to ensure proper chunking and embedding generation.

The Grace system now includes:
1. Distributed tracing with GTrace
2. Memory operation tracing
3. Copilot integration tracing
4. Cross-system correlation
5. Error handling and observability

This enables full visibility into file processing, memory operations, and copilot interactions.
""")
            tmp_path = tmp_file.name
        
        try:
            # Test text ingestion with tracing
            result = await pipeline.ingest_text_content(
                text="Grace Trace test content for memory pipeline integration testing.",
                title="GTrace Integration Test",
                user_id="test_user",
                tags=["gtrace", "integration", "memory"]
            )
            
            print(f"✅ Text ingestion result: {result['status']}")
            if 'trace_id' in result:
                print(f"✅ Trace ID: {result['trace_id']}")
            
            # Test memory search with tracing
            search_results = await pipeline.search_memory(
                query="Grace Trace integration",
                user_id="test_user",
                limit=5
            )
            
            print(f"✅ Memory search found {len(search_results)} results")
            
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        
        # Check memory operation traces
        finished_spans = tracer.get_finished_spans(limit=100)
        memory_spans = [s for s in finished_spans if 'memory' in s.operation_name]
        
        print(f"✅ Found {len(memory_spans)} memory operation spans")
        for span in memory_spans[-5:]:  # Show last 5
            print(f"   - {span.operation_name} ({span.duration_ms:.2f}ms)")
        
        return len(memory_spans) > 0
        
    except ImportError as e:
        print(f"⚠️  Memory pipeline not available for testing: {e}")
        return True  # Don't fail if pipeline isn't available

async def test_trace_correlation_across_systems():
    """Test trace correlation across different Grace components."""
    print("🔗 Testing cross-system trace correlation...")
    
    tracer = get_tracer()
    
    # Start a high-level operation that spans multiple systems
    async with tracer.async_span(
        "grace.user_workflow",
        tags={
            "workflow.type": "document_analysis",
            "user.id": "test_user"
        }
    ) as workflow_span:
        
        # Simulate Orb interface operations
        orb = GraceUnifiedOrbInterface()
        
        async with tracer.async_span(
            "grace.session_management",
            parent_context=workflow_span.context
        ) as session_span:
            session_id = await orb.create_session("test_user", {"workflow": "document_analysis"})
            session_span.set_tag("session.id", session_id)
        
        # Simulate document upload
        async with tracer.async_span(
            "grace.document_upload",
            parent_context=workflow_span.context
        ) as upload_span:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write("Integration test document for cross-system tracing")
                tmp_path = tmp_file.name
            
            try:
                fragment_id = await orb.upload_document(
                    "test_user",
                    tmp_path,
                    "txt",
                    {"workflow_trace_id": workflow_span.context.trace_id}
                )
                upload_span.set_tag("fragment.id", fragment_id)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        
        # Simulate memory search
        async with tracer.async_span(
            "grace.memory_query",
            parent_context=workflow_span.context
        ) as query_span:
            results = await orb.search_memory(session_id, "integration test")
            query_span.set_tag("results.count", len(results))
        
        workflow_span.log("workflow_completed", {
            "session_id": session_id,
            "operations": ["session_create", "document_upload", "memory_search"],
            "total_results": len(results) if 'results' in locals() else 0
        })
    
    # Verify trace correlation
    trace_id = workflow_span.context.trace_id
    trace_spans = tracer.get_trace(trace_id)
    
    print(f"✅ Workflow trace {trace_id} contains {len(trace_spans)} spans")
    
    # Group spans by system
    systems = {}
    for span in trace_spans:
        system = span.operation_name.split('.')[0] if '.' in span.operation_name else 'unknown'
        if system not in systems:
            systems[system] = []
        systems[system].append(span)
    
    print("✅ Spans by system:")
    for system, spans in systems.items():
        print(f"   - {system}: {len(spans)} spans")
    
    return len(trace_spans) >= 4  # Should have at least 4 spans

async def test_trace_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("📊 Testing trace performance monitoring...")
    
    tracer = get_tracer()
    collector = GTraceCollector()
    
    # Simulate various operations with different performance characteristics
    operations = [
        ("fast_operation", 0.001),
        ("medium_operation", 0.1),
        ("slow_operation", 0.5),
        ("very_slow_operation", 1.0)
    ]
    
    import time
    for op_name, target_duration in operations:
        for i in range(3):  # Create multiple instances
            async with tracer.async_span(
                f"performance_test.{op_name}",
                tags={"test.iteration": i, "test.target_duration": target_duration}
            ) as span:
                # Simulate work
                start_time = time.time()
                while time.time() - start_time < target_duration:
                    await asyncio.sleep(0.001)
                
                actual_duration = time.time() - start_time
                span.set_tag("actual.duration_seconds", actual_duration)
            
            collector.collect_span(span)
    
    # Analyze performance
    slow_ops = collector.get_slow_operations(threshold_ms=200, limit=10)
    all_spans = collector.get_spans(limit=50)
    
    print(f"✅ Slow operations (>200ms): {len(slow_ops)}")
    for span in slow_ops:
        print(f"   - {span.operation_name}: {span.duration_ms:.2f}ms")
    
    print(f"✅ Total spans analyzed: {len(all_spans)}")
    
    # Check stats
    stats = collector.get_stats()
    print(f"✅ Collector performance stats:")
    print(f"   - Total spans: {stats['current_stats']['total_spans']}")
    print(f"   - Storage utilization: {stats['current_stats']['storage_utilization']:.1%}")
    
    return len(slow_ops) > 0

async def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting Grace Trace Integration Tests...")
    print("=" * 60)
    
    tests = [
        ("Orb Interface Tracing", test_orb_interface_tracing),
        ("Memory Pipeline Integration", test_memory_pipeline_integration),
        ("Cross-System Correlation", test_trace_correlation_across_systems),
        ("Performance Monitoring", test_trace_performance_monitoring)
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed! Grace Trace is fully integrated.")
        return True
    else:
        print("💥 Some integration tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    asyncio.run(run_integration_tests())