# Grace Trace (gtrace) - Comprehensive Tracing System

Grace Trace (gtrace) is a sophisticated distributed tracing system specifically designed for Grace's memory operations, copilot integrations, and cross-system observability. It provides complete visibility into file processing, memory operations, and copilot interactions.

## Features

### Core Tracing Capabilities
- **Distributed Tracing**: Full request correlation across all Grace components
- **Memory Operation Tracing**: Specialized tracing for file ingestion, text processing, and vector operations
- **Copilot Integration Tracing**: Dedicated tracing for copilot memory operations
- **Error Handling**: Comprehensive error context and exception tracking
- **Performance Monitoring**: Automatic detection of slow operations and bottlenecks

### Advanced Features
- **Trace Correlation**: Automatic correlation of related operations across system boundaries
- **Span Relationships**: Parent-child span relationships for complex workflows
- **Rich Metadata**: Extensive tagging and logging for operational context
- **Flexible Querying**: Advanced filtering and search capabilities
- **Performance Analytics**: Built-in performance metrics and slow operation detection

## Architecture

### Core Components

1. **GTracer**: Main distributed tracer
2. **MemoryTracer**: Specialized tracer for memory operations
3. **GTraceCollector**: Trace collection, storage, and querying
4. **TraceContext**: Cross-system correlation context
5. **TraceSpan**: Individual operation tracking

### Integration Points

- **Memory Ingestion Pipeline**: Full file processing tracing
- **Orb Interface**: Copilot operation tracing
- **Vector Operations**: Embedding and search tracing
- **Database Operations**: Fragment creation and storage tracing

## Usage Examples

### Basic Tracing

```python
from grace.gtrace import get_tracer

tracer = get_tracer()

# Synchronous operation
with tracer.span("my_operation", tags={"user_id": "123"}) as span:
    span.log("processing_started", {"items": 10})
    # Do work
    span.set_tag("result", "success")

# Asynchronous operation
async with tracer.async_span("async_operation") as span:
    result = await some_async_work()
    span.set_tag("result_count", len(result))
```

### Memory Operation Tracing

```python
from grace.gtrace import get_tracer, MemoryTracer

tracer = get_tracer()
memory_tracer = MemoryTracer(tracer)

# File upload tracing
span = memory_tracer.trace_file_upload("document.pdf", "user123")
memory_tracer.add_file_metadata(span, file_size=1024, mime_type="application/pdf")
memory_tracer.add_processing_result(span, chunks=5, fragments=5, embedding_dim=384)
tracer.finish_span(span)

# Memory search tracing
search_span = memory_tracer.trace_memory_search("find documents about AI")
memory_tracer.add_search_result(search_span, results=3, time_ms=45.2, threshold=0.8)
tracer.finish_span(search_span)
```

### Copilot Operation Tracing

```python
# Copilot-specific operations
copilot_span = memory_tracer.trace_copilot_operation(
    "document_analysis",
    {
        "document_type": "pdf",
        "pages": 10,
        "confidence": 0.95
    }
)
tracer.finish_span(copilot_span)
```

### Error Handling

```python
try:
    # Some operation that might fail
    process_file(file_path)
except Exception as e:
    memory_tracer.add_error_context(
        span, 
        e, 
        "file_processing",
        {"file_path": file_path, "user_id": user_id}
    )
    raise
```

### Trace Collection and Querying

```python
from grace.gtrace import GTraceCollector

collector = GTraceCollector()

# Query spans
memory_spans = collector.get_memory_operations(limit=50)
error_spans = collector.get_error_spans(limit=20)
slow_ops = collector.get_slow_operations(threshold_ms=1000)

# Get trace summary
trace_summary = collector.get_trace_summary(trace_id)
print(f"Trace had {trace_summary['span_count']} spans in {trace_summary['duration_ms']}ms")

# Get statistics
stats = collector.get_stats()
print(f"Total spans: {stats['current_stats']['total_spans']}")
```

## Configuration

### Tracer Configuration

```python
from grace.gtrace import GTracer, set_tracer

# Create custom tracer
tracer = GTracer(
    service_name="grace_production",
    collector_endpoint="http://jaeger:14268/api/traces"
)

# Set as global tracer
set_tracer(tracer)
```

### Collector Configuration

```python
collector = GTraceCollector(
    max_traces=50000,      # Maximum traces to store
    retention_hours=48      # How long to keep traces
)
```

## Performance Considerations

### Storage Management
- Automatic span eviction when limits are reached
- Configurable retention policies
- Efficient indexing for fast queries

### Overhead Minimization
- Lightweight span creation
- Asynchronous trace collection
- Optional sampling for high-volume operations

### Memory Usage
- Bounded memory usage with configurable limits
- Automatic cleanup of old traces
- Efficient data structures for span storage

## Integration with Existing Systems

### Memory Ingestion Pipeline
- Automatic tracing of file uploads
- Text extraction and chunking traces
- Embedding generation tracking
- Vector storage operation traces

### Orb Interface
- Copilot operation tracing
- Memory search operation traces
- Document upload tracing
- Session correlation

### Cross-System Correlation
- Automatic trace ID propagation
- Parent-child span relationships
- Context passing between components

## Monitoring and Observability

### Built-in Metrics
- Operation duration percentiles
- Error rates by operation type
- Memory operation statistics
- Slow operation detection

### Performance Analytics
- Automatic bottleneck identification
- Resource utilization tracking
- Operation frequency analysis
- Trust score correlation

### Error Analysis
- Comprehensive error context
- Exception stack traces
- Error categorization
- Recovery action tracking

## Best Practices

### Span Naming
- Use hierarchical naming: `component.operation`
- Be specific: `memory.file_upload` vs `upload`
- Include context: `copilot.document_analysis`

### Tagging Strategy
- Add user context: `user.id`, `session.id`
- Include operation metadata: `file.type`, `content.length`
- Track performance: `duration_ms`, `result_count`

### Error Handling
- Always add error context
- Include recovery actions
- Tag error severity levels

### Performance Optimization
- Use sampling for high-frequency operations
- Batch span collection when possible
- Set appropriate retention policies

## Future Enhancements

### Planned Features
- Integration with OpenTelemetry
- Advanced analytics and ML-driven insights
- Real-time alerting and monitoring
- Custom trace visualization

### Export Capabilities
- Jaeger integration
- Zipkin compatibility
- Prometheus metrics export
- Custom webhook support

## Testing

The system includes comprehensive tests:

- **Unit Tests**: `grace/gtrace/test_gtrace.py`
- **Integration Tests**: `grace/gtrace/test_integration.py`

Run tests with:
```bash
cd /path/to/Grace-
PYTHONPATH=/path/to/Grace- python grace/gtrace/test_gtrace.py
PYTHONPATH=/path/to/Grace- python grace/gtrace/test_integration.py
```

## Conclusion

Grace Trace provides comprehensive observability for the Grace system, with special focus on memory operations and copilot integrations. It enables developers and operators to understand system behavior, identify performance bottlenecks, and troubleshoot issues effectively.

The system is designed to be lightweight, efficient, and easy to use while providing powerful insights into system operation. It integrates seamlessly with existing Grace components and provides the foundation for advanced monitoring and analytics capabilities.