"""
Memory Tracer - Specialized tracing for Grace memory operations and copilot integrations.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path

from .tracer import GTracer, TraceContext, TraceSpan, SpanKind, SpanStatus

logger = logging.getLogger(__name__)


class MemoryTracer:
    """Specialized tracer for Grace memory operations."""

    def __init__(self, base_tracer: GTracer):
        self.tracer = base_tracer

    def trace_file_upload(
        self,
        file_path: str,
        user_id: str,
        session_id: Optional[str] = None,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing a file upload operation."""
        tags = {
            "operation.type": "file_upload",
            "file.path": file_path,
            "file.name": Path(file_path).name,
            "file.extension": Path(file_path).suffix,
            "user.id": user_id,
            "memory.operation": "ingest",
        }

        if session_id:
            tags["session.id"] = session_id

        return self.tracer.start_trace(
            operation_name="memory.file_upload",
            parent_context=parent_context,
            kind=SpanKind.MEMORY_OPERATION,
            tags=tags,
        )

    def trace_text_ingestion(
        self,
        title: str,
        content_length: int,
        user_id: str,
        session_id: Optional[str] = None,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing a text content ingestion."""
        tags = {
            "operation.type": "text_ingestion",
            "content.title": title,
            "content.length": content_length,
            "user.id": user_id,
            "memory.operation": "ingest",
        }

        if session_id:
            tags["session.id"] = session_id

        return self.tracer.start_trace(
            operation_name="memory.text_ingestion",
            parent_context=parent_context,
            kind=SpanKind.MEMORY_OPERATION,
            tags=tags,
        )

    def trace_memory_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing a memory search operation."""
        tags = {
            "operation.type": "memory_search",
            "search.query": query[:100],  # Truncate long queries
            "search.query_length": len(query),
            "memory.operation": "search",
        }

        if user_id:
            tags["user.id"] = user_id
        if session_id:
            tags["session.id"] = session_id

        return self.tracer.start_trace(
            operation_name="memory.search",
            parent_context=parent_context,
            kind=SpanKind.MEMORY_OPERATION,
            tags=tags,
        )

    def trace_embedding_generation(
        self,
        text_chunks: int,
        model: str,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing embedding generation."""
        tags = {
            "operation.type": "embedding_generation",
            "embedding.model": model,
            "embedding.chunks": text_chunks,
            "memory.operation": "embed",
        }

        return self.tracer.start_trace(
            operation_name="memory.embedding_generation",
            parent_context=parent_context,
            kind=SpanKind.INTERNAL,
            tags=tags,
        )

    def trace_vector_storage(
        self,
        vector_count: int,
        dimension: int,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing vector storage operation."""
        tags = {
            "operation.type": "vector_storage",
            "vector.count": vector_count,
            "vector.dimension": dimension,
            "memory.operation": "store",
        }

        return self.tracer.start_trace(
            operation_name="memory.vector_storage",
            parent_context=parent_context,
            kind=SpanKind.INTERNAL,
            tags=tags,
        )

    def trace_text_extraction(
        self,
        file_path: str,
        extractor_type: str,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing text extraction from file."""
        tags = {
            "operation.type": "text_extraction",
            "file.path": file_path,
            "file.name": Path(file_path).name,
            "extractor.type": extractor_type,
            "memory.operation": "extract",
        }

        return self.tracer.start_trace(
            operation_name="memory.text_extraction",
            parent_context=parent_context,
            kind=SpanKind.INTERNAL,
            tags=tags,
        )

    def trace_text_chunking(
        self,
        content_length: int,
        chunker_type: str,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing text chunking operation."""
        tags = {
            "operation.type": "text_chunking",
            "content.length": content_length,
            "chunker.type": chunker_type,
            "memory.operation": "chunk",
        }

        return self.tracer.start_trace(
            operation_name="memory.text_chunking",
            parent_context=parent_context,
            kind=SpanKind.INTERNAL,
            tags=tags,
        )

    def trace_fragment_creation(
        self,
        fragment_id: str,
        content_type: str,
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing memory fragment creation."""
        tags = {
            "operation.type": "fragment_creation",
            "fragment.id": fragment_id,
            "fragment.content_type": content_type,
            "memory.operation": "create_fragment",
        }

        return self.tracer.start_trace(
            operation_name="memory.fragment_creation",
            parent_context=parent_context,
            kind=SpanKind.MEMORY_OPERATION,
            tags=tags,
        )

    def trace_copilot_operation(
        self,
        operation_type: str,
        context: Dict[str, Any],
        parent_context: Optional[TraceContext] = None,
    ) -> TraceSpan:
        """Start tracing a copilot-specific operation."""
        tags = {
            "operation.type": "copilot_operation",
            "copilot.operation": operation_type,
            "memory.operation": "copilot",
        }

        # Add context as tags (with safe conversion)
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                tags[f"copilot.{key}"] = value
            else:
                tags[f"copilot.{key}"] = str(value)

        return self.tracer.start_trace(
            operation_name=f"copilot.{operation_type}",
            parent_context=parent_context,
            kind=SpanKind.COPILOT_OPERATION,
            tags=tags,
        )

    def add_file_metadata(
        self,
        span: TraceSpan,
        file_size: int,
        mime_type: Optional[str] = None,
        checksum: Optional[str] = None,
    ) -> None:
        """Add file metadata to an existing span."""
        span.set_tag("file.size_bytes", file_size)
        if mime_type:
            span.set_tag("file.mime_type", mime_type)
        if checksum:
            span.set_tag("file.checksum", checksum)

    def add_processing_result(
        self,
        span: TraceSpan,
        chunks_created: int,
        fragments_created: int,
        embedding_dimension: int,
    ) -> None:
        """Add processing results to a span."""
        span.set_tag("processing.chunks_created", chunks_created)
        span.set_tag("processing.fragments_created", fragments_created)
        span.set_tag("processing.embedding_dimension", embedding_dimension)

        span.log(
            "processing_completed",
            {
                "chunks": chunks_created,
                "fragments": fragments_created,
                "embedding_dim": embedding_dimension,
            },
        )

    def add_search_result(
        self,
        span: TraceSpan,
        results_found: int,
        search_time_ms: float,
        similarity_threshold: float,
    ) -> None:
        """Add search results to a span."""
        span.set_tag("search.results_found", results_found)
        span.set_tag("search.time_ms", search_time_ms)
        span.set_tag("search.similarity_threshold", similarity_threshold)

        span.log(
            "search_completed",
            {
                "results": results_found,
                "duration_ms": search_time_ms,
                "threshold": similarity_threshold,
            },
        )

    def add_error_context(
        self,
        span: TraceSpan,
        error: Exception,
        operation_stage: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add error context to a span."""
        span.set_error(error)
        span.set_tag("error.stage", operation_stage)

        if context:
            for key, value in context.items():
                span.set_tag(f"error.context.{key}", str(value))

        span.log("error_occurred", {"stage": operation_stage, "context": context or {}})

    def get_memory_operation_stats(self) -> Dict[str, Any]:
        """Get statistics for memory operations."""
        finished_spans = self.tracer.get_finished_spans()

        memory_spans = [
            s
            for s in finished_spans
            if s.kind in [SpanKind.MEMORY_OPERATION, SpanKind.COPILOT_OPERATION]
        ]

        stats = {
            "total_memory_operations": len(memory_spans),
            "operations_by_type": {},
            "avg_duration_ms": 0.0,
            "error_rate": 0.0,
            "recent_errors": [],
        }

        if not memory_spans:
            return stats

        # Calculate statistics
        total_duration = sum(s.duration_ms or 0 for s in memory_spans)
        stats["avg_duration_ms"] = total_duration / len(memory_spans)

        error_count = sum(1 for s in memory_spans if s.status == SpanStatus.ERROR)
        stats["error_rate"] = error_count / len(memory_spans)

        # Group by operation type
        for span in memory_spans:
            op_type = span.tags.get("operation.type", "unknown")
            if op_type not in stats["operations_by_type"]:
                stats["operations_by_type"][op_type] = {
                    "count": 0,
                    "avg_duration_ms": 0.0,
                    "error_count": 0,
                }

            stats["operations_by_type"][op_type]["count"] += 1
            if span.status == SpanStatus.ERROR:
                stats["operations_by_type"][op_type]["error_count"] += 1

        # Calculate averages for each operation type
        for op_type, op_stats in stats["operations_by_type"].items():
            op_spans = [
                s for s in memory_spans if s.tags.get("operation.type") == op_type
            ]
            if op_spans:
                total_duration = sum(s.duration_ms or 0 for s in op_spans)
                op_stats["avg_duration_ms"] = total_duration / len(op_spans)

        # Get recent errors
        error_spans = [s for s in memory_spans[-50:] if s.status == SpanStatus.ERROR]
        stats["recent_errors"] = [
            {
                "operation": s.operation_name,
                "error_type": s.tags.get("error.type"),
                "error_message": s.tags.get("error.message"),
                "timestamp": s.end_time,
            }
            for s in error_spans[-10:]  # Last 10 errors
        ]

        return stats
