"""Tracing initialization."""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def init_tracing(app=None):
    """Initialize OpenTelemetry tracing for FastAPI, Postgres, Redis."""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"),
        insecure=True
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)
    if app:
        FastAPIInstrumentor.instrument_app(app)
    try:
        AsyncPGInstrumentor().instrument()
    except Exception:
        pass
    try:
        RedisInstrumentor().instrument()
    except Exception:
        pass