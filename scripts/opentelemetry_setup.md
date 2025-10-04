# OpenTelemetry Setup for Grace
# Add to backend/main.py and grace/worker/worker_service.py

# Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp

# Example FastAPI instrumentation (backend/main.py):
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import trace

provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
FastAPIInstrumentor.instrument_app(app)

# Example worker instrumentation (grace/worker/worker_service.py):
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

async def _process_task(...):
    with tracer.start_as_current_span(f"process_task_{queue_name}"):
        ...existing code...

# This wires traces from API to worker and can be extended to DB calls.
# Configure OTLP endpoint for your collector (e.g., Jaeger, Tempo, or OpenTelemetry Collector).
