"""
Performance benchmarks for KPI validation
"""

import pytest
import asyncio


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_event_emit_throughput(benchmark):
    """Benchmark: Event emission throughput"""
    from grace.integration.event_bus import EventBus
    from grace.schemas.events import GraceEvent
    
    bus = EventBus()
    
    async def emit_events():
        for i in range(100):
            event = GraceEvent(
                event_type="test.benchmark",
                source="benchmark",
                payload={"seq": i}
            )
            await bus.emit(event)
    
    # Benchmark should achieve > 100 events/sec
    result = benchmark(lambda: asyncio.run(emit_events()))


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_mcp_validation_performance(benchmark):
    """Benchmark: MCP message validation"""
    from grace.mcp import MCPClient, MCPMessageType
    from grace.integration.event_bus import EventBus
    
    bus = EventBus()
    client = MCPClient("benchmark_kernel", bus)
    
    async def send_mcp():
        await client.send_message(
            destination="target",
            payload={"test": "data"},
            message_type=MCPMessageType.REQUEST,
            trust_score=0.9
        )
    
    result = benchmark(lambda: asyncio.run(send_mcp()))


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_consensus_latency(benchmark):
    """Benchmark: Consensus request latency"""
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.kernels.mldl import MLDLKernel
    from grace.events.factory import GraceEventFactory
    
    bus = EventBus()
    mesh = TriggerMesh(bus)
    factory = GraceEventFactory()
    
    kernel = MLDLKernel(bus, factory, None, None, mesh)
    
    async def consensus():
        await kernel.start()
        # Simulate consensus request
        await asyncio.sleep(0.01)
        await kernel.stop()
    
    result = benchmark(lambda: asyncio.run(consensus()))
