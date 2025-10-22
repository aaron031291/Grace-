# Grace Examples

Quick start examples for common use cases.

## Table of Contents

1. [Simple Event Publishing](#simple-event-publishing)
2. [Memory Operations](#memory-operations)
3. [Kernel Management](#kernel-management)
4. [Governance Validation](#governance-validation)
5. [Complete Workflow](#complete-workflow)

## Simple Event Publishing

```python
# examples/01_simple_event.py
import asyncio
from grace.integration.event_bus import get_event_bus
from grace.schemas.events import GraceEvent

async def main():
    # Get event bus
    bus = get_event_bus()
    
    # Create event
    event = GraceEvent(
        event_type="user.action",
        source="my_app",
        payload={"action": "button_click", "user": "alice"}
    )
    
    # Emit event
    success = await bus.emit(event)
    print(f"Event emitted: {success}")
    print(f"Event ID: {event.event_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Memory Operations

```python
# examples/02_memory_operations.py
import asyncio
from grace.memory.core import MemoryCore
from grace.memory.async_lightning import AsyncLightningMemory

async def main():
    # Setup memory
    lightning = AsyncLightningMemory()
    await lightning.connect()
    
    # Write to memory
    await lightning.set("user_preference", {"theme": "dark"})
    
    # Read from memory
    value = await lightning.get("user_preference")
    print(f"Stored value: {value}")
    
    await lightning.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Kernel Management

```python
# examples/03_kernel_management.py
import asyncio
from grace.kernels.multi_os import MultiOSKernel
from grace.integration.event_bus import get_event_bus

async def main():
    # Create kernel
    bus = get_event_bus()
    kernel = MultiOSKernel(bus, None)
    
    # Start kernel
    await kernel.start()
    print("Kernel started")
    
    # Check health
    health = kernel.get_health()
    print(f"Health: {health}")
    
    # Stop kernel
    await kernel.stop()
    print("Kernel stopped")

if __name__ == "__main__":
    asyncio.run(main())
```
