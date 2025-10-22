"""
Grace Quick Start Example

This example demonstrates the most common use cases in a simple, intuitive way.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel

console = Console()


async def main():
    """Quick start demo"""
    
    console.print(Panel.fit(
        "[bold cyan]Grace Quick Start Demo[/bold cyan]\n"
        "This demo shows the most common operations",
        border_style="cyan"
    ))
    
    # 1. Setup
    console.print("\n[bold]1. Setting up Grace...[/bold]")
    from grace.integration.event_bus import get_event_bus
    from grace.schemas.events import GraceEvent
    
    bus = get_event_bus()
    console.print("[green]✓[/green] Event bus initialized")
    
    # 2. Emit an event
    console.print("\n[bold]2. Emitting an event...[/bold]")
    event = GraceEvent(
        event_type="demo.hello",
        source="quickstart",
        payload={"message": "Hello, Grace!"}
    )
    
    await bus.emit(event)
    console.print(f"[green]✓[/green] Event emitted: {event.event_id}")
    
    # 3. Subscribe to events
    console.print("\n[bold]3. Subscribing to events...[/bold]")
    received = []
    
    async def handler(evt):
        received.append(evt)
        console.print(f"[blue]→[/blue] Received: {evt.event_type}")
    
    bus.subscribe("demo.hello", handler)
    console.print("[green]✓[/green] Subscribed to demo.hello")
    
    # 4. Emit another event
    console.print("\n[bold]4. Testing subscription...[/bold]")
    test_event = GraceEvent(
        event_type="demo.hello",
        source="quickstart",
        payload={"test": True}
    )
    
    await bus.emit(test_event)
    await asyncio.sleep(0.2)
    
    console.print(f"[green]✓[/green] Received {len(received)} events")
    
    # 5. Get metrics
    console.print("\n[bold]5. Checking metrics...[/bold]")
    metrics = bus.get_metrics()
    console.print(f"[cyan]Published:[/cyan] {metrics['events_published']}")
    console.print(f"[cyan]Processed:[/cyan] {metrics['events_processed']}")
    
    console.print("\n[bold green]✓ Quick start complete![/bold green]")
    console.print("[dim]Try running: grace start[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
