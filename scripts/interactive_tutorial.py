"""
Interactive Grace Tutorial

Guides users through Grace features step-by-step
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

console = Console()


class GraceTutorial:
    """Interactive tutorial for Grace"""
    
    def __init__(self):
        self.console = console
    
    def welcome(self):
        """Welcome message"""
        welcome_text = """
# Welcome to Grace AI System! 🚀

This interactive tutorial will guide you through:

1. **Event System** - Publishing and subscribing to events
2. **Kernels** - Managing AI kernels
3. **Memory** - Working with the memory system
4. **Governance** - Constitutional compliance
5. **Security** - RBAC and encryption

Let's get started!
        """
        
        self.console.print(Panel(
            Markdown(welcome_text),
            border_style="cyan",
            padding=(1, 2)
        ))
    
    async def step_events(self):
        """Tutorial step: Events"""
        self.console.print("\n[bold cyan]Step 1: Event System[/bold cyan]\n")
        
        self.console.print("Events are the core of Grace communication.")
        self.console.print("Let's emit your first event!\n")
        
        if not Confirm.ask("Ready to continue?"):
            return
        
        from grace.integration.event_bus import get_event_bus
        from grace.schemas.events import GraceEvent
        
        bus = get_event_bus()
        
        event_type = Prompt.ask("Enter event type", default="tutorial.test")
        message = Prompt.ask("Enter a message", default="Hello, Grace!")
        
        event = GraceEvent(
            event_type=event_type,
            source="tutorial",
            payload={"message": message}
        )
        
        await bus.emit(event)
        
        self.console.print(f"\n[green]✓[/green] Event emitted!")
        self.console.print(f"[dim]Event ID: {event.event_id}[/dim]\n")
    
    async def step_kernels(self):
        """Tutorial step: Kernels"""
        self.console.print("\n[bold cyan]Step 2: Kernels[/bold cyan]\n")
        
        self.console.print("Kernels are specialized AI agents.")
        self.console.print("Available kernels: multi_os, mldl, resilience\n")
        
        if not Confirm.ask("Want to start a demo kernel?"):
            return
        
        from grace.kernels.multi_os import MultiOSKernel
        from grace.integration.event_bus import get_event_bus
        
        bus = get_event_bus()
        kernel = MultiOSKernel(bus, None)
        
        self.console.print("[dim]Starting kernel...[/dim]")
        await kernel.start()
        
        self.console.print("[green]✓[/green] Kernel started!\n")
        
        health = kernel.get_health()
        self.console.print(f"Status: {health['status']}")
        self.console.print(f"Running: {health['running']}\n")
        
        await kernel.stop()
        self.console.print("[dim]Kernel stopped[/dim]\n")
    
    async def step_memory(self):
        """Tutorial step: Memory"""
        self.console.print("\n[bold cyan]Step 3: Memory System[/bold cyan]\n")
        
        self.console.print("Grace has a multi-layer memory system:")
        self.console.print("• Lightning (cache) - Fast access")
        self.console.print("• Fusion (durable) - Persistent storage")
        self.console.print("• Vector (semantic) - Similarity search\n")
        
        if not Confirm.ask("Try storing something in memory?"):
            return
        
        from grace.memory.async_lightning import AsyncLightningMemory
        
        memory = AsyncLightningMemory()
        await memory.connect()
        
        key = Prompt.ask("Memory key", default="tutorial_data")
        value = Prompt.ask("Value to store", default="Grace is awesome!")
        
        await memory.set(key, value)
        self.console.print(f"[green]✓[/green] Stored: {key} = {value}\n")
        
        retrieved = await memory.get(key)
        self.console.print(f"[green]✓[/green] Retrieved: {retrieved}\n")
        
        await memory.disconnect()
    
    def completion(self):
        """Tutorial completion"""
        completion_text = """
# Tutorial Complete! 🎉

You've learned the basics of Grace:

- ✓ Events and messaging
- ✓ Kernel management
- ✓ Memory operations

## Next Steps:

1. Start the service: `grace start`
2. View the dashboard: http://localhost:8000
3. Explore API docs: http://localhost:8000/docs
4. Check examples: `examples/` directory

## Resources:

- Documentation: `documentation/`
- API Reference: `documentation/API_REFERENCE.md`
- Deployment: `documentation/DEPLOYMENT.md`

Happy building! 🚀
        """
        
        self.console.print(Panel(
            Markdown(completion_text),
            border_style="green",
            padding=(1, 2)
        ))
    
    async def run(self):
        """Run the tutorial"""
        self.welcome()
        
        if not Confirm.ask("\nStart tutorial?", default=True):
            return
        
        await self.step_events()
        await self.step_kernels()
        await self.step_memory()
        
        self.completion()


async def main():
    """Main tutorial entry point"""
    tutorial = GraceTutorial()
    await tutorial.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tutorial interrupted. Run again anytime![/yellow]")
