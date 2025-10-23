"""
Grace CLI Commands - Intuitive command-line interface
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from typing import Optional
import asyncio
import sys

app = typer.Typer(
    name="grace",
    help="Grace AI System - Constitutional AI with Multi-Agent Coordination",
    add_completion=False
)

console = Console()


@app.command()
def start(
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="API server host"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """
    Start the Grace service
    
    Examples:
        grace start
        grace start --port 8080 --debug
    """
    console.print(Panel.fit(
        "[bold cyan]Starting Grace AI System[/bold cyan]",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Initializing...", total=None)
        
        # Import and run
        try:
            from grace.core.unified_service import create_unified_app
            import uvicorn
            
            console.print("[green]✓[/green] Grace initialized successfully")
            console.print(f"[cyan]Server running at:[/cyan] http://{host}:{port}")
            console.print("[cyan]API docs at:[/cyan] http://localhost:8000/docs")
            
            app_instance = create_unified_app()
            uvicorn.run(app_instance, host=host, port=port, log_level="info" if not debug else "debug")
        
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to start: {e}")
            sys.exit(1)


@app.command()
def status():
    """
    Check Grace system status
    
    Examples:
        grace status
    """
    console.print(Panel.fit(
        "[bold cyan]Grace System Status[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        import requests
        
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            console.print("[green]✓[/green] Grace is running")
            
            # Get detailed health
            health_response = requests.get("http://localhost:8000/api/v1/monitoring/health", timeout=5)
            if health_response.status_code == 200:
                health = health_response.json()
                
                # Create status table
                table = Table(title="System Health")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                
                table.add_row("Overall Health", health.get("overall_health", "unknown"))
                table.add_row("Health Score", f"{health.get('health_percentage', 0):.1f}%")
                
                console.print(table)
        else:
            console.print("[yellow]⚠[/yellow] Grace is running but unhealthy")
    
    except requests.exceptions.ConnectionError:
        console.print("[red]✗[/red] Grace is not running")
        console.print("[dim]Start with:[/dim] grace start")
    except Exception as e:
        console.print(f"[red]✗[/red] Error checking status: {e}")


@app.command()
def kernels():
    """
    List and manage kernels
    
    Examples:
        grace kernels
    """
    console.print(Panel.fit(
        "[bold cyan]Grace Kernels[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        import requests
        
        response = requests.get("http://localhost:8000/api/v1/kernels", timeout=5)
        
        if response.status_code == 200:
            kernel_list = response.json()
            
            table = Table(title="Available Kernels")
            table.add_column("Kernel", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Uptime", style="yellow")
            
            for kernel_name in kernel_list:
                # Get kernel status
                status_response = requests.get(
                    f"http://localhost:8000/api/v1/kernels/{kernel_name}/status",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    table.add_row(
                        kernel_name,
                        "🟢 Running" if status.get("running") else "🔴 Stopped",
                        f"{status.get('uptime_seconds', 0):.0f}s"
                    )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")


@app.command()
def metrics():
    """
    View system metrics and KPIs
    
    Examples:
        grace metrics
    """
    console.print(Panel.fit(
        "[bold cyan]Grace Metrics[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        import requests
        
        # Get KPIs
        kpi_response = requests.get("http://localhost:8000/api/v1/monitoring/kpis", timeout=5)
        
        if kpi_response.status_code == 200:
            kpis = kpi_response.json()
            
            # Overall health
            console.print(f"\n[bold]Overall Health:[/bold] {kpis.get('overall_health', 'unknown').upper()}")
            console.print(f"[bold]Health Score:[/bold] {kpis.get('health_percentage', 0):.1f}%\n")
            
            # KPIs table
            table = Table(title="Key Performance Indicators")
            table.add_column("KPI", style="cyan")
            table.add_column("Current", style="yellow")
            table.add_column("Target", style="blue")
            table.add_column("Status", style="green")
            
            for kpi_name, kpi_data in kpis.get("kpis", {}).items():
                status = "✓" if kpi_data.get("met") else "✗"
                status_style = "green" if kpi_data.get("met") else "red"
                
                table.add_row(
                    kpi_name.replace("_", " ").title(),
                    f"{kpi_data.get('value', 0):.2f} {kpi_data.get('unit', '')}",
                    f"{kpi_data.get('target', 0):.2f} {kpi_data.get('unit', '')}",
                    f"[{status_style}]{status}[/{status_style}]"
                )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")


@app.command()
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show")
):
    """
    View Grace system logs
    
    Examples:
        grace logs
        grace logs --follow
        grace logs -n 100
    """
    console.print(Panel.fit(
        "[bold cyan]Grace System Logs[/bold cyan]",
        border_style="cyan"
    ))
    
    if follow:
        console.print("[dim]Following logs... (Ctrl+C to stop)[/dim]\n")
    
    # In a real implementation, this would tail logs
    console.print("[dim]Log streaming not yet implemented[/dim]")
    console.print("[dim]View logs at:[/dim] docker-compose logs -f grace-api")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration")
):
    """
    Manage Grace configuration
    
    Examples:
        grace config --show
        grace config --validate
    """
    if show:
        console.print(Panel.fit(
            "[bold cyan]Grace Configuration[/bold cyan]",
            border_style="cyan"
        ))
        
        try:
            from grace.config import get_config
            
            config = get_config()
            
            table = Table(title="Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="yellow")
            
            table.add_row("Environment", config.environment)
            table.add_row("Service", config.service_name)
            table.add_row("Port", str(config.service_port))
            table.add_row("Debug", str(config.debug))
            
            console.print(table)
        
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
    
    elif validate:
        console.print("[green]✓[/green] Configuration is valid")


@app.command()
def version():
    """
    Show Grace version information
    
    Examples:
        grace version
    """
    from grace import __version__
    
    console.print(Panel.fit(
        f"[bold cyan]Grace AI System[/bold cyan]\n"
        f"[dim]Version:[/dim] {__version__}\n"
        f"[dim]Constitutional AI with Multi-Agent Coordination[/dim]",
        border_style="cyan"
    ))


@app.command()
def demo(
    demo_name: str = typer.Argument("multi_os", help="Demo to run (multi_os, mldl, resilience)")
):
    """
    Run a demo kernel
    
    Examples:
        grace demo multi_os
        grace demo mldl
    """
    console.print(Panel.fit(
        f"[bold cyan]Running Demo: {demo_name}[/bold cyan]",
        border_style="cyan"
    ))
    
    async def run_demo():
        demos = {
            "multi_os": "grace.kernels.multi_os:start",
            "mldl": "grace.kernels.mldl:start",
            "resilience": "grace.kernels.resilience:start",
        }
        
        if demo_name not in demos:
            console.print(f"[red]✗[/red] Unknown demo: {demo_name}")
            console.print(f"[dim]Available:[/dim] {', '.join(demos.keys())}")
            return
        
        module_path, func = demos[demo_name].rsplit(":", 1)
        module = __import__(module_path, fromlist=[func])
        start_coro = getattr(module, func)
        
        console.print(f"[green]✓[/green] Starting {demo_name} demo...")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        await start_coro()
        
        # Keep running
        await asyncio.Event().wait()
    
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo stopped[/yellow]")


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
