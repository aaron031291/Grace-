"""
Configuration validation script
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace.config import get_settings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main():
    """Validate Grace configuration"""
    console.print("\n[bold blue]Grace Configuration Validation[/bold blue]\n")
    
    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load configuration:[/bold red] {e}")
        return 1
    
    # Display environment
    console.print(Panel(
        f"[bold]Environment:[/bold] {settings.environment}\n"
        f"[bold]Debug:[/bold] {settings.debug}\n"
        f"[bold]API Version:[/bold] {settings.api_version}",
        title="General Settings"
    ))
    
    # Validate production config
    if settings.environment == "production":
        console.print("\n[yellow]Validating production configuration...[/yellow]\n")
        issues = settings.validate_production_config()
        
        if issues:
            console.print("[bold red]⚠️  Configuration Issues Found:[/bold red]\n")
            for issue in issues:
                if "CRITICAL" in issue or "ERROR" in issue:
                    console.print(f"  [red]• {issue}[/red]")
                else:
                    console.print(f"  [yellow]• {issue}[/yellow]")
            
            has_critical = any("CRITICAL" in issue or "ERROR" in issue for issue in issues)
            if has_critical:
                console.print("\n[bold red]❌ Cannot deploy with critical issues![/bold red]")
                return 1
        else:
            console.print("[bold green]✅ Production configuration valid![/bold green]")
    
    # Feature table
    console.print("\n[bold]Enabled Features:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Configuration")
    
    features = [
        ("Authentication", "✅", f"JWT ({settings.auth.algorithm})"),
        ("Database", "✅", settings.database.url[:50] + "..."),
        ("Embeddings", "✅", settings.embedding.provider),
        ("Vector Store", "✅", settings.vector_store.type),
        ("Rate Limiting", "✅" if settings.rate_limit.enabled else "❌", 
         f"{settings.rate_limit.default_limit}/{settings.rate_limit.window_seconds}s"),
        ("Metrics", "✅" if settings.observability.metrics_enabled else "❌", 
         f"Port {settings.observability.metrics_port}"),
        ("Swarm", "✅" if settings.swarm.enabled else "❌", 
         settings.swarm.transport if settings.swarm.enabled else "Disabled"),
        ("Quantum", "✅" if settings.transcendence.quantum_enabled else "❌", ""),
        ("Discovery", "✅" if settings.transcendence.discovery_enabled else "❌", ""),
        ("Impact Eval", "✅" if settings.transcendence.impact_enabled else "❌", ""),
    ]
    
    for name, status, config in features:
        table.add_row(name, status, config)
    
    console.print(table)
    
    # Deployment info
    console.print("\n[bold]Deployment Information:[/bold]\n")
    info = settings.get_deployment_info()
    console.print(f"  Environment: {info['environment']}")
    console.print(f"  Debug Mode: {info['debug']}")
    console.print(f"  API Version: {info['api_version']}")
    
    console.print("\n[bold green]✅ Configuration validation complete![/bold green]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
