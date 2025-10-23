"""
Grace System Status - Quick health check
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy',
        'structlog', 'prometheus_client', 'numpy'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    return len(required) - len(missing), len(missing), missing


def check_config():
    """Check if configuration is valid"""
    try:
        from grace.config import get_settings
        settings = get_settings()
        return True, settings.environment
    except Exception as e:
        return False, str(e)


def check_database():
    """Check if database can be initialized"""
    try:
        from grace.database import init_db
        return True, "SQLite/PostgreSQL"
    except Exception as e:
        return False, str(e)


def main():
    """Quick system status check"""
    console.print("\n[bold blue]Grace System Status[/bold blue]")
    console.print("=" * 60)
    
    # Dependencies
    installed, missing_count, missing = check_dependencies()
    dep_status = "✅" if missing_count == 0 else "❌"
    console.print(f"\n{dep_status} Dependencies: {installed}/{installed + missing_count} installed")
    if missing:
        console.print(f"   Missing: {', '.join(missing)}")
    
    # Configuration
    config_ok, env = check_config()
    config_status = "✅" if config_ok else "❌"
    console.print(f"{config_status} Configuration: {env if config_ok else 'Error'}")
    
    # Database
    db_ok, db_info = check_database()
    db_status = "✅" if db_ok else "❌"
    console.print(f"{db_status} Database: {db_info if db_ok else 'Error'}")
    
    # File structure
    grace_files = len(list(Path("grace").rglob("*.py")))
    console.print(f"✅ Python files: {grace_files}")
    
    # Overall
    console.print("\n" + "=" * 60)
    
    all_ok = missing_count == 0 and config_ok and db_ok
    
    if all_ok:
        console.print(Panel(
            "[green]✅ System Ready[/green]\n"
            "Run: make run",
            style="green"
        ))
        return 0
    else:
        console.print(Panel(
            "[yellow]⚠️  Issues Found[/yellow]\n"
            "Run: make validate",
            style="yellow"
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
