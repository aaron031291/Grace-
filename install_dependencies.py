#!/usr/bin/env python3
"""
Automated Dependency Installer for Grace

Installs ALL required packages in correct order.
Handles optional dependencies gracefully.
"""

import subprocess
import sys

def install_package(package, optional=False):
    """Install a package"""
    print(f"  Installing {package}...", end=" ")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅")
        return True
    except:
        if optional:
            print("⚠️  (optional, skipped)")
        else:
            print("❌")
        return False

def main():
    print("\n🔧 Installing Grace Dependencies\n")
    
    # Core dependencies (REQUIRED)
    print("📦 Core Dependencies (Required):")
    core = [
        "fastapi",
        "uvicorn[standard]",
        "sqlalchemy[asyncio]",
        "asyncpg",
        "redis",
        "pydantic>=2.0",
        "pydantic-settings",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "python-multipart",
        "aiofiles"
    ]
    
    for pkg in core:
        install_package(pkg)
    
    # Integration dependencies
    print("\n📦 Integration Dependencies:")
    integration = [
        "requests",
        "aiohttp",
        "websockets",
        "python-dotenv"
    ]
    
    for pkg in integration:
        install_package(pkg)
    
    # Optional advanced features
    print("\n📦 Advanced Features (Optional):")
    advanced = [
        ("openai", True),  # For OpenAI API
        ("anthropic", True),  # For Claude API
        ("PyPDF2", True),  # For PDF processing
        ("beautifulsoup4", True),  # For web scraping
        ("sentence-transformers", True),  # For embeddings
    ]
    
    for pkg, optional in advanced:
        install_package(pkg, optional)
    
    # Very optional (large downloads)
    print("\n📦 Local AI Models (Very Optional - Large Downloads):")
    print("   ⚠️  These are large and take time to install")
    
    response = input("   Install local AI models? (y/n): ").lower()
    
    if response == 'y':
        local_ai = [
            ("torch", False),
            ("openai-whisper", True),
            ("llama-cpp-python", True)
        ]
        
        for pkg, optional in local_ai:
            install_package(pkg, optional)
    else:
        print("   ⏭️  Skipping local AI (can use cloud APIs instead)")
    
    print("\n" + "="*70)
    print("✅ DEPENDENCY INSTALLATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. python init_database.py  # Initialize database")
    print("  2. python start_grace_production.py  # Start Grace")
    print("")

if __name__ == "__main__":
    main()
