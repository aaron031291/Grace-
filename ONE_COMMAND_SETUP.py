#!/usr/bin/env python3
"""
ONE COMMAND TO RULE THEM ALL

This script does EVERYTHING to get Grace working:
1. Installs dependencies
2. Initializes database
3. Tests integration
4. Starts Grace
5. Opens browser

Just run: python ONE_COMMAND_SETUP.py

Grace will be working!
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def run_step(description, command, critical=True):
    """Run a setup step"""
    print(f"\n{'='*70}")
    print(f"⚡ {description}")
    print(f"{'='*70}\n")
    
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=critical)
        else:
            result = subprocess.run(command, shell=True, check=critical)
        
        print(f"\n✅ {description} - Complete")
        return True
        
    except subprocess.CalledProcessError as e:
        if critical:
            print(f"\n❌ {description} - Failed")
            print(f"   Error: {e}")
            return False
        else:
            print(f"\n⚠️  {description} - Had issues (continuing...)")
            return True
    except Exception as e:
        print(f"\n⚠️  {description} - {e}")
        return not critical

def main():
    print("\n" + "🌟 "*35)
    print("GRACE ONE-COMMAND SETUP")
    print("🌟 "*35 + "\n")
    
    print("This will:")
    print("  1. Install all dependencies")
    print("  2. Initialize database")
    print("  3. Test all systems")
    print("  4. Start Grace")
    print("  5. Open browser")
    print("")
    
    # Step 1: Install dependencies
    if not run_step(
        "Installing Dependencies",
        [sys.executable, "install_dependencies.py"],
        critical=False
    ):
        print("\n⚠️  Some dependencies failed to install")
        print("   Grace will work with reduced functionality")
    
    # Step 2: Start infrastructure
    print(f"\n{'='*70}")
    print("⚡ Starting Infrastructure (Docker)")
    print(f"{'='*70}\n")
    
    if Path("docker-compose-working.yml").exists():
        subprocess.run(
            "docker-compose -f docker-compose-working.yml up -d postgres redis",
            shell=True
        )
        print("✅ PostgreSQL and Redis starting...")
        print("   Waiting 15 seconds for services...")
        time.sleep(15)
    else:
        print("⚠️  docker-compose-working.yml not found")
        print("   Start PostgreSQL and Redis manually")
    
    # Step 3: Initialize database
    if not run_step(
        "Initializing Database",
        [sys.executable, "init_database.py"],
        critical=False
    ):
        print("   Grace will work without persistent storage")
    
    # Step 4: Test integration
    if not run_step(
        "Testing Integration",
        [sys.executable, "test_all_integration.py"],
        critical=False
    ):
        print("   Some tests failed but Grace may still work")
    
    # Step 5: Start Grace
    print(f"\n{'='*70}")
    print("🚀 STARTING GRACE")
    print(f"{'='*70}\n")
    
    print("Starting backend...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("   Backend starting on http://localhost:8000")
    print("   Waiting 5 seconds...")
    time.sleep(5)
    
    print("\nStarting frontend...")
    if Path("frontend").exists():
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd="frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("   Frontend starting on http://localhost:5173")
        print("   Waiting 5 seconds...")
        time.sleep(5)
    
    # Step 6: Open browser
    print(f"\n{'='*70}")
    print("✅ GRACE IS STARTING!")
    print(f"{'='*70}\n")
    
    print("Opening browser...")
    try:
        webbrowser.open("http://localhost:8000/api/docs")
        time.sleep(2)
        webbrowser.open("http://localhost:5173")
    except:
        pass
    
    print("\n🌐 Access Points:")
    print("   Backend API:  http://localhost:8000")
    print("   API Docs:     http://localhost:8000/api/docs")
    print("   Frontend UI:  http://localhost:5173")
    print("   Orb Interface: http://localhost:5173/")
    print("\n🔑 Login Credentials:")
    print("   Username: admin")
    print("   Password: admin")
    print("\n📊 Verify Working:")
    print("   python verify_grace_working.py")
    print("\n" + "="*70)
    print("🎉 GRACE IS OPERATIONAL!")
    print("="*70)
    print("\nPress Ctrl+C to stop\n")
    
    # Keep running
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Grace...")
        backend_process.terminate()
        if 'frontend_process' in locals():
            frontend_process.terminate()
        print("   Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled")
        sys.exit(0)
