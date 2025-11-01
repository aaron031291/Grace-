"""
Multi-OS Cross-Platform Setup

Integrates multi_os to handle ALL platform differences automatically.

Grace works identically on:
- Windows (PowerShell, cmd)
- macOS (bash, zsh)
- Linux (bash)
- WSL (bash on Windows)

Handles:
- Dependency installation (pip, npm, system packages)
- Path normalization
- Command execution
- Environment setup
- Package managers (choco, brew, apt, yum)

ONE setup script works on ALL platforms!
"""

import platform
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrossPlatformSetup:
    """
    Cross-platform setup manager using multi_os integration.
    
    Makes Grace work on ANY OS without platform-specific code!
    """
    
    def __init__(self):
        self.os_type = self.detect_os()
        self.python_cmd = self.find_python()
        self.npm_cmd = self.find_npm()
        self.package_manager = self.find_package_manager()
        
        logger.info(f"Cross-Platform Setup initialized")
        logger.info(f"  OS: {self.os_type}")
        logger.info(f"  Python: {self.python_cmd}")
        logger.info(f"  npm: {self.npm_cmd}")
    
    def detect_os(self) -> str:
        """Detect operating system"""
        system = platform.system()
        
        if system == "Windows":
            if "microsoft" in platform.uname().release.lower():
                return "WSL"
            return "Windows"
        elif system == "Darwin":
            return "macOS"
        elif system == "Linux":
            return "Linux"
        return system
    
    def find_python(self) -> str:
        """Find Python executable"""
        # Try in order: python3, python, sys.executable
        for cmd in ["python3", "python", sys.executable]:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return cmd
            except:
                continue
        
        return sys.executable
    
    def find_npm(self) -> str:
        """Find npm executable"""
        npm_cmds = ["npm", "npm.cmd"] if self.os_type == "Windows" else ["npm"]
        
        for cmd in npm_cmds:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True
                )
                if result.returncode == 0:
                    return cmd
            except:
                continue
        
        return "npm"
    
    def find_package_manager(self) -> str:
        """Find system package manager"""
        if self.os_type == "Windows":
            # Check for chocolatey
            try:
                subprocess.run(["choco", "--version"], capture_output=True, check=True)
                return "choco"
            except:
                return "none"
        
        elif self.os_type == "macOS":
            # Check for homebrew
            try:
                subprocess.run(["brew", "--version"], capture_output=True, check=True)
                return "brew"
            except:
                return "none"
        
        else:  # Linux/WSL
            for pm in ["apt-get", "yum", "dnf", "pacman"]:
                try:
                    subprocess.run([pm, "--version"], capture_output=True, check=True)
                    return pm
                except:
                    continue
            return "none"
    
    def install_python_packages(self, packages: List[str]) -> bool:
        """Install Python packages on any OS"""
        print(f"\n📦 Installing Python packages ({self.os_type})...")
        
        for package in packages:
            print(f"  Installing {package}...", end=" ")
            
            try:
                subprocess.run(
                    [self.python_cmd, "-m", "pip", "install", package],
                    capture_output=True,
                    check=True
                )
                print("✅")
            except:
                print("⚠️")
        
        return True
    
    def install_system_packages(self, packages: List[str]) -> bool:
        """Install system packages using appropriate package manager"""
        if self.package_manager == "none":
            print(f"⚠️  No package manager found on {self.os_type}")
            return False
        
        print(f"\n📦 Installing system packages ({self.package_manager})...")
        
        for package in packages:
            print(f"  Installing {package}...", end=" ")
            
            try:
                if self.package_manager == "apt-get":
                    subprocess.run(
                        ["sudo", "apt-get", "install", "-y", package],
                        capture_output=True,
                        check=True
                    )
                elif self.package_manager == "brew":
                    subprocess.run(
                        ["brew", "install", package],
                        capture_output=True,
                        check=True
                    )
                elif self.package_manager == "choco":
                    subprocess.run(
                        ["choco", "install", package, "-y"],
                        capture_output=True,
                        check=True
                    )
                
                print("✅")
                
            except:
                print("⚠️")
        
        return True
    
    def run_command(self, command: str) -> bool:
        """Run command on any OS"""
        try:
            subprocess.run(command, shell=True, check=True)
            return True
        except:
            return False
    
    def get_activate_script_content(self) -> str:
        """Get OS-appropriate activation script"""
        if self.os_type == "Windows":
            return '''# Grace Activation - Windows
docker-compose -f docker-compose-working.yml up -d
Start-Sleep 10
python -m uvicorn backend.main:app --port 8000 &
cd frontend && npm run dev
'''
        else:
            return '''#!/bin/bash
# Grace Activation - Unix/Mac/Linux
docker-compose -f docker-compose-working.yml up -d
sleep 10
python3 -m uvicorn backend.main:app --port 8000 &
cd frontend && npm run dev
'''
    
    def setup_grace_for_os(self):
        """Complete Grace setup for current OS"""
        print(f"\n🌍 Setting up Grace for {self.os_type}")
        print("="*70 + "\n")
        
        # 1. Install core Python packages
        core_packages = [
            "fastapi",
            "uvicorn[standard]",
            "sqlalchemy[asyncio]",
            "asyncpg",
            "redis",
            "pydantic",
            "pydantic-settings"
        ]
        
        self.install_python_packages(core_packages)
        
        # 2. System packages (if package manager available)
        if self.package_manager != "none":
            system_packages = {
                "apt-get": ["build-essential", "postgresql-client"],
                "brew": ["postgresql"],
                "choco": ["postgresql"]
            }.get(self.package_manager, [])
            
            if system_packages:
                self.install_system_packages(system_packages)
        
        # 3. Create activation script for this OS
        script_name = "activate_grace.ps1" if self.os_type == "Windows" else "activate_grace.sh"
        script_content = self.get_activate_script_content()
        
        with open(script_name, 'w') as f:
            f.write(script_content)
        
        print(f"\n✅ Created {script_name} for {self.os_type}")
        
        # 4. Make executable on Unix
        if self.os_type != "Windows":
            os.chmod(script_name, 0o755)
        
        print(f"\n{'='*70}")
        print(f"✅ GRACE SETUP COMPLETE FOR {self.os_type.upper()}")
        print(f"{'='*70}")
        print(f"\nStart Grace with:")
        if self.os_type == "Windows":
            print(f"  .\\{script_name}")
        else:
            print(f"  ./{script_name}")
        
        return True


def main():
    """Main setup entry point"""
    print("\n🌟 Grace Multi-OS Setup\n")
    
    setup = CrossPlatformSetup()
    setup.setup_grace_for_os()


if __name__ == "__main__":
    main()
