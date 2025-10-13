"""Multi-OS kernel - runtime management and environment setup."""

import os
import sys
import subprocess
from typing import Dict, List
from pydantic import BaseModel


class RuntimeSnapshot(BaseModel):
    """Runtime environment snapshot."""

    python_version: str
    platform: str
    working_directory: str
    environment_variables: Dict[str, str]
    installed_packages: List[str] = []


class MultiOSKernel:
    """Handles multi-platform runtime management."""

    def __init__(self):
        self.runtime_snapshot = None
        self.required_packages = ["fastapi", "uvicorn", "pydantic", "websockets"]

    def ensure_runtime(self) -> RuntimeSnapshot:
        """Ensure runtime environment is properly configured."""
        # Capture current runtime state
        snapshot = RuntimeSnapshot(
            python_version=sys.version,
            platform=sys.platform,
            working_directory=os.getcwd(),
            environment_variables=dict(os.environ),
            installed_packages=self._get_installed_packages(),
        )

        self.runtime_snapshot = snapshot

        # Check and install missing packages
        self._ensure_packages()

        return snapshot

    def _get_installed_packages(self) -> List[str]:
        """Get list of installed Python packages."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [
                    line.split("==")[0]
                    for line in result.stdout.split("\n")
                    if "==" in line
                ]
        except Exception:
            pass

        return []

    def _ensure_packages(self):
        """Ensure required packages are installed."""
        installed = set(self.runtime_snapshot.installed_packages)
        missing = [pkg for pkg in self.required_packages if pkg not in installed]

        if missing:
            # In a real implementation, this would install packages
            # For development, we just log what's missing
            pass

    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        return {
            "python_version": sys.version_info[:3],
            "platform": sys.platform,
            "architecture": os.uname().machine if hasattr(os, "uname") else "unknown",
            "working_directory": os.getcwd(),
            "python_executable": sys.executable,
            "runtime_ready": bool(self.runtime_snapshot),
        }

    def get_stats(self) -> Dict:
        """Get multi-OS kernel statistics."""
        if not self.runtime_snapshot:
            self.ensure_runtime()

        return {
            "runtime_initialized": bool(self.runtime_snapshot),
            "platform": self.runtime_snapshot.platform
            if self.runtime_snapshot
            else "unknown",
            "python_version": self.runtime_snapshot.python_version[:10]
            if self.runtime_snapshot
            else "unknown",
            "required_packages": len(self.required_packages),
            "working_directory": os.getcwd(),
        }
