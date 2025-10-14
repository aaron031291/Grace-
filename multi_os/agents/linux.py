"""
Linux Agent Adapter - Handles Linux-specific operations.
"""

import asyncio
import logging
import subprocess
import os
import shutil
from typing import Dict, Any, List, Optional
import sys

sys.path.append("/home/runner/work/Grace-/Grace-/multi_os")
from adapters.base import (
    ProcessAdapter,
    FSAdapter,
    NetAdapter,
    PkgAdapter,
    SnapshotAdapter,
)


logger = logging.getLogger(__name__)


class LinuxAdapter(ProcessAdapter, FSAdapter, NetAdapter, PkgAdapter, SnapshotAdapter):
    """
    Linux-specific adapter implementing Multi-OS capabilities.
    Uses subprocess, cgroups/namespace sandbox (nsjail/firejail), apt/yum/dnf,
    systemd services, Docker/Podman, NVIDIA CUDA.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.sandbox_backend = self.config.get("sandbox_backend", "nsjail")
        self.package_manager = self._detect_package_manager()
        self.container_runtime = self._detect_container_runtime()

        logger.info(
            f"Linux adapter initialized with {self.package_manager} package manager"
        )

    # ProcessAdapter implementation
    async def exec(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using Linux process management."""
        try:
            task_id = task.get("task_id", "unknown")
            command = task.get("command", "")
            args = task.get("args", [])
            cwd = task.get("cwd", os.getcwd())

            # Build command
            full_command = [command] + args if isinstance(args, list) else [command]

            # Apply sandbox if requested
            constraints = task.get("constraints", {})
            sandbox = constraints.get("sandbox", "none")

            if sandbox != "none":
                full_command = self._apply_sandbox(full_command, sandbox, constraints)

            # Execute process
            logger.info(f"Executing Linux task {task_id}: {' '.join(full_command)}")

            process = await asyncio.create_subprocess_exec(
                *full_command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._prepare_environment(task),
            )

            # Wait for completion with timeout
            timeout = constraints.get("max_runtime_s", 1800)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "exit_code": -1,
                    "error": "Task timed out",
                    "duration_seconds": timeout,
                }

            return {
                "success": process.returncode == 0,
                "pid": process.pid,
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
                "duration_seconds": timeout,  # Would track actual duration
            }

        except Exception as e:
            logger.error(f"Linux task execution failed: {e}")
            return {"success": False, "error": str(e), "exit_code": -1}

    async def kill(self, pid: int) -> bool:
        """Kill a Linux process by PID."""
        try:
            import signal

            os.kill(pid, signal.SIGTERM)

            # Wait a bit for graceful termination
            await asyncio.sleep(2)

            # Check if still running, force kill if necessary
            try:
                os.kill(pid, 0)  # Check if process exists
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Force killed Linux process {pid}")
            except ProcessLookupError:
                logger.info(f"Linux process {pid} terminated gracefully")

            return True

        except Exception as e:
            logger.error(f"Failed to kill Linux process {pid}: {e}")
            return False

    async def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get information about a Linux process."""
        try:
            # Use ps command to get process info
            result = await asyncio.create_subprocess_exec(
                "ps",
                "-p",
                str(pid),
                "-o",
                "pid,ppid,user,cpu,mem,cmd",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                return None

            lines = stdout.decode("utf-8").strip().split("\n")
            if len(lines) < 2:
                return None

            # Parse ps output (simplified)
            return {
                "pid": pid,
                "status": "running",
                "cpu_percent": 0.0,  # Would parse from ps output
                "memory_mb": 0,  # Would parse from ps output
                "command": "unknown",  # Would parse from ps output
            }

        except Exception as e:
            logger.error(f"Failed to get Linux process info for {pid}: {e}")
            return None

    # FSAdapter implementation
    async def apply(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filesystem action on Linux."""
        try:
            action_type = action.get("type")
            path = action.get("path")

            if action_type == "read":
                content = await self.read(path)
                return {"success": True, "content_b64": content}
            elif action_type == "write":
                content_b64 = action.get("content_b64", "")
                success = await self.write(
                    path, content_b64.encode() if content_b64 else b""
                )
                return {"success": success}
            elif action_type == "list":
                recursive = action.get("recursive", False)
                files = await self.list_dir(path, recursive)
                return {"success": True, "files": files}
            elif action_type == "delete":
                recursive = action.get("recursive", False)
                success = await self.delete(path, recursive)
                return {"success": success}
            elif action_type == "hash":
                import hashlib

                content = await self.read(path)
                hash_value = hashlib.sha256(content).hexdigest()
                return {"success": True, "hash": hash_value}
            else:
                return {
                    "success": False,
                    "error": f"Unsupported action type: {action_type}",
                }

        except Exception as e:
            logger.error(f"Linux filesystem action failed: {e}")
            return {"success": False, "error": str(e)}

    async def read(self, path: str, encoding: str = "utf-8") -> bytes:
        """Read file contents on Linux."""
        with open(path, "rb") as f:
            return f.read()

    async def write(self, path: str, content: bytes) -> bool:
        """Write file contents on Linux."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return False

    async def list_dir(self, path: str, recursive: bool = False) -> List[str]:
        """List directory contents on Linux."""
        try:
            if recursive:
                files = []
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
                return files
            else:
                return [os.path.join(path, f) for f in os.listdir(path)]
        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            return []

    async def exists(self, path: str) -> bool:
        """Check if path exists on Linux."""
        return os.path.exists(path)

    async def delete(self, path: str, recursive: bool = False) -> bool:
        """Delete file or directory on Linux."""
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                if recursive:
                    shutil.rmtree(path)
                else:
                    os.rmdir(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False

    # NetAdapter implementation
    async def apply(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply network action on Linux."""
        try:
            action_type = action.get("type")

            if action_type == "fetch":
                result = await self.fetch(
                    action.get("url"), action.get("timeout_s", 30)
                )
                return {"success": True, "result": result}
            elif action_type == "post":
                result = await self.post(
                    action.get("url"), action.get("body"), action.get("timeout_s", 30)
                )
                return {"success": True, "result": result}
            elif action_type == "port_check":
                url = action.get("url", "")
                # Parse host:port from URL
                if "://" in url:
                    host_port = url.split("://")[1].split("/")[0]
                else:
                    host_port = url

                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                    is_open = await self.port_check(
                        host, int(port), action.get("timeout_s", 5)
                    )
                    return {"success": True, "port_open": is_open}
                else:
                    return {
                        "success": False,
                        "error": "Invalid URL format for port check",
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported network action: {action_type}",
                }

        except Exception as e:
            logger.error(f"Linux network action failed: {e}")
            return {"success": False, "error": str(e)}

    async def fetch(
        self, url: str, timeout: int = 30, headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Fetch data from URL using Linux tools."""
        try:
            # Use curl for HTTP requests
            cmd = ["curl", "-s", "-L", "--max-time", str(timeout)]

            if headers:
                for key, value in headers.items():
                    cmd.extend(["-H", f"{key}: {value}"])

            cmd.append(url)

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            return {
                "status_code": result.returncode,
                "content": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None,
            }

        except Exception as e:
            return {"status_code": -1, "error": str(e)}

    async def post(
        self, url: str, data: Any, timeout: int = 30, headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Post data to URL using Linux tools."""
        try:
            cmd = ["curl", "-s", "-L", "-X", "POST", "--max-time", str(timeout)]

            if headers:
                for key, value in headers.items():
                    cmd.extend(["-H", f"{key}: {value}"])

            if data:
                cmd.extend(["-d", str(data)])

            cmd.append(url)

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            return {
                "status_code": result.returncode,
                "content": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None,
            }

        except Exception as e:
            return {"status_code": -1, "error": str(e)}

    async def port_check(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if port is open using Linux tools."""
        try:
            result = await asyncio.create_subprocess_exec(
                "nc",
                "-z",
                "-w",
                str(timeout),
                host,
                str(port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await result.communicate()
            return result.returncode == 0

        except Exception:
            return False

    # PkgAdapter implementation
    async def ensure(self, runtime_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure runtime environment on Linux."""
        try:
            runtime = runtime_spec.get("runtime")
            version = runtime_spec.get("version")
            packages = runtime_spec.get("packages", [])

            results = {
                "success": True,
                "runtime_installed": False,
                "packages_installed": [],
                "packages_failed": [],
            }

            # Install runtime if needed
            if runtime == "python":
                results["runtime_installed"] = await self._ensure_python(version)
            elif runtime == "node":
                results["runtime_installed"] = await self._ensure_node(version)
            elif runtime == "java":
                results["runtime_installed"] = await self._ensure_java(version)

            # Install packages
            for pkg in packages:
                pkg_name = pkg.get("name")
                pkg_version = pkg.get("version")
                pkg_manager = pkg.get("manager", "auto")

                success = await self.install_package(pkg_name, pkg_version, pkg_manager)
                if success:
                    results["packages_installed"].append(pkg_name)
                else:
                    results["packages_failed"].append(pkg_name)

            results["success"] = len(results["packages_failed"]) == 0
            return results

        except Exception as e:
            logger.error(f"Linux runtime setup failed: {e}")
            return {"success": False, "error": str(e)}

    async def install_package(
        self, name: str, version: Optional[str] = None, manager: str = "auto"
    ) -> bool:
        """Install package on Linux."""
        try:
            if manager == "auto":
                manager = self.package_manager

            if manager == "apt":
                pkg_spec = f"{name}={version}" if version else name
                cmd = ["sudo", "apt-get", "install", "-y", pkg_spec]
            elif manager == "yum":
                pkg_spec = f"{name}-{version}" if version else name
                cmd = ["sudo", "yum", "install", "-y", pkg_spec]
            elif manager == "pip":
                pkg_spec = f"{name}=={version}" if version else name
                cmd = ["pip", "install", pkg_spec]
            else:
                logger.error(f"Unsupported package manager: {manager}")
                return False

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            await result.communicate()
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Package installation failed: {e}")
            return False

    async def uninstall_package(self, name: str, manager: str = "auto") -> bool:
        """Uninstall package on Linux."""
        try:
            if manager == "auto":
                manager = self.package_manager

            if manager == "apt":
                cmd = ["sudo", "apt-get", "remove", "-y", name]
            elif manager == "yum":
                cmd = ["sudo", "yum", "remove", "-y", name]
            elif manager == "pip":
                cmd = ["pip", "uninstall", "-y", name]
            else:
                return False

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            await result.communicate()
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Package uninstallation failed: {e}")
            return False

    async def list_packages(self, manager: str = "auto") -> List[Dict[str, str]]:
        """List installed packages on Linux."""
        try:
            if manager == "auto":
                manager = self.package_manager

            if manager == "apt":
                cmd = ["dpkg", "-l"]
            elif manager == "yum":
                cmd = ["rpm", "-qa"]
            elif manager == "pip":
                cmd = ["pip", "list", "--format=freeze"]
            else:
                return []

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await result.communicate()

            if result.returncode != 0:
                return []

            # Parse output (simplified)
            packages = []
            lines = stdout.decode("utf-8").strip().split("\n")

            for line in lines:
                if manager == "pip" and "==" in line:
                    name, version = line.split("==", 1)
                    packages.append({"name": name, "version": version})
                # Would implement parsing for other package managers

            return packages

        except Exception as e:
            logger.error(f"Failed to list packages: {e}")
            return []

    # SnapshotAdapter implementation
    async def snapshot(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Create snapshot on Linux."""
        try:
            scope = target.get("scope", "agent")

            if scope == "agent":
                # Snapshot agent configuration and state
                return {
                    "snapshot_id": f"linux_agent_{int(asyncio.get_event_loop().time())}",
                    "scope": scope,
                    "data": {
                        "agent_version": "2.4.1",
                        "config": self.config,
                        "runtime_state": await self._capture_runtime_state(),
                    },
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported snapshot scope: {scope}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def rollback(self, snapshot_id: str) -> Dict[str, Any]:
        """Rollback to snapshot on Linux."""
        try:
            # In real implementation, would restore from snapshot
            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "restored_items": ["agent_config", "runtime_state"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_snapshots(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available snapshots on Linux."""
        # Mock implementation
        return [
            {
                "snapshot_id": "linux_agent_123456",
                "scope": "agent",
                "created_at": "2025-01-01T00:00:00Z",
                "size_mb": 10,
            }
        ]

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete snapshot on Linux."""
        return True  # Mock implementation

    # Helper methods
    def _detect_package_manager(self) -> str:
        """Detect available package manager."""
        if shutil.which("apt-get"):
            return "apt"
        elif shutil.which("yum"):
            return "yum"
        elif shutil.which("dnf"):
            return "dnf"
        else:
            return "unknown"

    def _detect_container_runtime(self) -> str:
        """Detect available container runtime."""
        if shutil.which("docker"):
            return "docker"
        elif shutil.which("podman"):
            return "podman"
        else:
            return "none"

    def _apply_sandbox(
        self, command: List[str], sandbox: str, constraints: Dict[str, Any]
    ) -> List[str]:
        """Apply sandbox wrapper to command."""
        if sandbox == "nsjail":
            return ["nsjail", "--mode", "o", "--chroot", "/tmp"] + command
        elif sandbox == "firejail":
            return ["firejail", "--quiet"] + command
        else:
            return command

    def _prepare_environment(self, task: Dict[str, Any]) -> Dict[str, str]:
        """Prepare environment variables for task."""
        env = os.environ.copy()
        runtime_env = task.get("runtime", {}).get("env", {})
        env.update(runtime_env)
        return env

    async def _ensure_python(self, version: str) -> bool:
        """Ensure Python version is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "python3",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False

    async def _ensure_node(self, version: str) -> bool:
        """Ensure Node.js version is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "node",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False

    async def _ensure_java(self, version: str) -> bool:
        """Ensure Java version is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "java",
                "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False

    async def _capture_runtime_state(self) -> Dict[str, Any]:
        """Capture current runtime state."""
        return {
            "package_manager": self.package_manager,
            "container_runtime": self.container_runtime,
            "sandbox_backend": self.sandbox_backend,
            "timestamp": asyncio.get_event_loop().time(),
        }
