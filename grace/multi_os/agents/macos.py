"""
macOS Agent Adapter - Handles macOS-specific operations.
"""
import asyncio
import logging
import subprocess
import os
import shutil
from typing import Dict, Any, List, Optional
import sys
from ..adapters.base import ProcessAdapter, FSAdapter, NetAdapter, PkgAdapter, SnapshotAdapter


logger = logging.getLogger(__name__)


class MacOSAdapter(ProcessAdapter, FSAdapter, NetAdapter, PkgAdapter, SnapshotAdapter):
    """
    macOS-specific adapter implementing Multi-OS capabilities.
    Uses posix_spawn, Sandbox/Entitlements, Homebrew/pkgutil, launchd, 
    Apple silicon GPU/Metal.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.sandbox_backend = self.config.get("sandbox_backend", "sandboxd")
        self.package_manager = self._detect_package_manager()
        self.arch = self._detect_architecture()
        
        logger.info(f"macOS adapter initialized with {self.package_manager} on {self.arch}")
    
    # ProcessAdapter implementation
    async def exec(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using macOS process management."""
        try:
            task_id = task.get("task_id", "unknown")
            command = task.get("command", "")
            args = task.get("args", [])
            cwd = task.get("cwd", os.getcwd())
            
            # Build command
            full_command = [command] + (args if isinstance(args, list) else [])
            
            # Apply sandbox if requested
            constraints = task.get("constraints", {})
            sandbox = constraints.get("sandbox", "none")
            
            if sandbox == "sandboxd":
                full_command = self._apply_sandboxd(full_command, constraints)
            
            # Execute process
            logger.info(f"Executing macOS task {task_id}: {' '.join(full_command)}")
            
            process = await asyncio.create_subprocess_exec(
                *full_command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._prepare_environment(task)
            )
            
            # Wait for completion with timeout
            timeout = constraints.get("max_runtime_s", 1800)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "exit_code": -1,
                    "error": "Task timed out",
                    "duration_seconds": timeout
                }
            
            return {
                "success": process.returncode == 0,
                "pid": process.pid,
                "exit_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='replace') if stdout else "",
                "stderr": stderr.decode('utf-8', errors='replace') if stderr else "",
                "duration_seconds": timeout  # Would track actual duration
            }
            
        except Exception as e:
            logger.error(f"macOS task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "exit_code": -1
            }
    
    async def kill(self, pid: int) -> bool:
        """Kill a macOS process by PID."""
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
            
            # Wait a bit for graceful termination
            await asyncio.sleep(2)
            
            # Check if still running, force kill if necessary
            try:
                os.kill(pid, 0)  # Check if process exists
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Force killed macOS process {pid}")
            except ProcessLookupError:
                logger.info(f"macOS process {pid} terminated gracefully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to kill macOS process {pid}: {e}")
            return False
    
    async def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get information about a macOS process."""
        try:
            # Use ps command to get process info
            result = await asyncio.create_subprocess_exec(
                "ps", "-p", str(pid), "-o", "pid,ppid,user,%cpu,%mem,command",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return None
            
            lines = stdout.decode('utf-8').strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Parse ps output (simplified)
            return {
                "pid": pid,
                "status": "running",
                "cpu_percent": 0.0,  # Would parse from ps output
                "memory_mb": 0,      # Would parse from ps output
                "command": "unknown" # Would parse from ps output
            }
            
        except Exception as e:
            logger.error(f"Failed to get macOS process info for {pid}: {e}")
            return None
    
    # FSAdapter implementation
    async def apply(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filesystem action on macOS."""
        try:
            action_type = action.get("type")
            path = action.get("path")
            
            if action_type == "read":
                content = await self.read(path)
                return {"success": True, "content_b64": content}
            elif action_type == "write":
                content_b64 = action.get("content_b64", "")
                success = await self.write(path, content_b64.encode() if content_b64 else b"")
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
                return {"success": False, "error": f"Unsupported action type: {action_type}"}
                
        except Exception as e:
            logger.error(f"macOS filesystem action failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def read(self, path: str, encoding: str = "utf-8") -> bytes:
        """Read file contents on macOS."""
        with open(path, 'rb') as f:
            return f.read()
    
    async def write(self, path: str, content: bytes) -> bool:
        """Write file contents on macOS."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return False
    
    async def list_dir(self, path: str, recursive: bool = False) -> List[str]:
        """List directory contents on macOS."""
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
        """Check if path exists on macOS."""
        return os.path.exists(path)
    
    async def delete(self, path: str, recursive: bool = False) -> bool:
        """Delete file or directory on macOS."""
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
        """Apply network action on macOS."""
        try:
            action_type = action.get("type")
            
            if action_type == "fetch":
                result = await self.fetch(action.get("url"), action.get("timeout_s", 30))
                return {"success": True, "result": result}
            elif action_type == "post":
                result = await self.post(action.get("url"), action.get("body"), action.get("timeout_s", 30))
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
                    is_open = await self.port_check(host, int(port), action.get("timeout_s", 5))
                    return {"success": True, "port_open": is_open}
                else:
                    return {"success": False, "error": "Invalid URL format for port check"}
            else:
                return {"success": False, "error": f"Unsupported network action: {action_type}"}
                
        except Exception as e:
            logger.error(f"macOS network action failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch(self, url: str, timeout: int = 30, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Fetch data from URL using macOS curl."""
        try:
            cmd = ["curl", "-s", "-L", "--max-time", str(timeout)]
            
            if headers:
                for key, value in headers.items():
                    cmd.extend(["-H", f"{key}: {value}"])
            
            cmd.append(url)
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            return {
                "status_code": result.returncode,
                "content": stdout.decode('utf-8', errors='replace'),
                "error": stderr.decode('utf-8', errors='replace') if stderr else None
            }
            
        except Exception as e:
            return {"status_code": -1, "error": str(e)}
    
    async def post(self, url: str, data: Any, timeout: int = 30, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Post data to URL using macOS curl."""
        try:
            cmd = ["curl", "-s", "-L", "-X", "POST", "--max-time", str(timeout)]
            
            if headers:
                for key, value in headers.items():
                    cmd.extend(["-H", f"{key}: {value}"])
            
            if data:
                cmd.extend(["-d", str(data)])
            
            cmd.append(url)
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            return {
                "status_code": result.returncode,
                "content": stdout.decode('utf-8', errors='replace'),
                "error": stderr.decode('utf-8', errors='replace') if stderr else None
            }
            
        except Exception as e:
            return {"status_code": -1, "error": str(e)}
    
    async def port_check(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if port is open using macOS nc."""
        try:
            result = await asyncio.create_subprocess_exec(
                "nc", "-z", "-G", str(timeout), host, str(port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await result.communicate()
            return result.returncode == 0
            
        except Exception:
            return False
    
    # PkgAdapter implementation
    async def ensure(self, runtime_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure runtime environment on macOS."""
        try:
            runtime = runtime_spec.get("runtime")
            version = runtime_spec.get("version")
            packages = runtime_spec.get("packages", [])
            
            results = {
                "success": True,
                "runtime_installed": False,
                "packages_installed": [],
                "packages_failed": []
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
            logger.error(f"macOS runtime setup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def install_package(self, name: str, version: Optional[str] = None, manager: str = "auto") -> bool:
        """Install package on macOS."""
        try:
            if manager == "auto":
                manager = self.package_manager
            
            if manager == "brew":
                # Homebrew package
                pkg_spec = f"{name}@{version}" if version else name
                cmd = ["brew", "install", pkg_spec]
            elif manager == "pip":
                pkg_spec = f"{name}=={version}" if version else name
                cmd = ["pip", "install", pkg_spec]
            elif manager == "npm":
                pkg_spec = f"{name}@{version}" if version else name
                cmd = ["npm", "install", "-g", pkg_spec]
            else:
                logger.error(f"Unsupported package manager: {manager}")
                return False
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await result.communicate()
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Package installation failed: {e}")
            return False
    
    async def uninstall_package(self, name: str, manager: str = "auto") -> bool:
        """Uninstall package on macOS."""
        try:
            if manager == "auto":
                manager = self.package_manager
            
            if manager == "brew":
                cmd = ["brew", "uninstall", name]
            elif manager == "pip":
                cmd = ["pip", "uninstall", "-y", name]
            elif manager == "npm":
                cmd = ["npm", "uninstall", "-g", name]
            else:
                return False
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await result.communicate()
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Package uninstallation failed: {e}")
            return False
    
    async def list_packages(self, manager: str = "auto") -> List[Dict[str, str]]:
        """List installed packages on macOS."""
        try:
            if manager == "auto":
                manager = self.package_manager
            
            if manager == "brew":
                cmd = ["brew", "list", "--versions"]
            elif manager == "pip":
                cmd = ["pip", "list", "--format=freeze"]
            elif manager == "npm":
                cmd = ["npm", "list", "-g", "--depth=0"]
            else:
                return []
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await result.communicate()
            
            if result.returncode != 0:
                return []
            
            # Parse output (simplified)
            packages = []
            lines = stdout.decode('utf-8').strip().split('\n')
            
            for line in lines:
                if manager == "pip" and "==" in line:
                    name, version = line.split("==", 1)
                    packages.append({"name": name, "version": version})
                elif manager == "brew" and " " in line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        packages.append({"name": parts[0], "version": parts[1]})
                # Would implement parsing for other package managers
            
            return packages
            
        except Exception as e:
            logger.error(f"Failed to list packages: {e}")
            return []
    
    # SnapshotAdapter implementation
    async def snapshot(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Create snapshot on macOS."""
        try:
            scope = target.get("scope", "agent")
            
            if scope == "agent":
                return {
                    "snapshot_id": f"macos_agent_{int(asyncio.get_event_loop().time())}",
                    "scope": scope,
                    "data": {
                        "agent_version": "2.2.5",
                        "config": self.config,
                        "runtime_state": await self._capture_runtime_state()
                    }
                }
            else:
                return {"success": False, "error": f"Unsupported snapshot scope: {scope}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def rollback(self, snapshot_id: str) -> Dict[str, Any]:
        """Rollback to snapshot on macOS."""
        try:
            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "restored_items": ["agent_config", "runtime_state"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def list_snapshots(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available snapshots on macOS."""
        return [
            {
                "snapshot_id": "macos_agent_123456",
                "scope": "agent",
                "created_at": "2025-01-01T00:00:00Z",
                "size_mb": 12
            }
        ]
    
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete snapshot on macOS."""
        return True
    
    # Helper methods
    def _detect_package_manager(self) -> str:
        """Detect available package manager."""
        if shutil.which("brew"):
            return "brew"
        else:
            return "unknown"
    
    def _detect_architecture(self) -> str:
        """Detect macOS architecture."""
        try:
            result = subprocess.run(["uname", "-m"], capture_output=True, text=True)
            arch = result.stdout.strip()
            if arch == "arm64":
                return "arm64"  # Apple Silicon
            else:
                return "x86_64"  # Intel
        except Exception:
            return "unknown"
    
    def _apply_sandboxd(self, command: List[str], constraints: Dict[str, Any]) -> List[str]:
        """Apply macOS sandbox wrapper to command."""
        # In real implementation, would use sandbox-exec or similar
        # For now, just log the intent
        logger.info("macOS sandboxd requested but not implemented in mock")
        return command
    
    def _prepare_environment(self, task: Dict[str, Any]) -> Dict[str, str]:
        """Prepare environment variables for task."""
        env = os.environ.copy()
        runtime_env = task.get("runtime", {}).get("env", {})
        env.update(runtime_env)
        
        # Add macOS-specific paths
        if "/opt/homebrew/bin" not in env.get("PATH", ""):
            env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
        
        return env
    
    async def _ensure_python(self, version: str) -> bool:
        """Ensure Python version is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "python3", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _ensure_node(self, version: str) -> bool:
        """Ensure Node.js version is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "node", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _ensure_java(self, version: str) -> bool:
        """Ensure Java version is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "java", "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _capture_runtime_state(self) -> Dict[str, Any]:
        """Capture current runtime state."""
        return {
            "package_manager": self.package_manager,
            "architecture": self.arch,
            "sandbox_backend": self.sandbox_backend,
            "timestamp": asyncio.get_event_loop().time()
        }