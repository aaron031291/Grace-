"""
Base adapter interfaces for Multi-OS capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class ProcessAdapter(ABC):
    """Base interface for process management across operating systems."""

    @abstractmethod
    async def exec(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task on the host system.

        Args:
            task: ExecTask specification

        Returns:
            Dict with execution results including pid, status, etc.
        """
        pass

    @abstractmethod
    async def kill(self, pid: int) -> bool:
        """
        Kill a process by PID.

        Args:
            pid: Process ID to kill

        Returns:
            True if process was successfully killed
        """
        pass

    @abstractmethod
    async def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a running process.

        Args:
            pid: Process ID

        Returns:
            Process information dict or None if not found
        """
        pass


class FSAdapter(ABC):
    """Base interface for filesystem operations across operating systems."""

    @abstractmethod
    async def apply(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a filesystem action.

        Args:
            action: FSAction specification

        Returns:
            Dict with action results
        """
        pass

    @abstractmethod
    async def read(self, path: str, encoding: str = "utf-8") -> bytes:
        """Read file contents."""
        pass

    @abstractmethod
    async def write(self, path: str, content: bytes) -> bool:
        """Write file contents."""
        pass

    @abstractmethod
    async def list_dir(self, path: str, recursive: bool = False) -> List[str]:
        """List directory contents."""
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    @abstractmethod
    async def delete(self, path: str, recursive: bool = False) -> bool:
        """Delete file or directory."""
        pass


class NetAdapter(ABC):
    """Base interface for network operations across operating systems."""

    @abstractmethod
    async def apply(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a network action.

        Args:
            action: NetAction specification

        Returns:
            Dict with action results
        """
        pass

    @abstractmethod
    async def fetch(
        self, url: str, timeout: int = 30, headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Fetch data from URL."""
        pass

    @abstractmethod
    async def post(
        self, url: str, data: Any, timeout: int = 30, headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Post data to URL."""
        pass

    @abstractmethod
    async def port_check(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if port is open."""
        pass


class PkgAdapter(ABC):
    """Base interface for package management across operating systems."""

    @abstractmethod
    async def ensure(self, runtime_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure runtime environment is available (idempotent).

        Args:
            runtime_spec: RuntimeSpec specification

        Returns:
            Dict with setup results
        """
        pass

    @abstractmethod
    async def install_package(
        self, name: str, version: Optional[str] = None, manager: str = "auto"
    ) -> bool:
        """Install a package."""
        pass

    @abstractmethod
    async def uninstall_package(self, name: str, manager: str = "auto") -> bool:
        """Uninstall a package."""
        pass

    @abstractmethod
    async def list_packages(self, manager: str = "auto") -> List[Dict[str, str]]:
        """List installed packages."""
        pass


class SnapshotAdapter(ABC):
    """Base interface for snapshot management across operating systems."""

    @abstractmethod
    async def snapshot(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a snapshot of the target system/component.

        Args:
            target: Target specification (agent, image, vm, container)

        Returns:
            Dict with snapshot information
        """
        pass

    @abstractmethod
    async def rollback(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Rollback to a previous snapshot.

        Args:
            snapshot_id: ID of snapshot to rollback to

        Returns:
            Dict with rollback results
        """
        pass

    @abstractmethod
    async def list_snapshots(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available snapshots."""
        pass

    @abstractmethod
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        pass


class GPUAdapter(ABC):
    """Base interface for GPU operations across operating systems."""

    @abstractmethod
    async def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about available GPUs."""
        pass

    @abstractmethod
    async def allocate_gpu(
        self, requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Allocate GPU resources."""
        pass

    @abstractmethod
    async def release_gpu(self, allocation_id: str) -> bool:
        """Release GPU resources."""
        pass


class ContainerAdapter(ABC):
    """Base interface for container operations."""

    @abstractmethod
    async def run_container(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run a container with specified configuration."""
        pass

    @abstractmethod
    async def stop_container(self, container_id: str) -> bool:
        """Stop a running container."""
        pass

    @abstractmethod
    async def list_containers(self) -> List[Dict[str, Any]]:
        """List containers."""
        pass


class VMAdapter(ABC):
    """Base interface for virtual machine operations."""

    @abstractmethod
    async def start_vm(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Start a virtual machine."""
        pass

    @abstractmethod
    async def stop_vm(self, vm_id: str) -> bool:
        """Stop a virtual machine."""
        pass

    @abstractmethod
    async def list_vms(self) -> List[Dict[str, Any]]:
        """List virtual machines."""
        pass
