"""
Multi-OS Host Inventory Registry - Manages host registration and capabilities.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import uuid


logger = logging.getLogger(__name__)


class Registry:
    """
    Host inventory registry for Multi-OS kernel.
    Manages host registration, capabilities, and health status.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.hosts = {}  # host_id -> HostDescriptor
        self.storage_path = storage_path or "/tmp/multi_os_hosts.json"
        self.health_check_interval = 60  # seconds
        self.host_timeout = 300  # seconds - when to mark host as offline
        self.capabilities_map = {
            "process": "Process execution and management",
            "fs": "Filesystem operations",
            "net": "Network operations", 
            "pkg": "Package management",
            "gpu": "GPU acceleration",
            "container": "Container runtime",
            "vm": "Virtual machine management",
            "sandbox": "Sandboxing and isolation"
        }
        
        # Load existing hosts
        self._load_hosts()
        
        logger.info("Multi-OS Host Registry initialized")
    
    def register_host(self, host_descriptor: Dict[str, Any]) -> str:
        """
        Register or update a host in the inventory.
        
        Args:
            host_descriptor: HostDescriptor specification
            
        Returns:
            Host ID
        """
        try:
            # Validate host descriptor
            self._validate_host_descriptor(host_descriptor)
            
            host_id = host_descriptor["host_id"]
            
            # Add timestamp and processing info
            host_descriptor["registered_at"] = datetime.utcnow().isoformat()
            host_descriptor["last_seen"] = datetime.utcnow().isoformat()
            host_descriptor["health_score"] = 1.0
            
            # If this is an update, preserve some historical data
            if host_id in self.hosts:
                old_host = self.hosts[host_id]
                host_descriptor["first_registered"] = old_host.get("first_registered", host_descriptor["registered_at"])
                host_descriptor["total_tasks"] = old_host.get("total_tasks", 0)
                host_descriptor["successful_tasks"] = old_host.get("successful_tasks", 0)
                host_descriptor["failed_tasks"] = old_host.get("failed_tasks", 0)
            else:
                host_descriptor["first_registered"] = host_descriptor["registered_at"]
                host_descriptor["total_tasks"] = 0
                host_descriptor["successful_tasks"] = 0  
                host_descriptor["failed_tasks"] = 0
            
            self.hosts[host_id] = host_descriptor
            self._save_hosts()
            
            logger.info(f"Host {host_id} registered: {host_descriptor['os']}/{host_descriptor['arch']} "
                       f"with {len(host_descriptor.get('capabilities', []))} capabilities")
            
            return host_id
            
        except Exception as e:
            logger.error(f"Host registration failed: {e}")
            raise
    
    def get_host(self, host_id: str) -> Optional[Dict[str, Any]]:
        """Get host descriptor by ID."""
        return self.hosts.get(host_id)
    
    def list_hosts(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List hosts with optional filtering.
        
        Args:
            filters: Optional filters like {"os": "linux", "status": "online"}
            
        Returns:
            List of matching host descriptors
        """
        hosts = list(self.hosts.values())
        
        if not filters:
            return hosts
            
        filtered_hosts = []
        for host in hosts:
            match = True
            
            for key, value in filters.items():
                if key == "os" and host.get("os") != value:
                    match = False
                    break
                elif key == "arch" and host.get("arch") != value:
                    match = False
                    break
                elif key == "status" and host.get("status") != value:
                    match = False
                    break
                elif key == "capability" and value not in host.get("capabilities", []):
                    match = False
                    break
                elif key == "label":
                    # Check if any label contains the filter value
                    if not any(value in label for label in host.get("labels", [])):
                        match = False
                        break
            
            if match:
                filtered_hosts.append(host)
        
        return filtered_hosts
    
    def update_host_status(self, host_id: str, status: str, health_metrics: Optional[Dict] = None) -> bool:
        """
        Update host status and health metrics.
        
        Args:
            host_id: Host identifier
            status: New status (online, degraded, offline)
            health_metrics: Optional health metrics dict
            
        Returns:
            True if update was successful
        """
        if host_id not in self.hosts:
            logger.warning(f"Attempted to update status for unknown host: {host_id}")
            return False
        
        host = self.hosts[host_id]
        old_status = host.get("status")
        
        host["status"] = status
        host["last_seen"] = datetime.utcnow().isoformat()
        
        if health_metrics:
            host["health_metrics"] = health_metrics
            host["health_score"] = self._calculate_health_score(health_metrics)
        
        if old_status != status:
            logger.info(f"Host {host_id} status changed: {old_status} -> {status}")
        
        self._save_hosts()
        return True
    
    def update_task_stats(self, host_id: str, success: bool) -> bool:
        """
        Update task execution statistics for a host.
        
        Args:
            host_id: Host identifier
            success: Whether the task was successful
            
        Returns:
            True if update was successful
        """
        if host_id not in self.hosts:
            return False
        
        host = self.hosts[host_id]
        host["total_tasks"] = host.get("total_tasks", 0) + 1
        
        if success:
            host["successful_tasks"] = host.get("successful_tasks", 0) + 1
        else:
            host["failed_tasks"] = host.get("failed_tasks", 0) + 1
        
        # Update success rate
        total = host["total_tasks"]
        successful = host["successful_tasks"]
        host["success_rate"] = successful / total if total > 0 else 0.0
        
        self._save_hosts()
        return True
    
    def unregister_host(self, host_id: str) -> bool:
        """
        Remove a host from the registry.
        
        Args:
            host_id: Host identifier
            
        Returns:
            True if host was removed
        """
        if host_id in self.hosts:
            del self.hosts[host_id]
            self._save_hosts()
            logger.info(f"Host {host_id} unregistered")
            return True
        
        return False
    
    def find_capable_hosts(self, required_capabilities: List[str], 
                          constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find hosts that have the required capabilities and meet constraints.
        
        Args:
            required_capabilities: List of required capability names
            constraints: Optional constraints dict
            
        Returns:
            List of matching host descriptors
        """
        matching_hosts = []
        
        for host in self.hosts.values():
            # Check status
            if host.get("status") != "online":
                continue
            
            # Check capabilities
            host_capabilities = set(host.get("capabilities", []))
            required_set = set(required_capabilities)
            
            if not required_set.issubset(host_capabilities):
                continue
            
            # Check constraints if provided
            if constraints:
                # OS constraint
                if "os" in constraints:
                    required_os = constraints["os"]
                    if isinstance(required_os, list):
                        if host.get("os") not in required_os:
                            continue
                    else:
                        if host.get("os") != required_os:
                            continue
                
                # Architecture constraint
                if "arch" in constraints:
                    required_arch = constraints["arch"]
                    if isinstance(required_arch, list):
                        if host.get("arch") not in required_arch:
                            continue
                    else:
                        if host.get("arch") != required_arch:
                            continue
                
                # GPU constraint
                if constraints.get("gpu_required", False):
                    if "gpu" not in host_capabilities:
                        continue
                    
                    # Check GPU labels
                    gpu_labels = [label for label in host.get("labels", []) if label.startswith("gpu:")]
                    if not gpu_labels or any("none" in label.lower() for label in gpu_labels):
                        continue
                
                # Label constraints
                if "labels" in constraints:
                    required_labels = constraints["labels"]
                    host_labels = host.get("labels", [])
                    
                    for req_label in required_labels:
                        if not any(req_label in host_label for host_label in host_labels):
                            continue
            
            matching_hosts.append(host)
        
        # Sort by health score and success rate
        matching_hosts.sort(
            key=lambda h: (h.get("health_score", 0), h.get("success_rate", 0)), 
            reverse=True
        )
        
        return matching_hosts
    
    def get_inventory_stats(self) -> Dict[str, Any]:
        """Get inventory statistics."""
        if not self.hosts:
            return {
                "total_hosts": 0,
                "by_os": {},
                "by_arch": {},
                "by_status": {},
                "capabilities": {},
                "avg_health_score": 0.0
            }
        
        stats = {
            "total_hosts": len(self.hosts),
            "by_os": {},
            "by_arch": {},
            "by_status": {},
            "capabilities": {},
            "avg_health_score": 0.0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0
        }
        
        health_scores = []
        
        for host in self.hosts.values():
            # Count by OS
            os_name = host.get("os", "unknown")
            stats["by_os"][os_name] = stats["by_os"].get(os_name, 0) + 1
            
            # Count by architecture
            arch = host.get("arch", "unknown")
            stats["by_arch"][arch] = stats["by_arch"].get(arch, 0) + 1
            
            # Count by status
            status = host.get("status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # Count capabilities
            for cap in host.get("capabilities", []):
                stats["capabilities"][cap] = stats["capabilities"].get(cap, 0) + 1
            
            # Collect health scores
            health_score = host.get("health_score", 0.0)
            health_scores.append(health_score)
            
            # Aggregate task stats
            stats["total_tasks"] += host.get("total_tasks", 0)
            stats["successful_tasks"] += host.get("successful_tasks", 0) 
            stats["failed_tasks"] += host.get("failed_tasks", 0)
        
        # Calculate average health score
        if health_scores:
            stats["avg_health_score"] = sum(health_scores) / len(health_scores)
        
        # Calculate overall success rate
        if stats["total_tasks"] > 0:
            stats["success_rate"] = stats["successful_tasks"] / stats["total_tasks"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def cleanup_stale_hosts(self) -> int:
        """
        Remove hosts that haven't been seen for a while.
        
        Returns:
            Number of hosts removed
        """
        now = datetime.utcnow()
        timeout_threshold = now - timedelta(seconds=self.host_timeout)
        
        stale_hosts = []
        
        for host_id, host in self.hosts.items():
            last_seen_str = host.get("last_seen")
            if last_seen_str:
                try:
                    last_seen = datetime.fromisoformat(last_seen_str.replace("Z", "+00:00"))
                    if last_seen.replace(tzinfo=None) < timeout_threshold:
                        stale_hosts.append(host_id)
                except ValueError:
                    # Invalid timestamp, consider stale
                    stale_hosts.append(host_id)
        
        # Remove stale hosts
        for host_id in stale_hosts:
            self.unregister_host(host_id)
            logger.info(f"Removed stale host: {host_id}")
        
        return len(stale_hosts)
    
    def _validate_host_descriptor(self, host_descriptor: Dict[str, Any]) -> None:
        """Validate host descriptor format."""
        required_fields = ["host_id", "os", "arch", "agent_version", "capabilities", "labels", "status"]
        
        for field in required_fields:
            if field not in host_descriptor:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate OS
        valid_os = ["linux", "windows", "macos"]
        if host_descriptor["os"] not in valid_os:
            raise ValueError(f"Invalid OS: {host_descriptor['os']}. Must be one of {valid_os}")
        
        # Validate architecture
        valid_arch = ["x86_64", "arm64"] 
        if host_descriptor["arch"] not in valid_arch:
            raise ValueError(f"Invalid architecture: {host_descriptor['arch']}. Must be one of {valid_arch}")
        
        # Validate status
        valid_status = ["online", "degraded", "offline"]
        if host_descriptor["status"] not in valid_status:
            raise ValueError(f"Invalid status: {host_descriptor['status']}. Must be one of {valid_status}")
        
        # Validate capabilities
        valid_capabilities = list(self.capabilities_map.keys())
        for cap in host_descriptor["capabilities"]:
            if cap not in valid_capabilities:
                raise ValueError(f"Invalid capability: {cap}. Must be one of {valid_capabilities}")
    
    def _calculate_health_score(self, health_metrics: Dict[str, Any]) -> float:
        """Calculate health score from metrics."""
        score = 1.0
        
        # CPU usage penalty
        cpu = health_metrics.get("cpu", 0.0)
        if cpu > 0.8:
            score -= 0.3
        elif cpu > 0.6:
            score -= 0.1
        
        # Memory usage penalty  
        mem_used = health_metrics.get("mem_used_mb", 0)
        mem_total = health_metrics.get("mem_total_mb", 1)
        mem_usage = mem_used / mem_total if mem_total > 0 else 0
        
        if mem_usage > 0.9:
            score -= 0.4
        elif mem_usage > 0.7:
            score -= 0.2
        
        # Disk space penalty
        disk_free = health_metrics.get("disk_free_mb", float('inf'))
        if disk_free < 1000:  # Less than 1GB free
            score -= 0.5
        elif disk_free < 5000:  # Less than 5GB free
            score -= 0.2
        
        return max(0.0, score)
    
    def _load_hosts(self) -> None:
        """Load hosts from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.hosts = data.get("hosts", {})
                logger.info(f"Loaded {len(self.hosts)} hosts from storage")
        except FileNotFoundError:
            logger.info("No existing host storage found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load hosts from storage: {e}")
    
    def _save_hosts(self) -> None:
        """Save hosts to storage."""
        try:
            data = {
                "hosts": self.hosts,
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save hosts to storage: {e}")