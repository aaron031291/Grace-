"""
Multi-OS Snapshots Manager - System state snapshots and rollback functionality.
"""
import logging
import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
import uuid


logger = logging.getLogger(__name__)


class SnapshotManager:
    """
    Manages system snapshots and rollback functionality for Multi-OS kernel.
    Handles agent versions, runtime caches, configurations, and golden images.
    """
    
    def __init__(self, storage_path: str = "/tmp/multi_os_snapshots"):
        self.storage_path = storage_path
        self.snapshots = {}  # snapshot_id -> snapshot data
        self.current_config = self._get_default_config()
        
        # Load existing snapshots
        self._load_snapshots()
        
        logger.info("Multi-OS Snapshot Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return {
            "agent_versions": {
                "linux": "2.4.1",
                "windows": "2.3.0", 
                "macos": "2.2.5"
            },
            "runtime_caches": {
                "python@3.11": {
                    "pip": "hash:abc123",
                    "wheels": ["numpy-1.24.0", "pydantic-2.5.0"]
                },
                "node@18": {
                    "npm": "hash:def456"
                }
            },
            "sandbox_profiles": {
                "linux": "nsjail_v5",
                "windows": "appcontainer_low", 
                "macos": "sandboxd_profile_3"
            },
            "placement_weights": {
                "capability_fit": 0.4,
                "latency": 0.25,
                "success": 0.25,
                "gpu": 0.1
            },
            "network_policies": {
                "default": "deny_all",
                "allowlist": ["api.company.local"]
            },
            "golden_images": {
                "ubuntu20.04": "ami-12345",
                "win11": "image-67890",
                "macos13": "apfs-snap-abcdef"
            },
            "timeouts": {
                "task_max_runtime_s": 1800
            },
            "rollout": {
                "strategy": "blue_green",
                "rings": ["canary:5%", "ring1:25%", "ring2:50%", "ring3:100%"]
            }
        }
    
    async def create_snapshot(self, scope: str = "agent", host_id: Optional[str] = None, 
                             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a snapshot of the current system state.
        
        Args:
            scope: Snapshot scope (agent, image, vm, container)
            host_id: Optional specific host to snapshot
            metadata: Optional additional metadata
            
        Returns:
            Snapshot information
        """
        try:
            snapshot_id = f"mos_{utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}"
            
            # Create snapshot payload based on scope
            if scope == "agent":
                payload = await self._create_agent_snapshot(host_id)
            elif scope == "image":
                payload = await self._create_image_snapshot(host_id)
            elif scope == "vm":
                payload = await self._create_vm_snapshot(host_id)
            elif scope == "container":
                payload = await self._create_container_snapshot(host_id)
            else:
                raise ValueError(f"Invalid snapshot scope: {scope}")
            
            # Add metadata
            payload.update({
                "snapshot_id": snapshot_id,
                "scope": scope,
                "host_id": host_id,
                "created_at": iso_format(),
                "metadata": metadata or {},
                "version": "1.0.0"
            })
            
            # Calculate hash
            payload_str = json.dumps(payload, sort_keys=True)
            payload["hash"] = hashlib.sha256(payload_str.encode()).hexdigest()
            
            # Store snapshot
            self.snapshots[snapshot_id] = payload
            self._save_snapshot(snapshot_id, payload)
            
            logger.info(f"Created {scope} snapshot {snapshot_id}")
            
            return {
                "snapshot_id": snapshot_id,
                "scope": scope,
                "uri": f"file://{self.storage_path}/{snapshot_id}.json",
                "hash": payload["hash"],
                "size_bytes": len(payload_str),
                "created_at": payload["created_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise
    
    async def rollback(self, snapshot_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Rollback system to a specific snapshot.
        
        Args:
            snapshot_id: Snapshot to rollback to
            dry_run: If True, only validate rollback without applying
            
        Returns:
            Rollback operation results
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        snapshot = self.snapshots[snapshot_id]
        rollback_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting rollback to snapshot {snapshot_id} (dry_run={dry_run})")
            
            rollback_plan = await self._create_rollback_plan(snapshot)
            
            if dry_run:
                return {
                    "rollback_id": rollback_id,
                    "snapshot_id": snapshot_id,
                    "dry_run": True,
                    "plan": rollback_plan,
                    "estimated_duration_seconds": rollback_plan.get("estimated_duration", 0),
                    "risks": rollback_plan.get("risks", [])
                }
            
            # Execute rollback
            results = await self._execute_rollback(snapshot, rollback_plan)
            
            # Update current configuration
            if results.get("success"):
                self._update_current_config(snapshot)
            
            logger.info(f"Rollback {rollback_id} completed with success={results.get('success')}")
            
            return {
                "rollback_id": rollback_id,
                "snapshot_id": snapshot_id,
                "success": results.get("success", False),
                "steps_completed": results.get("steps_completed", []),
                "errors": results.get("errors", []),
                "duration_seconds": results.get("duration_seconds", 0),
                "completed_at": iso_format()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {
                "rollback_id": rollback_id,
                "snapshot_id": snapshot_id,
                "success": False,
                "error": str(e),
                "completed_at": iso_format()
            }
    
    def list_snapshots(self, scope: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List available snapshots.
        
        Args:
            scope: Optional scope filter
            limit: Maximum number of snapshots to return
            
        Returns:
            List of snapshot summaries
        """
        snapshots = []
        
        for snapshot_id, snapshot in self.snapshots.items():
            if scope and snapshot.get("scope") != scope:
                continue
                
            summary = {
                "snapshot_id": snapshot_id,
                "scope": snapshot.get("scope"),
                "host_id": snapshot.get("host_id"),
                "created_at": snapshot.get("created_at"),
                "hash": snapshot.get("hash"),
                "metadata": snapshot.get("metadata", {}),
                "size_estimate": len(json.dumps(snapshot))
            }
            
            snapshots.append(summary)
        
        # Sort by creation time (newest first)
        snapshots.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return snapshots[:limit]
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get specific snapshot details."""
        return self.snapshots.get(snapshot_id)
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.
        
        Args:
            snapshot_id: Snapshot to delete
            
        Returns:
            True if snapshot was deleted
        """
        if snapshot_id not in self.snapshots:
            return False
        
        del self.snapshots[snapshot_id]
        
        # Remove from disk
        try:
            import os
            snapshot_file = f"{self.storage_path}/{snapshot_id}.json"
            if os.path.exists(snapshot_file):
                os.remove(snapshot_file)
        except Exception as e:
            logger.warning(f"Failed to delete snapshot file: {e}")
        
        logger.info(f"Deleted snapshot {snapshot_id}")
        return True
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return self.current_config.copy()
    
    async def _create_agent_snapshot(self, host_id: Optional[str]) -> Dict[str, Any]:
        """Create agent-level snapshot."""
        return {
            "agent_versions": self.current_config.get("agent_versions", {}),
            "runtime_caches": self.current_config.get("runtime_caches", {}),
            "sandbox_profiles": self.current_config.get("sandbox_profiles", {}),
            "placement_weights": self.current_config.get("placement_weights", {}),
            "network_policies": self.current_config.get("network_policies", {}),
            "rollout_config": self.current_config.get("rollout", {}),
            "timeouts": self.current_config.get("timeouts", {})
        }
    
    async def _create_image_snapshot(self, host_id: Optional[str]) -> Dict[str, Any]:
        """Create golden image snapshot."""
        return {
            "golden_images": self.current_config.get("golden_images", {}),
            "base_configurations": {
                "linux": {
                    "packages": ["systemd", "docker", "python3"],
                    "services": ["sshd", "docker"]
                },
                "windows": {
                    "features": ["IIS", "Containers"],
                    "services": ["winrm", "docker"]
                },
                "macos": {
                    "packages": ["homebrew", "python3"],
                    "services": ["sshd"]
                }
            },
            "security_policies": {
                "firewall_rules": ["allow 22/tcp", "allow 8080/tcp"],
                "user_accounts": ["multi-os-agent"],
                "certificates": ["multi-os-ca.crt"]
            }
        }
    
    async def _create_vm_snapshot(self, host_id: Optional[str]) -> Dict[str, Any]:
        """Create VM-level snapshot."""
        return {
            "vm_templates": {
                "linux": {
                    "image": "ubuntu-20.04",
                    "cpu": 4,
                    "memory_gb": 8,
                    "disk_gb": 50
                },
                "windows": {
                    "image": "windows-server-2022",
                    "cpu": 4,
                    "memory_gb": 16,
                    "disk_gb": 100
                }
            },
            "network_config": {
                "subnets": ["10.0.1.0/24", "10.0.2.0/24"],
                "security_groups": ["multi-os-sg"]
            },
            "storage_config": {
                "volumes": ["data", "logs", "cache"],
                "backup_policy": "daily"
            }
        }
    
    async def _create_container_snapshot(self, host_id: Optional[str]) -> Dict[str, Any]:
        """Create container-level snapshot."""
        return {
            "container_images": {
                "multi-os-agent": {
                    "tag": "2.4.1",
                    "digest": "sha256:abcdef123456"
                },
                "runtime-python": {
                    "tag": "3.11-slim",
                    "digest": "sha256:fedcba654321"
                }
            },
            "container_configs": {
                "agent": {
                    "limits": {"cpu": "1000m", "memory": "1Gi"},
                    "env": {"LOG_LEVEL": "INFO"},
                    "volumes": ["/tmp", "/var/log"]
                }
            },
            "registry_config": {
                "url": "registry.company.local",
                "auth": "token-based"
            }
        }
    
    async def _create_rollback_plan(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed rollback execution plan."""
        scope = snapshot.get("scope", "agent")
        
        plan = {
            "scope": scope,
            "steps": [],
            "estimated_duration": 0,
            "risks": [],
            "rollback_strategy": "blue_green"
        }
        
        if scope == "agent":
            plan["steps"] = [
                {"step": "freeze_task_placement", "duration": 5, "description": "Stop new task placements"},
                {"step": "drain_running_tasks", "duration": 120, "description": "Wait for running tasks to complete"}, 
                {"step": "update_agent_versions", "duration": 60, "description": "Deploy previous agent versions"},
                {"step": "restore_configurations", "duration": 30, "description": "Restore sandbox and network configs"},
                {"step": "restore_placement_weights", "duration": 10, "description": "Restore placement algorithm weights"},
                {"step": "resume_operations", "duration": 15, "description": "Resume normal operations"}
            ]
            plan["estimated_duration"] = 240  # 4 minutes
            plan["risks"] = [
                "Running tasks may be interrupted",
                "Brief service unavailability during agent updates"
            ]
        
        elif scope == "image":
            plan["steps"] = [
                {"step": "backup_current_images", "duration": 300, "description": "Backup current golden images"},
                {"step": "restore_golden_images", "duration": 600, "description": "Restore previous golden images"},
                {"step": "restart_hosts", "duration": 900, "description": "Restart hosts with restored images"},
                {"step": "verify_host_health", "duration": 180, "description": "Verify all hosts are healthy"}
            ]
            plan["estimated_duration"] = 1980  # 33 minutes
            plan["risks"] = [
                "Extended downtime during host restarts",
                "Potential data loss if images are corrupted",
                "Network partitioning during restart sequence"
            ]
        
        elif scope in ["vm", "container"]:
            plan["steps"] = [
                {"step": "create_current_backup", "duration": 180, "description": f"Backup current {scope} state"},
                {"step": "stop_services", "duration": 60, "description": "Stop running services"},
                {"step": f"restore_{scope}_config", "duration": 120, "description": f"Restore {scope} configuration"},
                {"step": "restart_services", "duration": 90, "description": "Restart services"},
                {"step": "health_check", "duration": 60, "description": "Verify service health"}
            ]
            plan["estimated_duration"] = 510  # 8.5 minutes
            plan["risks"] = [
                f"Service downtime during {scope} restoration",
                "Configuration drift if manual changes were made"
            ]
        
        return plan
    
    async def _execute_rollback(self, snapshot: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the rollback plan."""
        start_time = utc_now()
        results = {
            "success": True,
            "steps_completed": [],
            "errors": []
        }
        
        try:
            for step_info in plan["steps"]:
                step_name = step_info["step"]
                logger.info(f"Executing rollback step: {step_name}")
                
                # Simulate step execution (in real implementation, would call actual rollback functions)
                await asyncio.sleep(0.1)  # Simulate work
                
                # Mock success for most steps
                if step_name == "drain_running_tasks":
                    # Simulate longer operation
                    await asyncio.sleep(1)
                
                results["steps_completed"].append({
                    "step": step_name,
                    "description": step_info["description"],
                    "completed_at": iso_format(),
                    "success": True
                })
                
                logger.info(f"Completed rollback step: {step_name}")
        
        except Exception as e:
            logger.error(f"Rollback step failed: {e}")
            results["success"] = False
            results["errors"].append(str(e))
        
        # Calculate duration
        end_time = utc_now()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        return results
    
    def _update_current_config(self, snapshot: Dict[str, Any]) -> None:
        """Update current configuration from snapshot."""
        scope = snapshot.get("scope", "agent")
        
        if scope == "agent":
            # Restore agent-level configuration
            for key in ["agent_versions", "runtime_caches", "sandbox_profiles", 
                       "placement_weights", "network_policies", "rollout", "timeouts"]:
                if key in snapshot:
                    self.current_config[key] = snapshot[key]
        
        elif scope == "image":
            # Restore image-level configuration
            if "golden_images" in snapshot:
                self.current_config["golden_images"] = snapshot["golden_images"]
        
        # Save updated configuration
        self._save_current_config()
    
    def _save_current_config(self) -> None:
        """Save current configuration to disk."""
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            
            config_file = f"{self.storage_path}/current_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save current config: {e}")
    
    def _save_snapshot(self, snapshot_id: str, snapshot_data: Dict[str, Any]) -> None:
        """Save snapshot to disk."""
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            
            snapshot_file = f"{self.storage_path}/{snapshot_id}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save snapshot to disk: {e}")
    
    def _load_snapshots(self) -> None:
        """Load snapshots from disk."""
        try:
            import os
            if not os.path.exists(self.storage_path):
                return
            
            # Load current config
            config_file = f"{self.storage_path}/current_config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.current_config.update(json.load(f))
            
            # Load snapshots
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json') and filename != 'current_config.json':
                    snapshot_id = filename[:-5]  # Remove .json extension
                    
                    try:
                        with open(f"{self.storage_path}/{filename}", 'r') as f:
                            self.snapshots[snapshot_id] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load snapshot {filename}: {e}")
            
            logger.info(f"Loaded {len(self.snapshots)} snapshots from storage")
            
        except Exception as e:
            logger.error(f"Failed to load snapshots from storage: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get snapshot manager statistics."""
        snapshots_by_scope = {}
        total_size = 0
        
        for snapshot_id, snapshot in self.snapshots.items():
            scope = snapshot.get("scope", "unknown")
            snapshots_by_scope[scope] = snapshots_by_scope.get(scope, 0) + 1
            total_size += len(json.dumps(snapshot))
        
        return {
            "total_snapshots": len(self.snapshots),
            "by_scope": snapshots_by_scope,
            "total_size_bytes": total_size,
            "storage_path": self.storage_path,
            "current_config_size": len(json.dumps(self.current_config))
        }