"""
Grace Orchestration Kernel Bridges - Registry and integration for all Grace kernels.

Provides a centralized registry and bridge for communicating with all Grace kernels
including governance, memory, learning, intelligence, interface, ingress, etc.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class KernelStatus(Enum):
    UNKNOWN = "unknown"
    REGISTERING = "registering"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class KernelInfo:
    """Information about a registered kernel."""
    
    def __init__(self, kernel_name: str, kernel_instance: Any = None, 
                 endpoints: Dict[str, str] = None):
        self.kernel_name = kernel_name
        self.kernel_instance = kernel_instance
        self.endpoints = endpoints or {}
        
        self.status = KernelStatus.REGISTERING
        self.registered_at = datetime.now()
        self.last_heartbeat = self.registered_at
        self.version = "unknown"
        self.capabilities: Set[str] = set()
        
        # Communication statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.errors = 0
        self.last_error = None
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
    
    def record_message_sent(self):
        """Record a message sent to this kernel."""
        self.messages_sent += 1
    
    def record_message_received(self):
        """Record a message received from this kernel."""
        self.messages_received += 1
    
    def record_error(self, error: str):
        """Record an error with this kernel."""
        self.errors += 1
        self.last_error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "kernel_name": self.kernel_name,
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "version": self.version,
            "capabilities": list(self.capabilities),
            "endpoints": self.endpoints,
            "statistics": {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "errors": self.errors,
                "last_error": self.last_error
            }
        }


class KernelBridges:
    """Registry and bridge for all Grace kernels."""
    
    def __init__(self, event_publisher=None):
        self.event_publisher = event_publisher
        
        # Kernel registry
        self.kernels: Dict[str, KernelInfo] = {}
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Configuration
        self.heartbeat_timeout_seconds = 120  # 2 minutes
        self.auto_registration_enabled = True
        
        # Known Grace kernels and their expected capabilities
        self.known_kernels = {
            "governance": {
                "capabilities": ["policy_validation", "compliance_check", "audit"],
                "endpoints": {"validate": "/api/gov/v1/validate", "policies": "/api/gov/v1/policies"}
            },
            "memory": {
                "capabilities": ["store_experience", "retrieve_patterns", "memory_consolidation"],
                "endpoints": {"store": "/api/memory/v1/store", "query": "/api/memory/v1/query"}
            },
            "learning": {
                "capabilities": ["pattern_recognition", "adaptation", "meta_learning"],
                "endpoints": {"learn": "/api/learn/v1/learn", "adapt": "/api/learn/v1/adapt"}
            },
            "intelligence": {
                "capabilities": ["decision_making", "problem_solving", "reasoning"],
                "endpoints": {"analyze": "/api/intel/v1/analyze", "decide": "/api/intel/v1/decide"}
            },
            "interface": {
                "capabilities": ["api_gateway", "websocket", "ui"],
                "endpoints": {"api": "/api/interface/v1", "ws": "/ws"}
            },
            "ingress": {
                "capabilities": ["data_capture", "normalization", "routing"],
                "endpoints": {"ingest": "/api/ingress/v1/ingest", "sources": "/api/ingress/v1/sources"}
            },
            "mlt": {
                "capabilities": ["trust_calculation", "model_management", "learning_coordination"],
                "endpoints": {"trust": "/api/mlt/v1/trust", "models": "/api/mlt/v1/models"}
            },
            "multi_os": {
                "capabilities": ["cross_platform", "resource_management", "process_isolation"],
                "endpoints": {"health": "/api/os/v1/health", "resources": "/api/os/v1/resources"}
            },
            "resilience": {
                "capabilities": ["failure_recovery", "chaos_engineering", "circuit_breaker"],
                "endpoints": {"health": "/api/resilience/v1/health", "recover": "/api/resilience/v1/recover"}
            },
            "event_mesh": {
                "capabilities": ["event_routing", "message_queuing", "pub_sub"],
                "endpoints": {"publish": "/api/mesh/v1/publish", "subscribe": "/api/mesh/v1/subscribe"}
            }
        }
        
        self.running = False
        self._heartbeat_task = None
    
    async def start(self):
        """Start the kernel bridge registry."""
        if self.running:
            return
        
        logger.info("Starting Orchestration Kernel Bridges...")
        self.running = True
        
        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor_loop())
        
        # Auto-register known kernels if enabled
        if self.auto_registration_enabled:
            await self._auto_register_known_kernels()
        
        logger.info("Orchestration Kernel Bridges started")
    
    async def stop(self):
        """Stop the kernel bridge registry."""
        if not self.running:
            return
        
        logger.info("Stopping Orchestration Kernel Bridges...")
        self.running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Orchestration Kernel Bridges stopped")
    
    async def register_kernel(self, kernel_name: str, kernel_instance: Any = None,
                            endpoints: Dict[str, str] = None,
                            capabilities: Set[str] = None) -> bool:
        """Register a kernel with the bridge."""
        try:
            if kernel_name in self.kernels:
                logger.info(f"Re-registering existing kernel: {kernel_name}")
            else:
                logger.info(f"Registering new kernel: {kernel_name}")
            
            kernel_info = KernelInfo(kernel_name, kernel_instance, endpoints)
            
            # Set capabilities
            if capabilities:
                kernel_info.capabilities.update(capabilities)
            elif kernel_name in self.known_kernels:
                kernel_info.capabilities.update(self.known_kernels[kernel_name]["capabilities"])
            
            # Set default endpoints
            if not endpoints and kernel_name in self.known_kernels:
                kernel_info.endpoints.update(self.known_kernels[kernel_name]["endpoints"])
            
            # Detect version if possible
            if kernel_instance and hasattr(kernel_instance, 'get_version'):
                try:
                    kernel_info.version = await kernel_instance.get_version()
                except:
                    kernel_info.version = "unknown"
            
            self.kernels[kernel_name] = kernel_info
            kernel_info.status = KernelStatus.HEALTHY
            
            logger.info(f"Successfully registered kernel {kernel_name} with {len(kernel_info.capabilities)} capabilities")
            
            # Publish registration event
            if self.event_publisher:
                await self.event_publisher("KERNEL_REGISTERED", {
                    "kernel_name": kernel_name,
                    "capabilities": list(kernel_info.capabilities),
                    "timestamp": datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register kernel {kernel_name}: {e}")
            return False
    
    async def unregister_kernel(self, kernel_name: str) -> bool:
        """Unregister a kernel from the bridge."""
        if kernel_name not in self.kernels:
            logger.warning(f"Attempted to unregister unknown kernel: {kernel_name}")
            return False
        
        try:
            del self.kernels[kernel_name]
            logger.info(f"Unregistered kernel: {kernel_name}")
            
            # Publish unregistration event
            if self.event_publisher:
                await self.event_publisher("KERNEL_UNREGISTERED", {
                    "kernel_name": kernel_name,
                    "timestamp": datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister kernel {kernel_name}: {e}")
            return False
    
    async def send_message(self, kernel_name: str, method: str, 
                          params: Dict[str, Any] = None) -> Optional[Any]:
        """Send a message to a registered kernel."""
        if kernel_name not in self.kernels:
            logger.error(f"Cannot send message to unregistered kernel: {kernel_name}")
            return None
        
        kernel_info = self.kernels[kernel_name]
        
        try:
            if kernel_info.kernel_instance:
                # Direct method call
                if hasattr(kernel_info.kernel_instance, method):
                    method_func = getattr(kernel_info.kernel_instance, method)
                    
                    if asyncio.iscoroutinefunction(method_func):
                        if params:
                            result = await method_func(**params)
                        else:
                            result = await method_func()
                    else:
                        if params:
                            result = method_func(**params)
                        else:
                            result = method_func()
                    
                    kernel_info.record_message_sent()
                    kernel_info.update_heartbeat()
                    
                    return result
                else:
                    error_msg = f"Method {method} not found on kernel {kernel_name}"
                    kernel_info.record_error(error_msg)
                    logger.error(error_msg)
                    return None
            else:
                # Would implement HTTP/gRPC call to kernel endpoint
                logger.debug(f"Would call {kernel_name}.{method} via network")
                kernel_info.record_message_sent()
                return {"status": "simulated"}
        
        except Exception as e:
            error_msg = f"Failed to send message to {kernel_name}.{method}: {e}"
            kernel_info.record_error(error_msg)
            logger.error(error_msg)
            return None
    
    async def broadcast_message(self, method: str, params: Dict[str, Any] = None,
                              capability_filter: str = None) -> Dict[str, Any]:
        """Broadcast a message to all or filtered kernels."""
        results = {}
        
        target_kernels = []
        
        for kernel_name, kernel_info in self.kernels.items():
            if capability_filter and capability_filter not in kernel_info.capabilities:
                continue
            target_kernels.append(kernel_name)
        
        for kernel_name in target_kernels:
            result = await self.send_message(kernel_name, method, params)
            results[kernel_name] = result
        
        logger.debug(f"Broadcast {method} to {len(target_kernels)} kernels")
        return results
    
    async def get_kernel_status(self, kernel_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific kernel."""
        if kernel_name not in self.kernels:
            return None
        
        kernel_info = self.kernels[kernel_name]
        
        # Try to get live status
        try:
            status_result = await self.send_message(kernel_name, "get_status")
            if status_result:
                kernel_info.status = KernelStatus.HEALTHY
                return {
                    "kernel_info": kernel_info.to_dict(),
                    "live_status": status_result
                }
        except Exception as e:
            kernel_info.record_error(f"Status check failed: {e}")
            kernel_info.status = KernelStatus.UNHEALTHY
        
        return {"kernel_info": kernel_info.to_dict()}
    
    async def health_check_kernel(self, kernel_name: str) -> bool:
        """Perform health check on a specific kernel."""
        if kernel_name not in self.kernels:
            return False
        
        kernel_info = self.kernels[kernel_name]
        
        try:
            # Try health check method
            health_result = await self.send_message(kernel_name, "health_check")
            
            if health_result and health_result.get("healthy", False):
                kernel_info.status = KernelStatus.HEALTHY
                return True
            else:
                kernel_info.status = KernelStatus.UNHEALTHY
                return False
                
        except Exception as e:
            kernel_info.record_error(f"Health check failed: {e}")
            kernel_info.status = KernelStatus.UNHEALTHY
            return False
    
    async def _auto_register_known_kernels(self):
        """Attempt to auto-register known kernels."""
        for kernel_name in self.known_kernels:
            # Try to import and register known kernels
            try:
                # This would be replaced with actual kernel discovery/import logic
                logger.debug(f"Would attempt auto-registration of {kernel_name}")
            except Exception as e:
                logger.debug(f"Auto-registration failed for {kernel_name}: {e}")
    
    async def _heartbeat_monitor_loop(self):
        """Monitor kernel heartbeats."""
        try:
            while self.running:
                await self._check_kernel_heartbeats()
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            logger.debug("Heartbeat monitor loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Heartbeat monitor error: {e}", exc_info=True)
    
    async def _check_kernel_heartbeats(self):
        """Check heartbeats of all registered kernels."""
        current_time = datetime.now()
        
        for kernel_name, kernel_info in self.kernels.items():
            heartbeat_age = (current_time - kernel_info.last_heartbeat).total_seconds()
            
            if heartbeat_age > self.heartbeat_timeout_seconds:
                if kernel_info.status != KernelStatus.OFFLINE:
                    logger.warning(f"Kernel {kernel_name} heartbeat timeout ({heartbeat_age:.1f}s)")
                    kernel_info.status = KernelStatus.OFFLINE
                    
                    # Publish offline event
                    if self.event_publisher:
                        await self.event_publisher("KERNEL_OFFLINE", {
                            "kernel_name": kernel_name,
                            "last_heartbeat": kernel_info.last_heartbeat.isoformat(),
                            "timestamp": current_time.isoformat()
                        })
    
    def get_registered_kernels(self, status_filter: KernelStatus = None,
                             capability_filter: str = None) -> List[Dict[str, Any]]:
        """Get list of registered kernels with optional filtering."""
        kernels = []
        
        for kernel_info in self.kernels.values():
            # Apply status filter
            if status_filter and kernel_info.status != status_filter:
                continue
            
            # Apply capability filter
            if capability_filter and capability_filter not in kernel_info.capabilities:
                continue
            
            kernels.append(kernel_info.to_dict())
        
        return sorted(kernels, key=lambda k: k["kernel_name"])
    
    def get_kernel_by_capability(self, capability: str) -> List[str]:
        """Get kernels that have a specific capability."""
        matching_kernels = []
        
        for kernel_name, kernel_info in self.kernels.items():
            if capability in kernel_info.capabilities and kernel_info.status == KernelStatus.HEALTHY:
                matching_kernels.append(kernel_name)
        
        return matching_kernels
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge registry status."""
        status_counts = {}
        for status in KernelStatus:
            status_counts[status.value] = sum(
                1 for k in self.kernels.values() if k.status == status
            )
        
        capability_counts = {}
        for kernel_info in self.kernels.values():
            for capability in kernel_info.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        return {
            "running": self.running,
            "registered_kernels": len(self.kernels),
            "known_kernels": len(self.known_kernels),
            "status_distribution": status_counts,
            "capability_distribution": capability_counts,
            "auto_registration_enabled": self.auto_registration_enabled,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "kernels": {
                name: info.to_dict() for name, info in self.kernels.items()
            }
        }