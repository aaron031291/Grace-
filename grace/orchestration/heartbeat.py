"""
Heartbeat monitoring and fault tolerance
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatRecord:
    """Record of a heartbeat"""
    kernel_id: str
    timestamp: datetime
    status: str
    metadata: Dict[str, Any]


class HeartbeatMonitor:
    """
    Monitor kernel heartbeats and trigger recovery on failures
    """
    
    def __init__(
        self,
        ttl_seconds: int = 30,
        check_interval: int = 10,
        avn_core=None
    ):
        """
        Initialize heartbeat monitor
        
        Args:
            ttl_seconds: Time-to-live for heartbeats
            check_interval: How often to check for stale heartbeats
            avn_core: AVN core for triggering recovery
        """
        self.ttl_seconds = ttl_seconds
        self.check_interval = check_interval
        self.avn_core = avn_core
        
        self.heartbeats: Dict[str, HeartbeatRecord] = {}
        self.degraded_kernels: Dict[str, datetime] = {}
        self.recovery_callbacks: Dict[str, Callable] = {}
        
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(f"Heartbeat monitor initialized (TTL={ttl_seconds}s)")
    
    async def start(self):
        """Start heartbeat monitoring"""
        if self.running:
            return
        
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Heartbeat monitoring started")
    
    async def stop(self):
        """Stop heartbeat monitoring"""
        self.running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        
        logger.info("Heartbeat monitoring stopped")
    
    def register_kernel(
        self,
        kernel_id: str,
        recovery_callback: Optional[Callable] = None
    ):
        """Register a kernel for heartbeat monitoring"""
        if recovery_callback:
            self.recovery_callbacks[kernel_id] = recovery_callback
        
        logger.info(f"Registered kernel for heartbeat: {kernel_id}")
    
    def report_heartbeat(
        self,
        kernel_id: str,
        status: str = "healthy",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Report heartbeat from a kernel"""
        record = HeartbeatRecord(
            kernel_id=kernel_id,
            timestamp=datetime.now(timezone.utc),
            status=status,
            metadata=metadata or {}
        )
        
        self.heartbeats[kernel_id] = record
        
        # Remove from degraded if it was there
        if kernel_id in self.degraded_kernels:
            logger.info(f"Kernel recovered: {kernel_id}")
            del self.degraded_kernels[kernel_id]
        
        logger.debug(f"Heartbeat received: {kernel_id} ({status})")
    
    async def _monitor_loop(self):
        """Monitor heartbeats and detect failures"""
        while self.running:
            try:
                await self._check_heartbeats()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_heartbeats(self):
        """Check all heartbeats for staleness"""
        now = datetime.now(timezone.utc)
        ttl = timedelta(seconds=self.ttl_seconds)
        
        for kernel_id, record in list(self.heartbeats.items()):
            age = now - record.timestamp
            
            if age > ttl:
                logger.warning(
                    f"Kernel heartbeat timeout: {kernel_id} "
                    f"(last seen {age.total_seconds():.0f}s ago)"
                )
                
                # Mark as degraded
                if kernel_id not in self.degraded_kernels:
                    self.degraded_kernels[kernel_id] = now
                    await self._handle_kernel_failure(kernel_id, record)
    
    async def _handle_kernel_failure(
        self,
        kernel_id: str,
        last_record: HeartbeatRecord
    ):
        """Handle kernel failure"""
        logger.error(f"Handling kernel failure: {kernel_id}")
        
        # Trigger AVN healing if available
        if self.avn_core:
            try:
                self.avn_core.report_metrics(
                    kernel_id,
                    {
                        "latency": 10000,  # Very high to trigger healing
                        "error_rate": 1.0,
                        "source": "heartbeat_timeout"
                    }
                )
                logger.info(f"Triggered AVN healing for {kernel_id}")
            except Exception as e:
                logger.error(f"Error triggering AVN healing: {e}")
        
        # Call recovery callback if registered
        if kernel_id in self.recovery_callbacks:
            try:
                await self.recovery_callbacks[kernel_id](kernel_id, last_record)
                logger.info(f"Recovery callback executed for {kernel_id}")
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get heartbeat monitoring status"""
        now = datetime.now(timezone.utc)
        
        healthy_kernels = []
        degraded_kernels_list = []
        
        for kernel_id, record in self.heartbeats.items():
            age = (now - record.timestamp).total_seconds()
            
            if kernel_id in self.degraded_kernels:
                degraded_kernels_list.append({
                    "kernel_id": kernel_id,
                    "last_heartbeat": record.timestamp.isoformat(),
                    "age_seconds": age,
                    "degraded_since": self.degraded_kernels[kernel_id].isoformat()
                })
            elif age < self.ttl_seconds:
                healthy_kernels.append({
                    "kernel_id": kernel_id,
                    "last_heartbeat": record.timestamp.isoformat(),
                    "age_seconds": age,
                    "status": record.status
                })
        
        return {
            "total_kernels": len(self.heartbeats),
            "healthy_kernels": len(healthy_kernels),
            "degraded_kernels": len(degraded_kernels_list),
            "healthy": healthy_kernels,
            "degraded": degraded_kernels_list
        }
