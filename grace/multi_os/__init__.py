"""
Multi-OS Kernel - Unified execution layer across Linux/Windows/macOS.
"""
from .multi_os_service import MultiOSService
from .orchestrator.scheduler import Scheduler
from .inventory.registry import Registry
from .telemetry.collector import TelemetryCollector
from .snapshots.manager import SnapshotManager

__version__ = "1.0.0"
__all__ = [
    "MultiOSService",
    "Scheduler", 
    "Registry",
    "TelemetryCollector",
    "SnapshotManager"
]