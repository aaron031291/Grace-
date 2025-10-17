"""
Grace Memory System - Enhanced memory with production DB and health monitoring
"""

from .enhanced_memory_core import EnhancedMemoryCore, MemoryHealth, MemoryMetrics

__all__ = [
    'EnhancedMemoryCore',
    'MemoryHealth',
    'MemoryMetrics'
]

__version__ = '1.0.0'

# Enhanced memory components
from . import vector_db
from . import quantum_safe_storage
from . import enhanced_memory_bridge

# Legacy components
try:
    from . import api
    from . import fusion
    from . import lightning
    from . import librarian
except ImportError:
    # Gracefully handle missing legacy components
    pass
