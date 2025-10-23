"""
Grace AI System - Constitutional AI with Multi-Agent Coordination
"""

__version__ = "1.0.0"
__author__ = "Grace AI Team"

# No eager imports - everything is lazy loaded
__all__ = ['__version__', '__author__']

# Lazy imports to avoid circular dependencies at module level
def get_settings():
    """Get settings (lazy import)"""
    from grace.config import get_settings as _get_settings
    return _get_settings()


# Core Grace components
from . import core
from . import governance
from . import memory
from . import mldl

__all__ = ["core", "governance", "memory", "mldl"]
