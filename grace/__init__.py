"""
Grace AI System - Constitutional AI with Multi-Agent Coordination

A production-ready AI system with:
- Constitutional governance framework
- Multi-agent coordination
- Vector-based semantic search
- Real-time communication
- Self-healing capabilities
- Advanced reasoning (quantum-inspired, scientific discovery)
"""

__version__ = "1.0.0"
__author__ = "Grace AI Team"

# Core imports for convenience
from grace.config import get_settings

__all__ = [
    '__version__',
    'get_settings',
]

# Core Grace components
from . import core
from . import governance
from . import memory
from . import mldl

__all__ = ["core", "governance", "memory", "mldl"]
