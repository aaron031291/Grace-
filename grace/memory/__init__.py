"""Grace Memory subsystem - Enhanced memory infrastructure with vector databases and quantum-safe storage."""

# Enhanced memory components
try:
    from . import vector_db
except ImportError:
    # Gracefully handle missing numpy/vector dependencies
    vector_db = None

try:
    from . import quantum_safe_storage
except ImportError:
    quantum_safe_storage = None

try:
    from . import enhanced_memory_bridge
except ImportError:
    enhanced_memory_bridge = None

# Legacy components
try:
    from . import api
    from . import fusion
    from . import lightning
    from . import librarian
except ImportError:
    # Gracefully handle missing legacy components
    pass

__all__ = ["vector_db", "quantum_safe_storage", "enhanced_memory_bridge"]