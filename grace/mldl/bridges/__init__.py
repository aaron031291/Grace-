"""
MLDL Bridges - Connect to existing Grace kernels (Mesh, Gov, MLT, Intel, Memory, Ingress).
"""

# Import individual bridge classes
from .gov_bridge import MLDLGovernanceBridge  
from .mlt_bridge import MLDLMLTBridge
from .intel_bridge import MLDLIntelligenceBridge
from .memory_bridge import MLDLMemoryBridge
from .ingress_bridge import MLDLIngressBridge

# Create a placeholder for mesh bridge since we couldn't create it
class MLDLMeshBridge:
    """Placeholder MLDL Mesh Bridge."""
    def __init__(self, event_bus=None, trigger_mesh=None):
        self.event_bus = event_bus
        self.trigger_mesh = trigger_mesh
        
    async def start(self):
        pass
        
    async def stop(self):
        pass

__all__ = [
    'MLDLMeshBridge',
    'MLDLGovernanceBridge', 
    'MLDLMLTBridge',
    'MLDLIntelligenceBridge',
    'MLDLMemoryBridge',
    'MLDLIngressBridge'
]