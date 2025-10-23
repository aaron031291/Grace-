"""
Fix all __init__.py files to prevent circular imports
"""

from pathlib import Path

INIT_FILES = {
    'grace/__init__.py': '''"""Grace AI System"""
__version__ = "1.0.0"
''',
    
    'grace/config/__init__.py': '''"""Configuration"""
from .settings import get_settings, Settings
__all__ = ['get_settings', 'Settings']
''',
    
    'grace/events/__init__.py': '''"""Events"""
from .schema import GraceEvent
from .factory import GraceEventFactory
__all__ = ['GraceEvent', 'GraceEventFactory']
''',
    
    'grace/governance/__init__.py': '''"""Governance"""
from .engine import GovernanceEngine, ValidationResult, EscalationResult
__all__ = ['GovernanceEngine', 'ValidationResult', 'EscalationResult']
''',
    
    'grace/trust/__init__.py': '''"""Trust"""
from .core import TrustCoreKernel, TrustScore
__all__ = ['TrustCoreKernel', 'TrustScore']
''',
    
    'grace/memory/__init__.py': '''"""Memory"""
from .async_lightning import AsyncLightningMemory
from .async_fusion import AsyncFusionMemory
from .immutable_logs_async import AsyncImmutableLogs
__all__ = ['AsyncLightningMemory', 'AsyncFusionMemory', 'AsyncImmutableLogs']
''',
    
    'grace/integration/__init__.py': '''"""Integration"""
from .event_bus import EventBus
__all__ = ['EventBus']
''',
    
    'grace/llm/__init__.py': '''"""LLM"""
from .model_manager import ModelManager, ModelConfig
from .inference_router import InferenceRouter
from .private_llm import LLMProvider
__all__ = ['ModelManager', 'ModelConfig', 'InferenceRouter', 'LLMProvider']
''',
    
    'grace/core/__init__.py': '''"""Core"""
# Lazy import to avoid circular dependencies
__all__ = []
''',
    
    'grace/demo/__init__.py': '''"""Demos"""
__all__ = []
''',
    
    'grace/api/__init__.py': '''"""API"""
__all__ = []
''',
    
    'grace/middleware/__init__.py': '''"""Middleware"""
__all__ = []
''',
}

def main():
    """Fix all __init__.py files"""
    for filepath, content in INIT_FILES.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"✅ {filepath}")
    
    print(f"\n✅ Fixed {len(INIT_FILES)} __init__.py files")

if __name__ == "__main__":
    main()
