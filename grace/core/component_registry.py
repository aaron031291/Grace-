"""
Grace AI Component Registry - Dynamic component management
"""
import logging
import importlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for managing and reloading Grace's core components."""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.module_paths: Dict[str, str] = {}
    
    def register(self, name: str, component: Any, module_path: str = None):
        """Register a component."""
        self.components[name] = component
        if module_path:
            self.module_paths[name] = module_path
        logger.info(f"Registered component: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered component."""
        return self.components.get(name)
    
    def reload(self, name: str) -> bool:
        """Dynamically reload a component from disk."""
        if name not in self.module_paths:
            logger.warning(f"Cannot reload {name}: no module path registered")
            return False
        
        try:
            module_path = self.module_paths[name]
            module = importlib.import_module(module_path)
            importlib.reload(module)
            logger.info(f"Reloaded component: {name}")
            return True
        except Exception as e:
            logger.error(f"Error reloading {name}: {str(e)}")
            return False
    
    def list_components(self) -> Dict[str, str]:
        """List all registered components."""
        return {name: str(type(comp)) for name, comp in self.components.items()}
