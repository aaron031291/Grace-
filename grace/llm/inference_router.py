"""
Inference Router - Route requests to appropriate models
"""

from typing import Dict, Any, List, Optional
import logging

from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class InferenceRouter:
    """
    Routes inference requests to appropriate models
    
    Features:
    - Task-based routing
    - Load balancing
    - Fallback chains
    - Performance tracking
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.routing_rules = self._default_routing_rules()
        
        logger.info("InferenceRouter initialized")
    
    def _default_routing_rules(self) -> Dict[str, str]:
        """Default routing rules"""
        return {
            "general": "tiny",       # Fast for general queries
            "code": "code",          # Code-specialized
            "complex": "medium",     # Complex reasoning
            "chat": "tiny",          # Chat interface
            "embedding": "tiny"      # Embedding generation
        }
    
    def route(
        self,
        prompt: str,
        task_type: str = "general",
        preferred_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route request to appropriate model
        
        Args:
            prompt: Input text
            task_type: Type of task (general, code, complex, chat)
            preferred_model: Override routing with specific model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Determine which model to use
        if preferred_model:
            model_name = preferred_model
        else:
            model_name = self.routing_rules.get(task_type, "tiny")
        
        try:
            # Get model
            model = self.model_manager.get_model(model_name)
            
            # Generate
            result = model.generate(prompt, **kwargs)
            
            # Add routing metadata
            result["routed_to"] = model_name
            result["task_type"] = task_type
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            
            # Fallback to default model
            if model_name != self.model_manager.default_model:
                logger.info(f"Falling back to default model")
                return self.route(prompt, task_type="general", preferred_model=None, **kwargs)
            
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat completion with routing"""
        model_name = model or self.routing_rules.get("chat", "tiny")
        
        try:
            llm = self.model_manager.get_model(model_name)
            result = llm.chat(messages, **kwargs)
            result["routed_to"] = model_name
            return result
            
        except Exception as e:
            logger.error(f"Chat failed for model {model_name}: {e}")
            raise
