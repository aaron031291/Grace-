"""
Model Manager - Handle multiple LLM models and routing
"""

from typing import Dict, List, Optional
from pathlib import Path
import logging

from .private_llm import PrivateLLM, LLMProvider, ModelConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple LLM models
    
    Features:
    - Load/unload models dynamically
    - Route requests to appropriate model
    - Model fallback chains
    - Resource management
    """
    
    def __init__(self):
        self.models: Dict[str, PrivateLLM] = {}
        self.default_model: Optional[str] = None
        
        logger.info("ModelManager initialized")
    
    def register_model(
        self,
        name: str,
        config: ModelConfig,
        set_default: bool = False
    ) -> None:
        """Register a new model"""
        try:
            model = PrivateLLM(config)
            self.models[name] = model
            
            if set_default or self.default_model is None:
                self.default_model = name
            
            logger.info(f"Registered model: {name}")
            
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")
            raise
    
    def load_default_models(self):
        """Load default open-source models"""
        
        # Model 1: Small fast model (CPU-friendly)
        self.register_model(
            name="tiny",
            config=ModelConfig(
                name="TinyLlama-1.1B",
                provider=LLMProvider.TRANSFORMERS,
                model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                context_length=2048,
                max_tokens=256,
                gpu_layers=0  # CPU only
            ),
            set_default=True
        )
        
        # Model 2: Medium model (balanced)
        try:
            self.register_model(
                name="medium",
                config=ModelConfig(
                    name="Mistral-7B",
                    provider=LLMProvider.LLAMA_CPP,
                    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    context_length=4096,
                    max_tokens=512,
                    gpu_layers=32  # GPU if available
                )
            )
        except:
            logger.warning("Medium model not available")
        
        # Model 3: Code-specialized model
        try:
            self.register_model(
                name="code",
                config=ModelConfig(
                    name="CodeLlama-7B",
                    provider=LLMProvider.TRANSFORMERS,
                    model_path="codellama/CodeLlama-7b-Instruct-hf",
                    context_length=4096,
                    max_tokens=1024
                )
            )
        except:
            logger.warning("Code model not available")
    
    def get_model(self, name: Optional[str] = None) -> PrivateLLM:
        """Get model by name or default"""
        model_name = name or self.default_model
        
        if model_name is None:
            raise RuntimeError("No default model set")
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        return self.models[model_name]
    
    def list_models(self) -> List[Dict[str, str]]:
        """List all registered models"""
        return [
            {
                "name": name,
                "provider": model.config.provider.value,
                "model_path": model.config.model_path,
                "is_default": name == self.default_model
            }
            for name, model in self.models.items()
        ]
    
    def unload_model(self, name: str) -> None:
        """Unload a model from memory"""
        if name in self.models:
            del self.models[name]
            logger.info(f"Unloaded model: {name}")
            
            if self.default_model == name:
                self.default_model = next(iter(self.models), None)
