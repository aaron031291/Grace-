"""
Configuration module for Grace
"""

from .settings import (
    Settings,
    get_settings,
    DatabaseSettings,
    AuthSettings,
    EmbeddingSettings,
    VectorStoreSettings,
)

import os
from typing import Dict, Any

# A simple default configuration.
# In a real application, this would be loaded from a file (e.g., YAML, .env)
# or environment variables.
_CONFIG: Dict[str, Any] = {
    "APP_NAME": "Grace",
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    "API_HOST": "0.0.0.0",
    "API_PORT": 8000,
    "DATABASE_URL": os.environ.get("DATABASE_URL", "sqlite:///grace.db"),
}

def get_config() -> Dict[str, Any]:
    """
    Returns the application configuration dictionary.
    This is a simple placeholder. A real implementation would load from
    a file or environment variables and perform validation.
    """
    return _CONFIG.copy()

__all__ = [
    'Settings',
    'get_settings',
    'DatabaseSettings',
    'AuthSettings',
    'EmbeddingSettings',
    'VectorStoreSettings',
    'get_config',
]
