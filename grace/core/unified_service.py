"""
Unified Service - Main Grace service orchestrator
"""

from typing import Optional
from fastapi import FastAPI
import logging

from grace.api import create_app
from grace.config import get_settings

logger = logging.getLogger(__name__)


def create_unified_app(config: Optional[dict] = None) -> FastAPI:
    """
    Create unified Grace application
    
    This is the main entry point for service mode
    """
    logger.info("Creating unified Grace application")
    
    # Load settings
    settings = get_settings()
    
    # Create FastAPI app
    app = create_app()
    
    # Add any additional unified service configuration
    if config:
        app.state.custom_config = config
    
    logger.info(f"Unified app created: {settings.api_title} v{settings.api_version}")
    
    return app


# Backwards compatibility
UnifiedService = create_unified_app
