#!/usr/bin/env python3
"""
Grace AI System - Main application entry point
"""

import logging
import uvicorn
from grace.api import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = create_app()


def main():
    """
    Main entry point for running the application
    """
    logger.info("Starting Grace AI System")
    
    uvicorn.run(
        "grace.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
