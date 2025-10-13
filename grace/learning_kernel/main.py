#!/usr/bin/env python3
"""
Learning Kernel Main Entry Point.

Starts the Learning Kernel service.
"""

import asyncio
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for Learning kernel."""
    parser = argparse.ArgumentParser(description="Learning Kernel Service")
    parser.add_argument("--port", type=int, default=8005, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")

    args = parser.parse_args()
    logger.info(f"📚 Starting Learning Kernel on port {args.port}")

    logger.info("✅ Learning Kernel placeholder is running")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Learning Kernel stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Learning Kernel shutdown")
        sys.exit(0)
