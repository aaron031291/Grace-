#!/usr/bin/env python3
"""
Interface Kernel Main Entry Point.

Starts the Interface Kernel service for user interactions.
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
    """Main entry point for Interface kernel."""
    parser = argparse.ArgumentParser(description="Interface Kernel Service")
    parser.add_argument("--port", type=int, default=8004, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")

    args = parser.parse_args()
    logger.info(f"🖥️  Starting Interface Kernel on port {args.port}")

    logger.info("✅ Interface Kernel placeholder is running")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interface Kernel stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Interface Kernel shutdown")
        sys.exit(0)
