#!/usr/bin/env python3
"""
Resilience Kernel Main Entry Point.

Starts the Resilience Kernel service.
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
    """Main entry point for Resilience kernel."""
    parser = argparse.ArgumentParser(description="Resilience Kernel Service")
    parser.add_argument("--port", type=int, default=8011, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")

    args = parser.parse_args()
    logger.info(f"🛡️  Starting Resilience Kernel on port {args.port}")

    logger.info("✅ Resilience Kernel placeholder is running")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Resilience Kernel stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Resilience Kernel shutdown")
        sys.exit(0)
