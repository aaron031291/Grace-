#!/usr/bin/env python3
"""
MLDL Kernel Main Entry Point.

Starts the ML/DL (Machine Learning/Deep Learning) service.
"""
import asyncio
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for MLDL kernel."""
    parser = argparse.ArgumentParser(description="MLDL Kernel Service")
    parser.add_argument("--port", type=int, default=8006, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")
    
    args = parser.parse_args()
    logger.info(f"🤖 Starting MLDL Kernel on port {args.port}")
    
    try:
        from grace.mldl import mldl_service
        
        if args.mode == "demo":
            logger.info("📋 MLDL Kernel demo mode")
            logger.info("✅ MLDL Kernel is ready")
            
            try:
                while True:
                    await asyncio.sleep(5)
                    logger.info("MLDL Kernel running...")
            except KeyboardInterrupt:
                logger.info("Demo stopped")
        else:
            logger.info("✅ MLDL Kernel service placeholder is running")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("MLDL Kernel stopped")
                
    except ImportError as e:
        logger.error(f"Failed to import MLDL components: {e}")
        logger.info("✅ MLDL Kernel placeholder is running")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("MLDL Kernel stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 MLDL Kernel shutdown")
        sys.exit(0)