#!/usr/bin/env python3
"""
Intelligence Kernel Main Entry Point.

Starts the Intelligence Kernel service.
"""
import asyncio
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for Intelligence kernel."""
    parser = argparse.ArgumentParser(description="Intelligence Kernel Service")
    parser.add_argument("--port", type=int, default=8003, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")
    
    args = parser.parse_args()
    logger.info(f"🧠 Starting Intelligence Kernel on port {args.port}")
    
    try:
        from grace.intelligence import IntelligenceKernel
        
        if args.mode == "demo":
            kernel = IntelligenceKernel()
            logger.info("📋 Intelligence Kernel demo mode")
            logger.info("✅ Intelligence Kernel is ready")
            
            try:
                while True:
                    await asyncio.sleep(5)
                    logger.info("Intelligence Kernel running...")
            except KeyboardInterrupt:
                logger.info("Demo stopped")
        else:
            logger.info("✅ Intelligence Kernel service placeholder is running")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Intelligence Kernel stopped")
                
    except ImportError as e:
        logger.error(f"Failed to import Intelligence components: {e}")
        logger.info("✅ Intelligence Kernel placeholder is running")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Intelligence Kernel stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Intelligence Kernel shutdown")
        sys.exit(0)