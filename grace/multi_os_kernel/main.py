#!/usr/bin/env python3
"""
Multi-OS Kernel Main Entry Point.

Starts the Multi-OS Kernel service (grace version).
"""
import asyncio
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for Multi-OS kernel."""
    parser = argparse.ArgumentParser(description="Multi-OS Kernel Service")
    parser.add_argument("--port", type=int, default=8009, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")
    
    args = parser.parse_args()
    logger.info(f"🖥️  Starting Multi-OS Kernel on port {args.port}")
    
    try:
        # Add path for Grace imports if running from kernel directory
        import os
        import sys
        grace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if grace_root not in sys.path:
            sys.path.insert(0, grace_root)
            
        from grace.multi_os_kernel import MultiOSKernel
        
        if args.mode == "demo":
            kernel = MultiOSKernel()
            logger.info("📋 Multi-OS Kernel demo mode")
            logger.info("✅ Multi-OS Kernel is ready")
            
            try:
                while True:
                    await asyncio.sleep(5)
                    logger.info("Multi-OS Kernel running...")
            except KeyboardInterrupt:
                logger.info("Demo stopped")
        else:
            logger.info("✅ Multi-OS Kernel service placeholder is running")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Multi-OS Kernel stopped")
                
    except ImportError as e:
        logger.error(f"Failed to import Multi-OS components: {e}")
        logger.info("✅ Multi-OS Kernel placeholder is running")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Multi-OS Kernel stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Multi-OS Kernel shutdown")
        sys.exit(0)