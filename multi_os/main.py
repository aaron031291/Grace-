#!/usr/bin/env python3
"""
Multi-OS Kernel Main Entry Point.

Starts the Multi-OS service for cross-platform operations.
"""
import asyncio
import argparse
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for Multi-OS kernel."""
    parser = argparse.ArgumentParser(description="Multi-OS Kernel Service")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8001, 
        help="Service port"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--mode",
        choices=["service", "demo"],
        default="service",
        help="Run mode"
    )
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Multi-OS Kernel on port {args.port}")
    
    try:
        from multi_os_service import MultiOSService
        
        if args.mode == "demo":
            # Run demo mode
            service = MultiOSService()
            await service.start_background_services()
            
            logger.info("📋 Multi-OS Kernel demo mode")
            logger.info(f"Service stats: {service.get_service_stats()}")
            
            # Keep demo running
            try:
                while True:
                    await asyncio.sleep(10)
                    stats = service.get_service_stats()
                    logger.info(f"Service status: {len(stats['hosts'])} hosts")
            except KeyboardInterrupt:
                logger.info("Demo stopped")
        
        else:
            # Start as web service
            import uvicorn
            
            service = MultiOSService()
            app = service.get_app()
            
            # Start background services
            await service.start_background_services()
            
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=args.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
    except ImportError as e:
        logger.error(f"Failed to import Multi-OS service: {e}")
        logger.info("Multi-OS service not available, running placeholder")
        
        # Placeholder mode
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
    except Exception as e:
        logger.error(f"💥 Multi-OS Kernel error: {e}")
        sys.exit(1)