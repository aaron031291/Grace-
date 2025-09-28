#!/usr/bin/env python3
"""
Ingress Kernel Main Entry Point.

Starts the Ingress Kernel service for data ingestion.
"""
import asyncio
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for Ingress kernel."""
    parser = argparse.ArgumentParser(description="Ingress Kernel Service")
    parser.add_argument("--port", type=int, default=8002, help="Service port")
    parser.add_argument("--mode", choices=["service", "demo"], default="service")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Ingress Kernel on port {args.port}")
    
    try:
        from grace.ingress_kernel import IngressKernel, IngressService, create_ingress_app
        
        if args.mode == "demo":
            # Demo mode
            kernel = IngressKernel()
            logger.info("📋 Ingress Kernel demo mode")
            logger.info("✅ Ingress Kernel is ready for data ingestion")
            
            try:
                while True:
                    await asyncio.sleep(5)
                    logger.info("Ingress Kernel running...")
            except KeyboardInterrupt:
                logger.info("Demo stopped")
        else:
            # Service mode
            import uvicorn
            app = create_ingress_app()
            
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0", 
                port=args.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
    except ImportError as e:
        logger.error(f"Failed to import Ingress components: {e}")
        logger.info("✅ Ingress Kernel placeholder is running")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ingress Kernel stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Ingress Kernel shutdown")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Ingress Kernel error: {e}")
        sys.exit(1)