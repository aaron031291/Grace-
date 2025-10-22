"""
Main entry point for Grace
"""

import argparse
import asyncio
import sys
import logging

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_service():
    """
    Run Grace in service mode
    
    Fails fast if initialization fails
    """
    try:
        # Import here to catch import errors
        from grace.core.unified_service import create_unified_app
        from grace.config import get_config
        import uvicorn
        
        logger.info("Starting Grace Unified Service...")
        
        # Create app (will fail fast if initialization fails)
        app = create_unified_app()
        
        # Get configuration
        config = get_config()
        
        # Run server
        uvicorn.run(
            app,
            host=config.service_host,
            port=config.service_port,
            log_level=config.log_level.lower()
        )
    
    except ImportError as e:
        logger.critical(f"Failed to import required modules: {e}")
        logger.critical("Ensure all dependencies are installed: pip install -e .")
        sys.exit(1)
    
    except RuntimeError as e:
        logger.critical(f"Service initialization failed: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.critical(f"Unexpected error during service startup: {e}", exc_info=True)
        sys.exit(1)


async def run_demo(demo_name: str):
    """Run a specific demo (coroutine)"""
    try:
        # Map demo names to kernel start functions
        demos = {
            "multi_os": "grace.kernels.multi_os:start",
            "mldl": "grace.kernels.mldl:start",
            "resilience": "grace.kernels.resilience:start",
        }
        
        if demo_name not in demos:
            logger.error(f"Unknown demo: {demo_name}")
            logger.info(f"Available demos: {', '.join(demos.keys())}")
            sys.exit(1)
        
        module_path, func = demos[demo_name].rsplit(":", 1)
        module = __import__(module_path, fromlist=[func])
        start_coro = getattr(module, func)
        
        logger.info(f"Starting demo: {demo_name}")
        await start_coro()
        
        # Keep running
        logger.info("Demo running. Press Ctrl+C to stop.")
        await asyncio.Event().wait()
    
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    
    except Exception as e:
        logger.critical(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


def main(argv=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="grace",
        description="Grace AI System - Constitutional AI with Multi-Agent Coordination"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="service",
        choices=["service", "demo"],
        help="Run mode: service (default) or demo"
    )
    parser.add_argument(
        "--demo",
        default="multi_os",
        choices=["multi_os", "mldl", "resilience"],
        help="Demo to run when mode=demo"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args(argv)
    
    # Set debug if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run selected mode
    if args.mode == "service":
        run_service()
    elif args.mode == "demo":
        asyncio.run(run_demo(args.demo))


if __name__ == "__main__":
    main()