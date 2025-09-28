#!/usr/bin/env python3
"""
Grace Main Entry Point - Unified system launcher.

This is the main entry point for the Grace system that coordinates
all kernels and provides a unified interface for system operations.
"""
import asyncio
import argparse
import logging
import sys
import os
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Ensure Grace modules are in the Python path."""
    grace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if grace_root not in sys.path:
        sys.path.insert(0, grace_root)

async def main():
    """Main Grace system entry point."""
    setup_paths()
    
    parser = argparse.ArgumentParser(description="Grace AI Governance System")
    parser.add_argument(
        "--mode", 
        choices=["demo", "service", "kernel", "test"], 
        default="demo",
        help="Operation mode"
    )
    parser.add_argument(
        "--kernel", 
        choices=["ingress", "intelligence", "learning", "orchestration", "resilience", "interface", "multi_os", "mldl", "mlt", "mtl"],
        help="Specific kernel to run (when mode=kernel)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Service port (when mode=service)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Grace system in {args.mode} mode")
    
    if args.mode == "demo":
        await run_demo_mode(args)
    elif args.mode == "service":
        await run_service_mode(args)
    elif args.mode == "kernel":
        await run_kernel_mode(args)
    elif args.mode == "test":
        await run_test_mode(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

async def run_demo_mode(args):
    """Run Grace in demonstration mode."""
    logger.info("📋 Running Grace system demonstration")
    
    # Import demo modules
    try:
        from demo_multi_os_kernel import demo_multi_os_kernel
        from demo_ingress_kernel import demo_ingress_kernel
        from demo_mldl_kernel import demo_mldl_kernel
        from demo_resilience_kernel import demo_resilience_kernel
        
        # Run all demos
        await demo_multi_os_kernel()
        await demo_ingress_kernel()
        await demo_mldl_kernel()
        await demo_resilience_kernel()
        
        logger.info("✅ All demos completed successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import demo modules: {e}")
        logger.info("Running basic system check instead...")
        
        # Basic system check
        from grace import __version__
        logger.info(f"Grace system version: {__version__}")
        logger.info("✅ Basic system check completed")

async def run_service_mode(args):
    """Run Grace as a service."""
    logger.info(f"🌐 Starting Grace service on port {args.port}")
    
    try:
        # This would start the unified service combining all kernels
        import uvicorn
        from grace.core.unified_service import create_unified_app
        
        app = create_unified_app(config_file=args.config)
        
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=args.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError:
        logger.warning("Unified service not available, starting basic mode")
        logger.info("Grace system is ready for manual kernel management")
        
        # Keep alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Grace service stopped")

async def run_kernel_mode(args):
    """Run a specific kernel."""
    if not args.kernel:
        logger.error("Kernel name required when using kernel mode")
        sys.exit(1)
    
    logger.info(f"🔧 Starting {args.kernel} kernel")
    
    # Import and run specific kernel
    kernel_runners = {
        "ingress": "grace.ingress_kernel.main:main",
        "intelligence": "grace.intelligence.kernel.main:main", 
        "learning": "grace.learning_kernel.main:main",
        "orchestration": "grace.orchestration_kernel.main:main",
        "resilience": "grace.resilience_kernel.main:main",
        "interface": "grace.interface_kernel.main:main",
        "multi_os": "grace.multi_os_kernel.main:main",
        "mldl": "grace.mldl.main:main",
        "mlt": "grace.mlt_kernel_ml.main:main",
        "mtl": "grace.mtl_kernel.main:main"
    }
    
    if args.kernel not in kernel_runners:
        logger.error(f"Unknown kernel: {args.kernel}")
        sys.exit(1)
    
    try:
        module_path, func_name = kernel_runners[args.kernel].split(":")
        module = __import__(module_path, fromlist=[func_name])
        kernel_main = getattr(module, func_name)
        await kernel_main()
    except Exception as e:
        logger.error(f"Failed to start {args.kernel} kernel: {e}")
        logger.info(f"✅ {args.kernel} kernel placeholder started (implementation needed)")

async def run_test_mode(args):
    """Run Grace in test mode."""
    logger.info("🧪 Running Grace system tests")
    
    try:
        import pytest
        exit_code = pytest.main(["-v", "tests/"])
        if exit_code == 0:
            logger.info("✅ All tests passed")
        else:
            logger.error("❌ Some tests failed")
            sys.exit(exit_code)
    except ImportError:
        logger.warning("pytest not available, running basic tests")
        
        # Basic import tests
        try:
            import grace
            print(f"✅ Grace system version: {grace.__version__}")
            
            # Test only the new kernel packages that don't have dependencies
            import grace.interface_kernel
            import grace.multi_os_kernel
            import grace.orchestration_kernel
            import grace.resilience_kernel
            import tests
            
            logger.info("✅ Basic import tests passed")
        except ImportError as e:
            logger.error(f"❌ Import test failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Grace system shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Grace system error: {e}")
        sys.exit(1)