"""
Global exception handling and system watchdog
"""

import sys
import logging
import traceback
import signal
import asyncio
from typing import Optional, Callable, Any
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class GraceWatchdog:
    """
    System watchdog for exception handling and crash recovery
    """
    
    def __init__(self, restart_on_failure: bool = False):
        self.restart_on_failure = restart_on_failure
        self.crash_count = 0
        self.last_crash: Optional[datetime] = None
        self.shutdown_handlers: list[Callable] = []
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def register_shutdown_handler(self, handler: Callable):
        """Register a shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    async def shutdown(self):
        """Execute graceful shutdown"""
        logger.info("Watchdog initiating graceful shutdown")
        
        for handler in self.shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")
        
        logger.info("Graceful shutdown complete")
        sys.exit(0)
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        self.crash_count += 1
        self.last_crash = datetime.utcnow()
        
        logger.critical(
            "Unhandled exception caught by watchdog",
            exc_info=(exc_type, exc_value, exc_traceback),
            extra={
                "crash_count": self.crash_count,
                "exc_type": exc_type.__name__,
                "exc_value": str(exc_value)
            }
        )
        
        # Print to stderr for visibility
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        
        if self.restart_on_failure:
            logger.warning("Attempting restart due to crash")
            # In production, this would trigger process restart via supervisor
        else:
            logger.critical("System halted due to unrecoverable error")
            sys.exit(1)
    
    def install(self):
        """Install global exception handler"""
        sys.excepthook = self.handle_exception
        logger.info("Watchdog exception handler installed")
    
    def wrap_async(self, coro):
        """Wrap async function with error handling"""
        @wraps(coro)
        async def wrapper(*args, **kwargs):
            try:
                return await coro(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Async task error in {coro.__name__}: {e}")
                raise
        return wrapper


# Global watchdog instance
_watchdog: Optional[GraceWatchdog] = None


def get_watchdog() -> GraceWatchdog:
    """Get global watchdog instance"""
    global _watchdog
    if _watchdog is None:
        from grace.config import get_config
        config = get_config()
        _watchdog = GraceWatchdog(restart_on_failure=config.watchdog_restart_on_failure)
    return _watchdog


def install_watchdog():
    """Install global watchdog"""
    watchdog = get_watchdog()
    watchdog.install()
    return watchdog
