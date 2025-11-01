"""
Grace Integration Layer - Wire Autonomous System to Backend API

This is THE CRITICAL CONNECTOR that makes Grace's intelligence
accessible through the API.

Integrates:
- grace_autonomous.py → Backend startup
- All Grace systems → API endpoints
- Intelligence → Real-time responses
- Memory → Persistent across sessions
- All kernels → Health monitoring

After this integration, Grace's full cognition flows through the API!
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GraceIntegration:
    """
    Manages Grace autonomous system integration with backend.
    
    Singleton that initializes once and provides access to all
    Grace intelligence systems.
    """
    
    _instance: Optional['GraceIntegration'] = None
    
    def __init__(self):
        self.grace = None
        self.initialized = False
        self.initialization_error = None
        
    @classmethod
    def get_instance(cls) -> 'GraceIntegration':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = GraceIntegration()
        return cls._instance
    
    async def initialize(self):
        """Initialize Grace autonomous system"""
        if self.initialized:
            logger.info("Grace already initialized")
            return self.grace
        
        logger.info("="*70)
        logger.info("INITIALIZING GRACE AUTONOMOUS SYSTEM")
        logger.info("="*70)
        
        try:
            # Try to import and initialize Grace
            import sys
            from pathlib import Path
            
            # Add parent directory to path to find grace_autonomous
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            try:
                from grace_autonomous import GraceAutonomous
                
                logger.info("✅ Grace autonomous module found")
                logger.info("   Initializing Grace intelligence systems...")
                
                self.grace = GraceAutonomous()
                await self.grace.initialize()
                
                self.initialized = True
                
                logger.info("="*70)
                logger.info("✅ GRACE AUTONOMOUS SYSTEM INITIALIZED")
                logger.info("="*70)
                logger.info("   All intelligence systems operational")
                logger.info("   Grace is ready to assist!")
                logger.info("="*70)
                
                return self.grace
                
            except ImportError as e:
                logger.warning(f"⚠️  Grace autonomous module not available: {e}")
                logger.warning("   Backend will work with limited intelligence")
                logger.warning("   To enable full Grace: ensure grace_autonomous.py is accessible")
                
                self.initialization_error = f"Import error: {e}"
                self.initialized = False
                return None
                
        except Exception as e:
            logger.error(f"❌ Grace initialization failed: {e}")
            logger.error("   Backend will work but without Grace autonomous features")
            
            self.initialization_error = str(e)
            self.initialized = False
            return None
    
    def get_grace(self) -> Optional[object]:
        """Get Grace instance"""
        return self.grace
    
    def is_operational(self) -> bool:
        """Check if Grace is operational"""
        return self.initialized and self.grace is not None
    
    async def process_chat_message(
        self,
        message: str,
        session_id: str,
        context: Optional[dict] = None
    ) -> dict:
        """
        Process chat message through Grace.
        
        If Grace is operational, uses full intelligence.
        If not, returns graceful fallback.
        """
        if self.is_operational():
            try:
                # Use Grace's full intelligence
                if not self.grace.session_id or self.grace.session_id != session_id:
                    await self.grace.start_session(session_id)
                
                response = await self.grace.process_request(message, context)
                
                return {
                    "response": response.get("result", ""),
                    "source": response.get("source", "grace_autonomous"),
                    "autonomous": response.get("autonomous", False),
                    "llm_used": response.get("llm_used", False),
                    "grace_operational": True
                }
                
            except Exception as e:
                logger.error(f"Grace processing error: {e}")
                return self._fallback_response(message)
        else:
            return self._fallback_response(message)
    
    def _fallback_response(self, message: str) -> dict:
        """Fallback response when Grace not operational"""
        return {
            "response": f"I received your message: '{message}'. Grace autonomous system is not fully initialized. Basic API functionality is available. To enable full Grace intelligence, ensure all dependencies are installed and grace_autonomous.py is accessible.",
            "source": "fallback",
            "autonomous": False,
            "llm_used": False,
            "grace_operational": False,
            "initialization_error": self.initialization_error
        }
    
    async def get_all_systems_status(self) -> dict:
        """Get status of all Grace systems"""
        if self.is_operational():
            try:
                status = self.grace.get_status()
                return {
                    "grace_operational": True,
                    "systems": status,
                    "initialized": True
                }
            except Exception as e:
                logger.error(f"Status retrieval error: {e}")
                return self._fallback_status()
        else:
            return self._fallback_status()
    
    def _fallback_status(self) -> dict:
        """Fallback status when Grace not operational"""
        return {
            "grace_operational": False,
            "initialized": False,
            "error": self.initialization_error,
            "message": "Grace autonomous system not initialized. Backend API is functional with basic features.",
            "available_features": [
                "Basic API endpoints",
                "Authentication",
                "Health checks",
                "Orb interface"
            ],
            "unavailable_features": [
                "Full Grace intelligence",
                "Knowledge verification",
                "Multi-tasking",
                "Autonomous operation",
                "Voice interface (needs dependencies)"
            ]
        }


# Global instance
_grace_integration: Optional[GraceIntegration] = None


def get_grace_integration() -> GraceIntegration:
    """Get global Grace integration instance"""
    global _grace_integration
    if _grace_integration is None:
        _grace_integration = GraceIntegration.get_instance()
    return _grace_integration


# Helper function for easy access
async def initialize_grace_for_backend(app=None):
    """
    Initialize Grace and attach to FastAPI app.
    
    Usage in backend/main.py:
    
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            grace = await initialize_grace_for_backend(app)
            yield
    """
    integration = get_grace_integration()
    grace = await integration.initialize()
    
    if app and grace:
        app.state.grace_integration = integration
        app.state.grace = grace
        logger.info("✅ Grace attached to FastAPI app.state")
    
    return grace


if __name__ == "__main__":
    # Test integration
    async def test():
        print("🔌 Testing Grace Integration\n")
        
        integration = GraceIntegration()
        grace = await integration.initialize()
        
        if grace:
            print("✅ Grace initialized successfully!")
            print(f"   Operational: {integration.is_operational()}")
            
            # Test chat
            response = await integration.process_chat_message(
                "Hello Grace!",
                "test_session"
            )
            
            print(f"\n💬 Test Message Response:")
            print(f"   Source: {response['source']}")
            print(f"   Operational: {response['grace_operational']}")
            
        else:
            print("⚠️  Grace not initialized (this is OK - backend will still work)")
            print(f"   Error: {integration.initialization_error}")
    
    asyncio.run(test())
