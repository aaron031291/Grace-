import asyncio
import logging
from datetime import datetime
from typing import Optional, Any
import time

logger = logging.getLogger(__name__)


class MLDLKernel:
    """
    MLDL Kernel - Machine Learning / Deep Learning inference
    
    Dependencies injected by Unified Service
    """
    
    def __init__(self, event_bus, event_factory, model_manager=None, inference_router=None):
        self.event_bus = event_bus
        self.event_factory = event_factory
        self.model_manager = model_manager
        self.inference_router = inference_router
        
        # State
        self._running = False
        self._start_time: Optional[datetime] = None
        self._events_processed = 0
        self._error_count = 0
        self._inference_count = 0
        self._total_inference_time = 0.0
    
    async def start(self):
        """Start MLDL kernel"""
        if self._running:
            logger.warning("MLDL kernel already running")
            return
        
        self._running = True
        self._start_time = datetime.utcnow()
        
        logger.info("MLDL kernel starting", extra={"start_time": self._start_time.isoformat()})
        
        # Try to load LLM components if not provided
        if not self.model_manager or not self.inference_router:
            try:
                from grace.llm import ModelManager, InferenceRouter
                
                if not self.model_manager:
                    self.model_manager = ModelManager()
                    try:
                        self.model_manager.load_default_models()
                        logger.info("LLM models loaded")
                    except Exception as e:
                        logger.warning(f"LLM models not available: {e}")
                
                if not self.inference_router:
                    self.inference_router = InferenceRouter(self.model_manager)
            
            except ImportError as e:
                logger.info(f"LLM components not available (using fallback): {e}")
        
        # Subscribe to inference requests
        self.event_bus.subscribe("mldl.infer", self._handle_infer)
        
        logger.info("MLDL kernel started", extra={
            "llm_available": self.model_manager is not None
        })
    
    async def _handle_infer(self, event):
        """Handle inference request"""
        start = time.time()
        
        try:
            payload = event.payload or {}
            result = None
            
            # Try real LLM inference
            if self.inference_router and self.model_manager:
                try:
                    prompt = payload.get("input", "")
                    if not prompt:
                        raise ValueError("Missing 'input' in payload")
                    
                    task_type = payload.get("task_type", "general")
                    
                    llm_result = self.inference_router.route(
                        prompt=prompt,
                        task_type=task_type,
                        max_tokens=payload.get("max_tokens", 256)
                    )
                    
                    result = {
                        "prediction": llm_result.get("text", ""),
                        "confidence": 0.9,
                        "model": llm_result.get("routed_to", "unknown"),
                        "tokens": llm_result.get("tokens", 0),
                        "provider": llm_result.get("provider", "unknown")
                    }
                    
                    logger.info("LLM inference completed", extra={
                        "model": result["model"],
                        "tokens": result["tokens"]
                    })
                
                except Exception as e:
                    logger.warning(f"LLM inference failed: {e}, using fallback")
                    result = None
            
            # Fallback to echo
            if result is None:
                result = {
                    "prediction": f"Echo: {payload.get('input', 'no-input')}",
                    "confidence": 0.5,
                    "model": "echo-fallback",
                    "fallback": True
                }
            
            # Emit response
            response = self.event_factory.create_event(
                event_type="mldl.infer.response",
                payload={"result": result},
                correlation_id=event.correlation_id,
                source="mldl_kernel"
            )
            await self.event_bus.emit(response)
            
            # Update metrics
            self._events_processed += 1
            self._inference_count += 1
            elapsed = time.time() - start
            self._total_inference_time += elapsed
            
            logger.debug(f"Inference completed in {elapsed:.3f}s")
        
        except Exception as e:
            self._error_count += 1
            logger.error(f"Inference error: {e}", exc_info=True)
            
            # Send error response
            error_resp = self.event_factory.create_event(
                event_type="mldl.infer.response",
                payload={"error": str(e), "result": None},
                correlation_id=event.correlation_id,
                source="mldl_kernel"
            )
            await self.event_bus.emit(error_resp)
    
    async def stop(self):
        """Graceful shutdown"""
        if not self._running:
            logger.warning("MLDL kernel not running")
            return
        
        logger.info("MLDL kernel stopping", extra={
            "inference_count": self._inference_count,
            "errors": self._error_count
        })
        
        self._running = False
        
        # Unsubscribe
        self.event_bus.unsubscribe("mldl.infer", self._handle_infer)
        
        logger.info("MLDL kernel stopped")
    
    def get_health(self) -> dict:
        """Health check with actual metrics"""
        uptime = 0.0
        if self._running and self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        avg_inference_time = 0.0
        if self._inference_count > 0:
            avg_inference_time = self._total_inference_time / self._inference_count
        
        return {
            "status": "healthy" if self._running else "stopped",
            "running": self._running,
            "uptime_seconds": uptime,
            "inference_count": self._inference_count,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "total_inference_time_seconds": self._total_inference_time,
            "errors": self._error_count,
            "llm_available": self.model_manager is not None,
            "models_loaded": len(self.model_manager.models) if self.model_manager else 0,
            "events_processed": self._events_processed
        }


# Global instance for backwards compatibility
_instance: Optional[MLDLKernel] = None


async def start():
    """Legacy start function"""
    global _instance
    if _instance is None:
        from grace.integration.event_bus import get_event_bus
        from grace.events.factory import GraceEventFactory
        _instance = MLDLKernel(get_event_bus(), GraceEventFactory())
    await _instance.start()


async def stop():
    """Legacy stop function"""
    if _instance:
        await _instance.stop()


def get_health():
    """Legacy health function"""
    if _instance:
        return _instance.get_health()
    return {"status": "not_initialized", "running": False}
