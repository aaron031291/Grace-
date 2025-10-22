import asyncio
import logging
from datetime import datetime
from typing import Optional, Any

from grace.integration.event_bus import get_event_bus
from grace.events.factory import GraceEventFactory

logger = logging.getLogger(__name__)
_bus = get_event_bus()
_factory = GraceEventFactory()

_running = False
_start_time: Optional[datetime] = None
_events_processed = 0
_error_count = 0
_inference_count = 0
_avg_inference_time = 0.0
_model_manager: Optional[Any] = None
_inference_router: Optional[Any] = None

async def _handle_infer(event):
    global _events_processed, _inference_count, _error_count, _avg_inference_time
    
    try:
        import time
        start = time.time()
        
        payload = event.payload or {}
        
        # Try to use real LLM if available
        result = None
        
        if _inference_router and _model_manager:
            try:
                # Real inference through LLM
                prompt = payload.get("input", "")
                task_type = payload.get("task_type", "general")
                
                llm_result = _inference_router.route(
                    prompt=prompt,
                    task_type=task_type,
                    max_tokens=payload.get("max_tokens", 256)
                )
                
                result = {
                    "prediction": llm_result.get("text", ""),
                    "confidence": 0.9,
                    "model": llm_result.get("routed_to", "unknown"),
                    "tokens": llm_result.get("tokens", 0)
                }
            
            except Exception as e:
                logger.warning(f"LLM inference failed, using fallback: {e}")
                result = None
        
        # Fallback: simple echo
        if result is None:
            result = {
                "prediction": payload.get("input", "no-input"),
                "confidence": 0.9,
                "model": "fallback-echo"
            }
        
        # respond by publishing an event with same correlation_id
        resp = _factory.create_event(
            event_type=f"response.{event.correlation_id or event.event_id}",
            payload={"result": result},
            correlation_id=event.correlation_id
        )
        _bus.publish(resp)
        
        # Update metrics
        _events_processed += 1
        _inference_count += 1
        
        elapsed = time.time() - start
        _avg_inference_time = (_avg_inference_time * (_inference_count - 1) + elapsed) / _inference_count
        
    except Exception as e:
        _error_count += 1
        logger.error(f"Inference error: {e}")

async def start():
    global _running, _start_time, _model_manager, _inference_router
    if _running:
        return
    _running = True
    _start_time = datetime.utcnow()
    logger.info("mldl kernel starting")
    
    # Try to initialize real LLM components
    try:
        from grace.llm import ModelManager, InferenceRouter
        
        _model_manager = ModelManager()
        
        # Try to load default models (will gracefully fail if models not available)
        try:
            _model_manager.load_default_models()
            logger.info("LLM models loaded successfully")
        except Exception as e:
            logger.warning(f"LLM models not available: {e}")
        
        _inference_router = InferenceRouter(_model_manager)
        
    except Exception as e:
        logger.warning(f"LLM components not available, using fallback: {e}")
        _model_manager = None
        _inference_router = None
    
    # subscribe to inference requests
    _bus.subscribe("mldl.infer", _handle_infer)

async def stop():
    global _running
    _running = False
    logger.info("mldl kernel stopping")

def get_health():
    """Return kernel-specific health metrics"""
    return {
        "inference_count": _inference_count,
        "avg_inference_time_ms": _avg_inference_time * 1000,
        "errors": _error_count,
        "llm_available": _model_manager is not None,
        "models_loaded": len(_model_manager.models) if _model_manager else 0
    }
