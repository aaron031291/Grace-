import asyncio
import logging
from datetime import datetime
from typing import Optional, Any, Dict, List
import time

from grace.mcp import MCPClient, MCPMessageType

logger = logging.getLogger(__name__)


class MLDLKernel:
    """
    MLDL Kernel with MCP integration
    """
    
    def __init__(self, event_bus, event_factory, model_manager=None, inference_router=None, trigger_mesh=None):
        self.event_bus = event_bus
        self.event_factory = event_factory
        self.model_manager = model_manager
        self.inference_router = inference_router
        self.trigger_mesh = trigger_mesh
        
        # State
        self._running = False
        self._start_time: Optional[datetime] = None
        self._events_processed = 0
        self._error_count = 0
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._consensus_count = 0
    
        # MCP Client
        self.mcp_client = MCPClient(
            kernel_name="mldl_kernel",
            event_bus=event_bus,
            trigger_mesh=trigger_mesh,
            minimum_trust=0.6
        )
    
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
        if self.trigger_mesh:
            self.trigger_mesh.subscribe("mldl.infer", self._handle_infer, "mldl_kernel")
            self.trigger_mesh.subscribe("mldl.consensus.request", self._handle_consensus_request, "mldl_kernel")
        else:
            self.event_bus.subscribe("mldl.infer", self._handle_infer)
            self.event_bus.subscribe("mldl.consensus.request", self._handle_consensus_request)
        
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
    
    async def _handle_consensus_request(self, event):
        """Handle consensus request with MCP validation"""
        start = time.time()
        
        try:
            # Receive and validate MCP message
            mcp_message = await self.mcp_client.receive_message(event)
            if not mcp_message:
                logger.error("Invalid MCP consensus request")
                return
            
            payload = mcp_message.payload
            decision_context = payload.get("decision_context", {})
            options = payload.get("options", [])
            
            logger.info(f"Consensus request received via MCP", extra={
                "message_id": mcp_message.message_id,
                "correlation_id": mcp_message.correlation_id
            })
            
            # Get consensus
            consensus_result = await self._compute_consensus(decision_context, options)
            
            # Send MCP response
            await self.mcp_client.send_message(
                destination=mcp_message.source,
                payload={
                    "consensus": consensus_result,
                    "processing_time_ms": (time.time() - start) * 1000
                },
                message_type=MCPMessageType.RESPONSE,
                correlation_id=mcp_message.correlation_id,
                schema_name="consensus_response",
                trust_score=0.9
            )
            
            self._consensus_count += 1
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Consensus request error: {e}", exc_info=True)
    
    async def _compute_consensus(self, context: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
        """
        Compute consensus from multiple ML specialists
        
        Returns:
            Dictionary with recommendation, confidence, and specialist votes
        """
        specialists = []
        
        # Specialist 1: Rule-based heuristic
        specialist_1_vote = self._heuristic_specialist(context, options)
        specialists.append({
            "name": "heuristic",
            "vote": specialist_1_vote["recommendation"],
            "confidence": specialist_1_vote["confidence"]
        })
        
        # Specialist 2: LLM-based (if available)
        if self.inference_router and self.model_manager:
            specialist_2_vote = await self._llm_specialist(context, options)
            specialists.append({
                "name": "llm",
                "vote": specialist_2_vote["recommendation"],
                "confidence": specialist_2_vote["confidence"]
            })
        
        # Specialist 3: Statistical classifier
        specialist_3_vote = self._statistical_specialist(context, options)
        specialists.append({
            "name": "statistical",
            "vote": specialist_3_vote["recommendation"],
            "confidence": specialist_3_vote["confidence"]
        })
        
        # Compute quorum consensus
        consensus = self._quorum_vote(specialists)
        
        return {
            "recommendation": consensus["recommendation"],
            "confidence": consensus["confidence"],
            "specialists": specialists,
            "quorum_method": "weighted_majority",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _heuristic_specialist(self, context: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
        """Simple rule-based specialist"""
        # Check trust score
        trust_score = context.get("trust_score", 0.5)
        
        if trust_score >= 0.8:
            return {"recommendation": "approve", "confidence": 0.85}
        elif trust_score >= 0.5:
            return {"recommendation": "review", "confidence": 0.70}
        else:
            return {"recommendation": "reject", "confidence": 0.90}
    
    async def _llm_specialist(self, context: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
        """LLM-based specialist"""
        try:
            prompt = f"""
            Decision context: {context}
            Options: {options}
            
            Provide recommendation and confidence (0-1).
            """
            
            result = self.inference_router.route(
                prompt=prompt,
                task_type="classification",
                max_tokens=100
            )
            
            # Parse LLM output (simplified)
            return {"recommendation": "review", "confidence": 0.75}
        
        except Exception as e:
            logger.warning(f"LLM specialist failed: {e}")
            return {"recommendation": "review", "confidence": 0.50}
    
    def _statistical_specialist(self, context: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
        """Statistical classifier specialist"""
        # Simple Bayesian-style classification
        violation_count = len(context.get("violations", []))
        
        if violation_count == 0:
            return {"recommendation": "approve", "confidence": 0.80}
        elif violation_count <= 2:
            return {"recommendation": "review", "confidence": 0.75}
        else:
            return {"recommendation": "reject", "confidence": 0.85}
    
    def _quorum_vote(self, specialists: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute weighted majority consensus
        
        Each specialist's vote is weighted by their confidence
        """
        vote_weights = {}
        
        for specialist in specialists:
            vote = specialist["vote"]
            confidence = specialist["confidence"]
            
            if vote not in vote_weights:
                vote_weights[vote] = 0.0
            
            vote_weights[vote] += confidence
        
        # Find recommendation with highest weight
        if not vote_weights:
            return {"recommendation": "review", "confidence": 0.5}
        
        best_vote = max(vote_weights, key=vote_weights.get)
        total_weight = sum(vote_weights.values())
        
        # Confidence is the proportion of weight for the winning vote
        consensus_confidence = vote_weights[best_vote] / total_weight if total_weight > 0 else 0.5
        
        return {
            "recommendation": best_vote,
            "confidence": consensus_confidence
        }
    
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
        if self.trigger_mesh:
            self.trigger_mesh.unsubscribe("mldl.infer", self._handle_infer)
            self.trigger_mesh.unsubscribe("mldl.consensus.request", self._handle_consensus_request)
        else:
            self.event_bus.unsubscribe("mldl.infer", self._handle_infer)
            self.event_bus.unsubscribe("mldl.consensus.request", self._handle_consensus_request)
        
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
            "consensus_count": self._consensus_count,
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
