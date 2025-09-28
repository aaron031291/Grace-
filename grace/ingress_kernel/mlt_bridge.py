"""
Ingress-MLT Bridge - Connects Ingress Kernel to Meta Learning and Trust (MLT) System.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, Any, Optional, List
import uuid

from grace.contracts.ingress_contracts import IngressExperience
from grace.contracts.ingress_events import IngressEvent, IngressEventType, ExperiencePayload


logger = logging.getLogger(__name__)


class IngressMLTBridge:
    """Bridge between Ingress Kernel and MLT System."""
    
    def __init__(self, ingress_kernel, mlt_kernel=None, event_bus=None):
        """
        Initialize the MLT bridge.
        
        Args:
            ingress_kernel: The Ingress Kernel instance
            mlt_kernel: MLT kernel for meta-learning
            event_bus: Event bus for system communication
        """
        self.ingress_kernel = ingress_kernel
        self.mlt_kernel = mlt_kernel
        self.event_bus = event_bus
        
        # Experience tracking
        self.experience_buffer: List[IngressExperience] = []
        self.adaptation_plans: Dict[str, Dict[str, Any]] = {}
        
        # Metrics aggregation
        self.metrics_window = timedelta(minutes=15)  # 15-minute windows
        self.current_metrics = {}
        self.last_metrics_reset = utc_now()
        
        self.running = False
    
    async def start(self):
        """Start the MLT bridge."""
        if self.running:
            return
        
        logger.info("Starting Ingress-MLT Bridge...")
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._collect_experiences())
        asyncio.create_task(self._process_adaptation_plans())
        asyncio.create_task(self._emit_experiences())
        
        logger.info("Ingress-MLT Bridge started")
    
    async def stop(self):
        """Stop the MLT bridge."""
        logger.info("Stopping Ingress-MLT Bridge...")
        self.running = False
    
    async def emit_experience(self, stage: str, metrics: Dict[str, Any], 
                            source_id: str, samples: Optional[Dict[str, Any]] = None):
        """
        Emit an experience to MLT for learning.
        
        Args:
            stage: Ingress stage (capture, parse, normalize, validate, enrich, persist, publish)
            metrics: Performance metrics for the stage
            source_id: Source identifier
            samples: Optional sample data for accuracy evaluation
        """
        try:
            experience = IngressExperience(
                source_id=source_id,
                stage=stage,
                metrics=metrics,
                samples=samples
            )
            
            # Add to buffer
            self.experience_buffer.append(experience)
            
            # Emit event
            if self.event_bus:
                event = IngressEvent(
                    event_type=IngressEventType.EXPERIENCE,
                    correlation_id=experience.exp_id,
                    payload=ExperiencePayload(
                        schema_version="1.0.0",
                        experience=experience.dict()
                    )
                )
                await self.event_bus.publish(event.dict())
            
            # Send to MLT kernel
            if self.mlt_kernel:
                await self._send_experience_to_mlt(experience)
            
            logger.debug(f"Emitted experience for {stage} stage from {source_id}")
            
        except Exception as e:
            logger.error(f"Failed to emit experience: {e}")
    
    async def consume_adaptation_plan(self, plan: Dict[str, Any]) -> bool:
        """
        Consume an adaptation plan from MLT.
        
        Args:
            plan: Adaptation plan with actions
            
        Returns:
            Success status
        """
        try:
            plan_id = plan.get("plan_id", str(uuid.uuid4()))
            self.adaptation_plans[plan_id] = {
                "plan": plan,
                "status": "received",
                "received_at": utc_now()
            }
            
            # Apply plan actions
            success = await self._apply_adaptation_plan(plan)
            
            self.adaptation_plans[plan_id]["status"] = "applied" if success else "failed"
            self.adaptation_plans[plan_id]["applied_at"] = utc_now()
            
            logger.info(f"Applied adaptation plan: {plan_id}, success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to consume adaptation plan: {e}")
            return False
    
    async def get_parser_accuracy(self, parser_type: str, source_id: Optional[str] = None) -> float:
        """Get parser accuracy metrics for MLT feedback."""
        try:
            # Mock implementation - would calculate real accuracy from labeled data
            if parser_type == "pdf":
                return 0.92
            elif parser_type == "html":
                return 0.88
            elif parser_type == "json":
                return 0.98
            else:
                return 0.85
                
        except Exception as e:
            logger.error(f"Failed to get parser accuracy: {e}")
            return 0.5
    
    async def get_ner_metrics(self, source_id: Optional[str] = None) -> Dict[str, float]:
        """Get NER precision/recall metrics."""
        try:
            # Mock implementation - would calculate from evaluation data
            return {
                "precision": 0.87,
                "recall": 0.84,
                "f1": 0.85
            }
            
        except Exception as e:
            logger.error(f"Failed to get NER metrics: {e}")
            return {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    
    async def _collect_experiences(self):
        """Collect experiences from ingress operations."""
        while self.running:
            try:
                await self._collect_stage_metrics()
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Experience collection error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_stage_metrics(self):
        """Collect metrics from each ingress stage."""
        current_time = utc_now()
        
        # Reset metrics window if needed
        if current_time - self.last_metrics_reset > self.metrics_window:
            await self._emit_window_metrics()
            self.current_metrics = {}
            self.last_metrics_reset = current_time
        
        # Collect metrics from kernel
        for source_id in self.ingress_kernel.sources.keys():
            await self._collect_source_metrics(source_id)
    
    async def _collect_source_metrics(self, source_id: str):
        """Collect metrics for a specific source."""
        try:
            source_health = self.ingress_kernel.health_status.get(source_id, {})
            
            if source_id not in self.current_metrics:
                self.current_metrics[source_id] = {
                    "capture": {"events": 0, "errors": 0, "latency_sum": 0},
                    "parse": {"events": 0, "errors": 0, "accuracy_sum": 0},
                    "normalize": {"events": 0, "errors": 0, "quality_sum": 0},
                    "validate": {"events": 0, "errors": 0, "violations": 0},
                    "enrich": {"events": 0, "errors": 0, "enrichments": 0},
                    "persist": {"events": 0, "errors": 0, "size_sum": 0},
                    "publish": {"events": 0, "errors": 0, "topics": 0}
                }
            
            # Mock metric collection - would get real metrics from kernel
            metrics = self.current_metrics[source_id]
            metrics["capture"]["events"] += 1
            metrics["parse"]["events"] += 1
            metrics["normalize"]["events"] += 1
            
        except Exception as e:
            logger.error(f"Failed to collect source metrics for {source_id}: {e}")
    
    async def _emit_window_metrics(self):
        """Emit aggregated metrics as experiences."""
        for source_id, metrics in self.current_metrics.items():
            for stage, stage_metrics in metrics.items():
                if stage_metrics["events"] > 0:
                    await self._emit_stage_experience(source_id, stage, stage_metrics)
    
    async def _emit_stage_experience(self, source_id: str, stage: str, stage_metrics: Dict[str, Any]):
        """Emit experience for a specific stage."""
        try:
            events_count = stage_metrics["events"]
            error_count = stage_metrics["errors"]
            
            # Calculate metrics
            throughput_rps = events_count / (self.metrics_window.total_seconds())
            error_rate = error_count / max(events_count, 1)
            
            metrics = {
                "throughput_rps": throughput_rps,
                "error_rate": error_rate,
                "schema_violations": stage_metrics.get("violations", 0),
                "pii_incidents": 0,  # Would be calculated from validation results
                "avg_latency_ms": stage_metrics.get("latency_sum", 0) / max(events_count, 1),
                "dedup_rate": 0.05,  # Mock deduplication rate
                "trust_mean": 0.8,  # Mock trust score
                "completeness_mean": 0.9  # Mock completeness
            }
            
            # Add stage-specific metrics
            samples = {}
            if stage == "parse":
                samples["parser_accuracy"] = await self.get_parser_accuracy(
                    self.ingress_kernel.sources[source_id].parser,
                    source_id
                )
            elif stage == "enrich":
                ner_metrics = await self.get_ner_metrics(source_id)
                samples.update(ner_metrics)
            
            await self.emit_experience(stage, metrics, source_id, samples)
            
        except Exception as e:
            logger.error(f"Failed to emit stage experience: {e}")
    
    async def _process_adaptation_plans(self):
        """Process incoming adaptation plans."""
        while self.running:
            try:
                if self.mlt_kernel and hasattr(self.mlt_kernel, 'get_pending_plans'):
                    plans = await self.mlt_kernel.get_pending_plans("ingress")
                    
                    for plan in plans:
                        await self.consume_adaptation_plan(plan)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Adaptation plan processing error: {e}")
                await asyncio.sleep(60)
    
    async def _apply_adaptation_plan(self, plan: Dict[str, Any]) -> bool:
        """Apply adaptation plan actions."""
        try:
            actions = plan.get("actions", [])
            success_count = 0
            
            for action in actions:
                if await self._apply_action(action):
                    success_count += 1
            
            return success_count == len(actions)
            
        except Exception as e:
            logger.error(f"Failed to apply adaptation plan: {e}")
            return False
    
    async def _apply_action(self, action: Dict[str, Any]) -> bool:
        """Apply a single adaptation action."""
        try:
            action_type = action.get("type")
            
            if action_type == "hpo":
                return await self._apply_hyperparameter_optimization(action)
            elif action_type == "policy_delta":
                return await self._apply_policy_delta(action)
            elif action_type == "reweight_specialists":
                return await self._apply_specialist_reweighting(action)
            elif action_type == "canary":
                return await self._apply_canary_deployment(action)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply action: {e}")
            return False
    
    async def _apply_hyperparameter_optimization(self, action: Dict[str, Any]) -> bool:
        """Apply hyperparameter optimization action."""
        target = action.get("target")
        budget = action.get("budget", {})
        
        logger.info(f"Applying HPO to {target} with budget {budget}")
        
        # Mock implementation - would run actual HPO
        if target == "parsers.pdf":
            # Optimize PDF parser parameters
            pass
        elif target == "dedupe.threshold":
            # Optimize deduplication threshold
            new_threshold = 0.90  # Mock optimized value
            self.ingress_kernel.config["dedupe"]["threshold"] = new_threshold
            logger.info(f"Updated dedupe threshold to {new_threshold}")
        
        return True
    
    async def _apply_policy_delta(self, action: Dict[str, Any]) -> bool:
        """Apply policy delta action."""
        path = action.get("path")
        from_val = action.get("from")
        to_val = action.get("to")
        
        logger.info(f"Applying policy delta: {path} from {from_val} to {to_val}")
        
        # Update configuration
        if path == "ingress.validation.min_validity":
            self.ingress_kernel.config["validation"]["min_validity"] = to_val
        elif path == "ingress.validation.min_trust":
            self.ingress_kernel.config["validation"]["min_trust"] = to_val
        
        return True
    
    async def _apply_specialist_reweighting(self, action: Dict[str, Any]) -> bool:
        """Apply specialist reweighting action."""
        weights = action.get("weights", {})
        
        logger.info(f"Applying specialist reweighting: {weights}")
        
        # Mock implementation - would update specialist weights
        return True
    
    async def _apply_canary_deployment(self, action: Dict[str, Any]) -> bool:
        """Apply canary deployment action."""
        target_model = action.get("target_model")
        steps = action.get("steps", [])
        
        logger.info(f"Applying canary deployment: {target_model} with steps {steps}")
        
        # Mock implementation - would deploy new model version
        return True
    
    async def _send_experience_to_mlt(self, experience: IngressExperience):
        """Send experience to MLT kernel."""
        if not self.mlt_kernel:
            return
        
        try:
            # Convert to MLT format
            mlt_experience = {
                "component": "ingress_kernel",
                "stage": experience.stage,
                "metrics": experience.metrics,
                "samples": experience.samples,
                "timestamp": experience.timestamp.isoformat(),
                "source_id": experience.source_id
            }
            
            await self.mlt_kernel.process_experience(mlt_experience)
            
        except Exception as e:
            logger.error(f"Failed to send experience to MLT: {e}")
    
    async def _emit_experiences(self):
        """Emit batched experiences."""
        while self.running:
            try:
                if len(self.experience_buffer) > 100:
                    # Emit batch
                    batch = self.experience_buffer[:100]
                    self.experience_buffer = self.experience_buffer[100:]
                    
                    for exp in batch:
                        if self.mlt_kernel:
                            await self._send_experience_to_mlt(exp)
                
                await asyncio.sleep(30)  # Emit every 30 seconds
                
            except Exception as e:
                logger.error(f"Experience emission error: {e}")
                await asyncio.sleep(30)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "running": self.running,
            "experience_buffer_size": len(self.experience_buffer),
            "adaptation_plans": len(self.adaptation_plans),
            "metrics_sources": len(self.current_metrics),
            "last_metrics_reset": self.last_metrics_reset.isoformat()
        }