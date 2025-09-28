"""
Saga Engine - Compensating transaction framework for multi-step failures.

Provides transactional guarantees across distributed operations by implementing
the Saga pattern with compensating actions for rollback scenarios.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    """Saga execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class StepStatus(Enum):
    """Individual step status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """Individual step in a saga."""
    step_id: str
    name: str
    action: Callable
    compensation: Optional[Callable] = None
    parameters: Dict[str, Any] = None
    timeout_seconds: float = 30.0
    retry_count: int = 3
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class SagaDefinition:
    """Saga definition with ordered steps."""
    saga_id: str
    name: str
    description: str
    steps: List[SagaStep]
    timeout_seconds: float = 300.0
    correlation_id: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class SagaExecution:
    """Runtime saga execution state."""
    saga_def: SagaDefinition
    status: SagaStatus = SagaStatus.PENDING
    current_step_index: int = 0
    completed_steps: List[str] = None
    failed_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    compensation_started_at: Optional[datetime] = None
    compensation_completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []


class SagaEngine:
    """
    Saga engine implementing compensating transaction pattern.
    
    Manages saga execution, failure handling, and compensation workflows
    to provide transactional guarantees across distributed operations.
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.active_sagas: Dict[str, SagaExecution] = {}
        self.completed_sagas: Dict[str, SagaExecution] = {}
        self.saga_history: List[Dict[str, Any]] = []
        self.running = False
        
        # Configuration
        self.max_concurrent_sagas = 10
        self.saga_timeout_seconds = 300.0
        self.cleanup_interval_seconds = 3600.0  # 1 hour
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the saga engine."""
        if self.running:
            return
            
        self.running = True
        logger.info("Starting Saga Engine...")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Register for relevant events if event bus available
        if self.event_bus:
            await self.event_bus.subscribe("saga_request", self._handle_saga_request)
            await self.event_bus.subscribe("saga_step_completed", self._handle_step_completion)
            await self.event_bus.subscribe("saga_step_failed", self._handle_step_failure)
        
        logger.info("Saga Engine started successfully")
    
    async def stop(self):
        """Stop the saga engine."""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopping Saga Engine...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Compensate any in-progress sagas
        for saga_exec in self.active_sagas.values():
            if saga_exec.status == SagaStatus.IN_PROGRESS:
                logger.warning(f"Force compensating saga {saga_exec.saga_def.saga_id} due to shutdown")
                await self._compensate_saga(saga_exec)
        
        logger.info("Saga Engine stopped")
    
    async def execute_saga(self, saga_def: SagaDefinition) -> str:
        """
        Execute a saga with compensation support.
        Returns the saga execution ID.
        """
        if not self.running:
            raise RuntimeError("Saga engine not running")
        
        if len(self.active_sagas) >= self.max_concurrent_sagas:
            raise RuntimeError(f"Maximum concurrent sagas ({self.max_concurrent_sagas}) exceeded")
        
        # Create execution context
        saga_exec = SagaExecution(
            saga_def=saga_def,
            status=SagaStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )
        
        self.active_sagas[saga_def.saga_id] = saga_exec
        
        # Log saga start
        await self._log_saga_event("saga_started", saga_exec)
        
        logger.info(f"Starting saga execution: {saga_def.saga_id} ({saga_def.name})")
        
        # Execute saga asynchronously
        asyncio.create_task(self._execute_saga_steps(saga_exec))
        
        return saga_def.saga_id
    
    async def _execute_saga_steps(self, saga_exec: SagaExecution):
        """Execute saga steps in order."""
        try:
            steps = saga_exec.saga_def.steps
            
            for i, step in enumerate(steps):
                saga_exec.current_step_index = i
                
                # Check if saga was cancelled
                if saga_exec.status != SagaStatus.IN_PROGRESS:
                    break
                
                # Execute step
                success = await self._execute_step(saga_exec, step)
                
                if success:
                    saga_exec.completed_steps.append(step.step_id)
                    await self._log_saga_event("saga_step_completed", saga_exec, {
                        "step_id": step.step_id,
                        "step_name": step.name,
                        "result": step.result
                    })
                else:
                    # Step failed - start compensation
                    saga_exec.failed_step = step.step_id
                    saga_exec.status = SagaStatus.COMPENSATING
                    await self._log_saga_event("saga_step_failed", saga_exec, {
                        "step_id": step.step_id,
                        "error": step.error
                    })
                    
                    # Compensate previous steps
                    await self._compensate_saga(saga_exec)
                    return
            
            # All steps completed successfully
            saga_exec.status = SagaStatus.COMPLETED
            saga_exec.completed_at = datetime.utcnow()
            await self._complete_saga(saga_exec)
            
        except Exception as e:
            logger.error(f"Saga execution error {saga_exec.saga_def.saga_id}: {e}")
            saga_exec.status = SagaStatus.FAILED
            await self._log_saga_event("saga_failed", saga_exec, {"error": str(e)})
            await self._compensate_saga(saga_exec)
    
    async def _execute_step(self, saga_exec: SagaExecution, step: SagaStep) -> bool:
        """Execute an individual saga step."""
        step.status = StepStatus.EXECUTING
        step.started_at = datetime.utcnow()
        
        logger.info(f"Executing step {step.step_id} ({step.name}) in saga {saga_exec.saga_def.saga_id}")
        
        try:
            # Execute with timeout
            step.result = await asyncio.wait_for(
                self._call_step_action(step, saga_exec.saga_def.context),
                timeout=step.timeout_seconds
            )
            
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            
            logger.info(f"Step {step.step_id} completed successfully")
            return True
            
        except asyncio.TimeoutError:
            error_msg = f"Step {step.step_id} timed out after {step.timeout_seconds}s"
            step.error = error_msg
            step.status = StepStatus.FAILED
            logger.error(error_msg)
            return False
            
        except Exception as e:
            error_msg = f"Step {step.step_id} failed: {str(e)}"
            step.error = error_msg
            step.status = StepStatus.FAILED
            logger.error(error_msg)
            return False
    
    async def _call_step_action(self, step: SagaStep, context: Dict[str, Any]) -> Any:
        """Call step action with proper parameter handling."""
        if asyncio.iscoroutinefunction(step.action):
            return await step.action(step.parameters, context)
        else:
            return step.action(step.parameters, context)
    
    async def _compensate_saga(self, saga_exec: SagaExecution):
        """Compensate completed steps in reverse order."""
        saga_exec.status = SagaStatus.COMPENSATING
        saga_exec.compensation_started_at = datetime.utcnow()
        
        await self._log_saga_event("saga_compensation_started", saga_exec)
        
        logger.info(f"Starting compensation for saga {saga_exec.saga_def.saga_id}")
        
        # Compensate in reverse order
        completed_step_ids = saga_exec.completed_steps[::-1]
        
        for step_id in completed_step_ids:
            step = next((s for s in saga_exec.saga_def.steps if s.step_id == step_id), None)
            if step and step.compensation:
                await self._compensate_step(saga_exec, step)
        
        saga_exec.status = SagaStatus.COMPENSATED
        saga_exec.compensation_completed_at = datetime.utcnow()
        
        await self._complete_saga(saga_exec)
        
        logger.info(f"Compensation completed for saga {saga_exec.saga_def.saga_id}")
    
    async def _compensate_step(self, saga_exec: SagaExecution, step: SagaStep):
        """Compensate an individual step."""
        if not step.compensation:
            return
        
        step.status = StepStatus.COMPENSATING
        
        logger.info(f"Compensating step {step.step_id} in saga {saga_exec.saga_def.saga_id}")
        
        try:
            if asyncio.iscoroutinefunction(step.compensation):
                await step.compensation(step.parameters, saga_exec.saga_def.context, step.result)
            else:
                step.compensation(step.parameters, saga_exec.saga_def.context, step.result)
            
            step.status = StepStatus.COMPENSATED
            
            await self._log_saga_event("saga_step_compensated", saga_exec, {
                "step_id": step.step_id,
                "step_name": step.name
            })
            
            logger.info(f"Step {step.step_id} compensated successfully")
            
        except Exception as e:
            logger.error(f"Failed to compensate step {step.step_id}: {e}")
            # Continue with other compensations even if one fails
    
    async def _complete_saga(self, saga_exec: SagaExecution):
        """Complete saga and move to history."""
        # Move to completed sagas
        self.completed_sagas[saga_exec.saga_def.saga_id] = saga_exec
        if saga_exec.saga_def.saga_id in self.active_sagas:
            del self.active_sagas[saga_exec.saga_def.saga_id]
        
        # Add to history
        self.saga_history.append({
            "saga_id": saga_exec.saga_def.saga_id,
            "name": saga_exec.saga_def.name,
            "status": saga_exec.status.value,
            "started_at": saga_exec.started_at.isoformat() if saga_exec.started_at else None,
            "completed_at": saga_exec.completed_at.isoformat() if saga_exec.completed_at else None,
            "compensation_completed_at": saga_exec.compensation_completed_at.isoformat() if saga_exec.compensation_completed_at else None,
            "steps_completed": len(saga_exec.completed_steps),
            "total_steps": len(saga_exec.saga_def.steps),
            "failed_step": saga_exec.failed_step
        })
        
        await self._log_saga_event("saga_completed", saga_exec)
    
    async def _log_saga_event(self, event_type: str, saga_exec: SagaExecution, extra_data: Dict[str, Any] = None):
        """Log saga events to event bus."""
        if not self.event_bus:
            return
        
        event_data = {
            "saga_id": saga_exec.saga_def.saga_id,
            "saga_name": saga_exec.saga_def.name,
            "status": saga_exec.status.value,
            "current_step_index": saga_exec.current_step_index,
            "correlation_id": saga_exec.saga_def.correlation_id
        }
        
        if extra_data:
            event_data.update(extra_data)
        
        try:
            await self.event_bus.publish(event_type, event_data, correlation_id=saga_exec.saga_def.correlation_id)
        except Exception as e:
            logger.error(f"Failed to log saga event {event_type}: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old completed sagas."""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_old_sagas()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Saga cleanup error: {e}")
    
    async def _cleanup_old_sagas(self):
        """Clean up old completed sagas."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep for 24 hours
        
        to_remove = []
        for saga_id, saga_exec in self.completed_sagas.items():
            completed_time = saga_exec.completed_at or saga_exec.compensation_completed_at
            if completed_time and completed_time < cutoff_time:
                to_remove.append(saga_id)
        
        for saga_id in to_remove:
            del self.completed_sagas[saga_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old saga executions")
    
    def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a saga."""
        saga_exec = self.active_sagas.get(saga_id) or self.completed_sagas.get(saga_id)
        if not saga_exec:
            return None
        
        return {
            "saga_id": saga_id,
            "name": saga_exec.saga_def.name,
            "status": saga_exec.status.value,
            "current_step_index": saga_exec.current_step_index,
            "total_steps": len(saga_exec.saga_def.steps),
            "completed_steps": len(saga_exec.completed_steps),
            "failed_step": saga_exec.failed_step,
            "started_at": saga_exec.started_at.isoformat() if saga_exec.started_at else None,
            "completed_at": saga_exec.completed_at.isoformat() if saga_exec.completed_at else None
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get saga engine statistics."""
        return {
            "active_sagas": len(self.active_sagas),
            "completed_sagas": len(self.completed_sagas),
            "total_sagas_processed": len(self.saga_history),
            "max_concurrent_sagas": self.max_concurrent_sagas,
            "running": self.running
        }
    
    async def _handle_saga_request(self, event):
        """Handle external saga requests."""
        try:
            payload = event.get("payload", {})
            # This would be used for external saga triggering
            logger.info(f"Received saga request: {payload}")
        except Exception as e:
            logger.error(f"Error handling saga request: {e}")
    
    async def _handle_step_completion(self, event):
        """Handle step completion events."""
        try:
            payload = event.get("payload", {})
            logger.debug(f"Step completion event: {payload}")
        except Exception as e:
            logger.error(f"Error handling step completion: {e}")
    
    async def _handle_step_failure(self, event):
        """Handle step failure events.""" 
        try:
            payload = event.get("payload", {})
            logger.debug(f"Step failure event: {payload}")
        except Exception as e:
            logger.error(f"Error handling step failure: {e}")


# Helper functions for creating common saga patterns

def create_ingestion_saga(content_id: str, content: str, source: str) -> SagaDefinition:
    """Create saga for content ingestion pipeline."""
    saga_id = f"ingestion_{content_id}_{uuid.uuid4().hex[:8]}"
    
    async def validate_content(params, context):
        # Simulate content validation
        await asyncio.sleep(0.1)
        return {"valid": True, "content_id": params["content_id"]}
    
    async def compensate_validation(params, context, result):
        logger.info(f"Compensating validation for {params['content_id']}")
    
    async def store_content(params, context):
        # Simulate content storage
        await asyncio.sleep(0.1)
        return {"stored": True, "storage_id": f"storage_{params['content_id']}"}
    
    async def compensate_storage(params, context, result):
        logger.info(f"Compensating storage: removing {result.get('storage_id')}")
    
    async def index_content(params, context):
        # Simulate content indexing
        await asyncio.sleep(0.1)
        return {"indexed": True, "index_id": f"idx_{params['content_id']}"}
    
    async def compensate_indexing(params, context, result):
        logger.info(f"Compensating indexing: removing {result.get('index_id')}")
    
    steps = [
        SagaStep(
            step_id="validate",
            name="Content Validation",
            action=validate_content,
            compensation=compensate_validation,
            parameters={"content_id": content_id, "content": content}
        ),
        SagaStep(
            step_id="store",
            name="Content Storage", 
            action=store_content,
            compensation=compensate_storage,
            parameters={"content_id": content_id, "content": content}
        ),
        SagaStep(
            step_id="index",
            name="Content Indexing",
            action=index_content,
            compensation=compensate_indexing,
            parameters={"content_id": content_id}
        )
    ]
    
    return SagaDefinition(
        saga_id=saga_id,
        name="Content Ingestion",
        description=f"Ingest content from {source}",
        steps=steps,
        context={"source": source}
    )