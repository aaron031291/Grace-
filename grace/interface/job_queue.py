"""
Background job processing system for Grace interface.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime
from grace.utils.time import iso_now_utc
from dataclasses import dataclass, field
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Job:
    """Background job definition."""

    job_id: str
    job_type: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.MEDIUM
    created_at: str = field(default_factory=lambda: iso_now_utc())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    idempotency_key: Optional[str] = None  # For duplicate detection


class JobQueue:
    """Simple in-memory job queue with background processing."""

    def __init__(self, max_concurrent_jobs: int = 3):
        self.jobs: Dict[str, Job] = {}
        self.job_handlers: Dict[str, Callable] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self._processing = False
        self.idempotency_cache: Dict[str, str] = {}  # idempotency_key -> job_id

    def register_handler(self, job_type: str, handler: Callable):
        """Register a handler for a job type."""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    def create_idempotency_key(self, job_type: str, payload: Dict[str, Any]) -> str:
        """Create an idempotency key from job type and payload."""
        # Create a deterministic hash from job type and key payload fields
        key_data = {"job_type": job_type, **payload}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.MEDIUM,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """Enqueue a job for background processing."""
        # Generate or use provided idempotency key
        if idempotency_key is None:
            idempotency_key = self.create_idempotency_key(job_type, payload)

        # Check if job already exists (idempotent operation)
        if idempotency_key in self.idempotency_cache:
            existing_job_id = self.idempotency_cache[idempotency_key]
            logger.info(
                f"Job with idempotency key {idempotency_key} already exists: {existing_job_id}"
            )
            return existing_job_id

        # Create new job
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = Job(
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            priority=priority,
            idempotency_key=idempotency_key,
        )

        self.jobs[job_id] = job
        self.idempotency_cache[idempotency_key] = job_id

        logger.info(f"Enqueued job {job_id} of type {job_type}")

        # Start processing if not already running
        if not self._processing:
            asyncio.create_task(self._process_jobs())

        return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job details."""
        return self.jobs.get(job_id)

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        job = self.jobs.get(job_id)
        return job.status if job else None

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[Job]:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())

        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by priority (high first) then by created_at
        jobs.sort(key=lambda j: (-j.priority.value, j.created_at))

        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True
        elif job.status == JobStatus.RUNNING and job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            job.status = JobStatus.CANCELLED
            return True

        return False

    async def _process_jobs(self):
        """Background job processor."""
        if self._processing:
            return

        self._processing = True
        logger.info("Started background job processing")

        try:
            while True:
                # Get pending jobs
                pending_jobs = [
                    j for j in self.jobs.values() if j.status == JobStatus.PENDING
                ]

                if not pending_jobs:
                    await asyncio.sleep(1)  # Wait for new jobs
                    continue

                # Sort by priority
                pending_jobs.sort(key=lambda j: (-j.priority.value, j.created_at))

                # Start jobs up to concurrency limit
                free_slots = self.max_concurrent_jobs - len(self.running_jobs)
                jobs_to_start = pending_jobs[:free_slots]

                for job in jobs_to_start:
                    if job.job_type in self.job_handlers:
                        task = asyncio.create_task(self._execute_job(job))
                        self.running_jobs[job.job_id] = task
                    else:
                        logger.error(f"No handler for job type: {job.job_type}")
                        job.status = JobStatus.FAILED
                        job.error = f"No handler for job type: {job.job_type}"

                # Clean up completed tasks
                completed_tasks = [
                    job_id for job_id, task in self.running_jobs.items() if task.done()
                ]
                for job_id in completed_tasks:
                    del self.running_jobs[job_id]

                await asyncio.sleep(0.1)  # Small delay to prevent tight loop

        except Exception as e:
            logger.error(f"Job processing error: {e}")
        finally:
            self._processing = False

    async def _execute_job(self, job: Job):
        """Execute a single job."""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow().isoformat()

            logger.info(f"Executing job {job.job_id} of type {job.job_type}")

            handler = self.job_handlers[job.job_type]
            result = await handler(job.payload)

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow().isoformat()
            job.result = result

            logger.info(f"Completed job {job.job_id}")

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled job {job.job_id}")
            raise
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.error = str(e)

            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.PENDING  # Retry
                logger.info(
                    f"Retrying job {job.job_id} ({job.retry_count}/{job.max_retries})"
                )
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow().isoformat()
                logger.error(
                    f"Job {job.job_id} permanently failed after {job.retry_count} retries"
                )


# Global job queue instance
job_queue = JobQueue()


async def document_ingest_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for document ingestion jobs."""
    file_path = payload.get("file_path")
    file_type = payload.get("file_type")
    user_id = payload.get("user_id")

    logger.info(f"Processing document ingestion: {file_path}")

    # Import here to avoid circular imports
    from .orb_interface import GraceUnifiedOrbInterface

    # This would typically be passed in or injected
    # For now, create a new instance (in production, use dependency injection)
    interface = GraceUnifiedOrbInterface()

    # Call the intelligence ingest method
    result = await interface.grace_intelligence.ingest_batch_document(
        file_path, file_type
    )

    logger.info(f"Document ingestion completed: {file_path}")

    return {
        "file_path": file_path,
        "file_type": file_type,
        "user_id": user_id,
        "ingestion_result": result,
        "processed_at": datetime.utcnow().isoformat(),
    }


# Register the document ingestion handler
job_queue.register_handler("document_ingest", document_ingest_handler)
