"""
Grace Worker Service - Handles background tasks and queue processing.

This service consumes from queues: ingestion, embeddings, media
"""
import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional
import json

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from ..core.config import get_settings, setup_environment
from ..core.observability import setup_logging, ObservabilityContext

# Metrics
TASKS_PROCESSED = Counter('grace_worker_tasks_total', 'Total tasks processed', ['queue', 'status'])
TASK_DURATION = Histogram('grace_worker_task_duration_seconds', 'Task processing duration', ['queue'])
ACTIVE_TASKS = Gauge('grace_worker_active_tasks', 'Currently active tasks', ['queue'])
QUEUE_SIZE = Gauge('grace_worker_queue_size', 'Queue size', ['queue'])

logger = logging.getLogger(__name__)


class GraceWorker:
    """Grace background task worker."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        self.tasks: Dict[str, asyncio.Task] = {}
        self.queues = [q.strip() for q in self.settings.worker_queues.split(',')]
        
    async def start(self):
        """Start the worker service."""
        logger.info(f"Starting Grace Worker service with queues: {self.queues}")
        
        # Initialize Redis connection
        if self.settings.redis_url:
            self.redis_client = redis.from_url(self.settings.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        else:
            raise RuntimeError("Redis URL not configured")
        
        # Start metrics server
        start_http_server(8080)  # Expose metrics on port 8080
        logger.info("Metrics server started on port 8080")
        
        self.running = True
        
        # Start queue processors
        for queue in self.queues:
            task = asyncio.create_task(self._process_queue(queue))
            self.tasks[queue] = task
            logger.info(f"Started processor for queue: {queue}")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_queues())
        self.tasks['monitor'] = monitor_task
        
        logger.info("Grace Worker service started successfully")
    
    async def stop(self):
        """Stop the worker service."""
        logger.info("Stopping Grace Worker service...")
        self.running = False
        
        # Cancel all tasks
        for queue, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Cancelled task for queue: {queue}")
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
        
        logger.info("Grace Worker service stopped")
    
    async def _process_queue(self, queue_name: str):
        """Process tasks from a specific queue."""
        logger.info(f"Processing queue: {queue_name}")
        
        while self.running:
            try:
                # Block-pop from Redis list (queue)
                result = await self.redis_client.blpop(f"grace:queue:{queue_name}", timeout=5)
                
                if result is None:
                    continue  # Timeout, continue loop
                
                queue_key, task_data = result
                
                with ObservabilityContext(queue=queue_name, task_id=task_data[:8]):
                    await self._process_task(queue_name, task_data)
                    
            except asyncio.CancelledError:
                logger.info(f"Queue processor for {queue_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue processor for {queue_name}: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _process_task(self, queue_name: str, task_data: str):
        """Process a single task."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            ACTIVE_TASKS.labels(queue=queue_name).inc()
            
            # Parse task data
            try:
                task = json.loads(task_data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in task data: {task_data}")
                TASKS_PROCESSED.labels(queue=queue_name, status="error").inc()
                return
            
            task_type = task.get('type', 'unknown')
            task_id = task.get('id', 'unknown')
            
            logger.info(f"Processing task {task_id} of type {task_type} from queue {queue_name}")
            
            # Route to appropriate processor
            if queue_name == 'ingestion':
                await self._process_ingestion_task(task)
            elif queue_name == 'embeddings':
                await self._process_embeddings_task(task)
            elif queue_name == 'media':
                await self._process_media_task(task)
            else:
                logger.warning(f"Unknown queue: {queue_name}")
                TASKS_PROCESSED.labels(queue=queue_name, status="error").inc()
                return
            
            # Record success
            duration = asyncio.get_event_loop().time() - start_time
            TASK_DURATION.labels(queue=queue_name).observe(duration)
            TASKS_PROCESSED.labels(queue=queue_name, status="success").inc()
            
            logger.info(f"Task {task_id} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            # Record failure
            duration = asyncio.get_event_loop().time() - start_time
            TASK_DURATION.labels(queue=queue_name).observe(duration)
            TASKS_PROCESSED.labels(queue=queue_name, status="error").inc()
            
            logger.error(f"Task processing failed: {e}", exc_info=True)
            
        finally:
            ACTIVE_TASKS.labels(queue=queue_name).dec()
    
    async def _process_ingestion_task(self, task: dict):
        """Process file ingestion task."""
        logger.info("Processing ingestion task")
        
        from ..memory_ingestion.pipeline import get_memory_ingestion_pipeline
        from ..core.config import get_settings
        
        settings = get_settings()
        pipeline = get_memory_ingestion_pipeline(settings.vector_url)
        
        try:
            task_data = task.get('data', {})
            
            if 'file_path' in task_data:
                # File ingestion
                result = await pipeline.ingest_file(
                    file_path=task_data['file_path'],
                    session_id=task_data.get('session_id'),
                    user_id=task_data.get('user_id', 'system'),
                    tags=task_data.get('tags'),
                    trust_score=task_data.get('trust_score', 0.7)
                )
            elif 'text' in task_data:
                # Text ingestion
                result = await pipeline.ingest_text_content(
                    text=task_data['text'],
                    title=task_data.get('title', 'Text Content'),
                    session_id=task_data.get('session_id'),
                    user_id=task_data.get('user_id', 'system'),
                    tags=task_data.get('tags'),
                    trust_score=task_data.get('trust_score', 0.7)
                )
            else:
                logger.error("Ingestion task missing required data (file_path or text)")
                return
            
            logger.info(f"Ingestion task completed with status: {result['status']}")
            
            # Store result in Redis for retrieval
            if self.redis_client and 'task_id' in task:
                result_key = f"grace:task_result:{task['task_id']}"
                await self.redis_client.setex(result_key, 3600, str(result))  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Ingestion task failed: {e}", exc_info=True)
    
    async def _process_embeddings_task(self, task: dict):
        """Process embeddings generation task."""
        logger.info("Processing embeddings task")
        
        from ..memory_ingestion.embeddings import get_embedding_generator
        
        try:
            task_data = task.get('data', {})
            texts = task_data.get('texts', [])
            
            if not texts:
                logger.warning("Embeddings task has no texts to process")
                return
            
            generator = get_embedding_generator()
            embedding_results = await generator.generate_embeddings(texts)
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            
            # Store result in Redis for retrieval
            if self.redis_client and 'task_id' in task:
                result_key = f"grace:task_result:{task['task_id']}"
                result = {
                    'status': 'success',
                    'embeddings': embedding_results
                }
                await self.redis_client.setex(result_key, 3600, str(result))  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Embeddings task failed: {e}", exc_info=True)
    
    async def _process_media_task(self, task: dict):
        """Process media processing task."""
        logger.info("Processing media task")
        
        try:
            task_data = task.get('data', {})
            media_path = task_data.get('media_path')
            media_type = task_data.get('media_type', 'unknown')
            
            if not media_path:
                logger.warning("Media task missing media_path")
                return
            
            # Basic media processing placeholder
            # In a real implementation, this would handle:
            # - Image OCR for text extraction
            # - Audio transcription
            # - Video frame analysis
            # - Document conversion
            
            result = {
                'status': 'processed',
                'media_path': media_path,
                'media_type': media_type,
                'extracted_text': f"[Media content from {media_path} - processing placeholder]"
            }
            
            logger.info(f"Processed media file: {media_path}")
            
            # Store result in Redis for retrieval
            if self.redis_client and 'task_id' in task:
                result_key = f"grace:task_result:{task['task_id']}"
                await self.redis_client.setex(result_key, 3600, str(result))  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Media task failed: {e}", exc_info=True)
    
    async def _monitor_queues(self):
        """Monitor queue sizes and health."""
        logger.info("Starting queue monitoring")
        
        while self.running:
            try:
                for queue in self.queues:
                    queue_key = f"grace:queue:{queue}"
                    size = await self.redis_client.llen(queue_key)
                    QUEUE_SIZE.labels(queue=queue).set(size)
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Queue monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue monitoring: {e}", exc_info=True)
                await asyncio.sleep(5)


async def handle_shutdown(worker: GraceWorker):
    """Handle graceful shutdown signals."""
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker.stop())
    
    # Set up signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())


async def main():
    """Main entry point for the worker service."""
    setup_logging()
    setup_environment()
    
    logger.info("Starting Grace Worker Service...")
    
    worker = GraceWorker()
    
    # Set up graceful shutdown
    await handle_shutdown(worker)
    
    try:
        await worker.start()
        
        # Keep running until shutdown
        while worker.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker service error: {e}", exc_info=True)
    finally:
        await worker.stop()
        logger.info("Grace Worker Service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())