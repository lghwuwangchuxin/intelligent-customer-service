"""Async evaluation worker for background task processing."""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from services.common.logging import get_logger
from .evaluators import RAGASEvaluator, CustomMetrics
from .evaluators.ragas_evaluator import EvaluationSample

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Evaluation task status."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class EvaluationTask:
    """Evaluation task."""
    task_id: str
    samples: List[EvaluationSample]
    metrics: List[str]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EvaluationWorker:
    """
    Background worker for processing evaluation tasks.

    Features:
    - Async task queue processing
    - Concurrent task execution
    - Result storage and retrieval
    """

    def __init__(
        self,
        ragas_evaluator: Optional[RAGASEvaluator] = None,
        custom_metrics: Optional[CustomMetrics] = None,
        max_concurrent_tasks: int = 3,
        result_ttl_seconds: int = 3600,
    ):
        """
        Initialize evaluation worker.

        Args:
            ragas_evaluator: RAGAS evaluator instance
            custom_metrics: Custom metrics instance
            max_concurrent_tasks: Maximum concurrent tasks
            result_ttl_seconds: TTL for stored results
        """
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.custom_metrics = custom_metrics or CustomMetrics()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.result_ttl_seconds = result_ttl_seconds

        # Task queue and storage
        self._task_queue: asyncio.Queue[EvaluationTask] = asyncio.Queue()
        self._tasks: Dict[str, EvaluationTask] = {}
        self._running = False
        self._workers: List[asyncio.Task] = []

    async def start(self):
        """Start the worker."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting evaluation worker with {self.max_concurrent_tasks} workers")

        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker_task)

        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._workers.append(cleanup_task)

    async def stop(self):
        """Stop the worker."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping evaluation worker")

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit_task(
        self,
        samples: List[EvaluationSample],
        metrics: List[str],
    ) -> str:
        """
        Submit evaluation task.

        Args:
            samples: Evaluation samples
            metrics: Metrics to compute

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = EvaluationTask(
            task_id=task_id,
            samples=samples,
            metrics=metrics,
        )

        self._tasks[task_id] = task
        await self._task_queue.put(task)

        logger.info(f"Submitted evaluation task: {task_id}")
        return task_id

    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """
        Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            EvaluationTask if found
        """
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> List[EvaluationTask]:
        """
        List tasks.

        Args:
            status: Filter by status
            limit: Maximum tasks to return

        Returns:
            List of tasks
        """
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    async def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process task
                await self._process_task(task, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} stopped")

    async def _process_task(self, task: EvaluationTask, worker_id: int):
        """Process a single evaluation task."""
        logger.info(f"Worker {worker_id} processing task: {task.task_id}")

        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.utcnow()

        try:
            # Run RAGAS evaluation
            ragas_result = await self.ragas_evaluator.evaluate(
                samples=task.samples,
                metrics=task.metrics,
            )

            # Build result
            task.result = {
                "metrics": [
                    {
                        "metric_name": m.metric_name,
                        "score": m.score,
                        "details": m.details,
                    }
                    for m in ragas_result.metrics
                ],
                "sample_scores": ragas_result.sample_scores,
                "errors": ragas_result.errors,
            }

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            logger.info(f"Task {task.task_id} completed")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()

    async def _cleanup_loop(self):
        """Cleanup old tasks periodically."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                expired_ids = []

                for task_id, task in self._tasks.items():
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > self.result_ttl_seconds:
                            expired_ids.append(task_id)

                for task_id in expired_ids:
                    del self._tasks[task_id]

                if expired_ids:
                    logger.info(f"Cleaned up {len(expired_ids)} expired tasks")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


class RedisBackedWorker(EvaluationWorker):
    """
    Evaluation worker backed by Redis for persistence.

    Use this for production deployments with multiple instances.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ):
        """
        Initialize Redis-backed worker.

        Args:
            redis_url: Redis connection URL
            **kwargs: Additional arguments for base worker
        """
        super().__init__(**kwargs)
        self.redis_url = redis_url
        self._redis = None

    async def _connect_redis(self):
        """Connect to Redis."""
        if self._redis:
            return

        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis: {self.redis_url}")
        except ImportError:
            logger.warning("redis package not installed, using in-memory storage")

    async def submit_task(
        self,
        samples: List[EvaluationSample],
        metrics: List[str],
    ) -> str:
        """Submit task with Redis persistence."""
        await self._connect_redis()

        task_id = await super().submit_task(samples, metrics)

        # Store in Redis
        if self._redis:
            import json
            task = self._tasks[task_id]
            task_data = {
                "task_id": task.task_id,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "metrics": task.metrics,
                "samples": [
                    {
                        "question": s.question,
                        "answer": s.answer,
                        "contexts": s.contexts,
                        "ground_truth": s.ground_truth,
                    }
                    for s in task.samples
                ],
            }
            await self._redis.setex(
                f"eval_task:{task_id}",
                self.result_ttl_seconds,
                json.dumps(task_data),
            )

        return task_id

    async def _process_task(self, task: EvaluationTask, worker_id: int):
        """Process task with Redis updates."""
        await super()._process_task(task, worker_id)

        # Update in Redis
        if self._redis:
            import json
            task_data = {
                "task_id": task.task_id,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "result": task.result,
                "error": task.error,
            }
            await self._redis.setex(
                f"eval_task:{task.task_id}",
                self.result_ttl_seconds,
                json.dumps(task_data),
            )
