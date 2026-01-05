"""Evaluation Service - Main service implementation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from services.common.logging import get_logger
from services.common.config import get_service_config

from .evaluators import RAGASEvaluator, CustomMetrics
from .evaluators.ragas_evaluator import EvaluationSample
from .worker import EvaluationWorker, TaskStatus

logger = get_logger(__name__)


@dataclass
class EvaluationData:
    """Evaluation data for submission."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class EvaluationService:
    """
    Evaluation Service for async RAG evaluation.

    Features:
    - Async task submission
    - Background evaluation processing
    - Multiple metrics (RAGAS + custom)
    - Result retrieval
    """

    def __init__(
        self,
        worker: Optional[EvaluationWorker] = None,
        llm_client=None,
        embedding_model=None,
    ):
        """
        Initialize evaluation service.

        Args:
            worker: Evaluation worker instance
            llm_client: LLM client for evaluation
            embedding_model: Embedding model for similarity metrics
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model

        # Initialize evaluators
        self.ragas_evaluator = RAGASEvaluator(
            llm_client=llm_client,
            embedding_model=embedding_model,
        )
        self.custom_metrics = CustomMetrics(llm_client=llm_client)

        # Initialize worker
        self.worker = worker or EvaluationWorker(
            ragas_evaluator=self.ragas_evaluator,
            custom_metrics=self.custom_metrics,
        )

        self._initialized = False

    @classmethod
    def from_config(cls, config=None) -> "EvaluationService":
        """
        Create service from configuration.

        Args:
            config: Service configuration

        Returns:
            EvaluationService instance
        """
        if config is None:
            config = get_service_config("evaluation-service")

        # Initialize LLM client if configured
        llm_client = None
        if config.llm_base_url:
            from services.rag_service.pipeline.query_transform import SimpleLLMClient
            llm_client = SimpleLLMClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
            )

        return cls(llm_client=llm_client)

    async def initialize(self):
        """Initialize service and start worker."""
        if self._initialized:
            return

        await self.worker.start()
        self._initialized = True
        logger.info("Evaluation Service initialized")

    async def shutdown(self):
        """Shutdown service and stop worker."""
        await self.worker.stop()
        self._initialized = False
        logger.info("Evaluation Service shutdown")

    async def submit_evaluation(
        self,
        data: List[EvaluationData],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Submit evaluation task.

        Args:
            data: Evaluation data
            metrics: Metrics to compute

        Returns:
            Dictionary with task_id and status
        """
        await self.initialize()

        # Convert to samples
        samples = [
            EvaluationSample(
                question=d.question,
                answer=d.answer,
                contexts=d.contexts,
                ground_truth=d.ground_truth,
            )
            for d in data
        ]

        # Default metrics
        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy", "context_precision"]

        # Submit task
        task_id = await self.worker.submit_task(samples, metrics)

        return {
            "task_id": task_id,
            "status": "PENDING",
        }

    async def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get evaluation result.

        Args:
            task_id: Task ID

        Returns:
            Result dictionary
        """
        task = self.worker.get_task(task_id)

        if not task:
            return {
                "task_id": task_id,
                "status": "NOT_FOUND",
                "error": "Task not found",
            }

        result = {
            "task_id": task.task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
        }

        if task.started_at:
            result["started_at"] = task.started_at.isoformat()

        if task.completed_at:
            result["completed_at"] = task.completed_at.isoformat()

        if task.result:
            result["results"] = task.result.get("metrics", [])
            result["sample_scores"] = task.result.get("sample_scores", [])

        if task.error:
            result["error"] = task.error

        return result

    async def list_evaluations(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """
        List evaluations.

        Args:
            status: Filter by status
            page: Page number
            page_size: Page size

        Returns:
            Dictionary with evaluations list
        """
        task_status = None
        if status:
            try:
                task_status = TaskStatus(status)
            except ValueError:
                pass

        tasks = self.worker.list_tasks(status=task_status, limit=page * page_size)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_tasks = tasks[start:end]

        return {
            "evaluations": [
                {
                    "task_id": t.task_id,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat(),
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                }
                for t in page_tasks
            ],
            "total": len(tasks),
            "page": page,
            "page_size": page_size,
        }

    async def evaluate_sync(
        self,
        data: List[EvaluationData],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous evaluation (for testing).

        Args:
            data: Evaluation data
            metrics: Metrics to compute

        Returns:
            Evaluation results
        """
        samples = [
            EvaluationSample(
                question=d.question,
                answer=d.answer,
                contexts=d.contexts,
                ground_truth=d.ground_truth,
            )
            for d in data
        ]

        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy", "context_precision"]

        result = await self.ragas_evaluator.evaluate(samples, metrics)

        return {
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "score": m.score,
                    "details": m.details,
                }
                for m in result.metrics
            ],
            "sample_scores": result.sample_scores,
            "errors": result.errors,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status
        """
        pending_count = len(self.worker.list_tasks(status=TaskStatus.PENDING))
        processing_count = len(self.worker.list_tasks(status=TaskStatus.PROCESSING))

        return {
            "status": "healthy",
            "worker_running": self.worker._running,
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
        }
