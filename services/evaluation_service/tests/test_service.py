"""Tests for Evaluation Service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.evaluation_service.service import (
    EvaluationService,
    EvaluationData,
)
from services.evaluation_service.worker import TaskStatus


class TestEvaluationService:
    """Test cases for EvaluationService."""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock worker."""
        worker = MagicMock()
        worker._running = True
        worker.start = AsyncMock()
        worker.stop = AsyncMock()
        worker.submit_task = AsyncMock(return_value="task-123")
        worker.get_task = MagicMock(return_value=None)
        worker.list_tasks = MagicMock(return_value=[])
        return worker

    @pytest.fixture
    def service(self, mock_worker, mock_llm_client, mock_embedding_model):
        """Create an evaluation service with mocks."""
        return EvaluationService(
            worker=mock_worker,
            llm_client=mock_llm_client,
            embedding_model=mock_embedding_model,
        )

    @pytest.fixture
    def evaluation_data(self):
        """Create sample evaluation data."""
        return [
            EvaluationData(
                question="What is AI?",
                answer="AI is artificial intelligence.",
                contexts=["Artificial intelligence is a field of study."],
                ground_truth="AI is the simulation of human intelligence.",
            ),
        ]

    @pytest.mark.asyncio
    async def test_submit_evaluation(self, service, evaluation_data, mock_worker):
        """Test submitting an evaluation."""
        result = await service.submit_evaluation(
            data=evaluation_data,
            metrics=["faithfulness"],
        )

        assert result["task_id"] == "task-123"
        assert result["status"] == "PENDING"
        mock_worker.submit_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_evaluation_default_metrics(self, service, evaluation_data, mock_worker):
        """Test submitting evaluation with default metrics."""
        result = await service.submit_evaluation(data=evaluation_data)

        assert result["task_id"] is not None
        # Should use default metrics
        mock_worker.submit_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_result_not_found(self, service, mock_worker):
        """Test getting result for non-existent task."""
        mock_worker.get_task.return_value = None

        result = await service.get_result("nonexistent-task")

        assert result["status"] == "NOT_FOUND"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_result_found(self, service, mock_worker):
        """Test getting result for existing task."""
        from datetime import datetime
        from services.evaluation_service.worker import EvaluationTask

        mock_task = MagicMock()
        mock_task.task_id = "task-123"
        mock_task.status = TaskStatus.COMPLETED
        mock_task.created_at = datetime.utcnow()
        mock_task.started_at = datetime.utcnow()
        mock_task.completed_at = datetime.utcnow()
        mock_task.result = {
            "metrics": [{"metric_name": "faithfulness", "score": 0.85}],
            "sample_scores": [{"faithfulness": 0.85}],
        }
        mock_task.error = None

        mock_worker.get_task.return_value = mock_task

        result = await service.get_result("task-123")

        assert result["task_id"] == "task-123"
        assert result["status"] == "COMPLETED"
        assert "results" in result

    @pytest.mark.asyncio
    async def test_list_evaluations(self, service, mock_worker):
        """Test listing evaluations."""
        result = await service.list_evaluations(page=1, page_size=10)

        assert "evaluations" in result
        assert "total" in result
        assert "page" in result

    @pytest.mark.asyncio
    async def test_list_evaluations_with_status(self, service, mock_worker):
        """Test listing evaluations with status filter."""
        result = await service.list_evaluations(
            status="COMPLETED",
            page=1,
            page_size=10,
        )

        assert "evaluations" in result
        mock_worker.list_tasks.assert_called()

    @pytest.mark.asyncio
    async def test_health_check(self, service, mock_worker):
        """Test health check."""
        mock_worker.list_tasks.return_value = []

        result = await service.health_check()

        assert result["status"] == "healthy"
        assert "worker_running" in result
        assert "pending_tasks" in result

    @pytest.mark.asyncio
    async def test_shutdown(self, service, mock_worker):
        """Test service shutdown."""
        service._initialized = True

        await service.shutdown()

        mock_worker.stop.assert_called_once()
        assert service._initialized is False


class TestEvaluationData:
    """Test cases for EvaluationData dataclass."""

    def test_data_creation(self):
        """Test creating EvaluationData."""
        data = EvaluationData(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["Context 1", "Context 2"],
            ground_truth="Expected answer",
        )

        assert data.question == "What is AI?"
        assert data.answer == "AI is artificial intelligence."
        assert len(data.contexts) == 2
        assert data.ground_truth == "Expected answer"

    def test_data_without_ground_truth(self):
        """Test EvaluationData without ground truth."""
        data = EvaluationData(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["Context"],
        )

        assert data.ground_truth is None


class TestEvaluationServiceIntegration:
    """Integration tests for Evaluation Service."""

    @pytest.fixture
    def full_service(self, mock_llm_client, mock_embedding_model):
        """Create a fully initialized service."""
        return EvaluationService(
            llm_client=mock_llm_client,
            embedding_model=mock_embedding_model,
        )

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self, full_service):
        """Test complete evaluation flow."""
        data = [
            EvaluationData(
                question="What is machine learning?",
                answer="ML is a type of AI that learns from data.",
                contexts=["Machine learning uses algorithms to learn from data."],
            ),
        ]

        # Submit evaluation
        submit_result = await full_service.submit_evaluation(
            data=data,
            metrics=["faithfulness"],
        )

        assert "task_id" in submit_result
        assert submit_result["status"] == "PENDING"

        # Get result (may not be ready yet)
        result = await full_service.get_result(submit_result["task_id"])
        assert "task_id" in result

    @pytest.mark.asyncio
    async def test_sync_evaluation(self, full_service):
        """Test synchronous evaluation."""
        data = [
            EvaluationData(
                question="What is AI?",
                answer="AI is artificial intelligence.",
                contexts=["Artificial intelligence is a field of study."],
            ),
        ]

        result = await full_service.evaluate_sync(
            data=data,
            metrics=["faithfulness", "answer_relevancy"],
        )

        assert "metrics" in result
        assert "sample_scores" in result
        assert len(result["metrics"]) > 0
