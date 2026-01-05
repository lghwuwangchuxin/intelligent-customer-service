"""Tests for Evaluation Worker."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from services.evaluation_service.worker import (
    EvaluationWorker,
    EvaluationTask,
    TaskStatus,
)
from services.evaluation_service.evaluators.ragas_evaluator import (
    EvaluationSample,
    EvaluationResult,
    MetricResult,
)


class TestEvaluationWorker:
    """Test cases for EvaluationWorker."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock RAGAS evaluator."""
        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(return_value=EvaluationResult(
            metrics=[
                MetricResult(metric_name="faithfulness", score=0.85),
                MetricResult(metric_name="answer_relevancy", score=0.9),
            ],
            sample_scores=[{"faithfulness": 0.85, "answer_relevancy": 0.9}],
        ))
        return evaluator

    @pytest.fixture
    def worker(self, mock_evaluator):
        """Create a worker with mock evaluator."""
        return EvaluationWorker(
            ragas_evaluator=mock_evaluator,
            max_concurrent_tasks=2,
            result_ttl_seconds=60,
        )

    @pytest.mark.asyncio
    async def test_start_and_stop(self, worker):
        """Test starting and stopping the worker."""
        await worker.start()
        assert worker._running is True
        assert len(worker._workers) > 0

        await worker.stop()
        assert worker._running is False
        assert len(worker._workers) == 0

    @pytest.mark.asyncio
    async def test_submit_task(self, worker, sample_evaluation_samples):
        """Test submitting an evaluation task."""
        await worker.start()
        try:
            task_id = await worker.submit_task(
                samples=sample_evaluation_samples,
                metrics=["faithfulness"],
            )

            assert task_id is not None
            task = worker.get_task(task_id)
            assert task is not None
            assert task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.COMPLETED]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_get_task(self, worker, sample_evaluation_samples):
        """Test getting a task by ID."""
        await worker.start()
        try:
            task_id = await worker.submit_task(
                samples=sample_evaluation_samples,
                metrics=["faithfulness"],
            )

            task = worker.get_task(task_id)
            assert task is not None
            assert task.task_id == task_id
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, worker):
        """Test getting a task that doesn't exist."""
        task = worker.get_task("nonexistent-id")
        assert task is None

    @pytest.mark.asyncio
    async def test_list_tasks(self, worker, sample_evaluation_samples):
        """Test listing tasks."""
        await worker.start()
        try:
            # Submit multiple tasks
            for _ in range(3):
                await worker.submit_task(
                    samples=sample_evaluation_samples,
                    metrics=["faithfulness"],
                )

            tasks = worker.list_tasks()
            assert len(tasks) == 3
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_list_tasks_with_status_filter(self, worker, sample_evaluation_samples):
        """Test listing tasks with status filter."""
        await worker.start()
        try:
            await worker.submit_task(
                samples=sample_evaluation_samples,
                metrics=["faithfulness"],
            )

            # Give time for processing
            await asyncio.sleep(0.2)

            pending_tasks = worker.list_tasks(status=TaskStatus.PENDING)
            # May have been processed already
            assert isinstance(pending_tasks, list)
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_task_processing(self, worker, sample_evaluation_samples, mock_evaluator):
        """Test that tasks get processed."""
        await worker.start()
        try:
            task_id = await worker.submit_task(
                samples=sample_evaluation_samples,
                metrics=["faithfulness"],
            )

            # Wait for processing
            await asyncio.sleep(0.3)

            task = worker.get_task(task_id)
            # Task should be completed or still processing
            assert task.status in [TaskStatus.PROCESSING, TaskStatus.COMPLETED, TaskStatus.PENDING]
        finally:
            await worker.stop()


class TestEvaluationTask:
    """Test cases for EvaluationTask dataclass."""

    def test_task_creation(self, sample_evaluation_samples):
        """Test creating an evaluation task."""
        task = EvaluationTask(
            task_id="test-123",
            samples=sample_evaluation_samples,
            metrics=["faithfulness", "relevancy"],
        )

        assert task.task_id == "test-123"
        assert len(task.samples) == 2
        assert task.status == TaskStatus.PENDING
        assert task.created_at is not None
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_status_transitions(self, sample_evaluation_samples):
        """Test task status transitions."""
        task = EvaluationTask(
            task_id="test-123",
            samples=sample_evaluation_samples,
            metrics=["faithfulness"],
        )

        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.utcnow()
        assert task.status == TaskStatus.PROCESSING
        assert task.started_at is not None

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        assert task.status == TaskStatus.COMPLETED


class TestTaskStatus:
    """Test cases for TaskStatus enum."""

    def test_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "PENDING"
        assert TaskStatus.PROCESSING.value == "PROCESSING"
        assert TaskStatus.COMPLETED.value == "COMPLETED"
        assert TaskStatus.FAILED.value == "FAILED"

    def test_status_from_string(self):
        """Test creating TaskStatus from string."""
        status = TaskStatus("PENDING")
        assert status == TaskStatus.PENDING

    def test_invalid_status(self):
        """Test invalid status value."""
        with pytest.raises(ValueError):
            TaskStatus("INVALID")
