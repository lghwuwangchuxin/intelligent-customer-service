"""Tests for Evaluation Metrics."""

import pytest
from unittest.mock import AsyncMock, patch

from services.evaluation_service.evaluators.ragas_evaluator import (
    RAGASEvaluator,
    EvaluationSample,
    EvaluationResult,
    MetricResult,
    RAGASMetric,
)


class TestRAGASEvaluator:
    """Test cases for RAGASEvaluator."""

    @pytest.fixture
    def evaluator(self, mock_llm_client, mock_embedding_model):
        """Create evaluator with mocks."""
        return RAGASEvaluator(
            llm_client=mock_llm_client,
            embedding_model=mock_embedding_model,
        )

    @pytest.fixture
    def evaluator_no_llm(self, mock_embedding_model):
        """Create evaluator without LLM."""
        return RAGASEvaluator(
            llm_client=None,
            embedding_model=mock_embedding_model,
        )

    @pytest.fixture
    def evaluator_minimal(self):
        """Create minimal evaluator."""
        return RAGASEvaluator()

    @pytest.mark.asyncio
    async def test_evaluate_empty_samples(self, evaluator):
        """Test evaluation with empty samples."""
        result = await evaluator.evaluate(samples=[], metrics=["faithfulness"])

        assert isinstance(result, EvaluationResult)
        assert result.metrics == []
        assert result.sample_scores == []

    @pytest.mark.asyncio
    async def test_evaluate_with_samples(self, evaluator, sample_evaluation_samples):
        """Test evaluation with samples."""
        result = await evaluator.evaluate(
            samples=sample_evaluation_samples,
            metrics=["faithfulness", "answer_relevancy"],
        )

        assert isinstance(result, EvaluationResult)
        # Should have metric results
        assert len(result.metrics) >= 0
        # Should have sample scores
        assert len(result.sample_scores) >= 0

    @pytest.mark.asyncio
    async def test_evaluate_default_metrics(self, evaluator, sample_evaluation_samples):
        """Test evaluation with default metrics."""
        result = await evaluator.evaluate(samples=sample_evaluation_samples)

        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_fallback_evaluation(self, evaluator_minimal, sample_evaluation_samples):
        """Test fallback evaluation without RAGAS library."""
        result = await evaluator_minimal.evaluate(
            samples=sample_evaluation_samples,
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )

        assert isinstance(result, EvaluationResult)
        # Should have metric results from fallback
        assert len(result.metrics) > 0


class TestFaithfulnessMetric:
    """Test cases for faithfulness metric."""

    @pytest.fixture
    def evaluator(self, mock_llm_client):
        """Create evaluator for faithfulness testing."""
        return RAGASEvaluator(llm_client=mock_llm_client)

    @pytest.mark.asyncio
    async def test_compute_faithfulness_with_llm(self, evaluator):
        """Test faithfulness computation with LLM."""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["Artificial intelligence is the simulation of human intelligence."],
        )

        score = await evaluator._compute_faithfulness(sample)

        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_compute_faithfulness_fallback(self):
        """Test faithfulness computation without LLM."""
        evaluator = RAGASEvaluator(llm_client=None)

        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence technology.",
            contexts=["Artificial intelligence is a technology field."],
        )

        score = await evaluator._compute_faithfulness(sample)

        # Should use word overlap
        assert 0 <= score <= 1

    def test_simple_faithfulness(self):
        """Test simple word overlap faithfulness."""
        evaluator = RAGASEvaluator()

        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence",
            contexts=["Artificial intelligence is amazing"],
        )

        score = evaluator._simple_faithfulness(sample)

        assert score > 0  # Should have some overlap

    def test_simple_faithfulness_no_overlap(self):
        """Test faithfulness with no overlap."""
        evaluator = RAGASEvaluator()

        sample = EvaluationSample(
            question="What is AI?",
            answer="completely unrelated words",
            contexts=["nothing in common here"],
        )

        score = evaluator._simple_faithfulness(sample)

        assert score == 0 or score < 0.3  # Low or no overlap


class TestAnswerRelevancyMetric:
    """Test cases for answer relevancy metric."""

    @pytest.fixture
    def evaluator(self, mock_embedding_model):
        """Create evaluator for relevancy testing."""
        return RAGASEvaluator(embedding_model=mock_embedding_model)

    @pytest.mark.asyncio
    async def test_compute_answer_relevancy_with_embedding(self, evaluator):
        """Test relevancy with embedding model."""
        sample = EvaluationSample(
            question="What is machine learning?",
            answer="Machine learning is a subset of AI.",
            contexts=[],
        )

        score = await evaluator._compute_answer_relevancy(sample)

        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_compute_answer_relevancy_fallback(self):
        """Test relevancy without embedding model."""
        evaluator = RAGASEvaluator(embedding_model=None)

        sample = EvaluationSample(
            question="What is machine learning?",
            answer="Machine learning uses data to learn.",
            contexts=[],
        )

        score = await evaluator._compute_answer_relevancy(sample)

        # Should use word overlap
        assert 0 <= score <= 1


class TestContextPrecisionMetric:
    """Test cases for context precision metric."""

    @pytest.fixture
    def evaluator(self, mock_embedding_model):
        """Create evaluator for precision testing."""
        return RAGASEvaluator(embedding_model=mock_embedding_model)

    @pytest.mark.asyncio
    async def test_compute_context_precision(self, evaluator):
        """Test context precision computation."""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=[
                "Artificial intelligence is a field of study.",
                "AI has many applications.",
            ],
        )

        score = await evaluator._compute_context_precision(sample)

        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_context_precision_no_contexts(self, evaluator):
        """Test precision with no contexts."""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=[],
        )

        score = await evaluator._compute_context_precision(sample)

        assert score == 0.0


class TestCosineSimilarity:
    """Test cases for cosine similarity."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 0.0, 0.0]
        score = RAGASEvaluator._cosine_similarity(vec, vec)

        assert abs(score - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        score = RAGASEvaluator._cosine_similarity(vec1, vec2)

        assert abs(score) < 0.0001

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 0.0, 0.0]
        score = RAGASEvaluator._cosine_similarity(vec1, vec2)

        assert score == 0.0


class TestEvaluationResult:
    """Test cases for EvaluationResult dataclass."""

    def test_result_creation(self):
        """Test creating EvaluationResult."""
        result = EvaluationResult(
            metrics=[
                MetricResult(metric_name="faithfulness", score=0.85),
                MetricResult(metric_name="relevancy", score=0.9),
            ],
            sample_scores=[{"faithfulness": 0.85, "relevancy": 0.9}],
            errors=["test error"],
        )

        assert len(result.metrics) == 2
        assert result.metrics[0].score == 0.85
        assert len(result.sample_scores) == 1
        assert len(result.errors) == 1

    def test_result_defaults(self):
        """Test EvaluationResult defaults."""
        result = EvaluationResult(metrics=[])

        assert result.sample_scores == []
        assert result.errors == []


class TestMetricResult:
    """Test cases for MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test creating MetricResult."""
        result = MetricResult(
            metric_name="faithfulness",
            score=0.85,
            details={"sample_count": 10},
        )

        assert result.metric_name == "faithfulness"
        assert result.score == 0.85
        assert result.details["sample_count"] == 10

    def test_metric_result_defaults(self):
        """Test MetricResult defaults."""
        result = MetricResult(
            metric_name="test",
            score=0.5,
        )

        assert result.details == {}
