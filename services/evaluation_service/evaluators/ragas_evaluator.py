"""RAGAS Evaluator for RAG system evaluation."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from services.common.logging import get_logger

logger = get_logger(__name__)


class RAGASMetric(Enum):
    """RAGAS evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_RELEVANCY = "context_relevancy"
    ANSWER_CORRECTNESS = "answer_correctness"
    ANSWER_SIMILARITY = "answer_similarity"


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class MetricResult:
    """Result for a single metric."""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    metrics: List[MetricResult]
    sample_scores: List[Dict[str, float]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG systems.

    Metrics:
    - Faithfulness: How factually accurate is the answer based on contexts
    - Answer Relevancy: How relevant is the answer to the question
    - Context Precision: How precise are the retrieved contexts
    - Context Recall: How complete are the retrieved contexts
    - Answer Correctness: Correctness compared to ground truth
    """

    def __init__(
        self,
        llm_client=None,
        embedding_model=None,
        default_metrics: Optional[List[str]] = None,
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            llm_client: LLM client for evaluation
            embedding_model: Embedding model for similarity metrics
            default_metrics: Default metrics to compute
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.default_metrics = default_metrics or [
            RAGASMetric.FAITHFULNESS.value,
            RAGASMetric.ANSWER_RELEVANCY.value,
            RAGASMetric.CONTEXT_PRECISION.value,
        ]

        self._ragas = None
        self._initialized = False

    async def initialize(self):
        """Initialize RAGAS library."""
        if self._initialized:
            return

        try:
            # Try to import ragas
            import ragas
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            self._ragas = ragas
            self._initialized = True
            logger.info("RAGAS library initialized")
        except ImportError:
            logger.warning("RAGAS library not available, using fallback evaluation")
            self._initialized = True

    async def evaluate(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG samples.

        Args:
            samples: Evaluation samples
            metrics: Metrics to compute

        Returns:
            EvaluationResult with scores
        """
        await self.initialize()

        metrics = metrics or self.default_metrics
        result = EvaluationResult(metrics=[], sample_scores=[], errors=[])

        if not samples:
            return result

        # Try RAGAS evaluation first
        if self._ragas:
            try:
                return await self._evaluate_with_ragas(samples, metrics)
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed, using fallback: {e}")
                result.errors.append(f"RAGAS error: {str(e)}")

        # Fallback to custom evaluation
        return await self._evaluate_fallback(samples, metrics)

    async def _evaluate_with_ragas(
        self,
        samples: List[EvaluationSample],
        metrics: List[str],
    ) -> EvaluationResult:
        """Evaluate using RAGAS library."""
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Prepare dataset
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }

        if any(s.ground_truth for s in samples):
            data["ground_truth"] = [s.ground_truth or "" for s in samples]

        dataset = Dataset.from_dict(data)

        # Map metric names to RAGAS metrics
        metric_map = {
            RAGASMetric.FAITHFULNESS.value: faithfulness,
            RAGASMetric.ANSWER_RELEVANCY.value: answer_relevancy,
            RAGASMetric.CONTEXT_PRECISION.value: context_precision,
            RAGASMetric.CONTEXT_RECALL.value: context_recall,
        }

        ragas_metrics = [
            metric_map[m] for m in metrics
            if m in metric_map
        ]

        if not ragas_metrics:
            ragas_metrics = [faithfulness, answer_relevancy]

        # Run evaluation
        loop = asyncio.get_event_loop()
        result_df = await loop.run_in_executor(
            None,
            lambda: evaluate(dataset, metrics=ragas_metrics)
        )

        # Build result
        metric_results = []
        for metric_name in metrics:
            if metric_name in result_df:
                score = float(result_df[metric_name])
                metric_results.append(MetricResult(
                    metric_name=metric_name,
                    score=score,
                ))

        return EvaluationResult(
            metrics=metric_results,
            sample_scores=[dict(row) for row in result_df.to_dict(orient='records')],
        )

    async def _evaluate_fallback(
        self,
        samples: List[EvaluationSample],
        metrics: List[str],
    ) -> EvaluationResult:
        """Fallback evaluation without RAGAS."""
        metric_results = []
        sample_scores = []

        for sample in samples:
            scores = {}

            # Faithfulness (simple check)
            if RAGASMetric.FAITHFULNESS.value in metrics:
                score = await self._compute_faithfulness(sample)
                scores[RAGASMetric.FAITHFULNESS.value] = score

            # Answer Relevancy
            if RAGASMetric.ANSWER_RELEVANCY.value in metrics:
                score = await self._compute_answer_relevancy(sample)
                scores[RAGASMetric.ANSWER_RELEVANCY.value] = score

            # Context Precision
            if RAGASMetric.CONTEXT_PRECISION.value in metrics:
                score = await self._compute_context_precision(sample)
                scores[RAGASMetric.CONTEXT_PRECISION.value] = score

            sample_scores.append(scores)

        # Aggregate scores
        for metric in metrics:
            scores = [s.get(metric, 0) for s in sample_scores if metric in s]
            if scores:
                avg_score = sum(scores) / len(scores)
                metric_results.append(MetricResult(
                    metric_name=metric,
                    score=avg_score,
                    details={"sample_count": len(scores)},
                ))

        return EvaluationResult(
            metrics=metric_results,
            sample_scores=sample_scores,
        )

    async def _compute_faithfulness(self, sample: EvaluationSample) -> float:
        """Compute faithfulness score using LLM."""
        if not self.llm_client:
            return self._simple_faithfulness(sample)

        prompt = f"""Rate how faithfully the answer is supported by the contexts on a scale of 0 to 1.
0 means completely unsupported, 1 means fully supported.
Only respond with a number.

Question: {sample.question}

Contexts:
{chr(10).join(sample.contexts[:3])}

Answer: {sample.answer}

Faithfulness score (0-1):"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=5)
            score = float(response.strip())
            return min(max(score, 0), 1)
        except Exception as e:
            logger.warning(f"LLM faithfulness evaluation failed: {e}")
            return self._simple_faithfulness(sample)

    def _simple_faithfulness(self, sample: EvaluationSample) -> float:
        """Simple faithfulness using word overlap."""
        answer_words = set(sample.answer.lower().split())
        context_words = set()
        for ctx in sample.contexts:
            context_words.update(ctx.lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)

    async def _compute_answer_relevancy(self, sample: EvaluationSample) -> float:
        """Compute answer relevancy using embedding similarity."""
        if self.embedding_model:
            try:
                q_emb = await self.embedding_model.embed(sample.question)
                a_emb = await self.embedding_model.embed(sample.answer)
                return self._cosine_similarity(q_emb, a_emb)
            except Exception as e:
                logger.warning(f"Embedding relevancy failed: {e}")

        # Fallback to word overlap
        q_words = set(sample.question.lower().split())
        a_words = set(sample.answer.lower().split())

        if not q_words:
            return 0.0

        overlap = len(q_words & a_words)
        return overlap / len(q_words)

    async def _compute_context_precision(self, sample: EvaluationSample) -> float:
        """Compute context precision."""
        if not sample.contexts:
            return 0.0

        if self.embedding_model:
            try:
                q_emb = await self.embedding_model.embed(sample.question)
                ctx_embs = await self.embedding_model.embed_batch(sample.contexts)

                # Average similarity of contexts to question
                sims = [self._cosine_similarity(q_emb, c_emb) for c_emb in ctx_embs]
                return sum(sims) / len(sims)
            except Exception as e:
                logger.warning(f"Embedding precision failed: {e}")

        # Fallback
        q_words = set(sample.question.lower().split())
        total_overlap = 0
        for ctx in sample.contexts:
            ctx_words = set(ctx.lower().split())
            if q_words:
                total_overlap += len(q_words & ctx_words) / len(q_words)

        return total_overlap / len(sample.contexts) if sample.contexts else 0.0

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
