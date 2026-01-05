"""Custom evaluation metrics."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Result for a single metric."""
    metric_name: str
    score: float
    details: Dict[str, Any]


class CustomMetrics:
    """
    Custom evaluation metrics for RAG systems.

    Includes:
    - Latency metrics
    - Response quality metrics
    - Retrieval metrics
    """

    def __init__(self, llm_client=None):
        """
        Initialize custom metrics.

        Args:
            llm_client: LLM client for LLM-based metrics
        """
        self.llm_client = llm_client

    async def compute_response_length(
        self,
        responses: List[str],
    ) -> MetricResult:
        """
        Compute response length statistics.

        Args:
            responses: List of responses

        Returns:
            MetricResult with length stats
        """
        if not responses:
            return MetricResult(
                metric_name="response_length",
                score=0,
                details={"avg": 0, "min": 0, "max": 0},
            )

        lengths = [len(r) for r in responses]
        avg_length = sum(lengths) / len(lengths)

        return MetricResult(
            metric_name="response_length",
            score=avg_length,
            details={
                "avg": avg_length,
                "min": min(lengths),
                "max": max(lengths),
                "total_responses": len(responses),
            },
        )

    async def compute_latency_stats(
        self,
        latencies_ms: List[float],
    ) -> MetricResult:
        """
        Compute latency statistics.

        Args:
            latencies_ms: List of latencies in milliseconds

        Returns:
            MetricResult with latency stats
        """
        if not latencies_ms:
            return MetricResult(
                metric_name="latency",
                score=0,
                details={},
            )

        sorted_latencies = sorted(latencies_ms)
        p50_idx = len(sorted_latencies) // 2
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        return MetricResult(
            metric_name="latency",
            score=sum(latencies_ms) / len(latencies_ms),
            details={
                "avg_ms": sum(latencies_ms) / len(latencies_ms),
                "p50_ms": sorted_latencies[p50_idx] if sorted_latencies else 0,
                "p95_ms": sorted_latencies[p95_idx] if len(sorted_latencies) > p95_idx else sorted_latencies[-1] if sorted_latencies else 0,
                "p99_ms": sorted_latencies[p99_idx] if len(sorted_latencies) > p99_idx else sorted_latencies[-1] if sorted_latencies else 0,
                "min_ms": min(latencies_ms),
                "max_ms": max(latencies_ms),
            },
        )

    async def compute_retrieval_precision_at_k(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5,
    ) -> MetricResult:
        """
        Compute Precision@K for retrieval.

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of relevant document IDs per query
            k: Number of top documents to consider

        Returns:
            MetricResult with precision
        """
        if not retrieved_docs:
            return MetricResult(
                metric_name=f"precision_at_{k}",
                score=0,
                details={},
            )

        precisions = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            top_k = retrieved[:k]
            relevant_set = set(relevant)
            hits = sum(1 for doc in top_k if doc in relevant_set)
            precision = hits / k if k > 0 else 0
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)

        return MetricResult(
            metric_name=f"precision_at_{k}",
            score=avg_precision,
            details={
                "k": k,
                "query_count": len(precisions),
                "precisions": precisions,
            },
        )

    async def compute_mrr(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
    ) -> MetricResult:
        """
        Compute Mean Reciprocal Rank (MRR).

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of relevant document IDs per query

        Returns:
            MetricResult with MRR
        """
        if not retrieved_docs:
            return MetricResult(
                metric_name="mrr",
                score=0,
                details={},
            )

        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            relevant_set = set(relevant)
            rr = 0
            for rank, doc in enumerate(retrieved, start=1):
                if doc in relevant_set:
                    rr = 1 / rank
                    break
            reciprocal_ranks.append(rr)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        return MetricResult(
            metric_name="mrr",
            score=mrr,
            details={
                "query_count": len(reciprocal_ranks),
                "reciprocal_ranks": reciprocal_ranks,
            },
        )

    async def compute_hallucination_rate(
        self,
        answers: List[str],
        contexts: List[List[str]],
    ) -> MetricResult:
        """
        Compute hallucination rate (answers not supported by contexts).

        Args:
            answers: List of generated answers
            contexts: List of context lists

        Returns:
            MetricResult with hallucination rate
        """
        if not answers:
            return MetricResult(
                metric_name="hallucination_rate",
                score=0,
                details={},
            )

        hallucinated = 0

        for answer, ctx_list in zip(answers, contexts):
            # Simple heuristic: check if answer entities appear in contexts
            context_text = " ".join(ctx_list).lower()
            answer_lower = answer.lower()

            # Extract potential entities (capitalized words, numbers)
            entities = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', answer)

            if entities:
                unsupported = sum(1 for e in entities if e.lower() not in context_text)
                if unsupported / len(entities) > 0.5:
                    hallucinated += 1

        hallucination_rate = hallucinated / len(answers)

        return MetricResult(
            metric_name="hallucination_rate",
            score=hallucination_rate,
            details={
                "total_answers": len(answers),
                "hallucinated": hallucinated,
            },
        )

    async def compute_completeness(
        self,
        answers: List[str],
        ground_truths: List[str],
    ) -> MetricResult:
        """
        Compute answer completeness compared to ground truth.

        Args:
            answers: Generated answers
            ground_truths: Expected answers

        Returns:
            MetricResult with completeness score
        """
        if not answers or not ground_truths:
            return MetricResult(
                metric_name="completeness",
                score=0,
                details={},
            )

        completeness_scores = []

        for answer, truth in zip(answers, ground_truths):
            if not truth:
                continue

            # Key terms from ground truth
            truth_terms = set(truth.lower().split())
            answer_terms = set(answer.lower().split())

            # Remove common words
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            truth_terms -= common_words
            answer_terms -= common_words

            if truth_terms:
                recall = len(truth_terms & answer_terms) / len(truth_terms)
                completeness_scores.append(recall)

        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

        return MetricResult(
            metric_name="completeness",
            score=avg_completeness,
            details={
                "sample_count": len(completeness_scores),
            },
        )

    async def compute_llm_quality_score(
        self,
        questions: List[str],
        answers: List[str],
    ) -> MetricResult:
        """
        Use LLM to score answer quality.

        Args:
            questions: List of questions
            answers: List of answers

        Returns:
            MetricResult with LLM quality score
        """
        if not self.llm_client or not answers:
            return MetricResult(
                metric_name="llm_quality",
                score=0,
                details={"error": "LLM client not available"},
            )

        scores = []
        for question, answer in zip(questions, answers):
            try:
                score = await self._score_with_llm(question, answer)
                scores.append(score)
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")

        if not scores:
            return MetricResult(
                metric_name="llm_quality",
                score=0,
                details={"error": "All LLM scoring failed"},
            )

        avg_score = sum(scores) / len(scores)

        return MetricResult(
            metric_name="llm_quality",
            score=avg_score,
            details={
                "sample_count": len(scores),
                "scores": scores,
            },
        )

    async def _score_with_llm(self, question: str, answer: str) -> float:
        """Score a single Q&A pair with LLM."""
        prompt = f"""Rate the quality of the following answer on a scale of 0 to 10.
Consider: accuracy, completeness, clarity, and helpfulness.
Only respond with a number.

Question: {question}

Answer: {answer}

Quality score (0-10):"""

        response = await self.llm_client.generate(prompt, max_tokens=5)
        score = float(response.strip())
        return min(max(score, 0), 10) / 10.0  # Normalize to 0-1
