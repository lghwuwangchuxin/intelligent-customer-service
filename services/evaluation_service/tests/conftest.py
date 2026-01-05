"""Pytest fixtures for Evaluation service tests."""

import pytest
from typing import List
from unittest.mock import AsyncMock

from services.evaluation_service.evaluators.ragas_evaluator import EvaluationSample


@pytest.fixture
def sample_evaluation_samples():
    """Create sample evaluation data."""
    return [
        EvaluationSample(
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            contexts=[
                "Machine learning is a field of AI that uses algorithms to learn from data.",
                "ML systems improve their performance through experience.",
            ],
            ground_truth="Machine learning is a type of AI that learns from data.",
        ),
        EvaluationSample(
            question="How does deep learning work?",
            answer="Deep learning uses neural networks with multiple layers to process data.",
            contexts=[
                "Deep learning is based on artificial neural networks.",
                "Neural networks consist of layers of interconnected nodes.",
            ],
            ground_truth="Deep learning uses multi-layer neural networks.",
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()

    async def mock_generate(prompt: str, max_tokens: int = 200) -> str:
        if "faithfulness" in prompt.lower():
            return "0.85"
        elif "relevancy" in prompt.lower():
            return "0.9"
        else:
            return "0.8"

    client.generate = mock_generate
    return client


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock()

    async def mock_embed(text: str) -> List[float]:
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes[:768]]

    async def mock_embed_batch(texts: List[str]) -> List[List[float]]:
        return [await mock_embed(t) for t in texts]

    model.embed = mock_embed
    model.embed_batch = mock_embed_batch
    return model
