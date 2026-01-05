"""Pytest fixtures for RAG service tests."""

import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from services.rag_service.pipeline.hybrid_retriever import RetrievedDocument


@dataclass
class MockSearchResult:
    """Mock search result from vector store."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock()

    async def mock_embed(text: str) -> List[float]:
        # Generate deterministic mock embedding
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes[:768]]

    async def mock_embed_batch(texts: List[str]) -> List[List[float]]:
        return [await mock_embed(t) for t in texts]

    model.embed = mock_embed
    model.embed_batch = mock_embed_batch
    return model


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = AsyncMock()

    async def mock_search(vector, top_k, filter_expr=None):
        return [
            MockSearchResult(
                id="vec_1",
                content="Vector search result 1",
                score=0.95,
                metadata={"source": "test"},
            ),
            MockSearchResult(
                id="vec_2",
                content="Vector search result 2",
                score=0.85,
                metadata={"source": "test"},
            ),
        ]

    store.search = mock_search
    store.has_collection = AsyncMock(return_value=True)
    store.insert = AsyncMock()
    store.delete = AsyncMock()
    return store


@pytest.fixture
def mock_elasticsearch_client():
    """Create a mock Elasticsearch client."""
    client = AsyncMock()

    async def mock_search(index, body):
        return {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "es_1",
                        "_score": 10.5,
                        "_source": {
                            "content": "BM25 search result 1",
                            "metadata": {"source": "test"},
                        },
                    },
                    {
                        "_id": "es_2",
                        "_score": 8.3,
                        "_source": {
                            "content": "BM25 search result 2",
                            "metadata": {"source": "test"},
                        },
                    },
                ],
            }
        }

    client.search = mock_search
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.stats = AsyncMock(return_value={
        "indices": {
            "knowledge_base": {
                "primaries": {
                    "docs": {"count": 100},
                    "store": {"size_in_bytes": 1024000},
                }
            }
        }
    })
    client.index = AsyncMock()
    client.delete = AsyncMock()
    client.get = AsyncMock(return_value={
        "_source": {
            "content": "Test document",
            "metadata": {"source": "test"},
        }
    })
    client.bulk = AsyncMock()
    return client


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()

    async def mock_generate(prompt: str, max_tokens: int = 200) -> str:
        if "hypothetical" in prompt.lower() or "answer paragraph" in prompt.lower():
            return "This is a hypothetical answer about the topic."
        elif "related" in prompt.lower() or "expand" in prompt.lower():
            return "Related query 1\nRelated query 2\nRelated query 3"
        elif "rewrite" in prompt.lower():
            return "Improved query"
        elif "relevance" in prompt.lower():
            return "8"
        else:
            return "Mock response"

    client.generate = mock_generate
    return client


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        RetrievedDocument(
            id="doc1",
            content="This is the first document about AI and machine learning.",
            score=0.9,
            metadata={"source": "test1.txt"},
            source="vector",
        ),
        RetrievedDocument(
            id="doc2",
            content="This is the second document about natural language processing.",
            score=0.85,
            metadata={"source": "test2.txt"},
            source="vector",
        ),
        RetrievedDocument(
            id="doc3",
            content="This is the third document about deep learning.",
            score=0.8,
            metadata={"source": "test3.txt"},
            source="bm25",
        ),
        RetrievedDocument(
            id="doc4",
            content="This is the fourth document about neural networks.",
            score=0.75,
            metadata={"source": "test4.txt"},
            source="bm25",
        ),
    ]
