"""Tests for Reranker."""

import pytest
from unittest.mock import AsyncMock, patch

from services.rag_service.pipeline.reranker import Reranker, RerankResult, LLMReranker
from services.rag_service.pipeline.hybrid_retriever import RetrievedDocument


class TestReranker:
    """Test cases for Reranker."""

    @pytest.fixture
    def reranker(self):
        """Create a reranker with API mode."""
        return Reranker(
            model_name="test-model",
            top_k=5,
            use_api=True,
            api_key="test_key",
        )

    @pytest.fixture
    def reranker_no_api(self):
        """Create a reranker without API."""
        return Reranker(
            model_name="test-model",
            top_k=5,
            use_api=False,
        )

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, reranker):
        """Test reranking empty document list."""
        result = await reranker.rerank(
            query="test query",
            documents=[],
        )

        assert isinstance(result, RerankResult)
        assert result.documents == []
        assert result.original_count == 0
        assert result.reranked_count == 0

    @pytest.mark.asyncio
    async def test_rerank_with_api(self, reranker, sample_documents):
        """Test reranking with API."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.85},
                    {"index": 2, "relevance_score": 0.75},
                    {"index": 3, "relevance_score": 0.65},
                ]
            }
            mock_response.raise_for_status = AsyncMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await reranker.rerank(
                query="machine learning",
                documents=sample_documents,
                top_k=3,
            )

            assert isinstance(result, RerankResult)
            assert len(result.documents) <= 3
            assert result.original_count == 4
            assert result.reranked_count <= 3

    @pytest.mark.asyncio
    async def test_rerank_without_api_key(self, sample_documents):
        """Test reranking without API key returns original scores."""
        reranker = Reranker(
            model_name="test-model",
            use_api=True,
            api_key=None,
        )

        result = await reranker.rerank(
            query="test",
            documents=sample_documents,
        )

        # Should keep original order/scores
        assert len(result.documents) <= reranker.top_k

    @pytest.mark.asyncio
    async def test_rerank_api_error_handling(self, reranker, sample_documents):
        """Test error handling when API fails."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("API error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await reranker.rerank(
                query="test",
                documents=sample_documents,
            )

            # Should fallback to original scores
            assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_rerank_top_k_override(self, reranker, sample_documents):
        """Test top_k override parameter."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "results": [
                    {"index": i, "relevance_score": 1.0 - i * 0.1}
                    for i in range(4)
                ]
            }
            mock_response.raise_for_status = AsyncMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await reranker.rerank(
                query="test",
                documents=sample_documents,
                top_k=2,
            )

            assert len(result.documents) == 2


class TestLLMReranker:
    """Test cases for LLM-based reranker."""

    @pytest.fixture
    def llm_reranker(self, mock_llm_client):
        """Create an LLM reranker."""
        return LLMReranker(llm_client=mock_llm_client)

    @pytest.fixture
    def llm_reranker_no_client(self):
        """Create an LLM reranker without client."""
        return LLMReranker(llm_client=None)

    @pytest.mark.asyncio
    async def test_llm_rerank_success(self, llm_reranker, sample_documents):
        """Test LLM reranking."""
        result = await llm_reranker.rerank(
            query="machine learning",
            documents=sample_documents,
            top_k=3,
        )

        assert isinstance(result, RerankResult)
        assert len(result.documents) <= 3
        assert result.original_count == 4

    @pytest.mark.asyncio
    async def test_llm_rerank_empty_documents(self, llm_reranker):
        """Test LLM reranking with empty documents."""
        result = await llm_reranker.rerank(
            query="test",
            documents=[],
            top_k=5,
        )

        assert result.documents == []
        assert result.original_count == 0

    @pytest.mark.asyncio
    async def test_llm_rerank_no_client(self, llm_reranker_no_client, sample_documents):
        """Test LLM reranking without client."""
        result = await llm_reranker_no_client.rerank(
            query="test",
            documents=sample_documents,
            top_k=3,
        )

        # Should return original documents
        assert len(result.documents) == 3


class TestRerankResult:
    """Test cases for RerankResult dataclass."""

    def test_result_creation(self, sample_documents):
        """Test creating RerankResult."""
        result = RerankResult(
            documents=sample_documents[:2],
            original_count=4,
            reranked_count=2,
            latency_ms=50,
        )

        assert len(result.documents) == 2
        assert result.original_count == 4
        assert result.reranked_count == 2
        assert result.latency_ms == 50

    def test_result_defaults(self):
        """Test RerankResult default values."""
        result = RerankResult(
            documents=[],
            original_count=0,
            reranked_count=0,
        )

        assert result.latency_ms == 0
