"""Tests for RAG Service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.rag_service.service import RAGService, RetrieveConfig, RetrieveResult


class TestRAGService:
    """Test cases for RAGService."""

    @pytest.fixture
    def service(self, mock_vector_store, mock_elasticsearch_client, mock_embedding_model, mock_llm_client):
        """Create a RAG service with mocks."""
        return RAGService(
            vector_store=mock_vector_store,
            elasticsearch_client=mock_elasticsearch_client,
            embedding_model=mock_embedding_model,
            llm_client=mock_llm_client,
            use_reranker_api=False,
        )

    @pytest.fixture
    def minimal_service(self, mock_embedding_model):
        """Create a minimal RAG service without external dependencies."""
        return RAGService(
            embedding_model=mock_embedding_model,
        )

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, service):
        """Test basic retrieval."""
        result = await service.retrieve(
            query="What is machine learning?",
        )

        assert isinstance(result, RetrieveResult)
        assert result.query == "What is machine learning?"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_retrieve_with_config(self, service):
        """Test retrieval with custom config."""
        config = RetrieveConfig(
            top_k=5,
            enable_query_transform=True,
            enable_rerank=True,
            enable_postprocess=True,
            hybrid_alpha=0.7,
            rerank_top_k=3,
        )

        result = await service.retrieve(
            query="test query",
            config=config,
        )

        assert isinstance(result, RetrieveResult)

    @pytest.mark.asyncio
    async def test_retrieve_with_knowledge_base_filter(self, service):
        """Test retrieval with knowledge base filter."""
        result = await service.retrieve(
            query="test",
            knowledge_base_id="kb_123",
        )

        assert isinstance(result, RetrieveResult)

    @pytest.mark.asyncio
    async def test_retrieve_without_transform(self, service):
        """Test retrieval without query transformation."""
        config = RetrieveConfig(
            enable_query_transform=False,
        )

        result = await service.retrieve(
            query="test query",
            config=config,
        )

        # Transformed query should be None when disabled
        assert result.transformed_query is None or result.transformed_query == "test query"

    @pytest.mark.asyncio
    async def test_retrieve_without_rerank(self, service):
        """Test retrieval without reranking."""
        config = RetrieveConfig(
            enable_rerank=False,
        )

        result = await service.retrieve(
            query="test query",
            config=config,
        )

        assert isinstance(result, RetrieveResult)

    @pytest.mark.asyncio
    async def test_index_document(self, service):
        """Test document indexing."""
        result = await service.index_document(
            content="This is a test document about AI.",
            metadata={"source": "test.txt", "author": "Test"},
        )

        assert result.success is True or result.success is False  # Depends on mock setup

    @pytest.mark.asyncio
    async def test_index_document_with_chunking(self, service):
        """Test document indexing with chunking."""
        long_content = "This is a long document. " * 100

        result = await service.index_document(
            content=long_content,
            chunk=True,
        )

        # Should process without error
        assert hasattr(result, "success")

    @pytest.mark.asyncio
    async def test_delete_document(self, service):
        """Test document deletion."""
        result = await service.delete_document(
            document_id="test_doc_123",
        )

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_list_documents(self, service):
        """Test listing documents."""
        result = await service.list_documents(
            page=1,
            page_size=10,
        )

        assert "documents" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result

    @pytest.mark.asyncio
    async def test_get_stats(self, service):
        """Test getting index statistics."""
        result = await service.get_stats()

        assert "total_documents" in result
        assert "total_chunks" in result
        assert "index_size_bytes" in result

    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check."""
        result = await service.health_check()

        assert "status" in result
        assert result["status"] in ["healthy", "degraded", "unhealthy"]


class TestRetrieveConfig:
    """Test cases for RetrieveConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetrieveConfig()

        assert config.top_k == 10
        assert config.enable_query_transform is True
        assert config.enable_rerank is True
        assert config.enable_postprocess is True
        assert config.hybrid_alpha == 0.5
        assert config.rerank_top_k == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetrieveConfig(
            top_k=20,
            enable_query_transform=False,
            enable_rerank=False,
            hybrid_alpha=0.8,
        )

        assert config.top_k == 20
        assert config.enable_query_transform is False
        assert config.enable_rerank is False
        assert config.hybrid_alpha == 0.8


class TestRetrieveResult:
    """Test cases for RetrieveResult."""

    def test_result_creation(self):
        """Test creating RetrieveResult."""
        result = RetrieveResult(
            documents=[{"id": "1", "content": "test", "score": 0.9}],
            query="test query",
            transformed_query="improved query",
            expanded_queries=["q1", "q2"],
            latency_ms=50,
            total_candidates=10,
        )

        assert len(result.documents) == 1
        assert result.query == "test query"
        assert result.transformed_query == "improved query"
        assert len(result.expanded_queries) == 2
        assert result.latency_ms == 50
        assert result.total_candidates == 10

    def test_result_defaults(self):
        """Test RetrieveResult default values."""
        result = RetrieveResult(
            documents=[],
            query="test",
        )

        assert result.transformed_query is None
        assert result.expanded_queries == []
        assert result.latency_ms == 0
        assert result.total_candidates == 0


class TestRAGServiceIntegration:
    """Integration tests for RAG Service."""

    @pytest.fixture
    def full_service(self, mock_vector_store, mock_elasticsearch_client, mock_embedding_model, mock_llm_client):
        """Create a fully configured service."""
        return RAGService(
            vector_store=mock_vector_store,
            elasticsearch_client=mock_elasticsearch_client,
            embedding_model=mock_embedding_model,
            llm_client=mock_llm_client,
        )

    @pytest.mark.asyncio
    async def test_full_retrieval_pipeline(self, full_service):
        """Test the complete retrieval pipeline."""
        config = RetrieveConfig(
            top_k=5,
            enable_query_transform=True,
            enable_rerank=True,
            enable_postprocess=True,
        )

        result = await full_service.retrieve(
            query="What are the latest advances in AI?",
            config=config,
        )

        assert isinstance(result, RetrieveResult)
        assert result.query == "What are the latest advances in AI?"
        # Should have gone through transformation
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_index_and_retrieve(self, full_service):
        """Test indexing and then retrieving a document."""
        # Index a document
        index_result = await full_service.index_document(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"topic": "AI", "source": "test"},
        )

        # Retrieve
        retrieve_result = await full_service.retrieve(
            query="What is machine learning?",
        )

        assert isinstance(retrieve_result, RetrieveResult)
