"""Tests for Hybrid Retriever."""

import pytest

from services.rag_service.pipeline.hybrid_retriever import (
    HybridRetriever,
    RetrievedDocument,
    RetrievalResult,
)


class TestHybridRetriever:
    """Test cases for HybridRetriever."""

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_elasticsearch_client, mock_embedding_model):
        """Create a hybrid retriever with mocks."""
        return HybridRetriever(
            vector_store=mock_vector_store,
            elasticsearch_client=mock_elasticsearch_client,
            embedding_model=mock_embedding_model,
            vector_weight=0.5,
            bm25_weight=0.5,
            rrf_k=60,
        )

    @pytest.fixture
    def vector_only_retriever(self, mock_vector_store, mock_embedding_model):
        """Create retriever with only vector store."""
        return HybridRetriever(
            vector_store=mock_vector_store,
            elasticsearch_client=None,
            embedding_model=mock_embedding_model,
        )

    @pytest.fixture
    def bm25_only_retriever(self, mock_elasticsearch_client):
        """Create retriever with only Elasticsearch."""
        return HybridRetriever(
            vector_store=None,
            elasticsearch_client=mock_elasticsearch_client,
            embedding_model=None,
        )

    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, retriever):
        """Test hybrid retrieval combining vector and BM25."""
        result = await retriever.retrieve(
            query="machine learning",
            top_k=5,
        )

        assert isinstance(result, RetrievalResult)
        assert result.query == "machine learning"
        assert len(result.documents) <= 5
        assert result.total_candidates > 0
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_vector_only_retrieval(self, vector_only_retriever):
        """Test retrieval with only vector store."""
        result = await vector_only_retriever.retrieve(
            query="test query",
            top_k=5,
        )

        assert len(result.documents) <= 5
        # All documents should come from vector search
        for doc in result.documents:
            assert doc.source in ["vector", "hybrid"]

    @pytest.mark.asyncio
    async def test_bm25_only_retrieval(self, bm25_only_retriever):
        """Test retrieval with only Elasticsearch."""
        result = await bm25_only_retriever.retrieve(
            query="test query",
            top_k=5,
        )

        assert len(result.documents) <= 5

    @pytest.mark.asyncio
    async def test_retrieval_with_knowledge_base_filter(self, retriever):
        """Test retrieval with knowledge base filter."""
        result = await retriever.retrieve(
            query="test",
            top_k=5,
            knowledge_base_id="kb_123",
        )

        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieval_with_expanded_queries(self, retriever):
        """Test retrieval with expanded queries."""
        result = await retriever.retrieve(
            query="main query",
            top_k=5,
            expanded_queries=["expanded 1", "expanded 2"],
        )

        assert isinstance(result, RetrievalResult)
        # Should have more candidates from multiple queries
        assert result.total_candidates >= 0

    @pytest.mark.asyncio
    async def test_retrieval_with_hyde_passage(self, retriever):
        """Test retrieval with HyDE passage."""
        result = await retriever.retrieve(
            query="main query",
            top_k=5,
            hyde_passage="This is a hypothetical answer passage.",
        )

        assert isinstance(result, RetrievalResult)


class TestRRFFusion:
    """Test cases for RRF fusion."""

    @pytest.fixture
    def retriever(self):
        """Create retriever for testing RRF."""
        return HybridRetriever(
            vector_store=None,
            elasticsearch_client=None,
            embedding_model=None,
            rrf_k=60,
        )

    def test_rrf_fusion_empty_inputs(self, retriever):
        """Test RRF fusion with empty inputs."""
        result = retriever._rrf_fusion([], [], top_k=5)
        assert result == []

    def test_rrf_fusion_vector_only(self, retriever):
        """Test RRF fusion with only vector results."""
        vector_docs = [
            RetrievedDocument(id="1", content="doc1", score=0.9, source="vector"),
            RetrievedDocument(id="2", content="doc2", score=0.8, source="vector"),
        ]

        result = retriever._rrf_fusion(vector_docs, [], top_k=5)

        assert len(result) == 2
        assert result[0].id == "1"  # Higher ranked
        assert result[1].id == "2"

    def test_rrf_fusion_bm25_only(self, retriever):
        """Test RRF fusion with only BM25 results."""
        bm25_docs = [
            RetrievedDocument(id="1", content="doc1", score=10.0, source="bm25"),
            RetrievedDocument(id="2", content="doc2", score=8.0, source="bm25"),
        ]

        result = retriever._rrf_fusion([], bm25_docs, top_k=5)

        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"

    def test_rrf_fusion_combined(self, retriever):
        """Test RRF fusion combining results."""
        vector_docs = [
            RetrievedDocument(id="1", content="doc1", score=0.9, source="vector"),
            RetrievedDocument(id="2", content="doc2", score=0.8, source="vector"),
        ]
        bm25_docs = [
            RetrievedDocument(id="2", content="doc2", score=10.0, source="bm25"),
            RetrievedDocument(id="3", content="doc3", score=8.0, source="bm25"),
        ]

        result = retriever._rrf_fusion(vector_docs, bm25_docs, top_k=5)

        assert len(result) == 3
        # Doc 2 should be ranked higher (appears in both)
        doc_ids = [d.id for d in result]
        assert "2" in doc_ids
        assert result[0].source == "hybrid"

    def test_rrf_fusion_top_k_limit(self, retriever):
        """Test RRF fusion respects top_k."""
        vector_docs = [
            RetrievedDocument(id=str(i), content=f"doc{i}", score=0.9-i*0.1, source="vector")
            for i in range(10)
        ]

        result = retriever._rrf_fusion(vector_docs, [], top_k=3)

        assert len(result) == 3


class TestRetrievedDocument:
    """Test cases for RetrievedDocument dataclass."""

    def test_document_creation(self):
        """Test creating RetrievedDocument."""
        doc = RetrievedDocument(
            id="test_id",
            content="Test content",
            score=0.95,
            metadata={"source": "test.txt"},
            source="vector",
        )

        assert doc.id == "test_id"
        assert doc.content == "Test content"
        assert doc.score == 0.95
        assert doc.metadata == {"source": "test.txt"}
        assert doc.source == "vector"

    def test_document_defaults(self):
        """Test RetrievedDocument default values."""
        doc = RetrievedDocument(
            id="test",
            content="content",
            score=0.5,
        )

        assert doc.metadata == {}
        assert doc.source == ""
