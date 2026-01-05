"""Tests for Query Transformer."""

import pytest

from services.rag_service.pipeline.query_transform import QueryTransformer, TransformedQuery


class TestQueryTransformer:
    """Test cases for QueryTransformer."""

    @pytest.fixture
    def transformer_with_llm(self, mock_llm_client):
        """Create transformer with LLM client."""
        return QueryTransformer(
            llm_client=mock_llm_client,
            enable_hyde=True,
            enable_expansion=True,
            expansion_count=3,
        )

    @pytest.fixture
    def transformer_no_llm(self):
        """Create transformer without LLM client."""
        return QueryTransformer(
            llm_client=None,
            enable_hyde=True,
            enable_expansion=True,
        )

    @pytest.mark.asyncio
    async def test_transform_without_llm(self, transformer_no_llm):
        """Test transform without LLM client returns original query."""
        result = await transformer_no_llm.transform("test query")

        assert result.original_query == "test query"
        assert result.transformed_query == "test query"
        assert result.expanded_queries == []
        assert result.hyde_passage is None

    @pytest.mark.asyncio
    async def test_transform_with_llm(self, transformer_with_llm):
        """Test transform with LLM client."""
        result = await transformer_with_llm.transform("What is machine learning?")

        assert result.original_query == "What is machine learning?"
        # Should have transformed query from rewriting
        assert result.transformed_query is not None
        # Should have expanded queries
        assert len(result.expanded_queries) > 0
        # Should have HyDE passage
        assert result.hyde_passage is not None

    @pytest.mark.asyncio
    async def test_transform_hyde_only(self, mock_llm_client):
        """Test transform with only HyDE enabled."""
        transformer = QueryTransformer(
            llm_client=mock_llm_client,
            enable_hyde=True,
            enable_expansion=False,
        )

        result = await transformer.transform("test query")

        assert result.hyde_passage is not None
        assert result.expanded_queries == []

    @pytest.mark.asyncio
    async def test_transform_expansion_only(self, mock_llm_client):
        """Test transform with only expansion enabled."""
        transformer = QueryTransformer(
            llm_client=mock_llm_client,
            enable_hyde=False,
            enable_expansion=True,
        )

        result = await transformer.transform("test query")

        assert result.hyde_passage is None
        assert len(result.expanded_queries) > 0

    @pytest.mark.asyncio
    async def test_transform_error_handling(self, mock_llm_client):
        """Test error handling during transformation."""
        mock_llm_client.generate = lambda *args, **kwargs: (_ for _ in ()).throw(
            Exception("LLM error")
        )

        transformer = QueryTransformer(
            llm_client=mock_llm_client,
            enable_hyde=True,
            enable_expansion=True,
        )

        # Should not raise, just return original query
        result = await transformer.transform("test query")
        assert result.original_query == "test query"


class TestTransformedQuery:
    """Test cases for TransformedQuery dataclass."""

    def test_transformed_query_creation(self):
        """Test creating TransformedQuery."""
        result = TransformedQuery(
            original_query="original",
            transformed_query="transformed",
            expanded_queries=["q1", "q2"],
            hyde_passage="passage",
        )

        assert result.original_query == "original"
        assert result.transformed_query == "transformed"
        assert result.expanded_queries == ["q1", "q2"]
        assert result.hyde_passage == "passage"

    def test_transformed_query_defaults(self):
        """Test TransformedQuery default values."""
        result = TransformedQuery(
            original_query="query",
            transformed_query="query",
            expanded_queries=[],
        )

        assert result.hyde_passage is None
