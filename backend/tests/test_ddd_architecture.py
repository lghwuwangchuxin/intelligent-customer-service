"""
Tests for DDD Architecture Components.

Tests the domain base classes, interfaces, and infrastructure components.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Domain entities and interfaces
from app.domain.base.entities import (
    Message,
    MessageRole,
    Document,
    SearchResult,
    ToolCall,
    ToolCallStatus,
    AgentThought,
    ThoughtType,
    AgentResult,
    KnowledgeAddResult,
)
from app.domain.base.lifecycle import (
    ServiceStatus,
    ServiceLifecycle,
    CompositeService,
)
from app.domain.base.interfaces import (
    IEmbeddingService,
    ILLMService,
    IVectorStore,
)


class TestMessageEntity:
    """Test Message entity."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
            metadata={"source": "test"},
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there!"
        assert d["metadata"]["source"] == "test"

    def test_message_from_dict(self):
        """Test creating message from dict."""
        data = {
            "role": "user",
            "content": "Test message",
            "metadata": {"key": "value"},
        }
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"
        assert msg.metadata["key"] == "value"


class TestDocumentEntity:
    """Test Document entity."""

    def test_create_document(self):
        """Test creating a document."""
        doc = Document(
            id="doc-1",
            content="Test content",
            metadata={"source": "test.pdf", "title": "Test Doc"},
        )
        assert doc.id == "doc-1"
        assert doc.content == "Test content"
        assert doc.source == "test.pdf"
        assert doc.title == "Test Doc"

    def test_document_without_metadata(self):
        """Test document with minimal metadata."""
        doc = Document(id="doc-2", content="Content")
        assert doc.source is None
        assert doc.title is None

    def test_document_to_dict(self):
        """Test converting document to dict."""
        doc = Document(
            id="doc-3",
            content="Content",
            metadata={"source": "file.txt"},
            score=0.95,
        )
        d = doc.to_dict()
        assert d["id"] == "doc-3"
        assert d["score"] == 0.95


class TestSearchResultEntity:
    """Test SearchResult entity."""

    def test_create_search_result(self):
        """Test creating a search result."""
        docs = [
            Document(id="1", content="Result 1"),
            Document(id="2", content="Result 2"),
        ]
        result = SearchResult(
            documents=docs,
            query="test query",
            total_count=2,
            search_time_ms=15.5,
        )
        assert len(result.documents) == 2
        assert result.query == "test query"
        assert result.search_time_ms == 15.5


class TestToolCallEntity:
    """Test ToolCall entity."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tc = ToolCall(
            id="tc-1",
            name="search",
            arguments={"query": "test"},
        )
        assert tc.id == "tc-1"
        assert tc.status == ToolCallStatus.PENDING
        assert tc.result is None

    def test_tool_call_to_dict(self):
        """Test converting tool call to dict."""
        tc = ToolCall(
            id="tc-2",
            name="calculate",
            arguments={"x": 1, "y": 2},
            status=ToolCallStatus.SUCCESS,
            result=3,
            execution_time_ms=5.2,
        )
        d = tc.to_dict()
        assert d["name"] == "calculate"
        assert d["status"] == "success"
        assert d["result"] == 3


class TestAgentThoughtEntity:
    """Test AgentThought entity."""

    def test_create_thought(self):
        """Test creating an agent thought."""
        thought = AgentThought(
            type=ThoughtType.THINKING,
            content="Analyzing the problem...",
            iteration=1,
        )
        assert thought.type == ThoughtType.THINKING
        assert thought.iteration == 1

    def test_thought_with_tool_call(self):
        """Test thought with associated tool call."""
        tc = ToolCall(id="tc", name="search", arguments={})
        thought = AgentThought(
            type=ThoughtType.OBSERVATION,
            content="Got search results",
            tool_call=tc,
        )
        d = thought.to_dict()
        assert "tool_call" in d


class TestAgentResultEntity:
    """Test AgentResult entity."""

    def test_create_result(self):
        """Test creating an agent result."""
        result = AgentResult(
            response="Here is the answer",
            iterations=3,
        )
        assert result.success is True
        assert result.iterations == 3

    def test_result_with_error(self):
        """Test agent result with error."""
        result = AgentResult(
            response="",
            error="Something went wrong",
        )
        assert result.success is False


class TestKnowledgeAddResult:
    """Test KnowledgeAddResult entity."""

    def test_create_success_result(self):
        """Test successful knowledge add result."""
        result = KnowledgeAddResult(
            success=True,
            num_documents=5,
            num_chunks=25,
            source="document.pdf",
        )
        assert result.success is True
        assert result.num_chunks == 25

    def test_create_failure_result(self):
        """Test failed knowledge add result."""
        result = KnowledgeAddResult(
            success=False,
            error="File not found",
        )
        assert result.success is False
        assert result.error == "File not found"


class MockService(ServiceLifecycle):
    """Mock service for testing lifecycle."""

    def __init__(self, should_fail: bool = False):
        super().__init__()
        self.should_fail = should_fail
        self.init_called = False
        self.shutdown_called = False

    async def _do_init(self) -> None:
        self.init_called = True
        if self.should_fail:
            raise RuntimeError("Init failed")

    async def _do_shutdown(self) -> None:
        self.shutdown_called = True


class TestServiceLifecycle:
    """Test ServiceLifecycle base class."""

    @pytest.mark.asyncio
    async def test_successful_init(self):
        """Test successful service initialization."""
        service = MockService()
        assert service.status == ServiceStatus.NOT_INITIALIZED

        await service.async_init()

        assert service.status == ServiceStatus.READY
        assert service.is_initialized is True
        assert service.is_healthy is True
        assert service.init_called is True

    @pytest.mark.asyncio
    async def test_failed_init(self):
        """Test failed service initialization."""
        service = MockService(should_fail=True)

        with pytest.raises(RuntimeError):
            await service.async_init()

        assert service.status == ServiceStatus.ERROR
        assert service.is_initialized is False
        assert service.error == "Init failed"

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test service shutdown."""
        service = MockService()
        await service.async_init()
        await service.shutdown()

        assert service.status == ServiceStatus.SHUTDOWN
        assert service.shutdown_called is True

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        service = MockService()
        await service.async_init()

        health = service.health_check()
        assert health["status"] == "ready"
        assert health["healthy"] is True


class TestCompositeService:
    """Test CompositeService base class."""

    @pytest.mark.asyncio
    async def test_composite_init(self):
        """Test composite service initialization."""
        composite = CompositeService()
        service1 = MockService()
        service2 = MockService()

        composite.register_service("service1", service1)
        composite.register_service("service2", service2)

        await composite.async_init()

        assert service1.is_initialized is True
        assert service2.is_initialized is True
        assert composite.is_initialized is True

    @pytest.mark.asyncio
    async def test_composite_partial_failure(self):
        """Test composite with one failing service."""
        composite = CompositeService()
        service1 = MockService()
        service2 = MockService(should_fail=True)

        composite.register_service("service1", service1)
        composite.register_service("service2", service2)

        with pytest.raises(RuntimeError) as exc_info:
            await composite.async_init()

        assert "Failed to initialize service2" in str(exc_info.value)


class TestInterfaceProtocols:
    """Test that interface protocols work correctly."""

    def test_embedding_service_protocol(self):
        """Test IEmbeddingService protocol."""

        class MockEmbedding:
            @property
            def dimension(self) -> int:
                return 768

            async def embed_async(self, texts: List[str]) -> List[List[float]]:
                return [[0.0] * 768 for _ in texts]

            def embed(self, texts: List[str]) -> List[List[float]]:
                return [[0.0] * 768 for _ in texts]

            async def warmup(self) -> None:
                pass

        mock = MockEmbedding()
        assert isinstance(mock, IEmbeddingService)

    def test_llm_service_protocol(self):
        """Test ILLMService protocol."""

        class MockLLM:
            @property
            def provider(self) -> str:
                return "mock"

            @property
            def model(self) -> str:
                return "mock-model"

            @property
            def supports_tool_calling(self) -> bool:
                return True

            async def ainvoke(self, messages: List[Dict[str, Any]], **kwargs) -> str:
                return "response"

            def invoke(self, messages: List[Dict[str, Any]], **kwargs) -> str:
                return "response"

            def stream(self, messages: List[Dict[str, Any]], **kwargs):
                yield "chunk"

            async def astream(self, messages: List[Dict[str, Any]], **kwargs):
                yield "chunk"

            def get_info(self) -> Dict[str, Any]:
                return {"provider": "mock"}

        mock = MockLLM()
        assert isinstance(mock, ILLMService)


class TestServiceFactory:
    """Test ServiceFactory creation methods."""

    def test_factory_creates_llm_manager(self):
        """Test creating LLM manager via factory."""
        with patch('app.core.llm_manager.LLMManager') as mock_llm:
            from app.infrastructure.factory import ServiceFactory

            mock_llm.return_value = MagicMock()
            result = ServiceFactory.create_llm_manager(
                provider="ollama",
                model="test-model",
            )

            mock_llm.assert_called_once()
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["provider"] == "ollama"
            assert call_kwargs["model"] == "test-model"


class TestServiceRegistry:
    """Test ServiceRegistry singleton behavior."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        from app.infrastructure.factory import ServiceRegistry

        # Reset singleton for testing
        ServiceRegistry._instance = None

        registry1 = ServiceRegistry()
        registry2 = ServiceRegistry()

        assert registry1 is registry2

    def test_registry_reset(self):
        """Test registry reset."""
        from app.infrastructure.factory import get_registry

        registry = get_registry()
        registry.register("test_service", MagicMock())

        registry.reset()

        assert registry.is_initialized is False
        assert registry.get("test_service") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])