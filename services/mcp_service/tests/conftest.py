"""Pytest fixtures for MCP service tests."""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock

from services.mcp_service.tools.base import BaseTool, ToolParameter
from services.mcp_service.registry import ToolRegistry, clear_registry


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"
    tags = ["test", "mock"]
    parameters = [
        ToolParameter(
            name="input",
            type="string",
            description="Input value",
            required=True,
        ),
        ToolParameter(
            name="optional_param",
            type="string",
            description="Optional parameter",
            required=False,
            default="default_value",
        ),
    ]

    async def execute(self, input: str, optional_param: str = "default_value") -> Dict[str, Any]:
        return {
            "input": input,
            "optional_param": optional_param,
            "processed": True,
        }


class SlowTool(BaseTool):
    """Tool that takes time to execute (for timeout testing)."""

    name = "slow_tool"
    description = "A slow tool for timeout testing"
    tags = ["test", "slow"]
    timeout_seconds = 5
    parameters = [
        ToolParameter(
            name="delay",
            type="number",
            description="Delay in seconds",
            required=False,
            default=1,
        ),
    ]

    async def execute(self, delay: float = 1) -> Dict[str, Any]:
        await asyncio.sleep(delay)
        return {"completed": True, "delay": delay}


class FailingTool(BaseTool):
    """Tool that always fails."""

    name = "failing_tool"
    description = "A tool that always fails"
    tags = ["test", "fail"]
    parameters = []

    async def execute(self) -> Dict[str, Any]:
        raise RuntimeError("This tool always fails")


@pytest.fixture
def mock_tool():
    """Create a mock tool instance."""
    return MockTool()


@pytest.fixture
def slow_tool():
    """Create a slow tool instance."""
    return SlowTool()


@pytest.fixture
def failing_tool():
    """Create a failing tool instance."""
    return FailingTool()


@pytest.fixture
def registry():
    """Create a fresh tool registry."""
    clear_registry()
    return ToolRegistry()


@pytest.fixture
def populated_registry(registry, mock_tool, slow_tool, failing_tool):
    """Create a registry with pre-registered tools."""
    registry.register(mock_tool)
    registry.register(slow_tool)
    registry.register(failing_tool)
    return registry


@pytest.fixture
def mock_rag_client():
    """Create a mock RAG client."""
    client = AsyncMock()
    client.retrieve = AsyncMock(return_value={
        "documents": [
            {
                "id": "doc1",
                "content": "Test document content",
                "score": 0.95,
                "metadata": {"source": "test"},
                "source": "test.txt",
            }
        ],
        "transformed_query": "test query",
        "latency_ms": 50,
    })
    client.index_document = AsyncMock(return_value={
        "success": True,
        "document_id": "new_doc_1",
        "chunks_created": 3,
    })
    client.get_stats = AsyncMock(return_value={
        "total_documents": 100,
        "total_chunks": 500,
        "index_size_bytes": 1024000,
    })
    return client


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for web requests."""
    client = AsyncMock()
    return client
