"""Tests for MCP Service."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from services.mcp_service.service import MCPService
from services.mcp_service.registry import ToolRegistry, clear_registry


class TestMCPService:
    """Test cases for MCPService."""

    @pytest.fixture
    def service(self):
        """Create a fresh MCP service."""
        clear_registry()
        return MCPService(registry=ToolRegistry())

    @pytest.fixture
    def mock_rag_client(self):
        """Create a mock RAG client."""
        client = AsyncMock()
        client.retrieve = AsyncMock(return_value={
            "documents": [],
            "transformed_query": "test",
            "latency_ms": 10,
        })
        client.index_document = AsyncMock(return_value={
            "success": True,
            "document_id": "doc1",
            "chunks_created": 1,
        })
        return client

    def test_initialize_default_tools_all_enabled(self, service):
        """Test initializing all default tools."""
        service.initialize_default_tools(
            enable_web_search=True,
            enable_code_execution=True,
            enable_knowledge=True,
        )

        tool_names = service.registry.get_names()
        assert "web_search" in tool_names
        assert "web_fetch" in tool_names
        assert "code_executor" in tool_names
        assert "knowledge_search" in tool_names
        assert "knowledge_add" in tool_names

    def test_initialize_default_tools_selective(self, service):
        """Test initializing selective tools."""
        service.initialize_default_tools(
            enable_web_search=True,
            enable_code_execution=False,
            enable_knowledge=False,
        )

        tool_names = service.registry.get_names()
        assert "web_search" in tool_names
        assert "code_executor" not in tool_names
        assert "knowledge_search" not in tool_names

    def test_initialize_default_tools_idempotent(self, service):
        """Test that initialize is idempotent."""
        service.initialize_default_tools()
        initial_count = len(service.registry.get_names())

        service.initialize_default_tools()  # Second call
        assert len(service.registry.get_names()) == initial_count

    def test_list_tools(self, service):
        """Test listing tools."""
        service.initialize_default_tools(
            enable_web_search=True,
            enable_code_execution=False,
            enable_knowledge=False,
        )

        tools = service.list_tools()
        assert len(tools) == 2  # web_search and web_fetch
        assert all("name" in t for t in tools)
        assert all("description" in t for t in tools)
        assert all("input_schema" in t for t in tools)

    def test_list_tools_by_tags(self, service):
        """Test listing tools filtered by tags."""
        service.initialize_default_tools()

        web_tools = service.list_tools(tags=["web"])
        assert len(web_tools) == 2
        assert all(t["name"] in ["web_search", "web_fetch"] for t in web_tools)

    def test_list_tools_by_pattern(self, service):
        """Test listing tools filtered by pattern."""
        service.initialize_default_tools()

        knowledge_tools = service.list_tools(name_pattern="knowledge_*")
        assert len(knowledge_tools) == 2
        assert all(t["name"].startswith("knowledge_") for t in knowledge_tools)

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, service):
        """Test successful tool execution."""
        service.initialize_default_tools(
            enable_web_search=False,
            enable_code_execution=True,
            enable_knowledge=False,
        )

        result = await service.execute_tool(
            tool_name="code_executor",
            arguments=json.dumps({"code": "print('hello')"})
        )

        assert result["success"] is True
        assert "hello" in result["result"]["stdout"]

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_json(self, service):
        """Test tool execution with invalid JSON arguments."""
        service.initialize_default_tools()

        result = await service.execute_tool(
            tool_name="code_executor",
            arguments="not valid json{"
        )

        assert result["success"] is False
        assert result["error_code"] == "INVALID_JSON"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, service):
        """Test executing non-existent tool."""
        result = await service.execute_tool(
            tool_name="nonexistent",
            arguments="{}"
        )

        assert result["success"] is False
        assert result["error_code"] == "TOOL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_execute_tool_empty_arguments(self, service):
        """Test tool execution with empty arguments string."""
        service.initialize_default_tools(
            enable_web_search=False,
            enable_code_execution=True,
            enable_knowledge=False,
        )

        result = await service.execute_tool(
            tool_name="code_executor",
            arguments=""
        )

        # Should fail because code is required
        assert result["success"] is False

    def test_get_tool_schema(self, service):
        """Test getting schema for a specific tool."""
        service.initialize_default_tools(
            enable_web_search=True,
            enable_code_execution=False,
            enable_knowledge=False,
        )

        schema = service.get_tool_schema("web_search")
        assert schema is not None
        assert schema["name"] == "web_search"
        assert "input_schema" in schema

    def test_get_tool_schema_not_found(self, service):
        """Test getting schema for non-existent tool."""
        schema = service.get_tool_schema("nonexistent")
        assert schema is None

    def test_get_tool_schemas(self, service):
        """Test getting all tool schemas."""
        service.initialize_default_tools()

        schemas = service.get_tool_schemas()
        assert len(schemas) > 0
        assert all("name" in s for s in schemas)
        assert all("parameters" in s for s in schemas)

    def test_register_tool(self, service, mock_tool):
        """Test registering a custom tool."""
        service.register_tool(mock_tool)

        assert "mock_tool" in service.registry.get_names()

    def test_unregister_tool(self, service, mock_tool):
        """Test unregistering a tool."""
        service.register_tool(mock_tool)
        result = service.unregister_tool("mock_tool")

        assert result is True
        assert "mock_tool" not in service.registry.get_names()

    def test_unregister_nonexistent_tool(self, service):
        """Test unregistering a non-existent tool."""
        result = service.unregister_tool("nonexistent")
        assert result is False

    def test_get_langchain_tools(self, service):
        """Test getting LangChain tools."""
        service.initialize_default_tools(
            enable_web_search=True,
            enable_code_execution=False,
            enable_knowledge=False,
        )

        try:
            tools = service.get_langchain_tools()
            assert len(tools) == 2
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_service_with_rag_client(self, mock_rag_client):
        """Test service initialization with RAG client."""
        clear_registry()
        service = MCPService(
            registry=ToolRegistry(),
            rag_client=mock_rag_client
        )

        service.initialize_default_tools(
            enable_web_search=False,
            enable_code_execution=False,
            enable_knowledge=True,
        )

        # Knowledge tools should be registered with the RAG client
        tool_names = service.registry.get_names()
        assert "knowledge_search" in tool_names
        assert "knowledge_add" in tool_names


class TestMCPServiceIntegration:
    """Integration tests for MCP Service."""

    @pytest.fixture
    def full_service(self):
        """Create a fully initialized service."""
        clear_registry()
        service = MCPService(registry=ToolRegistry())
        service.initialize_default_tools()
        return service

    @pytest.mark.asyncio
    async def test_code_execution_flow(self, full_service):
        """Test complete code execution flow."""
        # Execute code
        result = await full_service.execute_tool(
            tool_name="code_executor",
            arguments=json.dumps({
                "code": "result = sum([1, 2, 3, 4, 5])\nprint(f'Sum: {result}')"
            })
        )

        assert result["success"] is True
        assert "Sum: 15" in result["result"]["stdout"]

    @pytest.mark.asyncio
    async def test_code_execution_with_math(self, full_service):
        """Test code execution with pre-loaded math module."""
        # Modules are pre-loaded in sandbox globals, not imported
        result = await full_service.execute_tool(
            tool_name="code_executor",
            arguments=json.dumps({
                "code": "print(math.pi)"
            })
        )

        assert result["success"] is True
        assert "3.14" in result["result"]["stdout"]

    @pytest.mark.asyncio
    async def test_code_execution_security(self, full_service):
        """Test that dangerous code is blocked."""
        dangerous_codes = [
            "import os",
            "import subprocess",
            "eval('1+1')",
            "open('/etc/passwd')",
        ]

        for code in dangerous_codes:
            result = await full_service.execute_tool(
                tool_name="code_executor",
                arguments=json.dumps({"code": code})
            )
            assert result["success"] is True  # Tool executed
            assert result["result"]["success"] is False  # But code was blocked
            assert "Dangerous" in result["result"]["error"]
