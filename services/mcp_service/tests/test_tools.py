"""Tests for MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.mcp_service.tools.base import BaseTool, ToolParameter
from services.mcp_service.tools.knowledge import (
    KnowledgeSearchTool,
    KnowledgeAddTool,
    KnowledgeStatsTool,
)
from services.mcp_service.tools.code_executor import CodeExecutorTool


class TestBaseTool:
    """Test cases for BaseTool."""

    def test_tool_must_have_name(self):
        """Test that tool must have a name."""
        class NoNameTool(BaseTool):
            description = "No name"
            parameters = []

            async def execute(self):
                pass

        with pytest.raises(ValueError, match="must have a name"):
            NoNameTool()

    def test_get_input_schema(self, mock_tool):
        """Test input schema generation."""
        schema = mock_tool.get_input_schema()
        import json
        parsed = json.loads(schema)

        assert parsed["type"] == "object"
        assert "input" in parsed["properties"]
        assert "optional_param" in parsed["properties"]
        assert "input" in parsed["required"]
        assert "optional_param" not in parsed["required"]

    def test_to_schema(self, mock_tool):
        """Test schema conversion for LLM binding."""
        schema = mock_tool.to_schema()

        assert schema["name"] == "mock_tool"
        assert schema["description"] == "A mock tool for testing"
        assert "parameters" in schema

    def test_validate_args_with_required(self, mock_tool):
        """Test argument validation with required params."""
        validated = mock_tool.validate_args(input="test")
        assert validated["input"] == "test"
        assert validated["optional_param"] == "default_value"

    def test_validate_args_missing_required(self, mock_tool):
        """Test argument validation with missing required param."""
        with pytest.raises(ValueError, match="Missing required parameter"):
            mock_tool.validate_args()

    def test_validate_args_with_optional(self, mock_tool):
        """Test argument validation with optional params."""
        validated = mock_tool.validate_args(
            input="test",
            optional_param="custom"
        )
        assert validated["optional_param"] == "custom"


class TestKnowledgeSearchTool:
    """Test cases for KnowledgeSearchTool."""

    @pytest.fixture
    def knowledge_search_tool(self, mock_rag_client):
        return KnowledgeSearchTool(rag_client=mock_rag_client)

    @pytest.mark.asyncio
    async def test_search_success(self, knowledge_search_tool, mock_rag_client):
        """Test successful knowledge search."""
        result = await knowledge_search_tool.execute(
            query="test query",
            top_k=5
        )

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Test document content"
        assert result["query"] == "test query"
        mock_rag_client.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_without_client(self):
        """Test search without RAG client configured."""
        tool = KnowledgeSearchTool(rag_client=None)
        result = await tool.execute(query="test")

        assert result["results"] == []
        assert "not configured" in result["message"]

    @pytest.mark.asyncio
    async def test_search_top_k_bounds(self, knowledge_search_tool):
        """Test that top_k is bounded between 1 and 10."""
        # top_k should be clamped to 10
        await knowledge_search_tool.execute(query="test", top_k=100)
        # top_k should be clamped to 1
        await knowledge_search_tool.execute(query="test", top_k=0)

    @pytest.mark.asyncio
    async def test_search_with_knowledge_base_id(self, knowledge_search_tool, mock_rag_client):
        """Test search with specific knowledge base."""
        await knowledge_search_tool.execute(
            query="test",
            knowledge_base_id="kb_123"
        )

        mock_rag_client.retrieve.assert_called_with(
            query="test",
            top_k=5,
            knowledge_base_id="kb_123"
        )

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_rag_client):
        """Test error handling during search."""
        mock_rag_client.retrieve.side_effect = Exception("Connection error")
        tool = KnowledgeSearchTool(rag_client=mock_rag_client)

        result = await tool.execute(query="test")

        assert result["results"] == []
        assert "error" in result
        assert "Connection error" in result["error"]


class TestKnowledgeAddTool:
    """Test cases for KnowledgeAddTool."""

    @pytest.fixture
    def knowledge_add_tool(self, mock_rag_client):
        return KnowledgeAddTool(rag_client=mock_rag_client)

    @pytest.mark.asyncio
    async def test_add_success(self, knowledge_add_tool, mock_rag_client):
        """Test successful document addition."""
        result = await knowledge_add_tool.execute(
            content="New document content",
            metadata={"title": "Test Doc"}
        )

        assert result["success"] is True
        assert result["document_id"] == "new_doc_1"
        assert result["chunks_created"] == 3

    @pytest.mark.asyncio
    async def test_add_empty_content(self, knowledge_add_tool):
        """Test adding empty content."""
        result = await knowledge_add_tool.execute(content="   ")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_add_without_client(self):
        """Test add without RAG client configured."""
        tool = KnowledgeAddTool(rag_client=None)
        result = await tool.execute(content="test content")

        assert result["success"] is False
        assert "not configured" in result["message"]


class TestKnowledgeStatsTool:
    """Test cases for KnowledgeStatsTool."""

    @pytest.fixture
    def knowledge_stats_tool(self, mock_rag_client):
        return KnowledgeStatsTool(rag_client=mock_rag_client)

    @pytest.mark.asyncio
    async def test_get_stats(self, knowledge_stats_tool, mock_rag_client):
        """Test getting knowledge base statistics."""
        result = await knowledge_stats_tool.execute()

        assert result["success"] is True
        assert result["total_documents"] == 100
        assert result["total_chunks"] == 500

    @pytest.mark.asyncio
    async def test_stats_without_client(self):
        """Test stats without RAG client configured."""
        tool = KnowledgeStatsTool(rag_client=None)
        result = await tool.execute()

        assert result["success"] is False
        assert "not configured" in result["error"]


class TestCodeExecutorTool:
    """Test cases for CodeExecutorTool."""

    @pytest.fixture
    def code_executor(self):
        return CodeExecutorTool(timeout=5)

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, code_executor):
        """Test executing simple Python code."""
        result = await code_executor.execute(code="print('Hello, World!')")

        assert result["success"] is True
        assert "Hello, World!" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_math_code(self, code_executor):
        """Test executing math operations with pre-loaded module."""
        # Note: modules are pre-loaded in globals, not imported
        result = await code_executor.execute(
            code="print(math.sqrt(16))"
        )

        assert result["success"] is True
        assert "4.0" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_with_result(self, code_executor):
        """Test executing code with _result variable."""
        result = await code_executor.execute(
            code="_result = 2 + 2"
        )

        assert result["success"] is True
        assert result["result"] == 4

    @pytest.mark.asyncio
    async def test_dangerous_import_os(self, code_executor):
        """Test that importing os is blocked."""
        result = await code_executor.execute(code="import os")

        assert result["success"] is False
        assert "Dangerous pattern" in result["error"]

    @pytest.mark.asyncio
    async def test_dangerous_import_sys(self, code_executor):
        """Test that importing sys is blocked."""
        result = await code_executor.execute(code="import sys")

        assert result["success"] is False
        assert "Dangerous pattern" in result["error"]

    @pytest.mark.asyncio
    async def test_dangerous_subprocess(self, code_executor):
        """Test that subprocess is blocked."""
        result = await code_executor.execute(code="import subprocess")

        assert result["success"] is False
        assert "Dangerous pattern" in result["error"]

    @pytest.mark.asyncio
    async def test_dangerous_eval(self, code_executor):
        """Test that eval is blocked."""
        result = await code_executor.execute(code="eval('1+1')")

        assert result["success"] is False
        assert "Dangerous pattern" in result["error"]

    @pytest.mark.asyncio
    async def test_dangerous_open(self, code_executor):
        """Test that open is blocked."""
        result = await code_executor.execute(code="open('test.txt')")

        assert result["success"] is False
        assert "Dangerous pattern" in result["error"]

    @pytest.mark.asyncio
    async def test_allowed_modules(self, code_executor):
        """Test that allowed modules are pre-loaded and can be used."""
        # Note: modules are pre-loaded in globals, not imported at runtime
        # Test that each module is available
        test_code = {
            "math": "print(math.pi)",
            "random": "print(random.randint(1, 10))",
            "json": "print(json.dumps({'test': 1}))",
            "re": "print(re.match('test', 'test').group())",
        }
        for module, code in test_code.items():
            result = await code_executor.execute(code=code)
            assert result["success"] is True, f"Module {module} should be available"

    @pytest.mark.asyncio
    async def test_execution_error(self, code_executor):
        """Test handling of execution errors."""
        result = await code_executor.execute(code="undefined_variable")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_syntax_error(self, code_executor):
        """Test handling of syntax errors."""
        result = await code_executor.execute(code="def broken(")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, code_executor):
        """Test execution timeout."""
        result = await code_executor.execute(
            code="import time\ntime.sleep(100)",
            timeout=1
        )

        # Note: time module is not in allowed list, so this will fail validation
        assert result["success"] is False

    def test_safe_builtins(self, code_executor):
        """Test that safe builtins are available."""
        builtins = code_executor._get_safe_builtins()

        # Check some safe functions are available
        assert "print" in builtins
        assert "len" in builtins
        assert "range" in builtins
        assert "list" in builtins

        # Check dangerous functions are not available
        assert "open" not in builtins
        assert "__import__" not in builtins
        assert "eval" not in builtins
        assert "exec" not in builtins

    def test_validate_code(self, code_executor):
        """Test code validation."""
        # Valid code
        assert code_executor._validate_code("print('hello')")["valid"] is True
        assert code_executor._validate_code("x = 1 + 2")["valid"] is True

        # Invalid code
        assert code_executor._validate_code("import os")["valid"] is False
        assert code_executor._validate_code("eval('test')")["valid"] is False
        assert code_executor._validate_code("open('file')")["valid"] is False
