"""Tests for Tool Registry."""

import pytest
import asyncio

from services.mcp_service.registry import ToolRegistry, get_tool_registry, clear_registry


class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def test_register_tool(self, registry, mock_tool):
        """Test registering a tool."""
        registry.register(mock_tool)
        assert "mock_tool" in registry.get_names()
        assert registry.get("mock_tool") is mock_tool

    def test_register_duplicate_tool(self, registry, mock_tool):
        """Test registering a duplicate tool overwrites."""
        registry.register(mock_tool)
        registry.register(mock_tool)  # Should not raise
        assert len(registry.get_names()) == 1

    def test_unregister_tool(self, registry, mock_tool):
        """Test unregistering a tool."""
        registry.register(mock_tool)
        result = registry.unregister("mock_tool")
        assert result is True
        assert "mock_tool" not in registry.get_names()

    def test_unregister_nonexistent_tool(self, registry):
        """Test unregistering a tool that doesn't exist."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_tool(self, populated_registry):
        """Test getting a tool by name."""
        tool = populated_registry.get("mock_tool")
        assert tool is not None
        assert tool.name == "mock_tool"

    def test_get_nonexistent_tool(self, registry):
        """Test getting a tool that doesn't exist."""
        tool = registry.get("nonexistent")
        assert tool is None

    def test_get_all_tools(self, populated_registry):
        """Test getting all tools."""
        tools = populated_registry.get_all()
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "mock_tool" in names
        assert "slow_tool" in names
        assert "failing_tool" in names

    def test_get_by_tags(self, populated_registry):
        """Test filtering tools by tags."""
        test_tools = populated_registry.get_by_tags(["test"])
        assert len(test_tools) == 3  # All have 'test' tag

        mock_tools = populated_registry.get_by_tags(["mock"])
        assert len(mock_tools) == 1
        assert mock_tools[0].name == "mock_tool"

    def test_get_by_name_pattern(self, populated_registry):
        """Test filtering tools by name pattern."""
        tools = populated_registry.get_by_name_pattern("*_tool")
        assert len(tools) == 3

        tools = populated_registry.get_by_name_pattern("mock*")
        assert len(tools) == 1
        assert tools[0].name == "mock_tool"

    def test_get_names(self, populated_registry):
        """Test getting tool names."""
        names = populated_registry.get_names()
        assert len(names) == 3
        assert set(names) == {"mock_tool", "slow_tool", "failing_tool"}


class TestToolExecution:
    """Test cases for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, populated_registry):
        """Test successful tool execution."""
        result = await populated_registry.execute(
            "mock_tool",
            {"input": "test_value"}
        )
        assert result["success"] is True
        assert result["result"]["input"] == "test_value"
        assert result["result"]["optional_param"] == "default_value"
        assert result["tool_name"] == "mock_tool"
        assert "execution_time_ms" in result

    @pytest.mark.asyncio
    async def test_execute_tool_with_optional_param(self, populated_registry):
        """Test tool execution with optional parameter."""
        result = await populated_registry.execute(
            "mock_tool",
            {"input": "test", "optional_param": "custom_value"}
        )
        assert result["success"] is True
        assert result["result"]["optional_param"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry):
        """Test executing a tool that doesn't exist."""
        result = await registry.execute("nonexistent", {})
        assert result["success"] is False
        assert result["error_code"] == "TOOL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_execute_tool_missing_required_param(self, populated_registry):
        """Test executing tool with missing required parameter."""
        result = await populated_registry.execute("mock_tool", {})
        assert result["success"] is False
        assert result["error_code"] == "INVALID_ARGUMENTS"

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, populated_registry):
        """Test tool execution timeout."""
        result = await populated_registry.execute(
            "slow_tool",
            {"delay": 10},
            timeout=0.1
        )
        assert result["success"] is False
        assert result["error_code"] == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_execute_failing_tool(self, populated_registry):
        """Test executing a tool that throws an exception."""
        result = await populated_registry.execute("failing_tool", {})
        assert result["success"] is False
        assert result["error_code"] == "EXECUTION_ERROR"
        assert "always fails" in result["error"]


class TestToolSchemas:
    """Test cases for tool schema generation."""

    def test_get_tool_schemas(self, populated_registry):
        """Test getting tool schemas."""
        schemas = populated_registry.get_tool_schemas()
        assert len(schemas) == 3

        mock_schema = next(s for s in schemas if s["name"] == "mock_tool")
        assert mock_schema["description"] == "A mock tool for testing"
        assert "parameters" in mock_schema

    def test_get_langchain_tools(self, populated_registry):
        """Test getting LangChain tools."""
        # This test may fail if langchain is not installed
        try:
            lc_tools = populated_registry.get_langchain_tools()
            assert len(lc_tools) == 3
        except ImportError:
            pytest.skip("LangChain not installed")


class TestGlobalRegistry:
    """Test cases for global registry functions."""

    def test_get_tool_registry_singleton(self):
        """Test that get_tool_registry returns singleton."""
        clear_registry()
        reg1 = get_tool_registry()
        reg2 = get_tool_registry()
        assert reg1 is reg2

    def test_clear_registry(self, mock_tool):
        """Test clearing the global registry."""
        clear_registry()
        reg = get_tool_registry()
        reg.register(mock_tool)
        assert len(reg.get_names()) == 1

        clear_registry()
        reg = get_tool_registry()
        assert len(reg.get_names()) == 0
