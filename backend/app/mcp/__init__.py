"""
MCP (Model Context Protocol) integration for intelligent customer service.
Provides tools for knowledge base operations, web search, and code execution.
"""

from app.mcp.registry import ToolRegistry, get_tool_registry
from app.mcp.tools.base import BaseMCPTool

__all__ = ["ToolRegistry", "get_tool_registry", "BaseMCPTool"]
