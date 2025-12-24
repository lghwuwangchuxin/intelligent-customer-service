"""
Tool Registry - Central management for MCP tools.
Handles tool registration, lookup, and execution.

性能优化
--------
- 工具执行超时控制 (默认 60 秒)
- 执行时间监控和日志记录
- 优雅的错误处理

Version: 1.1.0 (添加超时控制)
"""
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Type
from functools import lru_cache

from app.mcp.tools.base import BaseMCPTool

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_TOOL_TIMEOUT = 60.0  # 默认工具执行超时（秒）


class ToolRegistry:
    """
    Central registry for MCP tools.

    Provides:
    - Tool registration and lookup
    - Tool execution with error handling
    - Tool schema generation for LLM binding
    """

    def __init__(self):
        self._tools: Dict[str, BaseMCPTool] = {}
        self._initialized = False

    def register(self, tool: BaseMCPTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: Tool instance to register.
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def register_class(self, tool_class: Type[BaseMCPTool], **kwargs) -> None:
        """
        Register a tool by class, instantiating with kwargs.

        Args:
            tool_class: Tool class to instantiate and register.
            **kwargs: Arguments to pass to tool constructor.
        """
        tool = tool_class(**kwargs)
        self.register(tool)

    def get(self, name: str) -> Optional[BaseMCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[BaseMCPTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    async def execute(
        self,
        name: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with timeout control.

        Args:
            name: Tool name.
            timeout: Execution timeout in seconds. Defaults to 60s.
            **kwargs: Arguments to pass to tool.

        Returns:
            Dict with 'success', 'result' or 'error' keys, plus timing info.
        """
        tool = self.get(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' not found",
                "tool_name": name,
            }

        timeout = timeout or DEFAULT_TOOL_TIMEOUT
        start_time = time.time()

        try:
            # 带超时控制的工具执行
            result = await asyncio.wait_for(
                tool.execute(**kwargs),
                timeout=timeout
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[ToolRegistry] Tool '{name}' executed successfully in {elapsed_ms}ms"
            )

            return {
                "success": True,
                "result": result,
                "tool_name": name,
                "elapsed_ms": elapsed_ms,
            }

        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Tool '{name}' timed out after {timeout}s"
            logger.error(f"[ToolRegistry] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "tool_name": name,
                "elapsed_ms": elapsed_ms,
                "timeout": True,
            }

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[ToolRegistry] Tool '{name}' execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": name,
                "elapsed_ms": elapsed_ms,
            }

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get tool schemas for LLM binding.

        Returns:
            List of tool definitions compatible with function calling.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def get_langchain_tools(self) -> List:
        """
        Get tools as LangChain Tool objects.

        Returns:
            List of LangChain-compatible tools.
        """
        return [tool.to_langchain_tool() for tool in self._tools.values()]

    def initialize_default_tools(
        self,
        vector_store=None,
        document_processor=None,
        rag_service=None,
    ) -> None:
        """
        Initialize and register default tools.

        Args:
            vector_store: Vector store manager instance.
            document_processor: Document processor instance.
            rag_service: RAG service instance.
        """
        if self._initialized:
            return

        from config.settings import settings

        # Knowledge base tools
        if vector_store or rag_service:
            from app.mcp.tools.knowledge import (
                KnowledgeSearchTool,
                KnowledgeAddTextTool,
            )
            if rag_service:
                self.register(KnowledgeSearchTool(rag_service=rag_service))
                self.register(KnowledgeAddTextTool(rag_service=rag_service))

        # Web search tool
        if settings.MCP_WEB_SEARCH_ENABLED:
            from app.mcp.tools.web_search import WebSearchTool
            self.register(WebSearchTool())

        # Code execution tool
        if settings.MCP_CODE_EXECUTION_ENABLED:
            from app.mcp.tools.code_executor import CodeExecutorTool
            self.register(CodeExecutorTool(
                timeout=settings.MCP_CODE_EXECUTION_TIMEOUT
            ))

        self._initialized = True
        logger.info(f"Initialized {len(self._tools)} default tools")


# Global registry instance
_registry: Optional[ToolRegistry] = None


@lru_cache()
def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
