"""MCP Service - Business logic implementation."""

import json
from typing import List, Optional, Dict, Any

from services.common.logging import get_logger
from .registry import ToolRegistry, get_tool_registry
from .tools.base import BaseTool

logger = get_logger(__name__)


class MCPService:
    """MCP Tool Service implementation."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        rag_client=None,
    ):
        """
        Initialize MCP service.

        Args:
            registry: Tool registry (uses global if not provided)
            rag_client: HTTP client for RAG service
        """
        self.registry = registry or get_tool_registry()
        self.rag_client = rag_client
        self._initialized = False

    def initialize_default_tools(
        self,
        enable_web_search: bool = True,
        enable_code_execution: bool = True,
        enable_knowledge: bool = True,
        code_execution_timeout: int = 30,
    ):
        """
        Initialize default tools.

        Args:
            enable_web_search: Enable web search tools
            enable_code_execution: Enable code execution tool
            enable_knowledge: Enable knowledge base tools
            code_execution_timeout: Timeout for code execution
        """
        if self._initialized:
            return

        # Knowledge tools
        if enable_knowledge:
            from .tools.knowledge import KnowledgeSearchTool, KnowledgeAddTool
            self.registry.register(KnowledgeSearchTool(rag_client=self.rag_client))
            self.registry.register(KnowledgeAddTool(rag_client=self.rag_client))
            logger.info("Registered knowledge tools")

        # Web search tools
        if enable_web_search:
            from .tools.web_search import WebSearchTool, WebFetchTool
            self.registry.register(WebSearchTool())
            self.registry.register(WebFetchTool())
            logger.info("Registered web search tools")

        # Code execution tool
        if enable_code_execution:
            from .tools.code_executor import CodeExecutorTool
            self.registry.register(CodeExecutorTool(timeout=code_execution_timeout))
            logger.info("Registered code execution tool")

        self._initialized = True
        logger.info(
            f"MCP Service initialized with {len(self.registry.get_names())} tools: "
            f"{self.registry.get_names()}"
        )

    def list_tools(
        self,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available tools.

        Args:
            tags: Filter by tags
            name_pattern: Filter by name pattern

        Returns:
            List of tool definitions
        """
        if tags:
            tools = self.registry.get_by_tags(tags)
        elif name_pattern:
            tools = self.registry.get_by_name_pattern(name_pattern)
        else:
            tools = self.registry.get_all()

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.get_input_schema(),
                "tags": tool.tags,
                "is_async": tool.is_async,
                "timeout_seconds": tool.timeout_seconds,
            }
            for tool in tools
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool.

        Args:
            tool_name: Name of tool to execute
            arguments: JSON string of arguments
            timeout_seconds: Override timeout

        Returns:
            Execution result
        """
        # Parse arguments
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON arguments: {e}",
                "error_code": "INVALID_JSON",
            }

        # Execute tool
        result = await self.registry.execute(
            name=tool_name,
            arguments=args,
            timeout=timeout_seconds,
        )

        return result

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Tool name

        Returns:
            Tool schema or None
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.get_input_schema(),
            "tags": tool.tags,
            "is_async": tool.is_async,
            "timeout_seconds": tool.timeout_seconds,
        }

    def get_langchain_tools(self) -> List:
        """Get all tools as LangChain tools."""
        return self.registry.get_langchain_tools()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM binding."""
        return self.registry.get_tool_schemas()

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool."""
        self.registry.register(tool)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        return self.registry.unregister(name)
