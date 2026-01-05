"""Tool Registry - Central management for MCP tools."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Type

from services.common.logging import get_logger
from .tools.base import BaseTool

logger = get_logger(__name__)

DEFAULT_TOOL_TIMEOUT = 60.0


class ToolRegistry:
    """
    Central registry for MCP tools.

    Provides:
    - Tool registration and lookup
    - Tool execution with error handling and timeout
    - Tool schema generation for LLM binding
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def register_class(self, tool_class: Type[BaseTool], **kwargs) -> None:
        """Register a tool by class, instantiating with kwargs."""
        tool = tool_class(**kwargs)
        self.register(tool)

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_by_tags(self, tags: List[str]) -> List[BaseTool]:
        """Get tools filtered by tags."""
        if not tags:
            return self.get_all()

        return [
            tool for tool in self._tools.values()
            if any(tag in tool.tags for tag in tags)
        ]

    def get_by_name_pattern(self, pattern: str) -> List[BaseTool]:
        """Get tools matching a name pattern."""
        import fnmatch
        return [
            tool for tool in self._tools.values()
            if fnmatch.fnmatch(tool.name, pattern)
        ]

    def get_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with timeout control.

        Args:
            name: Tool name.
            arguments: Arguments dict to pass to tool.
            timeout: Execution timeout in seconds.

        Returns:
            Dict with 'success', 'result' or 'error' keys, plus timing info.
        """
        tool = self.get(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' not found",
                "error_code": "TOOL_NOT_FOUND",
                "tool_name": name,
            }

        timeout = timeout or tool.timeout_seconds or DEFAULT_TOOL_TIMEOUT
        start_time = time.time()

        try:
            # Validate arguments
            validated_args = tool.validate_args(**arguments)

            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(**validated_args),
                timeout=timeout
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(f"Tool '{name}' executed successfully in {elapsed_ms}ms")

            return {
                "success": True,
                "result": result,
                "tool_name": name,
                "execution_time_ms": elapsed_ms,
            }

        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Tool '{name}' timed out after {timeout}s"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_code": "TIMEOUT",
                "tool_name": name,
                "execution_time_ms": elapsed_ms,
            }

        except ValueError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Invalid arguments: {e}"
            logger.error(f"Tool '{name}' argument validation failed: {e}")
            return {
                "success": False,
                "error": error_msg,
                "error_code": "INVALID_ARGUMENTS",
                "tool_name": name,
                "execution_time_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Tool '{name}' execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "EXECUTION_ERROR",
                "tool_name": name,
                "execution_time_ms": elapsed_ms,
            }

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM binding."""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_langchain_tools(self) -> List:
        """Get tools as LangChain Tool objects."""
        return [tool.to_langchain_tool() for tool in self._tools.values()]


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def clear_registry():
    """Clear and reset the global registry."""
    global _registry
    _registry = None
