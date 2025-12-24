"""
Base MCP Tool - Abstract base class for all MCP tools.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str = "string"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


class BaseMCPTool(ABC):
    """
    Abstract base class for MCP tools.

    All tools must implement:
    - name: Unique identifier
    - description: Human-readable description
    - parameters: List of parameters
    - execute: Async execution method
    """

    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = []

    def __init__(self):
        if not self.name:
            raise ValueError("Tool must have a name")

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            Tool execution result.
        """
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert tool to JSON schema for LLM function calling.

        Returns:
            Dict conforming to OpenAI/Claude function calling format.
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_langchain_tool(self):
        """
        Convert to a LangChain-compatible tool.

        Returns:
            LangChain StructuredTool instance.
        """
        from langchain_core.tools import StructuredTool
        import asyncio

        # Create sync wrapper for async execute
        def sync_execute(**kwargs):
            return asyncio.run(self.execute(**kwargs))

        return StructuredTool.from_function(
            func=sync_execute,
            coroutine=self.execute,
            name=self.name,
            description=self.description,
        )

    def validate_args(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and normalize arguments.

        Args:
            **kwargs: Arguments to validate.

        Returns:
            Validated arguments dict.

        Raises:
            ValueError: If required argument is missing.
        """
        validated = {}
        for param in self.parameters:
            if param.name in kwargs:
                validated[param.name] = kwargs[param.name]
            elif param.required:
                if param.default is not None:
                    validated[param.name] = param.default
                else:
                    raise ValueError(f"Missing required parameter: {param.name}")
            elif param.default is not None:
                validated[param.name] = param.default

        return validated
