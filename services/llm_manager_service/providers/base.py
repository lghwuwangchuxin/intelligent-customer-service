"""Base LLM Provider interface."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from enum import Enum


class MessageRole(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ToolCall:
    """Tool call from LLM."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 300.0,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_kwargs = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name."""
        pass

    @property
    def supports_tool_calling(self) -> bool:
        """Whether provider supports tool calling."""
        return False

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a response."""
        pass

    def generate_sync(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous generate."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate(messages, temperature, max_tokens, **kwargs)
        )

    def stream_sync(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Synchronous stream."""
        async def collect():
            chunks = []
            async for chunk in self.stream(messages, temperature, max_tokens, **kwargs):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(collect())
        yield from chunks

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to provider."""
        try:
            response = await self.generate(
                messages=[Message(role="user", content="Hi")],
                max_tokens=10,
            )
            return {
                "status": "success",
                "model": self.model,
                "provider": self.provider_name,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.model,
                "provider": self.provider_name,
            }

    def get_info(self) -> Dict[str, Any]:
        """Get provider info."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
            "supports_tool_calling": self.supports_tool_calling,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
