"""OpenAI-compatible LLM Provider."""

import time
import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator

from .base import BaseLLMProvider, Message, LLMResponse, ToolCall


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    OpenAI-compatible LLM Provider.

    Works with:
    - OpenAI API
    - DeepSeek API
    - 阿里云通义千问 (DashScope)
    - 智谱AI
    - 月之暗面 Kimi
    - Any OpenAI-compatible API
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, api_key, **kwargs)
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def supports_tool_calling(self) -> bool:
        # Most OpenAI-compatible providers support tool calling
        return True

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using OpenAI-compatible API."""
        start_time = time.time()

        # Build request
        request_data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        # Add tools if provided
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add extra parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                request_data[key] = kwargs[key]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()

        # Parse response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Parse tool calls
        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", "{}"),
                ))

        # Parse usage
        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

        latency_ms = int((time.time() - start_time) * 1000)

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.provider_name,
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a response using OpenAI-compatible API."""
        request_data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=request_data,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

        models = []
        for model in data.get("data", []):
            models.append({
                "id": model.get("id"),
                "object": model.get("object"),
                "created": model.get("created"),
                "owned_by": model.get("owned_by"),
            })

        return models
