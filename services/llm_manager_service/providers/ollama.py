"""Ollama LLM Provider."""

import time
import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator

from .base import BaseLLMProvider, Message, LLMResponse, ToolCall


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM Provider for local models."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def supports_tool_calling(self) -> bool:
        # Ollama supports tool calling for some models
        return True

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using Ollama."""
        start_time = time.time()

        # Build request
        request_data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        # Add tools if provided
        if tools:
            request_data["tools"] = tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()

        # Parse response
        message = data.get("message", {})
        content = message.get("content", "")

        # Parse tool calls
        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=json.dumps(tc.get("function", {}).get("arguments", {})),
                ))

        latency_ms = int((time.time() - start_time) * 1000)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            finish_reason=data.get("done_reason", "stop"),
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a response using Ollama."""
        request_data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=request_data,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

        models = []
        for model in data.get("models", []):
            models.append({
                "name": model.get("name"),
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
                "digest": model.get("digest"),
            })

        return models

    async def check_health(self) -> Dict[str, Any]:
        """Check Ollama health."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "version": response.json().get("version"),
                    }
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def pull_model(self, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Pull a model from Ollama."""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model_name},
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
