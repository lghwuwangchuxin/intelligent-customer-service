"""
LLM Client for Multi-Agent System.
Provides a unified interface for calling LLM services.

Configuration Priority:
1. Constructor parameters
2. Environment variables
3. Config file defaults

Supported Providers:
- dashscope (Aliyun DashScope - default)
- openai (OpenAI / OpenAI-compatible)
- ollama (Local Ollama)
"""

import os
import logging
from typing import Any, Dict, List, Optional

import httpx

from .config import LLMConfig, get_llm_config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM Client for agent reasoning.

    Supports multiple LLM providers through a unified interface.
    Uses OpenAI-compatible API for cloud providers (DashScope, OpenAI).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Configuration priority: parameters > env vars > config defaults

        Args:
            base_url: LLM service base URL
            model: Model name
            api_key: API key for authentication
            timeout: Request timeout in seconds
            provider: LLM provider (dashscope/openai/ollama)
        """
        # Load default config (uses env vars + defaults)
        config = get_llm_config()

        # Override with constructor parameters if provided
        self.base_url = base_url or config.base_url
        self.model = model or config.model
        self.api_key = api_key or config.api_key
        self.timeout = timeout or config.timeout
        self.provider = provider or config.provider

        # Auto-detect provider from URL if not set
        if not self.provider:
            self._detect_provider()

        logger.info(
            f"LLM Client initialized: provider={self.provider}, "
            f"model={self.model}, timeout={self.timeout}s"
        )

    def _detect_provider(self) -> None:
        """Auto-detect provider from base URL."""
        url_lower = self.base_url.lower()
        if "dashscope" in url_lower:
            self.provider = "dashscope"
        elif "openai" in url_lower:
            self.provider = "openai"
        elif "anthropic" in url_lower:
            self.provider = "anthropic"
        elif "localhost" in url_lower or "11434" in url_lower:
            self.provider = "ollama"
        else:
            # Default to OpenAI-compatible API
            self.provider = "openai"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Chat with the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if self.provider == "ollama":
            return await self._chat_ollama(messages, temperature, max_tokens)
        else:
            # dashscope, openai, and other OpenAI-compatible APIs
            return await self._chat_openai_compatible(messages, temperature, max_tokens)

    async def _chat_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Call OpenAI-compatible API (works for DashScope, OpenAI, etc.)

        DashScope API: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
        OpenAI API: https://api.openai.com/v1/chat/completions
        """
        # Build URL - handle both with and without trailing slash
        base = self.base_url.rstrip("/")
        if "/chat/completions" not in base:
            url = f"{base}/chat/completions"
        else:
            url = base

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=30.0)) as client:
            try:
                logger.debug(f"Calling {self.provider} API: {url}")
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                # Extract content from response
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Log token usage if available
                usage = result.get("usage", {})
                if usage:
                    logger.debug(
                        f"Token usage: prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}"
                    )

                return content

            except httpx.TimeoutException as e:
                logger.error(f"{self.provider} API timeout after {self.timeout}s: {type(e).__name__}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"{self.provider} API HTTP error: {e.response.status_code} - "
                    f"{e.response.text[:200] if e.response.text else 'No response body'}"
                )
                raise
            except httpx.HTTPError as e:
                logger.error(f"{self.provider} API error: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                logger.error(f"{self.provider} API unexpected error: {type(e).__name__}: {e}")
                raise

    async def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Ollama API (local LLM)."""
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=30.0)) as client:
            try:
                logger.debug(f"Calling Ollama API: {url}")
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
            except httpx.TimeoutException as e:
                logger.error(f"Ollama API timeout after {self.timeout}s: {type(e).__name__}")
                raise
            except httpx.HTTPError as e:
                logger.error(f"Ollama API error: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                logger.error(f"Ollama API unexpected error: {type(e).__name__}: {e}")
                raise

    def get_info(self) -> Dict[str, Any]:
        """Get client configuration info."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }
