"""
LLM Client for Memory Service.
用于记忆摘要等LLM调用功能。
"""

from typing import List, Dict, Any, Optional

import httpx

from services.common.logging import get_logger

logger = get_logger(__name__)


class OllamaLLMClient:
    """
    Simple Ollama LLM client for memory service.
    Used for conversation summarization and other LLM operations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
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
        Chat with Ollama LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/chat"

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
                logger.debug(f"Calling Ollama API: {url}, model={self.model}")
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                content = data.get("message", {}).get("content", "")
                logger.debug(f"Ollama response: {content[:100]}..." if content else "Empty response")

                return content

            except httpx.TimeoutException as e:
                logger.error(f"Ollama API timeout after {self.timeout}s: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama API HTTP error: {e.response.status_code}")
                raise
            except Exception as e:
                logger.error(f"Ollama API error: {type(e).__name__}: {e}")
                raise

    async def summarize(
        self,
        text: str,
        max_length: int = 500,
    ) -> str:
        """
        Summarize text.

        Args:
            text: Text to summarize
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        prompt = f"""请将以下内容总结为简洁的要点，保留关键信息，总结长度不超过{max_length}字：

{text}

总结："""

        return await self.generate(prompt, temperature=0.3, max_tokens=max_length)
