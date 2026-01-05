"""LLM Manager Service - Main service implementation."""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field

from services.common.logging import get_logger

from .config import LLMConfig, PROVIDER_CONFIGS, ProviderConfig
from .providers import BaseLLMProvider, Message, LLMResponse, OllamaProvider, OpenAICompatibleProvider

logger = get_logger(__name__)

# Retry configuration
RETRYABLE_ERRORS = (asyncio.TimeoutError, ConnectionError, OSError)


@dataclass
class GenerateRequest:
    """Request for text generation."""
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict]] = None
    stream: bool = False


@dataclass
class GenerateResponse:
    """Response from text generation."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    tool_calls: List[Dict] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


class LLMManagerService:
    """
    LLM Manager Service.

    Provides unified interface for multiple LLM providers:
    - Ollama (local models)
    - OpenAI
    - Anthropic Claude
    - DeepSeek
    - 阿里云通义千问
    - 智谱AI
    - 月之暗面 Kimi
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM Manager Service.

        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig.from_env()
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._default_provider: Optional[BaseLLMProvider] = None
        self._initialized = False

    @classmethod
    def from_config(cls, config=None) -> "LLMManagerService":
        """Create service from configuration."""
        if config is None:
            llm_config = LLMConfig.from_env()
        else:
            llm_config = LLMConfig(
                provider=getattr(config, 'llm_provider', 'ollama'),
                model=getattr(config, 'llm_model', 'qwen3:latest'),
                base_url=getattr(config, 'llm_base_url', None),
                api_key=getattr(config, 'llm_api_key', None),
                temperature=getattr(config, 'llm_temperature', 0.7),
                max_tokens=getattr(config, 'llm_max_tokens', 2048),
            )
        return cls(llm_config)

    def _create_provider(
        self,
        provider: str,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        # Get provider config
        provider_config = PROVIDER_CONFIGS.get(provider)
        if not provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        # Use default base_url if not provided
        if not base_url:
            base_url = provider_config.base_url

        # Create provider instance
        if provider == "ollama":
            return OllamaProvider(
                model=model,
                base_url=base_url,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                timeout=kwargs.get("timeout", self.config.timeout),
            )
        else:
            # Use OpenAI-compatible provider for others
            return OpenAICompatibleProvider(
                model=model,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                timeout=kwargs.get("timeout", self.config.timeout),
            )

    def initialize(self):
        """Initialize service with default provider."""
        if self._initialized:
            return

        # Create default provider
        self._default_provider = self._create_provider(
            provider=self.config.provider,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        key = f"{self.config.provider}:{self.config.model}"
        self._providers[key] = self._default_provider

        self._initialized = True
        logger.info(f"LLM Manager initialized with {self.config.provider}:{self.config.model}")

    def get_provider(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BaseLLMProvider:
        """
        Get or create a provider instance.

        Args:
            provider: Provider name
            model: Model name
            base_url: Base URL
            api_key: API key

        Returns:
            Provider instance
        """
        self.initialize()

        # Use defaults if not provided
        provider = provider or self.config.provider
        model = model or self.config.model

        key = f"{provider}:{model}"

        # Return cached provider if available
        if key in self._providers:
            return self._providers[key]

        # Create new provider
        new_provider = self._create_provider(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        self._providers[key] = new_provider

        return new_provider

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> GenerateResponse:
        """
        Generate a response.

        Args:
            messages: Chat messages
            provider: Provider name
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Tools for function calling
            timeout: Request timeout
            max_retries: Maximum retries

        Returns:
            GenerateResponse
        """
        llm_provider = self.get_provider(provider, model)

        # Convert messages
        msg_objects = [Message.from_dict(m) for m in messages]

        # Retry logic
        max_retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    llm_provider.generate(
                        messages=msg_objects,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        **kwargs,
                    ),
                    timeout=timeout or self.config.timeout,
                )

                return GenerateResponse(
                    content=response.content,
                    model=response.model,
                    provider=response.provider,
                    finish_reason=response.finish_reason,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                )

            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
                continue

            except Exception as e:
                logger.error(f"Generate failed: {e}")
                raise

        # All retries exhausted
        raise last_error or Exception("All retries exhausted")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response.

        Args:
            messages: Chat messages
            provider: Provider name
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Yields:
            Response chunks
        """
        llm_provider = self.get_provider(provider, model)
        msg_objects = [Message.from_dict(m) for m in messages]

        async for chunk in llm_provider.stream(
            messages=msg_objects,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    def generate_sync(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> GenerateResponse:
        """Synchronous generate."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate(messages, **kwargs)
        )

    async def switch_model(
        self,
        provider: str,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Switch to a different model.

        Args:
            provider: Provider name
            model: Model name
            base_url: Base URL
            api_key: API key

        Returns:
            New model info
        """
        # Update config
        self.config.provider = provider
        self.config.model = model
        if base_url:
            self.config.base_url = base_url
        if api_key:
            self.config.api_key = api_key

        # Get or create provider
        new_provider = self.get_provider(provider, model, base_url, api_key)
        self._default_provider = new_provider

        logger.info(f"Switched to {provider}:{model}")

        return new_provider.get_info()

    async def test_connection(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test connection to provider."""
        llm_provider = self.get_provider(provider, model)
        return await llm_provider.test_connection()

    async def list_models(
        self,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available models for a provider."""
        provider = provider or self.config.provider
        llm_provider = self.get_provider(provider)

        if hasattr(llm_provider, 'list_models'):
            return await llm_provider.list_models()

        # Return configured models
        provider_config = PROVIDER_CONFIGS.get(provider)
        if provider_config:
            return [{"id": m, "provider": provider} for m in provider_config.models]

        return []

    def list_providers(self) -> List[Dict[str, Any]]:
        """List available providers."""
        return [
            {
                "name": config.name,
                "display_name": config.display_name,
                "base_url": config.base_url,
                "models": config.models,
                "requires_api_key": config.requires_api_key,
            }
            for config in PROVIDER_CONFIGS.values()
        ]

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.to_dict()

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        status = {
            "status": "healthy",
            "provider": self.config.provider,
            "model": self.config.model,
        }

        try:
            result = await self.test_connection()
            if result.get("status") != "success":
                status["status"] = "degraded"
                status["error"] = result.get("error")
        except Exception as e:
            status["status"] = "unhealthy"
            status["error"] = str(e)

        return status
