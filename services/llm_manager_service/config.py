"""LLM Manager configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    display_name: str
    base_url: str
    models: List[str]
    requires_api_key: bool = True
    default_model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.default_model is None and self.models:
            self.default_model = self.models[0]


# Provider configurations
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "ollama": ProviderConfig(
        name="ollama",
        display_name="Ollama (本地模型)",
        base_url="http://localhost:11434",
        models=["qwen3:latest", "llama3:latest", "deepseek-r1:7b", "mistral:latest"],
        requires_api_key=False,
    ),
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        requires_api_key=True,
    ),
    "claude": ProviderConfig(
        name="claude",
        display_name="Anthropic Claude",
        base_url="https://api.anthropic.com",
        models=["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        requires_api_key=True,
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        models=["deepseek-chat", "deepseek-coder"],
        requires_api_key=True,
    ),
    "dashscope": ProviderConfig(
        name="dashscope",
        display_name="阿里云通义千问",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=["qwen-turbo", "qwen-plus", "qwen-max"],
        requires_api_key=True,
    ),
    "zhipu": ProviderConfig(
        name="zhipu",
        display_name="智谱AI",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        models=["glm-4", "glm-4-flash", "glm-3-turbo"],
        requires_api_key=True,
    ),
    "moonshot": ProviderConfig(
        name="moonshot",
        display_name="月之暗面 Kimi",
        base_url="https://api.moonshot.cn/v1",
        models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        requires_api_key=True,
    ),
}


@dataclass
class LLMConfig:
    """Configuration for LLM instance."""

    provider: str = "ollama"
    model: str = "qwen3:latest"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Extended thinking (Claude)
    extended_thinking: bool = False
    thinking_budget: int = 10000

    # Extra parameters
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set default base_url from provider config
        if self.base_url is None and self.provider in PROVIDER_CONFIGS:
            self.base_url = PROVIDER_CONFIGS[self.provider].base_url

        # Load API key from environment if not provided
        if self.api_key is None:
            env_key = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_key, "")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            model=os.getenv("LLM_MODEL", "qwen3:latest"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=float(os.getenv("LLM_TIMEOUT", "300")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
