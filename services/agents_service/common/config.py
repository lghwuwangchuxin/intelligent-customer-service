"""
Configuration for Multi-Agent System.
Priority: Environment Variables > .env.agents File > Code Defaults

Automatically loads configuration from .env.agents file if present.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Auto-load .env.agents file
# =============================================================================

def _load_env_file():
    """Load environment variables from .env.agents file."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / ".env.agents",  # agents/.env.agents
        Path(__file__).parent.parent.parent / ".env.agents",  # project root/.env.agents
        Path(__file__).parent.parent.parent / "agents" / ".env.agents",  # explicit agents/.env.agents
    ]

    for env_path in possible_paths:
        if env_path.exists():
            logger.info(f"Loading configuration from: {env_path}")
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        # Parse key=value
                        if '=' in line:
                            key, _, value = line.partition('=')
                            key = key.strip()
                            value = value.strip()
                            # Only set if not already in environment (env vars take priority)
                            if key and key not in os.environ:
                                os.environ[key] = value
                return True
            except Exception as e:
                logger.warning(f"Failed to load {env_path}: {e}")

    logger.debug("No .env.agents file found, using defaults and environment variables")
    return False

# Load env file on module import
_load_env_file()

# =============================================================================
# Default Configuration Values (used if env vars not set)
# =============================================================================

# Aliyun DashScope (OpenAI-compatible)
DEFAULT_LLM_PROVIDER = "dashscope"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_LLM_API_KEY = ""  # Must be set via env var or override

# Alternative: Ollama (local)
# DEFAULT_LLM_PROVIDER = "ollama"
# DEFAULT_LLM_BASE_URL = "http://localhost:11434"
# DEFAULT_LLM_MODEL = "qwen2.5:7b"

# Alternative: OpenAI
# DEFAULT_LLM_PROVIDER = "openai"
# DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
# DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Request settings
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_LLM_MAX_TOKENS = 2048
DEFAULT_LLM_TEMPERATURE = 0.7


@dataclass
class LLMConfig:
    """LLM Configuration with environment variable support."""

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY))
    timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", str(DEFAULT_LLM_TIMEOUT))))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS))))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", str(DEFAULT_LLM_TEMPERATURE))))

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Auto-detect provider from URL if not explicitly set
        if not os.getenv("LLM_PROVIDER"):
            if "dashscope" in self.base_url.lower():
                self.provider = "dashscope"
            elif "openai" in self.base_url.lower():
                self.provider = "openai"
            elif "anthropic" in self.base_url.lower():
                self.provider = "anthropic"
            elif "localhost" in self.base_url.lower() or "11434" in self.base_url:
                self.provider = "ollama"

        # Log configuration (hide API key)
        logger.info(
            f"LLM Config: provider={self.provider}, model={self.model}, "
            f"base_url={self.base_url}, api_key={'***' if self.api_key else 'NOT SET'}"
        )

        # Warn if API key not set for cloud providers
        if self.provider in ("dashscope", "openai", "anthropic") and not self.api_key:
            logger.warning(
                f"API key not set for {self.provider}! "
                f"Set LLM_API_KEY environment variable."
            )


@dataclass
class NacosConfig:
    """Nacos service registry configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("NACOS_ENABLED", "true").lower() == "true")
    server_addresses: str = field(default_factory=lambda: os.getenv("NACOS_SERVER_ADDRESSES", "localhost:8848"))
    namespace: str = field(default_factory=lambda: os.getenv("NACOS_NAMESPACE", "public"))
    group: str = field(default_factory=lambda: os.getenv("NACOS_GROUP", "DEFAULT_GROUP"))
    username: str = field(default_factory=lambda: os.getenv("NACOS_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("NACOS_PASSWORD", ""))
    cluster_name: str = field(default_factory=lambda: os.getenv("NACOS_CLUSTER_NAME", "DEFAULT"))
    heartbeat_interval: int = field(default_factory=lambda: int(os.getenv("NACOS_HEARTBEAT_INTERVAL", "5")))

    def __post_init__(self):
        """Log configuration."""
        if self.enabled:
            logger.info(
                f"Nacos Config: server={self.server_addresses}, namespace={self.namespace}, "
                f"group={self.group}, cluster={self.cluster_name}"
            )
        else:
            logger.info("Nacos service registration is disabled")


@dataclass
class AgentConfig:
    """Agent-specific configuration."""

    # Agent service ports
    travel_assistant_port: int = field(default_factory=lambda: int(os.getenv("AGENT_TRAVEL_ASSISTANT_PORT", "9001")))
    charging_manager_port: int = field(default_factory=lambda: int(os.getenv("AGENT_CHARGING_MANAGER_PORT", "9002")))
    billing_advisor_port: int = field(default_factory=lambda: int(os.getenv("AGENT_BILLING_ADVISOR_PORT", "9003")))
    emergency_support_port: int = field(default_factory=lambda: int(os.getenv("AGENT_EMERGENCY_SUPPORT_PORT", "9004")))
    data_analyst_port: int = field(default_factory=lambda: int(os.getenv("AGENT_DATA_ANALYST_PORT", "9005")))
    maintenance_expert_port: int = field(default_factory=lambda: int(os.getenv("AGENT_MAINTENANCE_EXPERT_PORT", "9006")))
    energy_advisor_port: int = field(default_factory=lambda: int(os.getenv("AGENT_ENERGY_ADVISOR_PORT", "9007")))
    scheduling_advisor_port: int = field(default_factory=lambda: int(os.getenv("AGENT_SCHEDULING_ADVISOR_PORT", "9008")))

    # Agent host
    agent_host: str = field(default_factory=lambda: os.getenv("AGENT_HOST", "0.0.0.0"))


def get_llm_config() -> LLMConfig:
    """Get LLM configuration (singleton-like)."""
    return LLMConfig()


def get_agent_config() -> AgentConfig:
    """Get agent configuration (singleton-like)."""
    return AgentConfig()


def get_nacos_config() -> NacosConfig:
    """Get Nacos configuration (singleton-like)."""
    return NacosConfig()
