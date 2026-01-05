"""
Common utilities for multi-agent system.
Provides shared code for all specialized agents.
"""

from .mock_data import ChargingMockData, EnergyMockData
from .base_executor import BaseAgentExecutor
from .llm_client import LLMClient
from .config import get_llm_config, get_agent_config, get_nacos_config, NacosConfig
from .nacos_registry import (
    NacosServiceRegistry,
    get_registry,
    register_agent,
    deregister_agent,
    discover_agents,
)

__all__ = [
    "ChargingMockData",
    "EnergyMockData",
    "BaseAgentExecutor",
    "LLMClient",
    "get_llm_config",
    "get_agent_config",
    "get_nacos_config",
    "NacosConfig",
    "NacosServiceRegistry",
    "get_registry",
    "register_agent",
    "deregister_agent",
    "discover_agents",
]
