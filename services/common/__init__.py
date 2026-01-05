"""Common utilities for microservices."""

from .config import ServiceConfig, get_service_config
from .nacos_client import NacosClient, ServiceInstance
from .http_client import (
    HTTPClientBase,
    HTTPClientPool,
)
from .logging import setup_logging, get_logger
from .fastapi_nacos import (
    NacosServiceConfig,
    NacosServiceManager,
    create_nacos_lifespan,
    get_nacos_manager,
    ServiceRegistry,
    ServiceNames,
    SERVICE_PORTS,
)
from .service_client import (
    ServiceClient,
    ServiceClientConfig,
    LoadBalanceStrategy,
    LLMServiceClient,
    RAGServiceClient,
    MCPServiceClient,
    AgentServiceClient,
    get_service_client,
    close_service_client,
)

__all__ = [
    # Config
    "ServiceConfig",
    "get_service_config",
    # Service Discovery
    "NacosClient",
    "ServiceInstance",
    # FastAPI Nacos Integration
    "NacosServiceConfig",
    "NacosServiceManager",
    "create_nacos_lifespan",
    "get_nacos_manager",
    "ServiceRegistry",
    "ServiceNames",
    "SERVICE_PORTS",
    # Service Clients
    "ServiceClient",
    "ServiceClientConfig",
    "LoadBalanceStrategy",
    "LLMServiceClient",
    "RAGServiceClient",
    "MCPServiceClient",
    "AgentServiceClient",
    "get_service_client",
    "close_service_client",
    # HTTP Clients (legacy)
    "HTTPClientBase",
    "HTTPClientPool",
    # Logging
    "setup_logging",
    "get_logger",
]
