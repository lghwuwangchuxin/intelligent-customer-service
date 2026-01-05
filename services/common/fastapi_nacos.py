"""FastAPI integration with Nacos for service registration and discovery."""

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging import get_logger
from .nacos_client import NacosClient, ServiceInstance
from .config import ServiceConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)


@dataclass
class NacosServiceConfig:
    """Configuration for Nacos service registration."""

    # Service info
    service_name: str
    service_port: int
    service_host: Optional[str] = None  # Auto-detect if None

    # Nacos connection
    nacos_server_addresses: str = "localhost:8848"
    nacos_namespace: str = "public"
    nacos_group: str = "DEFAULT_GROUP"
    nacos_username: Optional[str] = None
    nacos_password: Optional[str] = None

    # Registration options
    enabled: bool = True
    weight: float = 1.0
    cluster_name: str = "DEFAULT"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    enable_heartbeat: bool = True

    # Health check
    health_check_path: str = "/health"
    health_check_interval: str = "30s"
    health_check_timeout: str = "10s"

    @classmethod
    def from_service_config(cls, config: ServiceConfig) -> "NacosServiceConfig":
        """Create from ServiceConfig."""
        return cls(
            service_name=config.service_name,
            service_port=config.http_port,
            service_host=config.host if config.host != "0.0.0.0" else None,
            nacos_server_addresses=config.nacos_server_addresses,
            nacos_namespace=config.nacos_namespace,
            nacos_group=config.nacos_group,
            nacos_username=config.nacos_username or None,
            nacos_password=config.nacos_password or None,
            enabled=config.nacos_enabled,
        )


class NacosServiceManager:
    """
    Manager for Nacos service registration and discovery.

    Integrates with FastAPI lifecycle to automatically register/deregister services.
    """

    def __init__(self, config: NacosServiceConfig):
        """
        Initialize Nacos service manager.

        Args:
            config: Nacos service configuration
        """
        self.config = config
        self._client: Optional[NacosClient] = None
        self._service_id: Optional[str] = None
        self._service_address: Optional[str] = None

    @property
    def client(self) -> Optional[NacosClient]:
        """Get Nacos client."""
        return self._client

    @property
    def service_id(self) -> Optional[str]:
        """Get registered service ID."""
        return self._service_id

    def _get_service_address(self) -> str:
        """Get service address for registration."""
        if self.config.service_host:
            return self.config.service_host

        # Try to get actual IP address
        hostname = socket.gethostname()
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return "127.0.0.1"

    async def startup(self):
        """
        Startup handler - register service with Nacos.

        Called during FastAPI startup.
        """
        if not self.config.enabled:
            logger.info("Nacos registration disabled")
            return

        try:
            self._client = NacosClient(
                server_addresses=self.config.nacos_server_addresses,
                namespace=self.config.nacos_namespace,
                username=self.config.nacos_username,
                password=self.config.nacos_password,
                group=self.config.nacos_group,
            )

            self._service_address = self._get_service_address()

            # Build metadata
            metadata = dict(self.config.metadata)
            if self.config.tags:
                metadata["tags"] = ",".join(self.config.tags)

            # Register service
            self._service_id = await self._client.register_service(
                name=self.config.service_name,
                port=self.config.service_port,
                address=self._service_address,
                tags=self.config.tags,
                meta=metadata,
                weight=self.config.weight,
                cluster_name=self.config.cluster_name,
                enable_heartbeat=self.config.enable_heartbeat,
                health_check_url=f"http://{self._service_address}:{self.config.service_port}{self.config.health_check_path}",
                health_check_interval=self.config.health_check_interval,
                health_check_timeout=self.config.health_check_timeout,
            )

            logger.info(
                f"Service registered with Nacos: {self.config.service_name} "
                f"(ID: {self._service_id}, Address: {self._service_address}:{self.config.service_port})"
            )

        except Exception as e:
            logger.error(f"Failed to register service with Nacos: {e}")
            # Don't raise - allow service to start even if Nacos is unavailable

    async def shutdown(self):
        """
        Shutdown handler - deregister service from Nacos.

        Called during FastAPI shutdown.
        """
        if not self.config.enabled or not self._client:
            return

        try:
            if self._service_id:
                await self._client.deregister_service(self._service_id)
                logger.info(f"Service deregistered from Nacos: {self._service_id}")

            await self._client.close()

        except Exception as e:
            logger.error(f"Error during Nacos shutdown: {e}")

    async def discover(
        self,
        service_name: str,
        tags: Optional[List[str]] = None,
        healthy_only: bool = True,
    ) -> List[ServiceInstance]:
        """
        Discover service instances.

        Args:
            service_name: Name of the service to discover
            tags: Filter by tags
            healthy_only: Only return healthy instances

        Returns:
            List of service instances
        """
        if not self._client:
            logger.warning("Nacos client not initialized")
            return []

        return await self._client.discover_service(
            name=service_name,
            tags=tags,
            passing_only=healthy_only,
        )

    async def get_instance(
        self,
        service_name: str,
        strategy: str = "round_robin",
    ) -> Optional[ServiceInstance]:
        """
        Get a healthy service instance using load balancing.

        Args:
            service_name: Name of the service
            strategy: Load balancing strategy (round_robin, random, weight)

        Returns:
            A service instance or None
        """
        if not self._client:
            logger.warning("Nacos client not initialized")
            return None

        return await self._client.get_healthy_instance(
            name=service_name,
            strategy=strategy,
        )

    async def get_service_url(
        self,
        service_name: str,
        strategy: str = "round_robin",
    ) -> Optional[str]:
        """
        Get base URL for a service.

        Args:
            service_name: Name of the service
            strategy: Load balancing strategy

        Returns:
            Base URL (e.g., http://host:port) or None
        """
        instance = await self.get_instance(service_name, strategy)
        if instance:
            return instance.base_url
        return None


def create_nacos_lifespan(
    config: NacosServiceConfig,
    additional_startup: Optional[Callable] = None,
    additional_shutdown: Optional[Callable] = None,
):
    """
    Create a lifespan context manager for FastAPI with Nacos integration.

    Args:
        config: Nacos service configuration
        additional_startup: Additional startup callback
        additional_shutdown: Additional shutdown callback

    Returns:
        Async context manager for FastAPI lifespan
    """
    manager = NacosServiceManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store manager in app state for access in routes
        app.state.nacos = manager

        # Startup
        await manager.startup()

        if additional_startup:
            if asyncio.iscoroutinefunction(additional_startup):
                await additional_startup(app)
            else:
                additional_startup(app)

        yield

        # Shutdown
        if additional_shutdown:
            if asyncio.iscoroutinefunction(additional_shutdown):
                await additional_shutdown(app)
            else:
                additional_shutdown(app)

        await manager.shutdown()

    return lifespan


def get_nacos_manager(request: Request) -> NacosServiceManager:
    """
    Get Nacos manager from request.

    Usage in FastAPI route:
        @app.get("/discover/{service_name}")
        async def discover(service_name: str, nacos: NacosServiceManager = Depends(get_nacos_manager)):
            instances = await nacos.discover(service_name)
            return instances
    """
    return request.app.state.nacos


class ServiceRegistry:
    """
    Global service registry for inter-service communication.

    Provides a convenient way to call other services.
    """

    _instance: Optional["ServiceRegistry"] = None
    _nacos: Optional[NacosServiceManager] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, nacos: NacosServiceManager):
        """Initialize registry with Nacos manager."""
        cls._nacos = nacos

    @classmethod
    async def get_service_url(cls, service_name: str) -> Optional[str]:
        """Get URL for a service."""
        if cls._nacos:
            return await cls._nacos.get_service_url(service_name)
        return None

    @classmethod
    async def discover(cls, service_name: str) -> List[ServiceInstance]:
        """Discover service instances."""
        if cls._nacos:
            return await cls._nacos.discover(service_name)
        return []


# Service name constants
class ServiceNames:
    """Standard service names for the microservices architecture."""

    API_GATEWAY = "api-gateway"
    MCP_SERVICE = "mcp-service"
    RAG_SERVICE = "rag-service"
    EVALUATION_SERVICE = "evaluation-service"
    MONITORING_SERVICE = "monitoring-service"
    SINGLE_AGENT_SERVICE = "single-agent-service"
    MULTI_AGENT_SERVICE = "multi-agent-service"
    LLM_MANAGER_SERVICE = "llm-manager-service"


# Default service ports
SERVICE_PORTS = {
    ServiceNames.API_GATEWAY: 8000,
    ServiceNames.MCP_SERVICE: 8001,
    ServiceNames.RAG_SERVICE: 8002,
    ServiceNames.EVALUATION_SERVICE: 8003,
    ServiceNames.MONITORING_SERVICE: 8004,
    ServiceNames.SINGLE_AGENT_SERVICE: 8005,
    ServiceNames.MULTI_AGENT_SERVICE: 8006,
    ServiceNames.LLM_MANAGER_SERVICE: 8007,
}
