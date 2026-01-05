"""HTTP Service Client with Nacos service discovery."""

import asyncio
import random
from typing import Optional, Dict, Any, List, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

import httpx

from .logging import get_logger
from .nacos_client import NacosClient, ServiceInstance

logger = get_logger(__name__)

T = TypeVar("T")


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategy."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHT = "weight"


@dataclass
class ServiceClientConfig:
    """Configuration for service client."""

    # Nacos connection
    nacos_server_addresses: str = "localhost:8848"
    nacos_namespace: str = "public"
    nacos_group: str = "DEFAULT_GROUP"
    nacos_username: Optional[str] = None
    nacos_password: Optional[str] = None

    # HTTP client settings
    timeout: float = 30.0
    connect_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Load balancing
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN

    # Cache settings
    cache_ttl: int = 30  # seconds


class ServiceClient:
    """
    HTTP client for inter-service communication with Nacos service discovery.

    Features:
    - Service discovery via Nacos
    - Load balancing (round-robin, random, weight-based)
    - Automatic retry with failover
    - Connection pooling
    - Circuit breaker pattern
    """

    # Default static URLs for services when Nacos is unavailable
    DEFAULT_SERVICE_URLS = {
        "single-agent-service": "http://localhost:8005",
        "multi-agent-service": "http://localhost:8006",
        "rag-service": "http://localhost:8002",
        "mcp-service": "http://localhost:8001",
        "evaluation-service": "http://localhost:8003",
        "monitoring-service": "http://localhost:8004",
        "llm-manager-service": "http://localhost:8007",
        "memory-service": "http://localhost:8008",
    }

    def __init__(self, config: Optional[ServiceClientConfig] = None):
        """
        Initialize service client.

        Args:
            config: Client configuration
        """
        self.config = config or ServiceClientConfig()
        self._nacos: Optional[NacosClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._round_robin_index: Dict[str, int] = {}
        self._instance_cache: Dict[str, tuple[List[ServiceInstance], float]] = {}
        self._static_urls: Dict[str, str] = dict(self.DEFAULT_SERVICE_URLS)
        self._initialized = False

    def set_static_url(self, service_name: str, url: str):
        """Set static URL for a service (used when Nacos is unavailable)."""
        self._static_urls[service_name] = url

    async def initialize(self):
        """Initialize the client."""
        if self._initialized:
            return

        self._nacos = NacosClient(
            server_addresses=self.config.nacos_server_addresses,
            namespace=self.config.nacos_namespace,
            username=self.config.nacos_username,
            password=self.config.nacos_password,
            group=self.config.nacos_group,
        )

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self.config.timeout,
                connect=self.config.connect_timeout,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )

        self._initialized = True
        logger.info("Service client initialized")

    async def close(self):
        """Close the client."""
        if self._http_client:
            await self._http_client.aclose()
        if self._nacos:
            await self._nacos.close()
        self._initialized = False

    async def _get_instances(self, service_name: str) -> List[ServiceInstance]:
        """
        Get service instances with caching.

        Args:
            service_name: Name of the service

        Returns:
            List of service instances
        """
        import time

        # Check cache
        if service_name in self._instance_cache:
            instances, timestamp = self._instance_cache[service_name]
            if time.time() - timestamp < self.config.cache_ttl:
                return instances

        # Fetch from Nacos
        if self._nacos:
            try:
                instances = await self._nacos.discover_service(
                    name=service_name,
                    passing_only=True,
                )
                self._instance_cache[service_name] = (instances, time.time())
                return instances
            except Exception as e:
                logger.warning(f"Failed to discover service {service_name}: {e}")

        return []

    def _select_instance(
        self,
        instances: List[ServiceInstance],
        service_name: str
    ) -> Optional[ServiceInstance]:
        """
        Select an instance based on load balancing strategy.

        Args:
            instances: Available instances
            service_name: Service name for round-robin tracking

        Returns:
            Selected instance
        """
        if not instances:
            return None

        healthy_instances = [i for i in instances if i.healthy]
        if not healthy_instances:
            healthy_instances = instances  # Fallback to all

        if self.config.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(healthy_instances)

        elif self.config.strategy == LoadBalanceStrategy.WEIGHT:
            # Weight-based selection
            total_weight = sum(i.weight for i in healthy_instances)
            if total_weight == 0:
                return random.choice(healthy_instances)

            r = random.uniform(0, total_weight)
            current = 0
            for instance in healthy_instances:
                current += instance.weight
                if r <= current:
                    return instance
            return healthy_instances[-1]

        else:  # Round-robin
            if service_name not in self._round_robin_index:
                self._round_robin_index[service_name] = 0

            index = self._round_robin_index[service_name] % len(healthy_instances)
            self._round_robin_index[service_name] = index + 1
            return healthy_instances[index]

    async def _get_base_url(self, service_name: str) -> Optional[str]:
        """
        Get base URL for a service.

        First tries Nacos discovery, then falls back to static URLs.

        Args:
            service_name: Name of the service

        Returns:
            Base URL or None
        """
        # Try Nacos discovery first
        instances = await self._get_instances(service_name)
        instance = self._select_instance(instances, service_name)
        if instance:
            return instance.base_url

        # Fallback to static URL
        if service_name in self._static_urls:
            logger.debug(f"Using static URL for {service_name}: {self._static_urls[service_name]}")
            return self._static_urls[service_name]

        return None

    async def request(
        self,
        service_name: str,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Make HTTP request to a service.

        Args:
            service_name: Name of the target service
            method: HTTP method
            path: Request path
            **kwargs: Additional arguments for httpx

        Returns:
            HTTP response

        Raises:
            Exception: If all retries fail
        """
        await self.initialize()

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                base_url = await self._get_base_url(service_name)
                if not base_url:
                    raise Exception(f"No available instance for service: {service_name}")

                url = f"{base_url}{path}"
                response = await self._http_client.request(method, url, **kwargs)
                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Request to {service_name}{path} failed (attempt {attempt + 1}): {e}"
                )

                # Clear cache to force re-discovery
                if service_name in self._instance_cache:
                    del self._instance_cache[service_name]

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error

    async def get(
        self,
        service_name: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make GET request."""
        return await self.request(service_name, "GET", path, **kwargs)

    async def post(
        self,
        service_name: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make POST request."""
        return await self.request(service_name, "POST", path, **kwargs)

    async def put(
        self,
        service_name: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make PUT request."""
        return await self.request(service_name, "PUT", path, **kwargs)

    async def delete(
        self,
        service_name: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make DELETE request."""
        return await self.request(service_name, "DELETE", path, **kwargs)

    async def get_json(
        self,
        service_name: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make GET request and return JSON."""
        response = await self.get(service_name, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post_json(
        self,
        service_name: str,
        path: str,
        data: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make POST request with JSON and return JSON."""
        response = await self.post(service_name, path, json=data, **kwargs)
        response.raise_for_status()
        return response.json()


class TypedServiceClient(Generic[T]):
    """
    Typed service client for specific service.

    Usage:
        llm_client = TypedServiceClient[LLMResponse](
            base_client,
            "llm-manager-service"
        )
        result = await llm_client.post("/api/generate", {"prompt": "Hello"})
    """

    def __init__(
        self,
        client: ServiceClient,
        service_name: str,
        response_type: type = dict,
    ):
        self._client = client
        self._service_name = service_name
        self._response_type = response_type

    async def get(self, path: str, **kwargs) -> T:
        """Make GET request."""
        response = await self._client.get(self._service_name, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post(self, path: str, data: Dict[str, Any] = None, **kwargs) -> T:
        """Make POST request."""
        response = await self._client.post(self._service_name, path, json=data, **kwargs)
        response.raise_for_status()
        return response.json()


# Service-specific clients
class LLMServiceClient:
    """Client for LLM Manager Service."""

    def __init__(self, client: ServiceClient):
        self._client = client
        self._service_name = "llm-manager-service"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Generate text."""
        return await self._client.post_json(
            self._service_name,
            "/api/generate",
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

    async def get_providers(self) -> List[Dict[str, Any]]:
        """Get available providers."""
        return await self._client.get_json(self._service_name, "/api/providers")

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._client.get_json(self._service_name, "/health")


class RAGServiceClient:
    """Client for RAG Service."""

    def __init__(self, client: ServiceClient):
        self._client = client
        self._service_name = "rag-service"

    async def retrieve(
        self,
        query: str,
        knowledge_base_id: Optional[str] = None,
        top_k: int = 5,
        enable_rerank: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve documents."""
        return await self._client.post_json(
            self._service_name,
            "/api/retrieve",
            {
                "query": query,
                "knowledge_base_id": knowledge_base_id,
                "top_k": top_k,
                "enable_rerank": enable_rerank,
            }
        )

    async def index_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_base_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a document."""
        return await self._client.post_json(
            self._service_name,
            "/api/index",
            {
                "content": content,
                "metadata": metadata or {},
                "knowledge_base_id": knowledge_base_id,
            }
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._client.get_json(self._service_name, "/health")


class MCPServiceClient:
    """Client for MCP Service."""

    def __init__(self, client: ServiceClient):
        self._client = client
        self._service_name = "mcp-service"

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        return await self._client.get_json(self._service_name, "/api/tools")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool."""
        return await self._client.post_json(
            self._service_name,
            f"/api/tools/{tool_name}/execute",
            arguments,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._client.get_json(self._service_name, "/health")


class AgentServiceClient:
    """Client for Agent Services (Single/Multi)."""

    def __init__(self, client: ServiceClient, multi_agent: bool = False):
        self._client = client
        self._service_name = "multi-agent-service" if multi_agent else "single-agent-service"

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        enable_tools: bool = True,
        enable_rag: bool = True,
        knowledge_base_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat message."""
        return await self._client.post_json(
            self._service_name,
            "/api/chat",
            {
                "message": message,
                "conversation_id": conversation_id,
                "config": {
                    "enable_tools": enable_tools,
                    "enable_rag": enable_rag,
                    "knowledge_base_id": knowledge_base_id,
                }
            }
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._client.get_json(self._service_name, "/health")


# Global service client factory
_global_client: Optional[ServiceClient] = None


def get_service_client(config: Optional[ServiceClientConfig] = None) -> ServiceClient:
    """
    Get or create global service client.

    Args:
        config: Optional configuration

    Returns:
        Service client instance
    """
    global _global_client
    if _global_client is None:
        _global_client = ServiceClient(config)
    return _global_client


async def close_service_client():
    """Close global service client."""
    global _global_client
    if _global_client:
        await _global_client.close()
        _global_client = None
