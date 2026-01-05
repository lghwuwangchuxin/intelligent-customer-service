"""HTTP client base class with service discovery."""

import asyncio
from typing import Optional, Any, Dict, TypeVar, Generic
import httpx

from .nacos_client import NacosClient, ServiceInstance
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class HTTPClientBase:
    """
    Base class for HTTP clients with Nacos service discovery.

    Example:
        class MCPClient(HTTPClientBase):
            def __init__(self, nacos: NacosClient):
                super().__init__(
                    nacos_client=nacos,
                    service_name="mcp-service",
                )

            async def list_tools(self):
                return await self.get("/api/tools")
    """

    def __init__(
        self,
        nacos_client: NacosClient,
        service_name: str,
        fallback_address: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize HTTP client.

        Args:
            nacos_client: Nacos client for service discovery
            service_name: Name of the service to connect to
            fallback_address: Fallback address if service discovery fails
            timeout: Default timeout for requests
            max_retries: Maximum number of retries
        """
        self.nacos = nacos_client
        self.service_name = service_name
        self.fallback_address = fallback_address
        self.timeout = timeout
        self.max_retries = max_retries

        self._client: Optional[httpx.AsyncClient] = None
        self._current_base_url: Optional[str] = None
        self._lock = asyncio.Lock()

    async def _get_service_address(self) -> str:
        """Get service address from Nacos or fallback."""
        instance = await self.nacos.get_healthy_instance(self.service_name)

        if instance:
            return instance.http_address

        if self.fallback_address:
            logger.warning(
                f"No healthy instance for {self.service_name}, "
                f"using fallback: {self.fallback_address}"
            )
            return self.fallback_address

        raise RuntimeError(f"No available instances for service: {self.service_name}")

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client."""
        async with self._lock:
            address = await self._get_service_address()
            base_url = f"http://{address}"

            # Reuse existing client if address hasn't changed
            if self._client and self._current_base_url == base_url:
                return self._client

            # Close existing client
            if self._client:
                await self._client.aclose()

            # Create new client
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            self._current_base_url = base_url

            logger.debug(f"Created HTTP client to {self.service_name} at {base_url}")
            return self._client

    async def request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                client = await self.get_client()
                response = await client.request(method, path, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    f"Request to {self.service_name}{path} failed (attempt {attempt + 1}): {e}"
                )
                # Reset connection on error
                await self.reset()
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors
                if 400 <= e.response.status_code < 500:
                    raise
                last_error = e
                logger.warning(
                    f"Request to {self.service_name}{path} failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise RuntimeError(
            f"Failed to call {self.service_name}{path} after {self.max_retries} attempts: {last_error}"
        )

    async def get(self, path: str, **kwargs) -> Any:
        """Make a GET request and return JSON."""
        response = await self.request("GET", path, **kwargs)
        return response.json()

    async def post(self, path: str, data: Optional[Dict] = None, **kwargs) -> Any:
        """Make a POST request and return JSON."""
        response = await self.request("POST", path, json=data, **kwargs)
        return response.json()

    async def put(self, path: str, data: Optional[Dict] = None, **kwargs) -> Any:
        """Make a PUT request and return JSON."""
        response = await self.request("PUT", path, json=data, **kwargs)
        return response.json()

    async def delete(self, path: str, **kwargs) -> Any:
        """Make a DELETE request and return JSON."""
        response = await self.request("DELETE", path, **kwargs)
        return response.json()

    async def stream_post(self, path: str, data: Optional[Dict] = None, **kwargs):
        """Make a streaming POST request."""
        client = await self.get_client()
        async with client.stream("POST", path, json=data, **kwargs) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def close(self):
        """Close the HTTP client."""
        async with self._lock:
            if self._client:
                await self._client.aclose()
                self._client = None
                self._current_base_url = None

    async def reset(self):
        """Reset connection (force reconnect on next call)."""
        await self.close()
        self.nacos.invalidate_cache(self.service_name)

    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            response = await self.get("/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Health check failed for {self.service_name}: {e}")
            return False

    async def __aenter__(self) -> "HTTPClientBase":
        """Async context manager entry."""
        await self.get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class HTTPClientPool:
    """Pool of HTTP clients for multiple services."""

    def __init__(self, nacos_client: NacosClient):
        self.nacos = nacos_client
        self._clients: dict[str, HTTPClientBase] = {}
        self._lock = asyncio.Lock()

    def register(self, name: str, client: HTTPClientBase):
        """Register a client."""
        self._clients[name] = client

    def get(self, name: str) -> Optional[HTTPClientBase]:
        """Get a registered client."""
        return self._clients.get(name)

    async def close_all(self):
        """Close all clients."""
        for client in self._clients.values():
            await client.close()

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered clients."""
        results = {}
        for name, client in self._clients.items():
            results[name] = await client.health_check()
        return results


# Service-specific clients
class MCPServiceClient(HTTPClientBase):
    """Client for MCP Service."""

    def __init__(self, nacos_client: NacosClient, fallback_address: str = "localhost:8001"):
        super().__init__(
            nacos_client=nacos_client,
            service_name="mcp-service",
            fallback_address=fallback_address,
        )

    async def list_tools(self, tags: Optional[list] = None) -> Dict:
        """List available tools."""
        params = {}
        if tags:
            params["tags"] = ",".join(tags)
        return await self.get("/api/tools", params=params)

    async def execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool."""
        return await self.post(
            "/api/tools/execute",
            data={"tool_name": tool_name, "arguments": arguments},
        )

    async def get_tool_schema(self, tool_name: str) -> Dict:
        """Get tool schema."""
        return await self.get(f"/api/tools/{tool_name}/schema")


class RAGServiceClient(HTTPClientBase):
    """Client for RAG Service."""

    def __init__(self, nacos_client: NacosClient, fallback_address: str = "localhost:8002"):
        super().__init__(
            nacos_client=nacos_client,
            service_name="rag-service",
            fallback_address=fallback_address,
            timeout=60.0,  # Longer timeout for RAG
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        knowledge_base_id: Optional[str] = None,
        enable_rerank: bool = True,
    ) -> Dict:
        """Retrieve documents."""
        return await self.post(
            "/api/retrieve",
            data={
                "query": query,
                "top_k": top_k,
                "knowledge_base_id": knowledge_base_id,
                "config": {"enable_rerank": enable_rerank},
            },
        )

    async def index_document(
        self,
        content: str,
        knowledge_base_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Index a document."""
        return await self.post(
            "/api/documents",
            data={
                "content": content,
                "knowledge_base_id": knowledge_base_id,
                "metadata": metadata or {},
            },
        )

    async def delete_document(self, document_id: str, knowledge_base_id: str) -> Dict:
        """Delete a document."""
        return await self.delete(
            f"/api/documents/{document_id}",
            params={"knowledge_base_id": knowledge_base_id},
        )


class EvaluationServiceClient(HTTPClientBase):
    """Client for Evaluation Service."""

    def __init__(self, nacos_client: NacosClient, fallback_address: str = "localhost:8003"):
        super().__init__(
            nacos_client=nacos_client,
            service_name="evaluation-service",
            fallback_address=fallback_address,
            timeout=120.0,  # Long timeout for evaluation
        )

    async def submit_evaluation(self, data: list, metrics: list) -> Dict:
        """Submit evaluation task."""
        return await self.post(
            "/api/evaluation/submit",
            data={"data": data, "metrics": metrics},
        )

    async def get_result(self, task_id: str) -> Dict:
        """Get evaluation result."""
        return await self.get(f"/api/evaluation/{task_id}")

    async def list_evaluations(self, status: Optional[str] = None) -> Dict:
        """List evaluations."""
        params = {}
        if status:
            params["status"] = status
        return await self.get("/api/evaluation", params=params)


class MonitoringServiceClient(HTTPClientBase):
    """Client for Monitoring Service."""

    def __init__(self, nacos_client: NacosClient, fallback_address: str = "localhost:8004"):
        super().__init__(
            nacos_client=nacos_client,
            service_name="monitoring-service",
            fallback_address=fallback_address,
        )

    async def record_trace(self, spans: list) -> Dict:
        """Record trace spans."""
        return await self.post("/api/traces", data={"spans": spans})

    async def record_metrics(self, metrics: list) -> Dict:
        """Record metrics."""
        return await self.post("/api/metrics", data={"metrics": metrics})

    async def get_health_summary(self) -> Dict:
        """Get health summary."""
        return await self.get("/api/health/summary")


class AgentServiceClient(HTTPClientBase):
    """Client for Agent Service."""

    def __init__(self, nacos_client: NacosClient, fallback_address: str = "localhost:8005"):
        super().__init__(
            nacos_client=nacos_client,
            service_name="single-agent-service",
            fallback_address=fallback_address,
            timeout=120.0,  # Long timeout for agent
        )

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        enable_tools: bool = True,
        enable_rag: bool = True,
    ) -> Dict:
        """Send chat message."""
        return await self.post(
            "/api/chat",
            data={
                "message": message,
                "conversation_id": conversation_id,
                "config": {
                    "enable_tools": enable_tools,
                    "enable_rag": enable_rag,
                },
            },
        )

    async def stream_chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        enable_tools: bool = True,
        enable_rag: bool = True,
    ):
        """Stream chat response."""
        async for chunk in self.stream_post(
            "/api/chat/stream",
            data={
                "message": message,
                "conversation_id": conversation_id,
                "config": {
                    "enable_tools": enable_tools,
                    "enable_rag": enable_rag,
                },
            },
        ):
            yield chunk

    async def get_history(self, conversation_id: str, limit: int = 50) -> Dict:
        """Get conversation history."""
        return await self.get(
            f"/api/conversations/{conversation_id}/history",
            params={"limit": limit},
        )


class MultiAgentServiceClient(HTTPClientBase):
    """Client for Multi-Agent Service."""

    def __init__(self, nacos_client: NacosClient, fallback_address: str = "localhost:8006"):
        super().__init__(
            nacos_client=nacos_client,
            service_name="multi-agent-service",
            fallback_address=fallback_address,
            timeout=180.0,  # Long timeout for multi-agent
        )

    async def route_and_execute(self, query: str, conversation_id: Optional[str] = None) -> Dict:
        """Route query to appropriate agent and execute."""
        return await self.post(
            "/api/route",
            data={
                "query": query,
                "conversation_id": conversation_id,
            },
        )

    async def list_agents(self, domain: Optional[str] = None) -> Dict:
        """List available agents."""
        params = {}
        if domain:
            params["domain"] = domain
        return await self.get("/api/agents", params=params)

    async def execute_on_agent(self, agent_id: str, query: str) -> Dict:
        """Execute query on specific agent."""
        return await self.post(
            f"/api/agents/{agent_id}/execute",
            data={"query": query},
        )
