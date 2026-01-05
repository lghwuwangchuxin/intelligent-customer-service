"""Multi-Agent Service HTTP Server using FastAPI."""

import json
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.common.logging import get_logger
from services.common.config import ServiceConfig
from services.common.fastapi_nacos import (
    NacosServiceConfig,
    NacosServiceManager,
    create_nacos_lifespan,
    get_nacos_manager,
    ServiceRegistry,
)
from .service import MultiAgentService

logger = get_logger(__name__)


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request."""
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from an agent."""
    agent_name: str
    agent_id: str
    response: str
    confidence: float = 0.0


class QueryResponse(BaseModel):
    """Query response."""
    message: str
    conversation_id: Optional[str] = None
    agent_responses: List[AgentResponse] = []
    sources: List[str] = []
    is_final: bool = True


class AgentInfo(BaseModel):
    """Agent information."""
    id: str
    name: str
    description: str
    capabilities: List[str] = []
    status: str = "unknown"
    endpoint: Optional[str] = None


class ListAgentsResponse(BaseModel):
    """Response for listing agents."""
    agents: List[AgentInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    online_agents: int = 0
    total_agents: int = 0
    details: Optional[Dict[str, Any]] = None


class MultiAgentServer:
    """Multi-Agent Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: MultiAgentService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["multi-agent", "a2a", "routing"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="Multi-Agent Service",
            description="A2A protocol based multi-agent coordination service",
            version="1.0.0",
            lifespan=create_nacos_lifespan(
                nacos_config,
                additional_startup=self._on_startup,
                additional_shutdown=self._on_shutdown,
            ),
        )

        # Register routes
        self._setup_routes()

    async def _on_startup(self, app: FastAPI):
        """Additional startup logic."""
        # Initialize ServiceRegistry for inter-service calls
        if hasattr(app.state, "nacos"):
            ServiceRegistry.initialize(app.state.nacos)
        logger.info("Multi-Agent Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        await self.service.shutdown()
        logger.info("Multi-Agent Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = await self.service.health_check()
                return HealthResponse(
                    status=health.get("status", "healthy"),
                    service="multi-agent-service",
                    message=f"Multi-Agent Service: {health.get('online_agents', 0)}/{health.get('total_agents', 0)} agents online",
                    online_agents=health.get("online_agents", 0),
                    total_agents=health.get("total_agents", 0),
                    details=health,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    service="multi-agent-service",
                    message=str(e),
                )

        @self.app.post("/api/query", response_model=QueryResponse)
        async def route_and_execute(request: QueryRequest):
            """Route and execute a query."""
            try:
                response = await self.service.process_query(
                    query=request.message,
                    session_id=request.session_id or request.conversation_id,
                )

                agent_responses = [
                    AgentResponse(
                        agent_name=ar["agent_name"],
                        agent_id=ar.get("agent_id", ""),
                        response=ar.get("response", ""),
                        confidence=ar.get("confidence", 0.0),
                    )
                    for ar in response.agent_responses
                ]

                sources = [ar["agent_name"] for ar in response.agent_responses]

                return QueryResponse(
                    message=response.message,
                    conversation_id=request.conversation_id,
                    agent_responses=agent_responses,
                    sources=sources,
                    is_final=True,
                )
            except Exception as e:
                logger.error(f"Route and execute failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/query/stream")
        async def stream_route_and_execute(request: QueryRequest):
            """Stream route and execute."""

            async def generate():
                try:
                    async for response in self.service.stream_query(
                        query=request.message,
                        session_id=request.session_id or request.conversation_id,
                    ):
                        agent_responses = [
                            {
                                "agent_name": ar["agent_name"],
                                "agent_id": ar.get("agent_id", ""),
                                "response": ar.get("response", ""),
                                "confidence": ar.get("confidence", 0.0),
                            }
                            for ar in response.agent_responses
                        ]

                        sources = [ar["agent_name"] for ar in response.agent_responses]

                        result = QueryResponse(
                            message=response.message,
                            conversation_id=request.conversation_id,
                            agent_responses=[AgentResponse(**ar) for ar in agent_responses],
                            sources=sources,
                            is_final=False,
                        )
                        yield result.model_dump_json() + "\n"

                except Exception as e:
                    logger.error(f"Stream query failed: {e}")
                    yield json.dumps({"error": str(e)}) + "\n"

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        @self.app.get("/api/agents", response_model=ListAgentsResponse)
        async def list_agents():
            """List available agents."""
            try:
                agents = await self.service.list_agents()

                agent_infos = [
                    AgentInfo(
                        id=agent["id"],
                        name=agent["name"],
                        description=agent.get("description", ""),
                        capabilities=agent.get("capabilities", []),
                        status=agent.get("status", "unknown"),
                        endpoint=agent.get("endpoint"),
                    )
                    for agent in agents
                ]

                return ListAgentsResponse(agents=agent_infos, total=len(agent_infos))
            except Exception as e:
                logger.error(f"List agents failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get agent by ID."""
            try:
                agent = await self.service.get_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

                return AgentInfo(
                    id=agent["id"],
                    name=agent["name"],
                    description=agent.get("description", ""),
                    capabilities=agent.get("capabilities", []),
                    status=agent.get("status", "unknown"),
                    endpoint=agent.get("endpoint"),
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get agent failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/agents/{agent_id}/invoke")
        async def invoke_agent(agent_id: str, request: QueryRequest):
            """Invoke a specific agent directly."""
            try:
                response = await self.service.invoke_agent(
                    agent_id=agent_id,
                    query=request.message,
                    session_id=request.session_id or request.conversation_id,
                )

                return {
                    "agent_id": agent_id,
                    "response": response.get("response", ""),
                    "success": response.get("success", True),
                }
            except Exception as e:
                logger.error(f"Invoke agent failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/services/{service_name}")
        async def discover_service(
            service_name: str,
            nacos: NacosServiceManager = Depends(get_nacos_manager),
        ):
            """Discover instances of a service."""
            try:
                instances = await nacos.discover(service_name)
                return {
                    "service": service_name,
                    "instances": [
                        {
                            "id": inst.service_id,
                            "address": inst.address,
                            "port": inst.port,
                            "healthy": inst.healthy,
                            "weight": inst.weight,
                        }
                        for inst in instances
                    ],
                    "count": len(instances),
                }
            except Exception as e:
                logger.error(f"Service discovery failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/service-info")
        async def get_service_info(
            nacos: NacosServiceManager = Depends(get_nacos_manager),
        ):
            """Get current service registration info."""
            return {
                "service_id": nacos.service_id,
                "service_name": self.config.service_name,
                "address": f"{self.config.host}:{self.config.http_port}",
                "nacos_enabled": self.config.nacos_enabled,
            }


def create_app(config: ServiceConfig, service: MultiAgentService) -> FastAPI:
    """Create FastAPI application."""
    server = MultiAgentServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: MultiAgentService):
    """Start the HTTP server."""
    import uvicorn

    app = create_app(config, service)

    server_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.http_port,
        log_level="info",
    )
    server = uvicorn.Server(server_config)
    await server.serve()
