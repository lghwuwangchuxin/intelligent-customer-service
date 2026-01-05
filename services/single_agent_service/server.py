"""Single Agent Service HTTP Server using FastAPI."""

import json
from typing import Optional, List, Dict, Any, AsyncIterator
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
from .service import SingleAgentService, ChatConfig

logger = get_logger(__name__)


# Request/Response Models
class ChatConfigRequest(BaseModel):
    """Chat configuration."""
    enable_tools: bool = True
    enable_rag: bool = True
    knowledge_base_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    conversation_id: Optional[str] = None
    config: Optional[ChatConfigRequest] = None


class ToolCallInfo(BaseModel):
    """Tool call information."""
    tool_name: str
    arguments: Optional[str] = None
    result: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""
    message: str
    conversation_id: str
    tool_calls: List[ToolCallInfo] = []
    sources: List[str] = []
    is_final: bool = True


class MessageInfo(BaseModel):
    """Message information."""
    role: str
    content: str
    timestamp: Optional[int] = None


class GetHistoryResponse(BaseModel):
    """Response for conversation history."""
    messages: List[MessageInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SingleAgentServer:
    """Single Agent Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: SingleAgentService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["agent", "langgraph", "chat"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="Single Agent Service",
            description="LangGraph-based intelligent agent service",
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
        logger.info("Single Agent Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        logger.info("Single Agent Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = await self.service.health_check()
                return HealthResponse(
                    status=health.get("status", "healthy"),
                    service="single-agent-service",
                    message="Single Agent Service is running",
                    details=health,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    service="single-agent-service",
                    message=str(e),
                )

        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Single chat request."""
            try:
                cfg = request.config or ChatConfigRequest()
                config = ChatConfig(
                    enable_tools=cfg.enable_tools,
                    enable_rag=cfg.enable_rag,
                    knowledge_base_id=cfg.knowledge_base_id,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )

                response = await self.service.chat(
                    message=request.message,
                    conversation_id=request.conversation_id,
                    config=config,
                )

                tool_calls = [
                    ToolCallInfo(
                        tool_name=tc["name"],
                        arguments=tc.get("arguments"),
                        result=tc.get("result"),
                    )
                    for tc in response.tool_calls
                ]

                return ChatResponse(
                    message=response.message,
                    conversation_id=response.conversation_id,
                    tool_calls=tool_calls,
                    sources=response.sources,
                    is_final=True,
                )
            except Exception as e:
                logger.error(f"Chat failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat/stream")
        async def stream_chat(request: ChatRequest):
            """Streaming chat request."""

            async def generate():
                try:
                    cfg = request.config or ChatConfigRequest()
                    config = ChatConfig(
                        enable_tools=cfg.enable_tools,
                        enable_rag=cfg.enable_rag,
                        knowledge_base_id=cfg.knowledge_base_id,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                    )

                    async for response in self.service.stream_chat(
                        message=request.message,
                        conversation_id=request.conversation_id,
                        config=config,
                    ):
                        tool_calls = [
                            {
                                "tool_name": tc["name"],
                                "arguments": tc.get("arguments"),
                                "result": tc.get("result"),
                            }
                            for tc in response.tool_calls
                        ]

                        result = ChatResponse(
                            message=response.message,
                            conversation_id=response.conversation_id,
                            tool_calls=[ToolCallInfo(**tc) for tc in tool_calls],
                            sources=response.sources,
                            is_final=response.is_final,
                        )
                        yield result.model_dump_json() + "\n"

                except Exception as e:
                    logger.error(f"Stream chat failed: {e}")
                    yield json.dumps({"error": str(e)}) + "\n"

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        @self.app.get("/api/conversations/{conversation_id}/history", response_model=GetHistoryResponse)
        async def get_history(conversation_id: str, limit: int = 50):
            """Get conversation history."""
            try:
                history = await self.service.get_history(
                    conversation_id=conversation_id,
                    limit=limit,
                )

                messages = [
                    MessageInfo(
                        role=msg["role"],
                        content=msg["content"],
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in history
                ]

                return GetHistoryResponse(messages=messages)
            except Exception as e:
                logger.error(f"Get history failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/conversations/{conversation_id}")
        async def clear_conversation(conversation_id: str):
            """Clear conversation history."""
            try:
                success = await self.service.clear_conversation(conversation_id)
                return {"success": success, "conversation_id": conversation_id}
            except Exception as e:
                logger.error(f"Clear conversation failed: {e}")
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


def create_app(config: ServiceConfig, service: SingleAgentService) -> FastAPI:
    """Create FastAPI application."""
    server = SingleAgentServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: SingleAgentService):
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
