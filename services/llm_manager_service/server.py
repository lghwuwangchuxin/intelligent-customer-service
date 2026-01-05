"""LLM Manager Service HTTP Server."""

import json
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
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
from .service import LLMManagerService
from .config import LLMConfig

logger = get_logger(__name__)


# Request/Response Models
class MessageModel(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Message role: system, user, assistant, tool")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name for tool messages")
    tool_calls: Optional[List[dict]] = Field(None, description="Tool calls")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID")


class GenerateRequest(BaseModel):
    """Request for text generation."""
    messages: List[MessageModel] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="Model name")
    provider: Optional[str] = Field(None, description="Provider name")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens")
    tools: Optional[List[dict]] = Field(None, description="Tools for function calling")
    stream: bool = Field(False, description="Stream response")


class GenerateResponse(BaseModel):
    """Response from text generation."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    tool_calls: List[dict] = []
    usage: dict = {}
    latency_ms: int = 0


class SwitchModelRequest(BaseModel):
    """Request to switch model."""
    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model name")
    base_url: Optional[str] = Field(None, description="Base URL")
    api_key: Optional[str] = Field(None, description="API key")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None


class LLMServer:
    """LLM Manager HTTP Server."""

    def __init__(self, config: ServiceConfig, service: LLMManagerService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["llm", "inference"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="LLM Manager Service",
            description="Unified LLM provider management and inference service",
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
        self.service.initialize()
        # Initialize ServiceRegistry for inter-service calls
        if hasattr(app.state, "nacos"):
            ServiceRegistry.initialize(app.state.nacos)
        logger.info("LLM Manager Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        logger.info("LLM Manager Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            health = await self.service.health_check()
            return HealthResponse(
                status=health.get("status", "healthy"),
                service="llm-manager-service",
                message="LLM Manager Service is running",
                provider=health.get("provider"),
                model=health.get("model"),
            )

        @self.app.post("/api/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Generate text using LLM."""
            try:
                messages = [m.model_dump() for m in request.messages]

                if request.stream:
                    # Return streaming response
                    async def generate_stream():
                        async for chunk in self.service.stream(
                            messages=messages,
                            provider=request.provider,
                            model=request.model,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                        ):
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/event-stream",
                    )

                response = await self.service.generate(
                    messages=messages,
                    provider=request.provider,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                )

                return GenerateResponse(**response.to_dict())

            except Exception as e:
                logger.error(f"Generate failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat")
        async def chat(request: GenerateRequest):
            """Chat endpoint (alias for generate)."""
            return await generate(request)

        @self.app.post("/api/stream")
        async def stream(request: GenerateRequest):
            """Stream text generation."""
            try:
                messages = [m.model_dump() for m in request.messages]

                async def generate_stream():
                    async for chunk in self.service.stream(
                        messages=messages,
                        provider=request.provider,
                        model=request.model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ):
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                )

            except Exception as e:
                logger.error(f"Stream failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/providers")
        async def list_providers():
            """List available LLM providers."""
            return {"providers": self.service.list_providers()}

        @self.app.get("/api/models")
        async def list_models(provider: Optional[str] = Query(None)):
            """List available models."""
            try:
                models = await self.service.list_models(provider)
                return {"models": models, "provider": provider or self.service.config.provider}
            except Exception as e:
                logger.error(f"List models failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/config")
        async def get_config():
            """Get current configuration."""
            return {"config": self.service.get_current_config()}

        @self.app.post("/api/switch")
        async def switch_model(request: SwitchModelRequest):
            """Switch to a different model."""
            try:
                result = await self.service.switch_model(
                    provider=request.provider,
                    model=request.model,
                    base_url=request.base_url,
                    api_key=request.api_key,
                )
                return {"success": True, "model_info": result}
            except Exception as e:
                logger.error(f"Switch model failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/test")
        async def test_connection(
            provider: Optional[str] = Query(None),
            model: Optional[str] = Query(None),
        ):
            """Test connection to provider."""
            try:
                result = await self.service.test_connection(provider, model)
                return result
            except Exception as e:
                logger.error(f"Test connection failed: {e}")
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


def create_app(config: ServiceConfig, service: LLMManagerService) -> FastAPI:
    """Create FastAPI application."""
    server = LLMServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: LLMManagerService):
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
