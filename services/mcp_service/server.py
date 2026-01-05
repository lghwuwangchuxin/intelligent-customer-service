"""MCP Service HTTP Server using FastAPI."""

import json
import time
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
from .service import MCPService

logger = get_logger(__name__)


# Request/Response Models
class ToolInfo(BaseModel):
    """Tool information."""
    name: str
    description: str
    input_schema: str
    output_schema: Optional[str] = None
    tags: List[str] = []
    is_async: bool = False
    timeout_seconds: int = 60


class ListToolsResponse(BaseModel):
    """Response for list tools."""
    tools: List[ToolInfo]
    total: int


class ExecuteToolRequest(BaseModel):
    """Request to execute a tool."""
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None


class ExecuteToolResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str


class MCPServer:
    """MCP Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: MCPService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["mcp", "tools"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="MCP Service",
            description="Tool management and execution service",
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
        logger.info("MCP Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        logger.info("MCP Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                service="mcp-service",
                message="MCP Service is running",
            )

        @self.app.get("/api/tools", response_model=ListToolsResponse)
        async def list_tools(
            tags: Optional[str] = Query(None, description="Comma-separated tags"),
            name_pattern: Optional[str] = Query(None, description="Name pattern filter"),
        ):
            """List available tools."""
            try:
                tag_list = tags.split(",") if tags else None
                tools = self.service.list_tools(
                    tags=tag_list,
                    name_pattern=name_pattern,
                )

                tool_infos = [
                    ToolInfo(
                        name=tool["name"],
                        description=tool["description"],
                        input_schema=tool["input_schema"],
                        output_schema=tool.get("output_schema"),
                        tags=tool.get("tags", []),
                        is_async=tool.get("is_async", False),
                        timeout_seconds=tool.get("timeout_seconds", 60),
                    )
                    for tool in tools
                ]

                return ListToolsResponse(tools=tool_infos, total=len(tool_infos))
            except Exception as e:
                logger.error(f"List tools failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/tools/{tool_name}/schema", response_model=ToolInfo)
        async def get_tool_schema(tool_name: str):
            """Get schema for a specific tool."""
            try:
                schema = self.service.get_tool_schema(tool_name)
                if not schema:
                    raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

                return ToolInfo(
                    name=schema["name"],
                    description=schema["description"],
                    input_schema=schema["input_schema"],
                    output_schema=schema.get("output_schema"),
                    tags=schema.get("tags", []),
                    is_async=schema.get("is_async", False),
                    timeout_seconds=schema.get("timeout_seconds", 60),
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get tool schema failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/tools/execute", response_model=ExecuteToolResponse)
        async def execute_tool(request: ExecuteToolRequest):
            """Execute a tool."""
            start_time = time.time()

            try:
                result = await self.service.execute_tool(
                    tool_name=request.tool_name,
                    arguments=json.dumps(request.arguments) if isinstance(request.arguments, dict) else request.arguments,
                    timeout_seconds=request.timeout_seconds,
                )

                execution_time_ms = int((time.time() - start_time) * 1000)

                return ExecuteToolResponse(
                    success=result.get("success", False),
                    result=result.get("result"),
                    error=result.get("error"),
                    error_code=result.get("error_code"),
                    execution_time_ms=result.get("execution_time_ms", execution_time_ms),
                )
            except Exception as e:
                logger.error(f"Execute tool failed: {e}")
                return ExecuteToolResponse(
                    success=False,
                    error=str(e),
                    error_code="INTERNAL_ERROR",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

        @self.app.post("/api/tools/execute/stream")
        async def execute_tool_stream(request: ExecuteToolRequest):
            """Execute a tool with streaming response."""

            async def generate():
                start_time = time.time()
                try:
                    result = await self.service.execute_tool(
                        tool_name=request.tool_name,
                        arguments=json.dumps(request.arguments) if isinstance(request.arguments, dict) else request.arguments,
                        timeout_seconds=request.timeout_seconds,
                    )

                    execution_time_ms = int((time.time() - start_time) * 1000)

                    response = ExecuteToolResponse(
                        success=result.get("success", False),
                        result=result.get("result"),
                        error=result.get("error"),
                        error_code=result.get("error_code"),
                        execution_time_ms=result.get("execution_time_ms", execution_time_ms),
                    )
                    yield response.model_dump_json() + "\n"
                except Exception as e:
                    error_response = ExecuteToolResponse(
                        success=False,
                        error=str(e),
                        error_code="INTERNAL_ERROR",
                        execution_time_ms=int((time.time() - start_time) * 1000),
                    )
                    yield error_response.model_dump_json() + "\n"

            return StreamingResponse(generate(), media_type="application/x-ndjson")

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


def create_app(config: ServiceConfig, service: MCPService) -> FastAPI:
    """Create FastAPI application."""
    server = MCPServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: MCPService):
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
