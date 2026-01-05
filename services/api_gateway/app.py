"""API Gateway FastAPI application."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.common.config import get_service_config
from services.common.logging import setup_logging, get_logger
from services.common.fastapi_nacos import (
    NacosServiceConfig,
    NacosServiceManager,
    create_nacos_lifespan,
    get_nacos_manager,
    ServiceRegistry,
    ServiceNames,
)
from services.common.service_client import (
    ServiceClient,
    ServiceClientConfig,
    LLMServiceClient,
    RAGServiceClient,
    MCPServiceClient,
    AgentServiceClient,
)

from .routes import chat_router, knowledge_router, tools_router, evaluation_router

logger = get_logger(__name__)


async def on_startup(app: FastAPI):
    """Application startup handler."""
    config = app.state.config

    # Setup logging
    setup_logging(
        service_name="api-gateway",
        level=config.log_level,
        log_format=config.log_format,
        log_to_console=config.log_to_console,
        log_to_file=config.log_to_file,
        log_dir=config.log_dir,
    )

    logger.info("Starting API Gateway...")

    # Initialize ServiceRegistry for inter-service calls
    if hasattr(app.state, "nacos"):
        ServiceRegistry.initialize(app.state.nacos)

    # Initialize unified service client
    service_client_config = ServiceClientConfig(
        nacos_server_addresses=config.nacos_server_addresses,
        nacos_namespace=config.nacos_namespace,
        nacos_group=config.nacos_group,
        nacos_username=config.nacos_username or None,
        nacos_password=config.nacos_password or None,
    )

    # Initialize HTTP clients using new ServiceClient
    try:
        app.state.service_client = ServiceClient(service_client_config)
        await app.state.service_client.initialize()

        # Create typed service clients
        app.state.llm_client = LLMServiceClient(app.state.service_client)
        app.state.rag_client = RAGServiceClient(app.state.service_client)
        app.state.mcp_client = MCPServiceClient(app.state.service_client)
        app.state.single_agent_client = AgentServiceClient(app.state.service_client, multi_agent=False)
        app.state.multi_agent_client = AgentServiceClient(app.state.service_client, multi_agent=True)

        logger.info("Service clients initialized")

    except Exception as e:
        logger.error(f"Failed to initialize service clients: {e}")

    logger.info("API Gateway started")


async def on_shutdown(app: FastAPI):
    """Application shutdown handler."""
    logger.info("Shutting down API Gateway...")

    # Close service client
    if hasattr(app.state, "service_client") and app.state.service_client:
        await app.state.service_client.close()

    logger.info("API Gateway stopped")


def create_app(config=None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config: Service configuration

    Returns:
        FastAPI application
    """
    if config is None:
        config = get_service_config("api-gateway")

    # Create Nacos service config for API Gateway
    nacos_config = NacosServiceConfig.from_service_config(config)
    nacos_config.tags = ["gateway", "api", "entry"]
    nacos_config.metadata = {"version": "1.0.0"}

    app = FastAPI(
        title="Intelligent Customer Service API",
        description="Unified API Gateway for microservices",
        version="1.0.0",
        lifespan=create_nacos_lifespan(
            nacos_config,
            additional_startup=on_startup,
            additional_shutdown=on_shutdown,
        ),
    )

    # Store config in app state
    app.state.config = config

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat_router)
    app.include_router(knowledge_router)
    app.include_router(tools_router)
    app.include_router(evaluation_router)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "api-gateway",
        }

    # Service status endpoint
    @app.get("/status")
    async def service_status(request: Request):
        """Get status of all services."""
        status = {
            "api_gateway": "healthy",
            "services": {},
        }

        # Check each service using the service client
        service_names = [
            (ServiceNames.MCP_SERVICE, "mcp"),
            (ServiceNames.RAG_SERVICE, "rag"),
            (ServiceNames.SINGLE_AGENT_SERVICE, "single_agent"),
            (ServiceNames.MULTI_AGENT_SERVICE, "multi_agent"),
            (ServiceNames.EVALUATION_SERVICE, "evaluation"),
            (ServiceNames.MONITORING_SERVICE, "monitoring"),
            (ServiceNames.LLM_MANAGER_SERVICE, "llm_manager"),
        ]

        service_client = getattr(request.app.state, "service_client", None)
        if service_client:
            for service_name, display_name in service_names:
                try:
                    response = await service_client.get(service_name, "/health")
                    if response.status_code == 200:
                        health = response.json()
                        status["services"][display_name] = health.get("status", "available")
                    else:
                        status["services"][display_name] = "unhealthy"
                except Exception:
                    status["services"][display_name] = "unavailable"
        else:
            for _, display_name in service_names:
                status["services"][display_name] = "client_not_initialized"

        return status

    # Service discovery endpoint
    @app.get("/api/services/{service_name}")
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
            return {"service": service_name, "instances": [], "count": 0, "error": str(e)}

    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


# For running with uvicorn directly
app = create_app()
