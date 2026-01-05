"""Monitoring Service HTTP Server using FastAPI."""

from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
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
from .service import MonitoringService

logger = get_logger(__name__)


# Request/Response Models
class SpanInfo(BaseModel):
    """Trace span information."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    attributes: Dict[str, str] = Field(default_factory=dict)
    status: Optional[str] = None


class RecordTraceRequest(BaseModel):
    """Request to record trace spans."""
    spans: List[SpanInfo]


class RecordTraceResponse(BaseModel):
    """Response from recording trace."""
    success: bool


class MetricInfo(BaseModel):
    """Metric information."""
    name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: Optional[int] = None


class RecordMetricsRequest(BaseModel):
    """Request to record metrics."""
    metrics: List[MetricInfo]


class RecordMetricsResponse(BaseModel):
    """Response from recording metrics."""
    success: bool


class GetMetricsResponse(BaseModel):
    """Response for getting metrics."""
    metrics: List[MetricInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    details: Optional[Dict[str, Any]] = None


class MonitoringServer:
    """Monitoring Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: MonitoringService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["monitoring", "tracing", "metrics"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="Monitoring Service",
            description="Observability and tracing service",
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
        logger.info("Monitoring Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        await self.service.shutdown()
        logger.info("Monitoring Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = await self.service.health_check()
                return HealthResponse(
                    status=health.get("status", "healthy"),
                    service="monitoring-service",
                    message="Monitoring Service is running",
                    details=health,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    service="monitoring-service",
                    message=str(e),
                )

        @self.app.post("/api/traces", response_model=RecordTraceResponse)
        async def record_trace(request: RecordTraceRequest):
            """Record trace spans."""
            try:
                spans = [
                    {
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "parent_span_id": span.parent_span_id,
                        "name": span.name,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "attributes": span.attributes,
                        "status": span.status,
                    }
                    for span in request.spans
                ]

                success = await self.service.record_trace(spans)
                return RecordTraceResponse(success=success)
            except Exception as e:
                logger.error(f"Record trace failed: {e}")
                return RecordTraceResponse(success=False)

        @self.app.post("/api/metrics", response_model=RecordMetricsResponse)
        async def record_metrics(request: RecordMetricsRequest):
            """Record metrics."""
            try:
                metrics = [
                    {
                        "name": m.name,
                        "value": m.value,
                        "labels": m.labels,
                        "timestamp": m.timestamp,
                    }
                    for m in request.metrics
                ]

                success = await self.service.record_metrics(metrics)
                return RecordMetricsResponse(success=success)
            except Exception as e:
                logger.error(f"Record metrics failed: {e}")
                return RecordMetricsResponse(success=False)

        @self.app.get("/api/metrics", response_model=GetMetricsResponse)
        async def get_metrics(
            metric_name: Optional[str] = Query(None),
            start_time: Optional[int] = Query(None),
            end_time: Optional[int] = Query(None),
        ):
            """Get metrics."""
            try:
                result = await self.service.get_metrics(
                    name=metric_name,
                    start_time=start_time,
                    end_time=end_time,
                )

                metrics = [
                    MetricInfo(
                        name=m.get("name", ""),
                        value=m["value"],
                        labels=m.get("labels", {}),
                        timestamp=m.get("timestamp"),
                    )
                    for m in result.get("metrics", [])
                ]

                return GetMetricsResponse(metrics=metrics)
            except Exception as e:
                logger.error(f"Get metrics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/traces/{trace_id}")
        async def get_trace(trace_id: str):
            """Get trace by ID."""
            try:
                result = await self.service.get_trace(trace_id)
                if not result:
                    raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get trace failed: {e}")
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


def create_app(config: ServiceConfig, service: MonitoringService) -> FastAPI:
    """Create FastAPI application."""
    server = MonitoringServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: MonitoringService):
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
