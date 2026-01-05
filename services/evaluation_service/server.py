"""Evaluation Service HTTP Server using FastAPI."""

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
from .service import EvaluationService, EvaluationData

logger = get_logger(__name__)


# Request/Response Models
class EvaluationDataRequest(BaseModel):
    """Evaluation data item."""
    question: str
    answer: str
    contexts: List[str] = []
    ground_truth: Optional[str] = None


class SubmitEvaluationRequest(BaseModel):
    """Request to submit evaluation."""
    data: List[EvaluationDataRequest]
    metrics: Optional[List[str]] = None


class SubmitEvaluationResponse(BaseModel):
    """Response from submitting evaluation."""
    task_id: str
    status: str


class MetricResult(BaseModel):
    """Single metric result."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None


class GetEvaluationResultResponse(BaseModel):
    """Response for evaluation result."""
    task_id: str
    status: str
    results: List[MetricResult] = []
    error: Optional[str] = None
    completed_at: Optional[str] = None


class EvaluationSummary(BaseModel):
    """Evaluation summary."""
    task_id: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class ListEvaluationsResponse(BaseModel):
    """Response for listing evaluations."""
    evaluations: List[EvaluationSummary]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    details: Optional[Dict[str, Any]] = None


class EvaluationServer:
    """Evaluation Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: EvaluationService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["evaluation", "ragas", "async"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="Evaluation Service",
            description="Async evaluation service with RAGAS metrics",
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
        logger.info("Evaluation Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        await self.service.shutdown()
        logger.info("Evaluation Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = await self.service.health_check()
                return HealthResponse(
                    status=health.get("status", "healthy"),
                    service="evaluation-service",
                    message="Evaluation Service is running",
                    details=health,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    service="evaluation-service",
                    message=str(e),
                )

        @self.app.post("/api/evaluations", response_model=SubmitEvaluationResponse)
        async def submit_evaluation(request: SubmitEvaluationRequest):
            """Submit evaluation task."""
            try:
                # Convert request data
                data = [
                    EvaluationData(
                        question=item.question,
                        answer=item.answer,
                        contexts=item.contexts,
                        ground_truth=item.ground_truth,
                    )
                    for item in request.data
                ]

                result = await self.service.submit_evaluation(data, request.metrics)

                return SubmitEvaluationResponse(
                    task_id=result["task_id"],
                    status=result["status"],
                )
            except Exception as e:
                logger.error(f"Submit evaluation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/evaluations/{task_id}", response_model=GetEvaluationResultResponse)
        async def get_evaluation_result(task_id: str):
            """Get evaluation result."""
            try:
                result = await self.service.get_result(task_id)

                if not result:
                    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

                metrics = [
                    MetricResult(
                        metric_name=m["metric_name"],
                        score=m["score"],
                        details=m.get("details"),
                    )
                    for m in result.get("results", [])
                ]

                return GetEvaluationResultResponse(
                    task_id=result["task_id"],
                    status=result["status"],
                    results=metrics,
                    error=result.get("error"),
                    completed_at=result.get("completed_at"),
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get evaluation result failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/evaluations", response_model=ListEvaluationsResponse)
        async def list_evaluations(
            status: Optional[str] = Query(None),
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
        ):
            """List evaluations."""
            try:
                result = await self.service.list_evaluations(
                    status=status,
                    page=page,
                    page_size=page_size,
                )

                evaluations = [
                    EvaluationSummary(
                        task_id=e["task_id"],
                        status=e["status"],
                        created_at=e.get("created_at"),
                        completed_at=e.get("completed_at"),
                    )
                    for e in result["evaluations"]
                ]

                return ListEvaluationsResponse(
                    evaluations=evaluations,
                    total=result["total"],
                )
            except Exception as e:
                logger.error(f"List evaluations failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/evaluations/{task_id}")
        async def cancel_evaluation(task_id: str):
            """Cancel evaluation task."""
            try:
                success = await self.service.cancel_evaluation(task_id)
                return {"success": success, "task_id": task_id}
            except Exception as e:
                logger.error(f"Cancel evaluation failed: {e}")
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


def create_app(config: ServiceConfig, service: EvaluationService) -> FastAPI:
    """Create FastAPI application."""
    server = EvaluationServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: EvaluationService):
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
