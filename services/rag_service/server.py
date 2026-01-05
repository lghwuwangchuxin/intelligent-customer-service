"""RAG Service HTTP Server using FastAPI."""

import time
from typing import Optional, List, Dict, Any
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
from .service import RAGService, RetrieveConfig

logger = get_logger(__name__)


# Request/Response Models
class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, str] = {}
    chunk_id: Optional[str] = None
    source: Optional[str] = None


class RetrieveRequestConfig(BaseModel):
    """Retrieve configuration."""
    enable_query_transform: bool = True
    enable_rerank: bool = True
    hybrid_alpha: float = 0.5
    rerank_top_k: int = 5
    min_score: float = 0.0


class RetrieveRequest(BaseModel):
    """Request for document retrieval."""
    query: str
    top_k: int = 10
    knowledge_base_id: Optional[str] = None
    config: Optional[RetrieveRequestConfig] = None


class RetrieveResponse(BaseModel):
    """Response from retrieval."""
    documents: List[DocumentInfo]
    original_query: str
    transformed_query: Optional[str] = None
    latency_ms: int = 0


class IndexDocumentRequest(BaseModel):
    """Request to index a document."""
    content: str
    knowledge_base_id: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    source: Optional[str] = None


class IndexDocumentResponse(BaseModel):
    """Response from document indexing."""
    document_id: str
    success: bool
    chunks_created: int = 0
    error: Optional[str] = None


class DeleteDocumentResponse(BaseModel):
    """Response from document deletion."""
    success: bool
    error: Optional[str] = None


class ListDocumentsRequest(BaseModel):
    """Request to list documents."""
    knowledge_base_id: Optional[str] = None
    page: int = 1
    page_size: int = 20


class ListDocumentsResponse(BaseModel):
    """Response from listing documents."""
    documents: List[DocumentInfo]
    total: int
    page: int
    page_size: int


class StatsResponse(BaseModel):
    """Index statistics."""
    total_documents: int = 0
    total_chunks: int = 0
    index_size_bytes: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    details: Optional[Dict[str, Any]] = None


class RAGServer:
    """RAG Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: RAGService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["rag", "search", "retrieval"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="RAG Service",
            description="Retrieval Augmented Generation service with hybrid search",
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
        # Initialize RAG service
        await self.service.initialize()
        # Initialize ServiceRegistry for inter-service calls
        if hasattr(app.state, "nacos"):
            ServiceRegistry.initialize(app.state.nacos)
        logger.info("RAG Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        logger.info("RAG Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = await self.service.health_check()
                return HealthResponse(
                    status=health.get("status", "healthy"),
                    service="rag-service",
                    message="RAG Service is running",
                    details=health,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    service="rag-service",
                    message=str(e),
                )

        @self.app.post("/api/retrieve", response_model=RetrieveResponse)
        async def retrieve(request: RetrieveRequest):
            """Retrieve documents for a query."""
            start_time = time.time()

            try:
                # Build config
                cfg = request.config or RetrieveRequestConfig()
                config = RetrieveConfig(
                    top_k=request.top_k,
                    enable_query_transform=cfg.enable_query_transform,
                    enable_rerank=cfg.enable_rerank,
                    hybrid_alpha=cfg.hybrid_alpha,
                    rerank_top_k=cfg.rerank_top_k,
                )

                result = await self.service.retrieve(
                    query=request.query,
                    knowledge_base_id=request.knowledge_base_id,
                    config=config,
                )

                documents = [
                    DocumentInfo(
                        id=doc["id"],
                        content=doc["content"],
                        score=doc["score"],
                        metadata=doc.get("metadata", {}),
                        source=doc.get("source"),
                    )
                    for doc in result.documents
                ]

                return RetrieveResponse(
                    documents=documents,
                    original_query=request.query,
                    transformed_query=result.transformed_query,
                    latency_ms=result.latency_ms,
                )
            except Exception as e:
                logger.error(f"Retrieve failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/retrieve/stream")
        async def retrieve_stream(request: RetrieveRequest):
            """Retrieve documents with streaming response."""

            async def generate():
                try:
                    cfg = request.config or RetrieveRequestConfig()
                    config = RetrieveConfig(
                        top_k=request.top_k,
                        enable_query_transform=cfg.enable_query_transform,
                        enable_rerank=cfg.enable_rerank,
                        hybrid_alpha=cfg.hybrid_alpha,
                        rerank_top_k=cfg.rerank_top_k,
                    )

                    result = await self.service.retrieve(
                        query=request.query,
                        knowledge_base_id=request.knowledge_base_id,
                        config=config,
                    )

                    for doc in result.documents:
                        doc_info = DocumentInfo(
                            id=doc["id"],
                            content=doc["content"],
                            score=doc["score"],
                            metadata=doc.get("metadata", {}),
                        )
                        yield doc_info.model_dump_json() + "\n"

                except Exception as e:
                    logger.error(f"Stream retrieve failed: {e}")
                    yield '{"error": "' + str(e) + '"}\n'

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        @self.app.post("/api/documents", response_model=IndexDocumentResponse)
        async def index_document(request: IndexDocumentRequest):
            """Index a document."""
            try:
                result = await self.service.index_document(
                    content=request.content,
                    metadata=request.metadata,
                    knowledge_base_id=request.knowledge_base_id,
                )

                return IndexDocumentResponse(
                    document_id=result.document_id,
                    success=result.success,
                    chunks_created=getattr(result, 'chunks_created', 0),
                )
            except Exception as e:
                logger.error(f"Index document failed: {e}")
                return IndexDocumentResponse(
                    document_id="",
                    success=False,
                    error=str(e),
                )

        @self.app.delete("/api/documents/{document_id}", response_model=DeleteDocumentResponse)
        async def delete_document(
            document_id: str,
            knowledge_base_id: Optional[str] = Query(None),
        ):
            """Delete a document."""
            try:
                success = await self.service.delete_document(document_id)
                return DeleteDocumentResponse(success=success)
            except Exception as e:
                logger.error(f"Delete document failed: {e}")
                return DeleteDocumentResponse(success=False, error=str(e))

        @self.app.get("/api/documents", response_model=ListDocumentsResponse)
        async def list_documents(
            knowledge_base_id: Optional[str] = Query(None),
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
        ):
            """List documents."""
            try:
                result = await self.service.list_documents(
                    knowledge_base_id=knowledge_base_id,
                    page=page,
                    page_size=page_size,
                )

                documents = [
                    DocumentInfo(
                        id=doc["id"],
                        content=doc["content"],
                        score=0.0,
                        metadata=doc.get("metadata", {}),
                    )
                    for doc in result["documents"]
                ]

                return ListDocumentsResponse(
                    documents=documents,
                    total=result["total"],
                    page=page,
                    page_size=page_size,
                )
            except Exception as e:
                logger.error(f"List documents failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/stats", response_model=StatsResponse)
        async def get_stats(knowledge_base_id: Optional[str] = Query(None)):
            """Get index statistics."""
            try:
                stats = await self.service.get_stats(knowledge_base_id)
                return StatsResponse(
                    total_documents=stats.get("total_documents", 0),
                    total_chunks=stats.get("total_chunks", 0),
                    index_size_bytes=stats.get("index_size_bytes", 0),
                )
            except Exception as e:
                logger.error(f"Get stats failed: {e}")
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


def create_app(config: ServiceConfig, service: RAGService) -> FastAPI:
    """Create FastAPI application."""
    server = RAGServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: RAGService):
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
