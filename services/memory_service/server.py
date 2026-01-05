"""Memory Service HTTP Server using FastAPI."""

import time
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
from .service import MemoryService

logger = get_logger(__name__)


# ==================== Request/Response Models ====================

# --- Conversation Models ---
class CreateConversationRequest(BaseModel):
    """Request to create a conversation."""
    conversation_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddMessageRequest(BaseModel):
    """Request to add a message."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class MessageInfo(BaseModel):
    """Message information."""
    role: str
    content: str
    timestamp: Optional[str] = None


class ConversationInfo(BaseModel):
    """Conversation information."""
    conversation_id: str
    title: Optional[str] = None
    message_count: int = 0
    has_summary: bool = False
    created_at: str
    updated_at: str


class ConversationDetail(BaseModel):
    """Detailed conversation information."""
    conversation_id: str
    title: Optional[str] = None
    messages: List[Dict[str, str]] = []
    summary: Optional[str] = None
    message_count: int = 0
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ListConversationsResponse(BaseModel):
    """Response for listing conversations."""
    conversations: List[ConversationInfo]
    total: int
    page: int
    page_size: int


class AddMessageResponse(BaseModel):
    """Response for adding a message."""
    conversation_id: str
    message_count: int
    has_summary: bool
    summary_triggered: bool


class SummarizeResponse(BaseModel):
    """Response for summarization."""
    conversation_id: str
    summary: Optional[str] = None
    success: bool


# --- Long-term Memory Models ---
class SetPreferenceRequest(BaseModel):
    """Request to set user preference."""
    key: str
    value: Any
    category: str = "general"


class StoreEntityRequest(BaseModel):
    """Request to store an entity."""
    entity_id: str
    entity_type: str
    name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, str]] = Field(default_factory=list)


class StoreKnowledgeRequest(BaseModel):
    """Request to store knowledge."""
    topic: str
    content: str
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class SetSessionDataRequest(BaseModel):
    """Request to set session data."""
    key: str
    value: Any


class SearchRequest(BaseModel):
    """Generic search request."""
    query: str
    limit: int = 10
    tags: List[str] = Field(default_factory=list)


# --- Stats and Health ---
class MemoryStatsResponse(BaseModel):
    """Memory statistics response."""
    total_conversations: int
    total_messages: int
    total_memories: int
    namespaces: List[str]
    memory_types: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ==================== Server Implementation ====================

class MemoryServer:
    """Memory Service HTTP Server."""

    def __init__(self, config: ServiceConfig, service: MemoryService):
        self.config = config
        self.service = service

        # Create Nacos service config
        nacos_config = NacosServiceConfig.from_service_config(config)
        nacos_config.tags = ["memory", "conversation", "storage"]
        nacos_config.metadata = {"version": "1.0.0"}

        # Create FastAPI app with Nacos lifespan
        self.app = FastAPI(
            title="Memory Service",
            description="Short-term and long-term memory management service",
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
        # Initialize memory service
        await self.service.initialize()
        # Initialize ServiceRegistry for inter-service calls
        if hasattr(app.state, "nacos"):
            ServiceRegistry.initialize(app.state.nacos)
        logger.info("Memory Service started")

    async def _on_shutdown(self, app: FastAPI):
        """Additional shutdown logic."""
        await self.service.shutdown()
        logger.info("Memory Service shutting down")

    def _setup_routes(self):
        """Setup HTTP routes."""

        # ==================== Health & Info ====================

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = await self.service.health_check()
                return HealthResponse(
                    status=health.get("status", "healthy"),
                    service="memory-service",
                    message="Memory Service is running",
                    details=health,
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    service="memory-service",
                    message=str(e),
                )

        @self.app.get("/api/stats", response_model=MemoryStatsResponse)
        async def get_stats():
            """Get memory statistics."""
            try:
                stats = await self.service.get_stats()
                return MemoryStatsResponse(
                    total_conversations=stats.total_conversations,
                    total_messages=stats.total_messages,
                    total_memories=stats.total_memories,
                    namespaces=stats.namespaces,
                    memory_types=stats.memory_types,
                )
            except Exception as e:
                logger.error(f"Get stats failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== Conversation Endpoints ====================

        @self.app.post("/api/conversations", response_model=ConversationInfo)
        async def create_conversation(request: CreateConversationRequest):
            """Create a new conversation."""
            try:
                result = await self.service.create_conversation(
                    conversation_id=request.conversation_id,
                    title=request.title,
                    metadata=request.metadata,
                )
                return ConversationInfo(
                    conversation_id=result.conversation_id,
                    title=result.title,
                    message_count=result.message_count,
                    has_summary=result.has_summary,
                    created_at=result.created_at,
                    updated_at=result.updated_at,
                )
            except Exception as e:
                logger.error(f"Create conversation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/conversations", response_model=ListConversationsResponse)
        async def list_conversations(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
            user_id: Optional[str] = Query(None),
        ):
            """List conversations."""
            try:
                result = await self.service.list_conversations(
                    page=page,
                    page_size=page_size,
                    user_id=user_id,
                )
                return ListConversationsResponse(
                    conversations=[
                        ConversationInfo(**c) for c in result["conversations"]
                    ],
                    total=result["total"],
                    page=result["page"],
                    page_size=result["page_size"],
                )
            except Exception as e:
                logger.error(f"List conversations failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
        async def get_conversation(conversation_id: str):
            """Get conversation details."""
            try:
                result = await self.service.get_conversation(conversation_id)
                if not result:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                return ConversationDetail(
                    conversation_id=result["conversation_id"],
                    title=result.get("title"),
                    messages=result.get("messages", []),
                    summary=result.get("summary"),
                    message_count=result.get("message_count", 0),
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                    metadata=result.get("metadata", {}),
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get conversation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/conversations/{conversation_id}/messages", response_model=AddMessageResponse)
        async def add_message(conversation_id: str, request: AddMessageRequest):
            """Add a message to conversation."""
            try:
                result = await self.service.add_message(
                    conversation_id=conversation_id,
                    role=request.role,
                    content=request.content,
                )
                return AddMessageResponse(
                    conversation_id=result["conversation_id"],
                    message_count=result["message_count"],
                    has_summary=result["has_summary"],
                    summary_triggered=result["summary_triggered"],
                )
            except Exception as e:
                logger.error(f"Add message failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/conversations/{conversation_id}/context")
        async def get_context(
            conversation_id: str,
            max_messages: int = Query(10, ge=1, le=100),
        ):
            """Get conversation context for LLM."""
            try:
                messages = await self.service.get_context(conversation_id, max_messages)
                return {"conversation_id": conversation_id, "messages": messages}
            except Exception as e:
                logger.error(f"Get context failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/conversations/{conversation_id}/summarize", response_model=SummarizeResponse)
        async def summarize_conversation(
            conversation_id: str,
            force: bool = Query(False),
        ):
            """Summarize conversation."""
            try:
                summary = await self.service.summarize_conversation(conversation_id, force)
                return SummarizeResponse(
                    conversation_id=conversation_id,
                    summary=summary,
                    success=summary is not None,
                )
            except Exception as e:
                logger.error(f"Summarize conversation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/conversations/{conversation_id}")
        async def delete_conversation(conversation_id: str):
            """Delete conversation."""
            try:
                success = await self.service.delete_conversation(conversation_id)
                return {"success": success, "conversation_id": conversation_id}
            except Exception as e:
                logger.error(f"Delete conversation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/conversations/{conversation_id}/clear")
        async def clear_conversation(conversation_id: str):
            """Clear conversation messages but keep metadata."""
            try:
                success = await self.service.clear_conversation(conversation_id)
                return {"success": success, "conversation_id": conversation_id}
            except Exception as e:
                logger.error(f"Clear conversation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== User Preference Endpoints ====================

        @self.app.put("/api/users/{user_id}/preferences/{key}")
        async def set_user_preference(
            user_id: str,
            key: str,
            request: SetPreferenceRequest,
        ):
            """Set user preference."""
            try:
                success = await self.service.set_user_preference(
                    user_id=user_id,
                    key=key,
                    value=request.value,
                    category=request.category,
                )
                return {"success": success, "user_id": user_id, "key": key}
            except Exception as e:
                logger.error(f"Set preference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/users/{user_id}/preferences/{key}")
        async def get_user_preference(user_id: str, key: str):
            """Get user preference."""
            try:
                value = await self.service.get_user_preference(user_id, key)
                if value is None:
                    raise HTTPException(status_code=404, detail="Preference not found")
                return {"user_id": user_id, "key": key, "value": value}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get preference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/users/{user_id}/preferences")
        async def get_user_preferences(user_id: str):
            """Get all user preferences."""
            try:
                preferences = await self.service.get_user_preferences(user_id)
                return {"user_id": user_id, "preferences": preferences}
            except Exception as e:
                logger.error(f"Get preferences failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/users/{user_id}/context")
        async def get_user_context(user_id: str):
            """Get full user context."""
            try:
                context = await self.service.get_user_context(user_id)
                return {"user_id": user_id, "context": context}
            except Exception as e:
                logger.error(f"Get user context failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== Entity Endpoints ====================

        @self.app.post("/api/entities")
        async def store_entity(request: StoreEntityRequest):
            """Store an entity."""
            try:
                success = await self.service.store_entity(
                    entity_id=request.entity_id,
                    entity_type=request.entity_type,
                    name=request.name,
                    attributes=request.attributes,
                    relationships=request.relationships,
                )
                return {"success": success, "entity_id": request.entity_id}
            except Exception as e:
                logger.error(f"Store entity failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/entities/{entity_type}/{entity_id}")
        async def get_entity(entity_type: str, entity_id: str):
            """Get entity by type and ID."""
            try:
                entity = await self.service.get_entity(entity_id, entity_type)
                if not entity:
                    raise HTTPException(status_code=404, detail="Entity not found")
                return entity
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get entity failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/entities")
        async def search_entities(
            entity_type: Optional[str] = Query(None),
            query: Optional[str] = Query(None),
            limit: int = Query(10, ge=1, le=100),
        ):
            """Search entities."""
            try:
                entities = await self.service.search_entities(
                    entity_type=entity_type,
                    query=query,
                    limit=limit,
                )
                return {"entities": entities, "count": len(entities)}
            except Exception as e:
                logger.error(f"Search entities failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== Knowledge Endpoints ====================

        @self.app.post("/api/knowledge")
        async def store_knowledge(request: StoreKnowledgeRequest):
            """Store knowledge item."""
            try:
                success = await self.service.store_knowledge(
                    topic=request.topic,
                    content=request.content,
                    source=request.source,
                    tags=request.tags,
                )
                return {"success": success, "topic": request.topic}
            except Exception as e:
                logger.error(f"Store knowledge failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/knowledge")
        async def search_knowledge(
            query: str = Query(...),
            tags: Optional[str] = Query(None),
            limit: int = Query(10, ge=1, le=100),
        ):
            """Search knowledge items."""
            try:
                tag_list = tags.split(",") if tags else None
                items = await self.service.search_knowledge(
                    query=query,
                    tags=tag_list,
                    limit=limit,
                )
                return {"items": items, "count": len(items)}
            except Exception as e:
                logger.error(f"Search knowledge failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== Session Data Endpoints ====================

        @self.app.put("/api/sessions/{session_id}/data/{key}")
        async def set_session_data(
            session_id: str,
            key: str,
            request: SetSessionDataRequest,
        ):
            """Set session data."""
            try:
                success = await self.service.set_session_data(
                    session_id=session_id,
                    key=key,
                    value=request.value,
                )
                return {"success": success, "session_id": session_id, "key": key}
            except Exception as e:
                logger.error(f"Set session data failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/sessions/{session_id}/data/{key}")
        async def get_session_data(session_id: str, key: str):
            """Get session data."""
            try:
                value = await self.service.get_session_data(session_id, key)
                if value is None:
                    raise HTTPException(status_code=404, detail="Session data not found")
                return {"session_id": session_id, "key": key, "value": value}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get session data failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== Service Discovery ====================

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


def create_app(config: ServiceConfig, service: MemoryService) -> FastAPI:
    """Create FastAPI application."""
    server = MemoryServer(config, service)
    return server.app


async def serve(config: ServiceConfig, service: MemoryService):
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
