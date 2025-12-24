"""
API Routes - Central routing module for the intelligent customer service system.

This module re-exports routers from domain-specific endpoint modules for
backward compatibility. New code should import directly from
app.api.endpoints or app.infrastructure.factory.

Architecture:
- app.api.endpoints.*: Domain-specific route handlers
- app.api.schemas.*: Request/Response Pydantic models
- app.infrastructure.factory: Service creation and management
"""

import logging

# Re-export routers from domain-specific modules
from app.api.endpoints import (
    chat_router,
    knowledge_router,
    agent_router,
    mcp_router,
    system_router,
    config_router,
)

# Re-export service initialization functions for backward compatibility
from app.infrastructure.factory import (
    async_init_services,
    get_services,
    get_registry,
)

# Re-export schemas for backward compatibility
from app.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    KnowledgeAddRequest,
    KnowledgeResponse,
    SearchRequest,
    SearchResponse,
    BatchUploadRequest,
    BatchUploadResponse,
    AgentChatRequest,
    AgentChatResponse,
    LangGraphChatRequest,
    LangGraphChatResponse,
    AgentCapabilities,
    ToolExecuteRequest,
    ToolInfo,
    SystemInfoResponse,
    ProviderInfo,
    ModelConfigRequest,
    ModelConfigResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
    ModelSwitchRequest,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Routers
    "chat_router",
    "knowledge_router",
    "agent_router",
    "mcp_router",
    "system_router",
    "config_router",
    # Service functions
    "async_init_services",
    "get_services",
    "get_registry",
    # Schemas (for backward compatibility)
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "KnowledgeAddRequest",
    "KnowledgeResponse",
    "SearchRequest",
    "SearchResponse",
    "BatchUploadRequest",
    "BatchUploadResponse",
    "AgentChatRequest",
    "AgentChatResponse",
    "LangGraphChatRequest",
    "LangGraphChatResponse",
    "AgentCapabilities",
    "ToolExecuteRequest",
    "ToolInfo",
    "SystemInfoResponse",
    "ProviderInfo",
    "ModelConfigRequest",
    "ModelConfigResponse",
    "ValidateConfigRequest",
    "ValidateConfigResponse",
    "ModelSwitchRequest",
]