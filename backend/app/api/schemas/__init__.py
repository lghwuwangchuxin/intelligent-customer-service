"""
API Schemas Module - Request/Response models for API endpoints.

Centralized location for all Pydantic models used in API endpoints.
"""

from .chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from .knowledge import (
    KnowledgeAddRequest,
    KnowledgeResponse,
    SearchRequest,
    SearchResponse,
    BatchUploadRequest,
    BatchUploadResponse,
)
from .agent import (
    AgentChatRequest,
    AgentChatResponse,
    LangGraphChatRequest,
    LangGraphChatResponse,
    AgentCapabilities,
    ToolExecuteRequest,
    ToolInfo,
    # Conversation History
    ConversationSummary,
    ConversationListResponse,
    InteractionDetail,
    ConversationDetail,
    UpdateConversationRequest,
    ConversationExportResponse,
    # Long-term Memory
    UserPreferenceRequest,
    UserPreferenceResponse,
    EntityRequest,
    EntityResponse,
    KnowledgeRequest,
    KnowledgeResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStatsResponse,
)
from .system import (
    SystemInfoResponse,
    HealthResponse,
)
from .config import (
    ProviderInfo,
    ModelConfigRequest,
    ModelConfigResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
    ModelSwitchRequest,
)

__all__ = [
    # Chat
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    # Knowledge
    "KnowledgeAddRequest",
    "KnowledgeResponse",
    "SearchRequest",
    "SearchResponse",
    "BatchUploadRequest",
    "BatchUploadResponse",
    # Agent
    "AgentChatRequest",
    "AgentChatResponse",
    "LangGraphChatRequest",
    "LangGraphChatResponse",
    "AgentCapabilities",
    "ToolExecuteRequest",
    "ToolInfo",
    # Conversation History
    "ConversationSummary",
    "ConversationListResponse",
    "InteractionDetail",
    "ConversationDetail",
    "UpdateConversationRequest",
    "ConversationExportResponse",
    # Long-term Memory
    "UserPreferenceRequest",
    "UserPreferenceResponse",
    "EntityRequest",
    "EntityResponse",
    "KnowledgeRequest",
    "KnowledgeResponse",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemoryStatsResponse",
    # System
    "SystemInfoResponse",
    "HealthResponse",
    # Config
    "ProviderInfo",
    "ModelConfigRequest",
    "ModelConfigResponse",
    "ValidateConfigRequest",
    "ValidateConfigResponse",
    "ModelSwitchRequest",
]