"""
Agent API Schemas - Request/Response models for agent endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .chat import ChatMessage


class AgentChatRequest(BaseModel):
    """Agent chat request model."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for memory")
    user_id: Optional[str] = Field(None, description="User ID for personalized long-term memory")
    history: Optional[List[ChatMessage]] = Field(None, description="Conversation history")
    stream: bool = Field(False, description="Whether to stream response")


class AgentChatResponse(BaseModel):
    """Agent chat response model."""
    response: str = Field(..., description="Agent response")
    conversation_id: Optional[str] = None
    thoughts: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    iterations: int = 0
    error: Optional[str] = None


class LangGraphChatRequest(BaseModel):
    """LangGraph agent chat request model."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for memory")
    user_id: Optional[str] = Field(None, description="User ID for personalized long-term memory")
    history: Optional[List[ChatMessage]] = Field(None, description="Conversation history")
    stream: bool = Field(False, description="Whether to stream response")
    enable_planning: bool = Field(True, description="Enable task planning for complex queries")


class LangGraphChatResponse(BaseModel):
    """LangGraph agent chat response model."""
    response: str = Field(..., description="Agent response")
    conversation_id: Optional[str] = None
    plan: Optional[List[Dict[str, Any]]] = None
    thoughts: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    iterations: int = 0
    parallel_executions: int = 0
    error_recoveries: int = 0
    error: Optional[str] = None


class AgentCapabilities(BaseModel):
    """Agent capabilities response."""
    react_agent: bool
    langgraph_agent: bool
    planning_enabled: bool
    parallel_tools_enabled: bool
    error_recovery_enabled: bool
    max_iterations: int
    available_tools: int


class ToolExecuteRequest(BaseModel):
    """Tool execution request."""
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str
    parameters: List[Dict[str, Any]]


# ============== Conversation History Schemas ==============

class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""
    conversation_id: str
    title: Optional[str] = None
    message_count: int = 0
    interaction_count: int = 0
    has_summary: bool = False
    created_at: str
    updated_at: str
    last_message: Optional[Dict[str, Any]] = None


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: List[ConversationSummary]
    total: int
    limit: int
    offset: int


class InteractionDetail(BaseModel):
    """Detailed information about an agent interaction."""
    interaction_id: str
    timestamp: str
    question: str
    response: str
    thoughts: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


class ConversationDetail(BaseModel):
    """Detailed conversation with full history."""
    conversation_id: str
    title: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    interactions: List[InteractionDetail] = Field(default_factory=list)
    summary: Optional[str] = None
    message_count: int = 0
    interaction_count: int = 0
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateConversationRequest(BaseModel):
    """Request to update conversation properties."""
    title: Optional[str] = Field(None, description="New title for the conversation")


class ConversationExportResponse(BaseModel):
    """Response for exporting a conversation."""
    conversation_id: str
    data: Dict[str, Any]
    exported_at: str


# ============== Long-term Memory Schemas ==============

class UserPreferenceRequest(BaseModel):
    """Request to set a user preference."""
    key: str = Field(..., description="Preference key")
    value: Any = Field(..., description="Preference value")
    category: str = Field(default="general", description="Preference category")


class UserPreferenceResponse(BaseModel):
    """User preference response."""
    user_id: str
    key: str
    value: Any
    category: str
    updated_at: str


class EntityRequest(BaseModel):
    """Request to store an entity."""
    entity_id: str = Field(..., description="Unique entity identifier")
    entity_type: str = Field(..., description="Entity type (person, product, etc.)")
    name: str = Field(..., description="Entity name")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, str]] = Field(default_factory=list)


class EntityResponse(BaseModel):
    """Entity response."""
    entity_id: str
    entity_type: str
    name: str
    attributes: Dict[str, Any]
    relationships: List[Dict[str, str]]


class KnowledgeRequest(BaseModel):
    """Request to store knowledge."""
    topic: str = Field(..., description="Knowledge topic")
    content: str = Field(..., description="Knowledge content")
    source: Optional[str] = Field(None, description="Knowledge source")
    tags: List[str] = Field(default_factory=list)


class KnowledgeResponse(BaseModel):
    """Knowledge response."""
    key: str
    topic: str
    content: str
    source: Optional[str]
    tags: List[str]
    relevance_score: Optional[float] = None


class MemorySearchRequest(BaseModel):
    """Request to search memory."""
    query: str = Field(..., description="Search query")
    memory_type: Optional[str] = Field(None, description="Filter by memory type")
    limit: int = Field(default=10, ge=1, le=100)


class MemorySearchResponse(BaseModel):
    """Memory search response."""
    results: List[Dict[str, Any]]
    total: int
    query: str


class MemoryStatsResponse(BaseModel):
    """Memory store statistics."""
    store_type: str
    namespace_count: int
    total_items: int
    namespaces: List[str]
