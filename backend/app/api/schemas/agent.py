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
