"""
Chat API Schemas - Request/Response models for chat endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    history: Optional[List[ChatMessage]] = Field(None, description="Conversation history")
    use_rag: bool = Field(True, description="Whether to use RAG")
    stream: bool = Field(False, description="Whether to stream response")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Assistant response")
    conversation_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None