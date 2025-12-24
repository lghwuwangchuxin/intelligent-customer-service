"""
Knowledge API Schemas - Request/Response models for knowledge base endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeAddRequest(BaseModel):
    """Request to add text knowledge."""
    text: str = Field(..., description="Text content to add")
    title: Optional[str] = Field(None, description="Document title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class KnowledgeResponse(BaseModel):
    """Knowledge operation response."""
    success: bool
    message: str
    num_documents: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Knowledge search request."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results")


class SearchResponse(BaseModel):
    """Knowledge search response."""
    results: List[Dict[str, Any]]
    query: str


class BatchUploadRequest(BaseModel):
    """Request to create a batch upload task."""
    files: List[Dict[str, Any]] = Field(
        ...,
        description="List of file info dicts with filename and file_size"
    )


class BatchUploadResponse(BaseModel):
    """Batch upload task response."""
    task_id: str
    total_files: int
    files: List[Dict[str, Any]]