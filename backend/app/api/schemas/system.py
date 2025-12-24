"""
System API Schemas - Request/Response models for system endpoints.
"""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class SystemInfoResponse(BaseModel):
    """System information response."""
    app_name: str
    version: str
    llm_info: Dict[str, Any]
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime