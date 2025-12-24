"""
Config API Schemas - Request/Response models for configuration endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """LLM provider information."""
    id: str
    name: str
    models: List[str]
    requires_api_key: bool
    base_url: str


class ModelConfigRequest(BaseModel):
    """Request to update model configuration."""
    provider: str = Field(..., description="LLM provider ID")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key (required for some providers)")
    base_url: Optional[str] = Field(None, description="Custom base URL (optional)")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Temperature (0-2)")
    max_tokens: Optional[int] = Field(None, ge=1, le=128000, description="Max tokens")


class ModelConfigResponse(BaseModel):
    """Response with current model configuration."""
    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    supports_tools: bool
    extended_thinking: Optional[bool] = None


class ValidateConfigRequest(BaseModel):
    """Request to validate model configuration."""
    provider: str
    api_key: Optional[str] = None


class ValidateConfigResponse(BaseModel):
    """Response for configuration validation."""
    valid: bool
    error: Optional[str] = None
    provider: Optional[str] = None
    name: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    """Request to switch model."""
    provider: str = Field(..., description="Provider ID (ollama, openai, etc.)")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key (if required)")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    test_before_switch: bool = Field(True, description="Test connection before switching")