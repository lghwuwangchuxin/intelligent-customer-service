"""
System Endpoints - Handles system info and health checks.

Bounded Context: Core Infrastructure Domain
"""

import datetime
import logging

from fastapi import APIRouter

from app.api.schemas import SystemInfoResponse
from app.infrastructure.factory import get_registry
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    """
    Get system information and status.
    """
    registry = get_registry()

    return SystemInfoResponse(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        llm_info=registry.get("llm").get_info(),
        status="running",
    )


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": str(datetime.datetime.now()),
    }


@router.get("/config")
async def get_config():
    """
    Get current configuration (non-sensitive).
    """
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "milvus_collection": settings.MILVUS_COLLECTION,
        "rag_top_k": settings.RAG_TOP_K,
        "rag_chunk_size": settings.RAG_CHUNK_SIZE,
        "agent_enabled": settings.AGENT_ENABLED,
        "agent_max_iterations": settings.AGENT_MAX_ITERATIONS,
    }