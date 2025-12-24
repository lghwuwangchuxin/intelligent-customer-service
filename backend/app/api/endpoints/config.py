"""
Config Endpoints - Handles model configuration and switching.

Bounded Context: Core Infrastructure Domain
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ModelConfigRequest,
    ModelConfigResponse,
    ModelSwitchRequest,
    ProviderInfo,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from app.core.llm_manager import LLMManager, PROVIDER_CONFIGS
from app.infrastructure.factory import get_registry
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["Configuration"])


@router.get("/providers", response_model=List[ProviderInfo])
async def get_available_providers():
    """
    Get list of all available LLM providers.
    """
    providers = LLMManager.get_available_providers()
    return [ProviderInfo(**p) for p in providers]


@router.get("/providers/{provider_id}/models")
async def get_provider_models(provider_id: str):
    """
    Get available models for a specific provider.
    """
    models = LLMManager.get_provider_models(provider_id)
    if not models:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")

    return {"provider": provider_id, "models": models}


@router.post("/validate", response_model=ValidateConfigResponse)
async def validate_model_config(request: ValidateConfigRequest):
    """
    Validate model configuration before applying.
    """
    result = LLMManager.validate_provider_config(request.provider, request.api_key)
    return ValidateConfigResponse(**result)


@router.get("/current", response_model=ModelConfigResponse)
async def get_current_config():
    """
    Get current model configuration.
    """
    registry = get_registry()
    llm = registry.get("llm")

    return ModelConfigResponse(
        provider=llm.provider,
        model=llm.model,
        base_url=llm.base_url or "",
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        supports_tools=llm.supports_tool_calling,
        extended_thinking=getattr(llm, "extended_thinking", None),
    )


@router.post("/update", response_model=ModelConfigResponse)
async def update_model_config(request: ModelConfigRequest):
    """
    Update model configuration and reinitialize LLM.

    Note: This updates the in-memory configuration. For persistent changes,
    update environment variables or configuration files.
    """
    validation = LLMManager.validate_provider_config(request.provider, request.api_key)
    if not validation.get("valid"):
        raise HTTPException(status_code=400, detail=validation.get("error"))

    try:
        config = PROVIDER_CONFIGS.get(request.provider.lower())
        if not config:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

        base_url = request.base_url or config["base_url"]

        # Update LLM via registry
        registry = get_registry()
        new_llm = registry.update_llm(
            provider=request.provider,
            model=request.model,
            base_url=base_url,
            api_key=request.api_key,
            temperature=request.temperature if request.temperature is not None else settings.LLM_TEMPERATURE,
            max_tokens=request.max_tokens if request.max_tokens is not None else settings.LLM_MAX_TOKENS,
        )

        logger.info(f"Model configuration updated: {request.provider}/{request.model}")

        return ModelConfigResponse(
            provider=new_llm.provider,
            model=new_llm.model,
            base_url=new_llm.base_url or "",
            temperature=new_llm.temperature,
            max_tokens=new_llm.max_tokens,
            supports_tools=new_llm.supports_tool_calling,
            extended_thinking=getattr(new_llm, "extended_thinking", None),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_model_config(request: ModelConfigRequest):
    """
    Test model configuration by sending a simple prompt.
    """
    validation = LLMManager.validate_provider_config(request.provider, request.api_key)
    if not validation.get("valid"):
        raise HTTPException(status_code=400, detail=validation.get("error"))

    try:
        config = PROVIDER_CONFIGS.get(request.provider.lower())
        if not config:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

        base_url = request.base_url or config["base_url"]

        # Create temporary LLM manager for testing
        test_llm = LLMManager(
            provider=request.provider,
            model=request.model,
            base_url=base_url,
            api_key=request.api_key,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 100,
        )

        response = await test_llm.ainvoke([
            {"role": "user", "content": "Say 'Hello' in one word."}
        ])

        return {
            "success": True,
            "provider": request.provider,
            "model": request.model,
            "response": response[:100],
        }

    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return {
            "success": False,
            "provider": request.provider,
            "model": request.model,
            "error": str(e),
        }


# ============== Ollama Model Management ==============

@router.get("/ollama/models")
async def get_ollama_models():
    """
    Get list of locally available Ollama models.
    """
    result = await LLMManager.fetch_ollama_models()
    if not result.get("success"):
        raise HTTPException(
            status_code=503,
            detail=result.get("error", "Failed to fetch Ollama models")
        )
    return result


@router.get("/ollama/health")
async def check_ollama_health():
    """
    Check Ollama service health and version.
    """
    result = await LLMManager.check_ollama_health()
    return result


@router.get("/models/all")
async def get_all_models():
    """
    Get all available models from all providers.
    """
    result = await LLMManager.get_all_available_models(include_ollama=True)
    return result


@router.post("/switch")
async def switch_model(request: ModelSwitchRequest):
    """
    Switch to a different model with optional connection testing.
    """
    validation = LLMManager.validate_provider_config(request.provider, request.api_key)
    if not validation.get("valid"):
        raise HTTPException(status_code=400, detail=validation.get("error"))

    # Test connection if requested
    if request.test_before_switch:
        test_result = await LLMManager.test_provider_connection(
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
            timeout=15.0,
        )
        if not test_result.get("success"):
            return {
                "success": False,
                "error": f"Connection test failed: {test_result.get('error')}",
                "test_result": test_result,
            }

    try:
        config = PROVIDER_CONFIGS.get(request.provider.lower())
        if not config:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

        base_url = request.base_url or config["base_url"]

        # Update LLM via registry
        registry = get_registry()
        registry.update_llm(
            provider=request.provider,
            model=request.model,
            base_url=base_url,
            api_key=request.api_key,
        )

        logger.info(f"Model switched to: {request.provider}/{request.model}")

        return {
            "success": True,
            "provider": request.provider,
            "model": request.model,
            "base_url": base_url,
            "message": f"Successfully switched to {request.provider}/{request.model}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-connection")
async def test_current_connection():
    """
    Test connection to the currently configured model.
    """
    registry = get_registry()
    llm = registry.get("llm")
    result = await llm.test_connection(timeout=15.0)
    return result