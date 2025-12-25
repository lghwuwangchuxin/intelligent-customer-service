"""
Agent Endpoints - Handles ReAct and LangGraph agent requests.

Bounded Context: Agent Domain
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agent import LANGGRAPH_AVAILABLE
from app.api.schemas import (
    AgentCapabilities,
    AgentChatRequest,
    AgentChatResponse,
    LangGraphChatRequest,
    LangGraphChatResponse,
    # Conversation History
    ConversationSummary,
    ConversationListResponse,
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
from app.infrastructure.factory import get_registry
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Chat with the ReAct agent.

    The agent can use tools to search knowledge base, browse web, execute code, etc.
    """
    logger.info(
        f"[API] /agent/chat - message: {request.message[:50]}..., "
        f"conversation_id: {request.conversation_id}, user_id: {request.user_id}, stream: {request.stream}"
    )

    if not settings.AGENT_ENABLED:
        logger.warning("[API] Agent mode is disabled")
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()

    try:
        if request.stream:
            async def generate():
                try:
                    history = [
                        {"role": m.role, "content": m.content}
                        for m in (request.history or [])
                    ]
                    async for event in registry.get("agent").stream(
                        question=request.message,
                        conversation_id=request.conversation_id,
                        history=history,
                    ):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            logger.info("[API] Agent processing request")
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            result = await registry.get("agent").run(
                question=request.message,
                conversation_id=request.conversation_id,
                history=history,
            )

            logger.info(
                f"[API] Agent completed - iterations: {result.get('iterations', 0)}, "
                f"tool_calls: {len(result.get('tool_calls', []))}, "
                f"response_length: {len(result.get('response', ''))}"
            )
            return AgentChatResponse(
                response=result["response"],
                conversation_id=result.get("conversation_id"),
                thoughts=result.get("thoughts"),
                tool_calls=result.get("tool_calls"),
                iterations=result.get("iterations", 0),
                error=result.get("error"),
            )

    except Exception as e:
        logger.error(f"[API] Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """
    Stream agent response with intermediate steps (thoughts, actions, observations).
    """
    if not settings.AGENT_ENABLED:
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()

    async def generate():
        try:
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            async for event in registry.get("agent").stream(
                question=request.message,
                conversation_id=request.conversation_id,
                history=history,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.get("/memory/{conversation_id}")
async def get_agent_memory(conversation_id: str):
    """
    Get agent memory for a conversation.
    """
    registry = get_registry()

    memory = registry.get("memory_manager").get_memory(conversation_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "messages": memory.get_recent_messages(20),
        "summary": memory.summary,
        "message_count": len(memory.messages),
    }


@router.delete("/memory/{conversation_id}")
async def clear_agent_memory(conversation_id: str):
    """
    Clear agent memory for a conversation.
    """
    registry = get_registry()

    success = registry.get("memory_manager").delete_memory(conversation_id)
    return {"success": success, "conversation_id": conversation_id}


# ============== LangGraph Agent Endpoints ==============

@router.get("/capabilities", response_model=AgentCapabilities)
async def get_agent_capabilities():
    """
    Get agent capabilities and status.
    """
    registry = get_registry()

    return AgentCapabilities(
        react_agent=registry.get("agent") is not None,
        langgraph_agent=registry.get("langgraph_agent") is not None,
        planning_enabled=LANGGRAPH_AVAILABLE and registry.get("langgraph_agent") is not None,
        parallel_tools_enabled=LANGGRAPH_AVAILABLE and registry.get("langgraph_agent") is not None,
        error_recovery_enabled=LANGGRAPH_AVAILABLE and registry.get("langgraph_agent") is not None,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        available_tools=len(registry.get("tool_registry").get_all()),
    )


@router.post("/langgraph/chat", response_model=LangGraphChatResponse)
async def langgraph_agent_chat(request: LangGraphChatRequest):
    """
    Chat with the LangGraph agent.

    The LangGraph agent provides advanced features:
    - Task planning for complex queries
    - Parallel tool execution
    - Intelligent error recovery
    - State-based workflow management
    - Long-term memory (when user_id is provided)
    """
    logger.info(
        f"[API] /langgraph/chat - message: {request.message[:50]}..., "
        f"conversation_id: {request.conversation_id}, user_id: {request.user_id}, stream: {request.stream}"
    )

    if not settings.AGENT_ENABLED:
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()
    langgraph_agent = registry.get("langgraph_agent")

    if not langgraph_agent:
        # Fallback to ReAct agent
        logger.warning("LangGraph agent not available, falling back to ReAct agent")
        history = [
            {"role": m.role, "content": m.content}
            for m in (request.history or [])
        ]
        result = await registry.get("agent").run(
            question=request.message,
            conversation_id=request.conversation_id,
            history=history,
        )
        return LangGraphChatResponse(
            response=result["response"],
            conversation_id=result.get("conversation_id"),
            thoughts=result.get("thoughts"),
            tool_calls=result.get("tool_calls"),
            iterations=result.get("iterations", 0),
            error=result.get("error"),
        )

    try:
        if request.stream:
            async def generate():
                try:
                    history = [
                        {"role": m.role, "content": m.content}
                        for m in (request.history or [])
                    ]
                    async for event in langgraph_agent.stream(
                        question=request.message,
                        conversation_id=request.conversation_id,
                        user_id=request.user_id,
                        history=history,
                    ):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            result = await langgraph_agent.run(
                question=request.message,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                history=history,
            )

            return LangGraphChatResponse(
                response=result.get("response", ""),
                conversation_id=result.get("conversation_id"),
                plan=result.get("plan"),
                thoughts=result.get("thoughts"),
                tool_calls=result.get("tool_calls"),
                iterations=result.get("iterations", 0),
                parallel_executions=result.get("parallel_executions", 0),
                error_recoveries=result.get("error_recoveries", 0),
                error=result.get("error"),
            )

    except Exception as e:
        logger.error(f"LangGraph agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/langgraph/chat/stream")
async def langgraph_agent_chat_stream(request: LangGraphChatRequest):
    """
    Stream LangGraph agent response with intermediate steps.
    """
    if not settings.AGENT_ENABLED:
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()
    langgraph_agent = registry.get("langgraph_agent")

    if not langgraph_agent:
        raise HTTPException(
            status_code=503,
            detail="LangGraph agent is not available. Use /agent/chat/stream instead."
        )

    async def generate():
        try:
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            async for event in langgraph_agent.stream(
                question=request.message,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                history=history,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"LangGraph stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# ============== Conversation History Endpoints ==============

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of conversations to return"),
    offset: int = Query(default=0, ge=0, description="Number of conversations to skip"),
    sort_by: str = Query(default="updated_at", description="Field to sort by (created_at, updated_at)"),
    descending: bool = Query(default=True, description="Sort in descending order"),
):
    """
    List all conversations with pagination.

    Returns a list of conversation summaries that can be displayed in a sidebar or list view.
    """
    registry = get_registry()
    memory_manager = registry.get("memory_manager")

    conversations = memory_manager.list_conversations(
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        descending=descending,
    )

    # Convert to response model
    conversation_summaries = [
        ConversationSummary(**conv) for conv in conversations
    ]

    return ConversationListResponse(
        conversations=conversation_summaries,
        total=len(memory_manager.get_all_conversation_ids()),
        limit=limit,
        offset=offset,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(conversation_id: str):
    """
    Get detailed conversation information including all messages and interactions.

    This endpoint returns the full conversation history with:
    - All messages (user and assistant)
    - All agent interactions (with thoughts, tool calls, etc.)
    - Conversation summary (if available)
    - Metadata
    """
    registry = get_registry()
    memory_manager = registry.get("memory_manager")

    detail = memory_manager.get_conversation_detail(conversation_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetail(**detail)


@router.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: UpdateConversationRequest):
    """
    Update conversation properties (e.g., title).
    """
    registry = get_registry()
    memory_manager = registry.get("memory_manager")

    if request.title:
        success = memory_manager.update_conversation_title(conversation_id, request.title)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "conversation_id": conversation_id}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its history.
    """
    registry = get_registry()
    memory_manager = registry.get("memory_manager")

    success = memory_manager.delete_memory(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "conversation_id": conversation_id}


@router.post("/conversations/{conversation_id}/export", response_model=ConversationExportResponse)
async def export_conversation(conversation_id: str):
    """
    Export a conversation as JSON data.

    This can be used for:
    - Backing up conversations
    - Sharing conversations
    - Importing conversations later
    """
    registry = get_registry()
    memory_manager = registry.get("memory_manager")

    data = memory_manager.export_memory(conversation_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationExportResponse(
        conversation_id=conversation_id,
        data=data,
        exported_at=datetime.utcnow().isoformat(),
    )


@router.post("/conversations/import")
async def import_conversation(data: dict):
    """
    Import a conversation from exported JSON data.
    """
    registry = get_registry()
    memory_manager = registry.get("memory_manager")

    try:
        memory = memory_manager.import_memory(data)
        return {
            "success": True,
            "conversation_id": memory.conversation_id,
            "message_count": len(memory.messages),
            "interaction_count": len(memory.interactions),
        }
    except Exception as e:
        logger.error(f"Import conversation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid conversation data: {str(e)}")


# ============== Long-term Memory Endpoints ==============

@router.get("/store/stats", response_model=MemoryStatsResponse)
async def get_memory_store_stats():
    """
    Get long-term memory store statistics.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    stats = store_manager.get_stats()
    return MemoryStatsResponse(
        store_type=stats.get("type", "unknown"),
        namespace_count=stats.get("namespace_count", 0),
        total_items=stats.get("total_items", 0),
        namespaces=stats.get("namespaces", []),
    )


@router.post("/store/users/{user_id}/preferences", response_model=UserPreferenceResponse)
async def set_user_preference(user_id: str, request: UserPreferenceRequest):
    """
    Set a user preference in long-term memory.

    User preferences are stored persistently and can be used to personalize
    agent responses and behavior.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    await store_manager.set_user_preference(
        user_id=user_id,
        key=request.key,
        value=request.value,
        category=request.category,
    )

    return UserPreferenceResponse(
        user_id=user_id,
        key=request.key,
        value=request.value,
        category=request.category,
        updated_at=datetime.utcnow().isoformat(),
    )


@router.get("/store/users/{user_id}/preferences")
async def get_user_preferences(
    user_id: str,
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """
    Get all preferences for a user.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    prefs = await store_manager.get_all_user_preferences(user_id, category)
    return {
        "user_id": user_id,
        "preferences": prefs,
        "category": category,
    }


@router.get("/store/users/{user_id}/preferences/{key}")
async def get_user_preference(user_id: str, key: str):
    """
    Get a specific user preference.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    value = await store_manager.get_user_preference(user_id, key)
    if value is None:
        raise HTTPException(status_code=404, detail="Preference not found")

    return {
        "user_id": user_id,
        "key": key,
        "value": value,
    }


@router.post("/store/entities", response_model=EntityResponse)
async def store_entity(request: EntityRequest):
    """
    Store an entity in long-term memory.

    Entities can be people, products, organizations, events, etc.
    They are stored globally and can be retrieved by type and ID.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    await store_manager.store_entity(
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        name=request.name,
        attributes=request.attributes,
        relationships=request.relationships,
    )

    return EntityResponse(
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        name=request.name,
        attributes=request.attributes,
        relationships=request.relationships,
    )


@router.get("/store/entities/{entity_type}/{entity_id}")
async def get_entity(entity_type: str, entity_id: str):
    """
    Get an entity by type and ID.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    entity = await store_manager.get_entity(entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    return entity


@router.get("/store/entities/search")
async def search_entities(
    query: str = Query(..., description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(default=10, ge=1, le=100),
):
    """
    Search for entities by query.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    entities = await store_manager.search_entities(query, entity_type, limit)
    return {
        "query": query,
        "entity_type": entity_type,
        "results": entities,
        "count": len(entities),
    }


@router.post("/store/knowledge", response_model=KnowledgeResponse)
async def store_knowledge(request: KnowledgeRequest):
    """
    Store a knowledge item in long-term memory.

    Knowledge items are facts, information, or learned content that
    can be retrieved later to enhance agent responses.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    key = await store_manager.store_knowledge(
        topic=request.topic,
        content=request.content,
        source=request.source,
        tags=request.tags,
    )

    return KnowledgeResponse(
        key=key,
        topic=request.topic,
        content=request.content,
        source=request.source,
        tags=request.tags,
    )


@router.post("/store/knowledge/search", response_model=MemorySearchResponse)
async def search_knowledge(request: MemorySearchRequest):
    """
    Search for knowledge items.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    results = await store_manager.search_knowledge(
        query=request.query,
        limit=request.limit,
    )

    return MemorySearchResponse(
        results=results,
        total=len(results),
        query=request.query,
    )


@router.delete("/store/users/{user_id}")
async def clear_user_memory(user_id: str):
    """
    Clear all memory for a user (preferences, entities, etc.).
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    count = await store_manager.store.clear_namespace(f"user:{user_id}")
    return {
        "success": True,
        "user_id": user_id,
        "deleted_items": count,
    }


@router.get("/store/users/{user_id}/context")
async def get_user_context(
    user_id: str,
    include_preferences: bool = Query(default=True),
    include_entities: bool = Query(default=True),
):
    """
    Get the full context for a user.

    This includes preferences, entities, and other relevant information
    that can be used to personalize agent interactions.
    """
    registry = get_registry()
    store_manager = registry.get("store_manager")

    if not store_manager:
        raise HTTPException(status_code=503, detail="Memory store is not available")

    context = await store_manager.get_context_for_user(
        user_id,
        include_preferences=include_preferences,
        include_entities=include_entities,
    )

    return context